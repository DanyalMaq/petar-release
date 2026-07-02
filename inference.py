"""
Example command to run
python inference.py \
  --pet /input/pet.nii.gz \
  --ct /input/ct.nii.gz \
  --segmentation /path/to/segmentation.nii.gz \
  --segment-name liver

You may need to change paths to the nifits/model weights/imports depending on your setup
"""

import argparse
import os
import sys
import json
import random
import base64
import gzip
import traceback
import warnings
from typing import Optional

import numpy as np
import nibabel as nib
import torch
from nilearn.image import resample_img
import monai.transforms as mtf
from monai.data.meta_obj import set_track_meta
from transformers import AutoTokenizer

from model.language_model import *       # registers LamedPhi3ForCausalLM

from utils import crop_image_around_lesion, focal_crop_around_mask

warnings.filterwarnings("ignore", category=UserWarning, message="Casting data from int16 to float32")

SAVE_DIR = "/data"
DEFAULT_CT_PATH = "/input/ct.nii.gz"
DEFAULT_PET_PATH = "/input/pet.nii.gz"

Seg_template = [
    "What organ is shown in the masked region?",
    "Which organ is segmented in this image?",
    "What structure is highlighted by the mask?",
    "Identify the organ marked by the mask.",
    "What anatomical region does the mask correspond to?",
    "Which body part is covered by the highlighted area?",
    "What is the masked area indicating?",
    "What organ lies within the masked region?",
    "Can you name the organ selected by the segmentation mask?",
    "What organ is enclosed by the mask?",
    "Please identify the organ highlighted in the image.",
    "What does the mask represent in this scan?",
    "Tell me which organ is being segmented.",
    "Which anatomical structure has been masked?",
    "What region of the body does the mask refer to?"
]


# -----------------------------
# Model config
# -----------------------------
MODEL_DIR         = "/petar"
MODALITY_KEYS     = ["pet", "ct", "mask", "pet_focal", "ct_focal", "mask_focal"]
PROJ_OUT_NUM      = 256                      # must match the checkpoint
MAX_LENGTH        = 768
MAX_NEW_TOKENS    = 256
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image-token prefix the model expects
IMAGE_TOKENS = "<im_patch>" * PROJ_OUT_NUM

# MONAI: convert .npy arrays to float tensors (no augmentation — inference mode)
set_track_meta(False)
infer_transform = mtf.Compose([
    mtf.ToTensord(keys=MODALITY_KEYS, dtype=torch.float)
])

# -----------------------------
# Load model + tokenizer once at startup
# -----------------------------
print(f"Loading model from {MODEL_DIR} …", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    model_max_length=MAX_LENGTH,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True,
)
model = LamedPhi3ForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    device_map="auto"
).to(DEVICE)
model.eval()
print("Model loaded ✓", file=sys.stderr)

# -----------------------------
# NIfTI processing settings
# -----------------------------
TARGET_AFFINE = np.diag([3, 3, 3])
CROP_XY = 200
CROP_Z = 350
SUV_THRESHOLD = 5
CT_THRESHOLD = 1000

# -----------------------------
# MONAI transforms
# -----------------------------
transforms_global = mtf.Compose([
    mtf.CropForegroundd(keys=["pet", "mask", "ct"], source_key="pet"),
    mtf.Resized(
        keys=["pet", "mask", "ct"],
        spatial_size=[32, 256, 256],
        mode=["trilinear", "nearest", "trilinear"]
    )
])

transforms_focal = mtf.Compose([
    mtf.CropForegroundd(keys=["pet_focal", "mask_focal", "ct_focal"], source_key="pet_focal"),
    mtf.Resized(
        keys=["pet_focal", "mask_focal", "ct_focal"],
        spatial_size=[32, 256, 256],
        mode=["trilinear", "nearest", "trilinear"]
    )
])


# -----------------------------
# Slicer decode / save helpers
# -----------------------------
def b64_to_numpy(b64_str: str, shape_kji, dtype_str: str) -> np.ndarray:
    compressed = base64.b64decode(b64_str)
    raw = gzip.decompress(compressed)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(tuple(shape_kji))
    return arr


def save_nifti_from_slicer_kji(arr_kji: np.ndarray, ijk_to_ras_4x4, out_path: str):
    arr_ijk = np.transpose(arr_kji, (2, 1, 0))
    affine = np.array(ijk_to_ras_4x4, dtype=np.float64)
    img = nib.Nifti1Image(arr_ijk, affine)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=0)
    nib.save(img, out_path)


# -----------------------------
# NIfTI processing helpers
# -----------------------------
def find_z_plane_above_threshold(threshold, data):
    midpoint_x = data.shape[0] // 2
    midpoint_y = data.shape[1] // 2

    x_start = max(midpoint_x - 50, 0)
    x_end   = min(midpoint_x + 50, data.shape[0])
    y_start = max(midpoint_y - 150, 0)
    y_end   = min(midpoint_y + 50, data.shape[1])

    for z in reversed(range(data.shape[2])):
        if np.max(data[x_start:x_end, y_start:y_end, z]) > threshold:
            return z
    return None


def compute_crop_offset_from_head(pet_img, ct_img):
    z_plane = find_z_plane_above_threshold(SUV_THRESHOLD, pet_img.get_fdata())
    if z_plane is not None:
        return int(pet_img.shape[2] - z_plane)

    z_plane = find_z_plane_above_threshold(CT_THRESHOLD, ct_img.get_fdata())
    if z_plane is not None:
        return int(ct_img.shape[2] - z_plane)

    return 0


def crop_z_axis(img, crop_offset):
    if crop_offset == 0:
        return img
    return img.slicer[:, :, :-crop_offset]


def crop_center_with_offset(img, z_offset=0):
    x, y, z = img.shape
    crop_x = min(CROP_XY, x)
    crop_y = min(CROP_XY, y)
    crop_z = min(CROP_Z, z)

    start_x = max((x - crop_x) // 2, 0)
    start_y = max((y - crop_y) // 2, 0)
    start_z = max(z - crop_z - z_offset, 0)

    return img.slicer[start_x:start_x + crop_x, start_y:start_y + crop_y, start_z:start_z + crop_z]


def pad_image_symmetrically(img, target_size=200, fill_value=0):
    data = img.get_fdata()
    padding = []
    for dim in data.shape[:2]:
        if dim < target_size:
            total_pad = target_size - dim
            pad_before = total_pad // 2
            pad_after  = total_pad - pad_before
            padding.append((pad_before, pad_after))
        else:
            padding.append((0, 0))
    padding.append((0, 0))
    padded_data = np.pad(data, padding, mode="constant", constant_values=fill_value)
    return nib.Nifti1Image(padded_data, img.affine, img.header)


def process_niftis(pet_path: str, ct_path: str, mask_path: str, out_dir: str):
    """
    Stage 1: resample + spatial crop → save pet_processed, ct_processed, mask_processed.nii.gz
    Returns paths to the three processed files plus diagnostics.
    """
    pet_img  = nib.load(pet_path)
    ct_img   = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    # Head-detection z-crop
    crop_offset = max(compute_crop_offset_from_head(pet_img, ct_img) - 1, 0)
    pet_img  = crop_z_axis(pet_img,  crop_offset)
    ct_img   = crop_z_axis(ct_img,   crop_offset)
    mask_img = crop_z_axis(mask_img, crop_offset)

    # Resample to 3×3×3 mm (nearest-neighbour for mask to preserve labels)
    pet_img  = resample_img(pet_img,  target_affine=TARGET_AFFINE, interpolation="linear",  force_resample=True, copy_header=True)
    ct_img   = resample_img(ct_img,   target_affine=TARGET_AFFINE, interpolation="linear",  force_resample=True, copy_header=True)
    mask_img = resample_img(mask_img, target_affine=TARGET_AFFINE, interpolation="nearest", force_resample=True, copy_header=True)

    # Final centre crop
    pet_img  = crop_center_with_offset(pet_img)
    ct_img   = crop_center_with_offset(ct_img)
    mask_img = crop_center_with_offset(mask_img)

    # Symmetric XY padding if undersized
    if ct_img.shape[0] < CROP_XY or ct_img.shape[1] < CROP_XY:
        ct_img   = pad_image_symmetrically(ct_img,   target_size=CROP_XY, fill_value=-1000)
        pet_img  = pad_image_symmetrically(pet_img,  target_size=CROP_XY, fill_value=0)
        mask_img = pad_image_symmetrically(mask_img, target_size=CROP_XY, fill_value=0)

    pet_out  = os.path.join(out_dir, "pet_processed.nii.gz")
    ct_out   = os.path.join(out_dir, "ct_processed.nii.gz")
    mask_out = os.path.join(out_dir, "mask_processed.nii.gz")

    nib.save(pet_img,  pet_out)
    nib.save(ct_img,   ct_out)
    nib.save(mask_img, mask_out)

    return pet_out, ct_out, mask_out, crop_offset, pet_img.shape


def generate_npy(pet_path: str, ct_path: str, mask_path: str, out_dir: str):
    """
    Stage 2: load processed NIfTIs → lesion crop → focal crop → MONAI resize → .npy

    Output files (all shape (1, 32, 256, 256) after transforms):
        pet.npy / ct.npy / mask.npy          — global crop centred on lesion
        pet_focal.npy / ct_focal.npy /
        mask_focal.npy                       — tight bounding-box crop around mask
    """
    def load_as_channel_first(path):
        """Load NIfTI → flip Y/Z axes (matching original script) → (1, Z, Y, X)."""
        data = nib.load(path).get_fdata()
        data = data[:, ::-1, ::-1]
        return data.transpose(2, 1, 0)[np.newaxis]   # (1, Z, Y, X)

    pet  = load_as_channel_first(pet_path).astype(np.float32)
    ct   = load_as_channel_first(ct_path).astype(np.float32)
    mask = load_as_channel_first(mask_path).astype(np.float32)
    mask = np.clip(mask, 0, 1)

    # Global lesion crop:
    #   CT  windowed to [-300, 400] HU (normalised to [0, 1])
    #   PET windowed to [0, 12] SUV  (normalised to [0, 1])
    print(pet.shape, ct.shape, mask.shape, file=sys.stderr)
    ct_cropped,  _            = crop_image_around_lesion(ct,  mask, margin=80, variance=15, clip_val_min=-300, clip_val_max=400)
    pet_cropped, mask_cropped = crop_image_around_lesion(pet, mask, margin=80, variance=15, clip_val_min=0,    clip_val_max=12)

    # Tight focal crop around mask bounding box
    pet_focal, mask_focal, ct_focal = focal_crop_around_mask(pet_cropped, mask_cropped, ct_cropped, margin=20)

    # MONAI: foreground crop + resize to (32, 256, 256)
    transformed = transforms_global({
        "pet":  pet_cropped,
        "mask": mask_cropped,
        "ct":   ct_cropped,
    })
    transformed_focal = transforms_focal({
        "pet_focal":  pet_focal,
        "mask_focal": mask_focal,
        "ct_focal":   ct_focal,
    })

    np.save(os.path.join(out_dir, "pet.npy"),         transformed["pet"])
    np.save(os.path.join(out_dir, "ct.npy"),          transformed["ct"])
    np.save(os.path.join(out_dir, "mask.npy"),        transformed["mask"])
    np.save(os.path.join(out_dir, "pet_focal.npy"),   transformed_focal["pet_focal"])
    np.save(os.path.join(out_dir, "ct_focal.npy"),    transformed_focal["ct_focal"])
    np.save(os.path.join(out_dir, "mask_focal.npy"),  transformed_focal["mask_focal"])


# -----------------------------
# Stage 3: model inference
# -----------------------------
def run_inference(npy_dir: str) -> tuple[str, str]:
    """
    Load the six .npy files from npy_dir, run the model with a randomly chosen
    prompt from Seg_template, and return (generated_text, prompt_used).
    """
    # Load arrays
    item = {}
    for key in MODALITY_KEYS:
        item[key] = np.load(os.path.join(npy_dir, f"{key}.npy"))

    # Apply inference transform (ToTensor only — no augmentation)
    item = infer_transform(item)

    # Pick a prompt at inference time, same as the dataset does during training
    prompt = random.choice(Seg_template)

    # Build question string: image tokens + prompt
    question = IMAGE_TOKENS + prompt

    # Tokenise question → input_ids for the model
    input_ids = tokenizer(
        question,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    )["input_ids"].to(DEVICE)

    # Move image tensors to device and add batch dim (model expects B, C, Z, Y, X)
    pet       = item["pet"      ].unsqueeze(0).to(DEVICE)
    ct        = item["ct"       ].unsqueeze(0).to(DEVICE)
    mask      = item["mask"     ].unsqueeze(0).to(DEVICE)
    pet_focal = item["pet_focal"].unsqueeze(0).to(DEVICE)
    ct_focal  = item["ct_focal" ].unsqueeze(0).to(DEVICE)
    mask_focal= item["mask_focal"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        generation = model.generate(
            pet, mask, ct,
            pet_focal, mask_focal, ct_focal,
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_text = tokenizer.decode(generation[0], skip_special_tokens=True).strip()
    return generated_text, prompt

# -----------------------------
# Standalone direct-NIfTI entry point
# -----------------------------
def analyze_niftis(
    pet_path: str,
    ct_path: str,
    mask_path: str,
    segment_name: str,
    save_dir: str = SAVE_DIR,
) -> dict:
    print("Received direct inference request", file=sys.stderr)

    for label, path in (("PET", pet_path), ("CT", ct_path), ("segmentation mask", mask_path)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    out_dir = os.path.join(save_dir, segment_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Stage 1: resample + spatial crop → processed NIfTIs ───────────────
    pet_proc, ct_proc, mask_proc, crop_offset, proc_shape = process_niftis(
        pet_path, ct_path, mask_path, out_dir
    )
    print(f"[{segment_name}] Stage 1 ✓ — processed NIfTIs saved | crop_offset={crop_offset} | shape={proc_shape}", file=sys.stderr)

    # ── Stage 2: lesion crop + MONAI resize → .npy arrays ─────────────────
    generate_npy(pet_proc, ct_proc, mask_proc, out_dir)
    print(f"[{segment_name}] Stage 2 ✓ — .npy files saved", file=sys.stderr)

    # ── Stage 3: model inference ───────────────────────────────────────────
    inference_result, prompt_used = run_inference(out_dir)
    print(f"[{segment_name}] Stage 3 ✓ — inference complete", file=sys.stderr)
    print(f"[{segment_name}] Prompt: {prompt_used}", file=sys.stderr)
    print(f"[{segment_name}] Result: {inference_result}", file=sys.stderr)

    # Save result to disk alongside the other outputs
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump({
            "segment_name": segment_name,
            "prompt":       prompt_used,
            "result":       inference_result,
        }, f, indent=2)

    return {
        "status":           "success",
        "result":           inference_result,
        "prompt":           prompt_used,
        "processed_shape":  list(proc_shape),
        "crop_offset":      crop_offset,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the former Flask /analyze inference path as a standalone direct-NIfTI script."
    )
    parser.add_argument(
        "--pet",
        "--pet-path",
        dest="pet_path",
        required=True,
        help="Path to the PET NIfTI file, e.g. /input/pet.nii.gz.",
    )
    parser.add_argument(
        "--ct",
        "--ct-path",
        dest="ct_path",
        required=True,
        help="Path to the CT NIfTI file, e.g. /input/ct.nii.gz.",
    )
    parser.add_argument(
        "--segmentation",
        "--mask",
        "--mask-path",
        dest="mask_path",
        required=True,
        help="Path to the segmentation mask NIfTI file.",
    )
    parser.add_argument(
        "--segment-name",
        required=True,
        help="Name used for the output directory under --save-dir.",
    )
    parser.add_argument(
        "--save-dir",
        default=SAVE_DIR,
        help="Directory where segment outputs are written. Defaults to /data, matching the server.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional file path for the final response JSON. The response is always printed to stdout.",
    )
    args = parser.parse_args()

    try:
        response = analyze_niftis(
            pet_path=args.pet_path,
            ct_path=args.ct_path,
            mask_path=args.mask_path,
            segment_name=args.segment_name,
            save_dir=args.save_dir,
        )
        exit_code = 0
    except Exception as e:
        tb = traceback.format_exc()
        print("=== ERROR ===", file=sys.stderr)
        print(tb, file=sys.stderr)
        sys.stderr.flush()
        response = {"status": "error", "error": str(e), "traceback": tb}
        exit_code = 1

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(response, f, indent=2)

    print(json.dumps(response))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
