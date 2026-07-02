import os
import numpy as np
import json
import nibabel as nib
from PIL import Image
import concurrent.futures
from tqdm import tqdm
from collections import Counter
import unicodedata
import monai.transforms as mtf
from multiprocessing import Pool
import shutil
# from unidecode import unidecode

import numpy as np

def crop_image_around_lesion(
    image: np.ndarray,
    mask_image: np.ndarray,
    margin: int = 60,
    variance: int = 15,
    clip_val_min: float = 0,
    clip_val_max: float = 12,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crops the Medical scan and segmentation mask around the lesion along the z-axis
    with randomized margin variation.

    Parameters:
    - image (np.ndarray): Medical scan of shape (1, Z, Y, X)
    - mask_image (np.ndarray): Segmentation mask of shape (1, Z, Y, X)
    - margin (int): Base number of slices to include above and below the lesion center
    - variance (int): Max random variation (+/-) added to margin
    - clip_val_min (float): Minimum PET value for clipping
    - clip_val_max (float): Maximum PET value for clipping
    - normalize (bool): Whether to normalize PET to [0, 1] after clipping

    Returns:
    - cropped_pet (np.ndarray): Cropped and normalized PET scan
    - cropped_mask (np.ndarray): Cropped mask
    """
    assert image.shape == mask_image.shape, (
        f"PET and mask must have the same shape, got Image: {image.shape} vs Mask: {mask_image.shape}"
    )
    assert image.ndim == 4, "Expected shape (1, Z, Y, X)"

    # Remove batch dimension
    image, mask = image[0], mask_image[0]

    # Clip and normalize PET
    if normalize:
        image = np.clip(image, clip_val_min, clip_val_max)
        image = (image - clip_val_min) / (clip_val_max - clip_val_min)

    # Detect lesion bounds
    lesion_slices = np.any(mask, axis=(1, 2))
    if not lesion_slices.any():
        raise ValueError("No lesion found in the segmentation mask.")

    z_indices = np.where(lesion_slices)[0]
    center = (z_indices[0] + z_indices[-1]) // 2

    # Apply random variation to margin
    random_margin = margin #+ np.random.randint(-variance, variance + 1)
    start = max(0, center - random_margin)
    random_margin = margin #+ np.random.randint(-variance, variance + 1)
    end = min(image.shape[0], center + random_margin + 1)

    # Crop and reintroduce batch dimension
    return (
        image[np.newaxis, start:end],
        mask[np.newaxis, start:end]
    )


def focal_crop_around_mask(
    pet: np.ndarray,
    mask: np.ndarray,
    ct: np.ndarray,
    margin: int = 20
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crops CT, PET, and mask volumes around the mask foreground (value == 1).
    
    Parameters:
    - ct (np.ndarray): CT scan, shape (1, Z, Y, X)
    - pet (np.ndarray): PET scan, shape (1, Z, Y, X)
    - mask (np.ndarray): Binary mask, shape (1, Z, Y, X)
    - margin (int): Number of voxels to include around the mask boundary in each direction

    Returns:
    - cropped_ct (np.ndarray): Cropped CT scan
    - cropped_pet (np.ndarray): Cropped PET scan
    - cropped_mask (np.ndarray): Cropped mask
    """
    assert ct.shape == pet.shape == mask.shape, "CT, PET, and mask must have same shape"
    assert ct.ndim == 4, "Expected shape (1, Z, Y, X)"
    
    # Remove batch dimension
    ct, pet, mask = ct[0], pet[0], mask[0]

    # Find bounding box of non-zero mask region
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("Mask contains no positive values.")
    
    zmin, ymin, xmin = np.maximum(indices.min(axis=0) - margin, 0)
    zmax, ymax, xmax = np.minimum(indices.max(axis=0) + margin + 1, mask.shape)

    # Crop all volumes
    cropped_ct = ct[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_pet = pet[zmin:zmax, ymin:ymax, xmin:xmax]
    cropped_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax]

    # Reintroduce batch dimension
    return (
        cropped_pet[np.newaxis],
        cropped_mask[np.newaxis],
        cropped_ct[np.newaxis]
    )


def image_info(name, img):
    # pass
    print("-----------------------")
    print(f"{name} shape:", img.shape)
    print(f"{name} unique values:", np.unique(img))
    print(f"{name} sum", np.sum(img))
    print("-----------------------")

tsclasses = {
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "right adrenal gland",
    9: "left adrenal gland",
    10: "left upper lung lobe",
    11: "left lower lung lobe",
    12: "right upper lung lobe",
    13: "right middle lung lobe",
    14: "right lower lung lobe",
    15: "esophagus",
    16: "trachea",
    17: "thyroid gland",
    18: "small bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary bladder",
    22: "prostate",
    23: "left kidney cyst",
    24: "right kidney cyst",
    25: "sacrum",
    26: "S1 vertebra",
    27: "L5 vertebra",
    28: "L4 vertebra",
    29: "L3 vertebra",
    30: "L2 vertebra",
    31: "L1 vertebra",
    32: "T12 vertebra",
    33: "T11 vertebra",
    34: "T10 vertebra",
    35: "T9 vertebra",
    36: "T8 vertebra",
    37: "T7 vertebra",
    38: "T6 vertebra",
    39: "T5 vertebra",
    40: "T4 vertebra",
    41: "T3 vertebra",
    42: "T2 vertebra",
    43: "T1 vertebra",
    44: "C7 vertebra",
    45: "C6 vertebra",
    46: "C5 vertebra",
    47: "C4 vertebra",
    48: "C3 vertebra",
    49: "C2 vertebra",
    50: "C1 vertebra",
    51: "heart",
    52: "aorta",
    53: "pulmonary vein",
    54: "brachiocephalic trunk",
    55: "right subclavian artery",
    56: "left subclavian artery",
    57: "right common carotid artery",
    58: "left common carotid artery",
    59: "left brachiocephalic vein",
    60: "right brachiocephalic vein",
    61: "left atrial appendage",
    62: "superior vena cava",
    63: "inferior vena cava",
    64: "portal and splenic veins",
    65: "left iliac artery",
    66: "right iliac artery",
    67: "left iliac vein",
    68: "right iliac vein",
    69: "left humerus",
    70: "right humerus",
    71: "left scapula",
    72: "right scapula",
    73: "left clavicle",
    74: "right clavicle",
    75: "left femur",
    76: "right femur",
    77: "left hip",
    78: "right hip",
    79: "spinal cord",
    80: "left gluteus maximus",
    81: "right gluteus maximus",
    82: "left gluteus medius",
    83: "right gluteus medius",
    84: "left gluteus minimus",
    85: "right gluteus minimus",
    86: "left autochthonous back muscle",
    87: "right autochthonous back muscle",
    88: "left iliopsoas",
    89: "right iliopsoas",
    90: "brain",
    91: "skull",
    92: "left 1st rib",
    93: "left 2nd rib",
    94: "left 3rd rib",
    95: "left 4th rib",
    96: "left 5th rib",
    97: "left 6th rib",
    98: "left 7th rib",
    99: "left 8th rib",
    100: "left 9th rib",
    101: "left 10th rib",
    102: "left 11th rib",
    103: "left 12th rib",
    104: "right 1st rib",
    105: "right 2nd rib",
    106: "right 3rd rib",
    107: "right 4th rib",
    108: "right 5th rib",
    109: "right 6th rib",
    110: "right 7th rib",
    111: "right 8th rib",
    112: "right 9th rib",
    113: "right 10th rib",
    114: "right 11th rib",
    115: "right 12th rib",
    116: "sternum",
    117: "costal cartilages"
}


sentence_templates = [
    "The segmented region is the {}.",
    "You are looking at the {}.",
    "This scan shows the {}.",
    "Identified structure: {}.",
    "Segmented area corresponds to the {}.",
    "Here's the {} detected in the image.",
    "What you see here is the {}.",
    "The highlighted region is the {}.",
    "That would be the {}.",
    "This region belongs to the {}."
]

if __name__ == "__main__":
    print(tscalsses[4])