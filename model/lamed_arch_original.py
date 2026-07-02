from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False
        print("Meta model is being initialized")

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
        self.config.qkv_bias = model_args.qkv_bias
        self.config.classification = model_args.classification
        self.config.pos_embed = model_args.pos_embed
        self.config.use_ct = getattr(model_args, 'use_ct', False)
        print("CT value is", self.config.use_ct)
             
        use_mask = getattr(model_args, 'use_mask', None)
        if use_mask is not None:
            self.config.use_mask = use_mask
        else:
            print("Use mask value is None in lamed_arch, setting it to False")
            self.config.use_mask = False


        # vision tower
        if self.get_vision_tower() is None:
            print("Rebuilding from scratch completely")
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        # Vision tower pretrain
        if model_args.pretrain_vision_model is not None:
            # Load the vision model weights
            print("Loading weights and rebuilding vision tower")
            self.vision_tower = build_vision_tower(self.config)
            vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=False)
            if self.config.use_ct:
                self.vision_tower.ct_tower.load_state_dict(vision_model_weights, strict=False)
            # # if True: #model_args.rebuild                
            #     # print("Rebuilding vision tower from scratch")
            #     # print(self.config)
            #     loaded_state_dict = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            #     model_state_dict = self.vision_tower.vision_tower.state_dict()
            #     # for ((k1, v1), (k2, v2)) in zip(model_state_dict.items(), loaded_state_dict.items()):
            #     #     print(f"{k1}: {v1.shape}")
            #     #     print(f"{k2}: {v2.shape}")

            #     # Pop the cls_token weight from the original state_dict and store it
            #     cls_token_weight = model_state_dict.pop("cls_token", None)

            #     # Rename the keys in the new state dict for patch_embeddings
            #     new_state_dict = {}
            #     for key, value in loaded_state_dict.items():
            #         if "patch_embedding.patch_embeddings.weight" in key:
            #             # Rename patch_embeddings weights with correct indexing (e.g., 1 -> 0)
            #             new_key = key.replace("patch_embedding.patch_embeddings.weight", "patch_embedding.patch_embeddings.1.weight")
            #             new_state_dict[new_key] = value
            #         elif "patch_embedding.patch_embeddings.bias" in key:
            #             # Rename patch_embeddings biases with correct indexing (e.g., 1 -> 0)
            #             new_key = key.replace("patch_embedding.patch_embeddings.bias", "patch_embedding.patch_embeddings.1.bias")
            #             new_state_dict[new_key] = value
            #         else:
            #             # If it's not a patch_embedding, just retain the original key
            #             new_state_dict[key] = value

            #     # If cls_token was popped, add it to the new state_dict
            #     if cls_token_weight is not None:
            #         new_state_dict["cls_token"] = cls_token_weight

            #     # Load the updated state dict into your model
            #     self.vision_tower = build_vision_tower(self.config)
            #     # for ((k1, v1), (k2, v2)) in zip(self.vision_tower.vision_tower.state_dict().items(), loaded_state_dict.items()):
            #     #     print(f"{k1}: {v1.shape}")
            #     #     print(f"{k2}: {v2.shape}")
            #     self.vision_tower.vision_tower.load_state_dict(loaded_state_dict, strict=False)
            # else:

        self.config.mm_hidden_size = self.vision_tower.hidden_size
        # print("Hidden size", self.vision_tower.hidden_size)

        # mm_projector
        self.mm_projector = build_mm_projector(self.config)

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, pets, masks, cts=None, pet_focals=None, mask_focals=None, ct_focals=None):
        image_features = self.get_model().get_vision_tower()(pets, masks, cts)
        if pet_focals is not None and mask_focals is not None and ct_focals is not None:
            focal_image_features = self.get_model().get_vision_tower()(pet_focals, mask_focals, ct_focals)
            image_features = image_features + focal_image_features
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        pets, masks, cts, pet_focals=None, mask_focals=None, ct_focals=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or pets is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            image_features = self.encode_images(pets, masks, cts, pet_focals, mask_focals, ct_focals)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            # print("Embed tokens weight:", embed_tokens_weight)

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")