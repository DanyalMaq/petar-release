# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

from .cross_attn import EfficientCrossAttention

# from merlin import Merlin

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        use_mask: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()
        print("The bias term is:", qkv_bias)

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.use_mask = use_mask
        self.qkv_bias = qkv_bias
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        if self.use_mask:
            self.patch_embedding_mask = PatchEmbeddingBlock(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                pos_embed=pos_embed,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            )
            for name, param in self.patch_embedding_mask.named_parameters():
                if "weight" in name or "bias" in name:
                    torch.nn.init.constant_(param, 0.0)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, self.qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x, mask=None):
        # print(x.shape)
        if self.use_mask:
            print("Using mask")
            x = self.patch_embedding(x) + self.patch_embedding_mask(mask)
        else:
            x = self.patch_embedding(x)
        # print(x.shape)
        # print("mask included")
        # print("x shape", x.shape)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            # print("x shape again", x.shape)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.classification = getattr(config, 'classification', True)
        self.pos_embed = getattr(config, 'pos_embed', 'perceptron')
        self.use_mask = getattr(config, 'use_mask', False)
        self.qkv_bias = getattr(config, 'qkv_bias', False)
        self.use_ct = getattr(config, 'use_ct', False)

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed=self.pos_embed,
            spatial_dims=len(self.config.patch_size),
            classification=self.classification,
            use_mask = self.use_mask,
            qkv_bias = self.qkv_bias
        )
        if self.use_ct:
            self.ct_tower = ViT(
                in_channels=self.config.image_channel,
                img_size=self.config.image_size,
                patch_size=self.config.patch_size,
                pos_embed=self.pos_embed,
                spatial_dims=len(self.config.patch_size),
                classification=False,
                use_mask = False,
                qkv_bias = self.qkv_bias
            )

    def forward(self, pet, masks, ct=None):
        last_feature, hidden_states = self.vision_tower(pet, masks)
        # print("Last_feature shape:", last_feature.shape)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            if self.use_ct:
                if ct is None: raise Exception("No CTs passed even though CT is true")
                print("Using CT")
                last_feature_ct, hidden_states_ct = self.ct_tower(ct)
                image_features = torch.cat([image_features, last_feature_ct], dim=-1)
        elif self.select_feature == 'no_cls_patch':
            image_features = image_features
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size
    
class ViTMerlin3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        use_mask = getattr(config, 'use_mask', None)
        if use_mask is not None:
            self.use_mask = use_mask
        else:
            print("Use mask value is None in vit, setting it to False")
            self.use_mask = False

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
            use_mask=self.use_mask
        )

        self.merlin = Merlin()

        self.cross_attention = CrossAttentionBlock(embed_dim=768, num_heads=8)

    def forward(self, images, masks):
        # Separately encode images
        last_feature, hidden_states = self.vision_tower(images, masks) # (bsz, 2048, 768)
        merlin_features = self.merlin(images) # (bsz, 2048), mayb project this to the same embedding dim

        # Determine the layer of the vision transformer to extract features from (Have fusion here later as well)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        # Determine if we're including the classification patch or not
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        # Logic to combine the image_features and merlin (or future) features
        image_features = self.cross_attention(image_features, merlin_features)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size