from .vit import ViT3DTower, ViTMerlin3DTower


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    # use_contour = getattr(config, 'use_contour', None)
    # print(kwargs)
    # print("Contour value:", use_contour)
    # if use_contour is None:
    #     config.use_contour = False)
    if 'vit3d' in vision_tower.lower():
        return ViT3DTower(config, **kwargs)
    elif 'vitmerlin3d' in vision_tower.lower():
        return ViTMerlin3DTower(config, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')