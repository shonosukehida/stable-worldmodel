"""Channels-first raw RGB -> antialiased letterbox -> DIAMOND's [-1, 1] grid."""
import torch
import torch.nn.functional as F


def image_spec(views):
    if list(views) not in (["pixels"], ["pixels", "wrist_pixels"]):
        raise ValueError("Expected overhead RGB, optionally followed by wrist RGB")
    return dict(views=list(views), layout="view-major RGB CHW", resolution=[64, 64],
                resize="bilinear antialias align_corners=False", padding_rgb=0,
                rounding="nearest byte after resize", raw_range=[0, 255], model_range=[-1, 1],
                inverse="remove centered padding then bilinear resize to source H,W")


def geometry(height, width):
    if min(height, width) < 1:
        raise ValueError("Empty image")
    scale = 64 / max(height, width)
    h, w = max(1, round(height * scale)), max(1, round(width * scale))
    return h, w, (64-h)//2, (64-w)//2


def pack_views(raw, views):
    """raw maps view names to (...,3,H,W) byte or 0..255 floating tensors."""
    image_spec(views)
    images = [torch.as_tensor(raw[key]) for key in views]
    shape = images[0].shape
    if len(shape) < 3 or shape[-3] != 3 or any(x.shape != shape for x in images):
        raise ValueError("RGB views must have identical leading and spatial dimensions")
    x = torch.cat(images, dim=-3).float()
    if not x.isfinite().all() or x.min() < 0 or x.max() > 255:
        raise ValueError("Raw RGB must be finite and in [0, 255]")
    h, w, top, left = geometry(*shape[-2:])
    channels = len(images)*3
    x = F.interpolate(x.reshape(-1, channels, *shape[-2:]), size=(h, w), mode="bilinear",
                      antialias=True, align_corners=False).round()
    x = F.pad(x, (left, 64-w-left, top, 64-h-top))
    return x.reshape(*shape[:-3], channels, 64, 64).div(255).mul(2).sub(1)


def unpack_views(image, views, original_hw=None):
    """Return named RGB uint8 tensors, optionally restored to the original field of view."""
    image_spec(views)
    if image.shape[-3:] != (3*len(views), 64, 64):
        raise ValueError("Unexpected packed image layout")
    x = image.clamp(-1, 1).add(1).mul(127.5)
    if original_hw is not None:
        h, w, top, left = geometry(*original_hw)
        crop = x[..., top:top+h, left:left+w]
        x = F.interpolate(crop.reshape(-1, x.shape[-3], h, w), size=original_hw,
                          mode="bilinear", align_corners=False).reshape(*x.shape[:-2], *original_hw)
    return dict(zip(views, x.round().byte().split(3, dim=-3)))
