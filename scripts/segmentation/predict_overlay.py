import argparse
import os
from functools import partial

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor

import dinov2.eval.segmentation.models


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = (size + self.multiple - 1) // self.multiple * self.multiple
        pad_size = new_size - size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    @torch.inference_mode()
    def forward(self, x):
        import itertools
        import torch.nn.functional as F

        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)


def build_head_only_model(cfg, backbone_name, device):
    model = build_segmentor(cfg.model)

    backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    def forward_no_grad(*args, **kwargs):
        with torch.no_grad():
            return backbone_model.get_intermediate_layers(*args, **kwargs)

    model.backbone.forward = partial(
        forward_no_grad,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )

    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )

    model.to(device)
    model.eval()
    return model


def read_split_images(data_root, img_dir, img_suffix, split_file):
    img_root = os.path.join(data_root, img_dir)
    if split_file:
        split_path = os.path.join(data_root, split_file)
        with open(split_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return [os.path.join(img_root, name + img_suffix) for name in names]

    images = []
    for name in os.listdir(img_root):
        if name.endswith(img_suffix):
            images.append(os.path.join(img_root, name))
    return sorted(images)


def build_palette_bgr(palette):
    pal = np.array(palette, dtype=np.uint8)
    if pal.ndim != 2 or pal.shape[1] != 3:
        raise ValueError("palette should be list of RGB triplets")
    return pal[:, ::-1]


def overlay_segmentation(image_bgr, seg_map, palette_bgr, alpha):
    color_mask = palette_bgr[seg_map]
    color_mask = color_mask.astype(np.float32)
    image = image_bgr.astype(np.float32)
    blended = image * (1.0 - alpha) + color_mask * alpha
    return blended.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Overlay segmentation predictions and save results")
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--backbone", default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency (0-1)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    test_cfg = cfg.data.test
    data_root = test_cfg.data_root
    img_dir = test_cfg.img_dir
    img_suffix = test_cfg.get("img_suffix", ".jpg")
    split_file = test_cfg.get("split")

    palette = cfg.get("palette") or test_cfg.get("palette")
    if palette is None:
        raise ValueError("palette is missing in config")

    palette_bgr = build_palette_bgr(palette)

    model = build_head_only_model(cfg, args.backbone, args.device)
    model.cfg = cfg
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = read_split_images(data_root, img_dir, img_suffix, split_file)
    if not image_paths:
        raise ValueError("no images found for inference")

    for img_path in image_paths:
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        seg_map = inference_segmentor(model, image_bgr)[0]
        overlay = overlay_segmentation(image_bgr, seg_map, palette_bgr, args.alpha)
        out_path = os.path.join(args.out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, overlay)


if __name__ == "__main__":
    main()
