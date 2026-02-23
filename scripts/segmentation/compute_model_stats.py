import argparse
from functools import partial

import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.runner import load_checkpoint
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


def format_number(value, unit):
    if unit == "params":
        return f"{value / 1e6:.2f} M"
    if unit == "flops":
        return f"{value / 1e9:.2f} GFLOPs"
    return str(value)


def get_multiscale_settings(cfg):
    try:
        pipeline = cfg.data.test.pipeline
    except Exception:
        return [], False

    for step in pipeline:
        if step.get("type") == "MultiScaleFlipAug":
            ratios = step.get("img_ratios", [])
            flip = bool(step.get("flip", False))
            return ratios, flip
    return [], False


def compute_flops(model, height, width, device):
    input_shape = (3, height, width)

    if hasattr(model, "forward_dummy"):
        class DummyWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, img):
                return self.inner.forward_dummy(img)

        flops_model = DummyWrapper(model).to(device)

        def input_constructor(input_res):
            _, h, w = input_res
            img = torch.zeros(1, 3, h, w, device=device)
            return dict(img=img)
    else:
        flops_model = model

        def input_constructor(input_res):
            _, h, w = input_res
            img = torch.zeros(1, 3, h, w, device=device)
            img_metas = [
                {
                    "img_shape": (h, w, 3),
                    "ori_shape": (h, w, 3),
                    "pad_shape": (h, w, 3),
                    "scale_factor": 1.0,
                    "flip": False,
                    "flip_direction": None,
                }
            ]
            return dict(img=img, img_metas=img_metas)

    flops, _ = get_model_complexity_info(
        flops_model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        input_constructor=input_constructor,
    )
    return flops


def main():
    parser = argparse.ArgumentParser(
        description="Compute params and FLOPs for DINOv2 ViT-S/14 + MS head model"
    )
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--backbone", default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--checkpoint", default=None, help="Path to head checkpoint (.pth)")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (H W)",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)

    model = build_head_only_model(cfg, args.backbone, device)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    params = sum(p.numel() for p in model.parameters())
    base_h, base_w = args.input_size
    flops_single = compute_flops(model, base_h, base_w, device)

    ratios, flip = get_multiscale_settings(cfg)
    flops_multiscale = None
    flops_per_scale = []
    if ratios:
        flops_total = 0.0
        for ratio in ratios:
            height = int(round(base_h * ratio))
            width = int(round(base_w * ratio))
            flops_scale = compute_flops(model, height, width, device)
            flops_per_scale.append((ratio, height, width, flops_scale))
            flops_total += flops_scale
        if flip:
            flops_total *= 2.0
        flops_multiscale = flops_total

    print("Model stats")
    print(f"Params: {format_number(params, 'params')} ({params:,})")
    print(f"FLOPs (single): {format_number(flops_single, 'flops')} ({int(flops_single):,})")
    if flops_multiscale is not None:
        print(f"FLOPs (multi):  {format_number(flops_multiscale, 'flops')} ({int(flops_multiscale):,})")
        for ratio, height, width, flops_scale in flops_per_scale:
            print(
                "  - ratio %.2f -> %dx%d: %s (%d)"
                % (ratio, height, width, format_number(flops_scale, "flops"), int(flops_scale))
            )


if __name__ == "__main__":
    main()
