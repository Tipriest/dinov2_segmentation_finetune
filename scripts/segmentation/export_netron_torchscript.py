import argparse
import math
from pathlib import Path

import torch
from mmcv import Config
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


def build_head_only_model(cfg, device):
    model = build_segmentor(cfg.model)
    model.to(device)
    model.eval()
    return model


class HeadWrapper(torch.nn.Module):
    def __init__(self, decode_head):
        super().__init__()
        self.decode_head = decode_head

    def forward(self, feat0, feat1, feat2, feat3):
        return self.decode_head([feat0, feat1, feat2, feat3])


def main():
    parser = argparse.ArgumentParser(
        description="Export DINOv2 + segmentation head as TorchScript for Netron"
    )
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--checkpoint", required=True, help="Path to head checkpoint (.pth)")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (H W)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size used by the backbone (for feature map size)",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--output",
        default="work_dirs/dinov2_vits14_ctem_ms/model_torchscript.pt",
        help="Output TorchScript path",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)

    model = build_head_only_model(cfg, device)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    decode_head = model.decode_head
    decode_head.eval()
    wrapper = HeadWrapper(decode_head).to(device)

    feat_h = math.ceil(args.input_size[0] / args.patch_size)
    feat_w = math.ceil(args.input_size[1] / args.patch_size)
    cfg_channels = cfg.model.decode_head.get("in_channels", None)
    if isinstance(cfg_channels, (list, tuple)):
        channels = list(cfg_channels)
    else:
        channels = decode_head.in_channels
        if isinstance(channels, (list, tuple)):
            channels = list(channels)
        else:
            in_index = getattr(decode_head, "in_index", [0, 1, 2, 3])
            if (
                getattr(decode_head, "input_transform", None) == "resize_concat"
                and isinstance(getattr(decode_head, "channels", None), int)
                and len(in_index) > 0
            ):
                per_channel = decode_head.channels // len(in_index)
                channels = [per_channel] * len(in_index)
            else:
                channels = [channels] * len(in_index)

    feat0 = torch.zeros(1, channels[0], feat_h, feat_w, device=device)
    feat1 = torch.zeros(1, channels[1], feat_h, feat_w, device=device)
    feat2 = torch.zeros(1, channels[2], feat_h, feat_w, device=device)
    feat3 = torch.zeros(1, channels[3], feat_h, feat_w, device=device)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (feat0, feat1, feat2, feat3), strict=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Saved TorchScript to: {output_path}")


if __name__ == "__main__":
    main()
