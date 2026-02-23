import argparse
from functools import partial

import torch
from mmcv import Config
from mmseg.apis import train_segmentor
from mmseg.datasets import build_dataset
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


def build_head_only_model(cfg, backbone_name):
    model = build_segmentor(cfg.model)

    backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    def forward_no_grad(*args, **kwargs):
        with torch.no_grad():
            return backbone_model.get_intermediate_layers(*args, **kwargs)

    model.backbone.forward = partial(
        forward_no_grad,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )

    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    model.cuda()
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 segmentation head on CTEM")
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--backbone", default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--work-dir", default=None, help="Override work_dir in config")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir

    model = build_head_only_model(cfg, args.backbone)
    datasets = [build_dataset(cfg.data.train)]
    train_segmentor(model, datasets, cfg, distributed=False, validate=True)


if __name__ == "__main__":
    main()
