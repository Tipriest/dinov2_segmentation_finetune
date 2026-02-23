import argparse
from functools import partial

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel

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
    model.cfg = cfg
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 segmentation head on test split")
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--backbone", default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
    )

    model = build_head_only_model(cfg, args.backbone, args.device)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.device.startswith("cuda"):
        model = MMDataParallel(model, device_ids=[0])

    outputs = single_gpu_test(model, data_loader, show=False)
    metrics = dataset.evaluate(outputs, metric="mIoU")

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
