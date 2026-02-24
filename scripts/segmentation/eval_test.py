import argparse
import os
import subprocess
from functools import partial

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel

import dinov2.eval.segmentation.models

# 新增 thop 用于 FLOPs 统计
from thop import profile

# ===== 新增 nvidia-smi 显存查询 =====
def nvidia_smi_used_mem_mb():
    """查询当前进程显存占用（MiB）"""
    try:
        pid = os.getpid()
        cmd = f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd.split()).decode().strip().splitlines()
        for line in out:
            cols = [c.strip() for c in line.split(',')]
            if len(cols) >= 2 and str(pid) == cols[0]:
                return int(cols[1])
        return -1
    except Exception:
        return -1
# ==================================

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
    # 1. 先建立 mmseg 的分割模型（包含一个空的 backbone 和 head）
    model = build_segmentor(cfg.model)

    # 2. 加载预训练 DINOv2 backbone
    backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    # 3. 把 backbone_model 挂到 model.backbone 下，这样参数可以被统计
    #    注意：这样可以避免丢失参数树
    model.backbone.dino = backbone_model

    # 保存 patch_size 信息，后面 padding 用
    if hasattr(backbone_model, "patch_size"):
        model.backbone.patch_size = backbone_model.patch_size

    # 4. 重写 model.backbone 的 forward
    def backbone_forward(self, x):
        # 如果 patch_size 存在，先居中 padding
        if hasattr(self, "patch_size"):
            x = CenterPadding(self.patch_size)(x)

        # 从 dino 提取中间层特征
        with torch.no_grad():
            feats = self.dino.get_intermediate_layers(
                x,
                n=cfg.model.backbone.out_indices,
                reshape=True
            )

        return feats
    # 将新的 forward 绑定到 model.backbone
    import types
    model.backbone.forward = types.MethodType(backbone_forward, model.backbone)

    # 5. 放到对应设备上
    model.to(device)
    model.eval()
    model.cfg = cfg
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_submodule_parameters(module):
    return sum(p.numel() for p in module.parameters())


def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:60} {param_count:,} ({param_count/1e6:.3f} M)")
    print("="*80)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f} M)")
def reset_cuda_mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def print_cuda_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        smi_used = nvidia_smi_used_mem_mb()
        print(f"[CUDA Mem] {tag} | "
              f"allocated={alloc:.1f} MB, reserved={reserved:.1f} MB, peak={peak:.1f} MB, "
              f"nvidia-smi used={smi_used} MB")
def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 segmentation head on test split")
    parser.add_argument("--config", required=True, help="Path to mmseg config file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--backbone", default="dinov2_vits14", help="DINOv2 backbone name")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--input_size", type=int, nargs=2, default=[960, 540],
                        help="Input size for FLOPs calculation: [width height]")
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

    total_params, trainable_params = count_parameters(model)
    backbone_params = count_submodule_parameters(model.backbone.dino)
    head_params = count_submodule_parameters(model.decode_head)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone parameters: {backbone_params/1e6:.3f} M")
    print(f"Head parameters: {head_params/1e6:.3f} M")

    print(f"Effective batch size: {cfg.data.samples_per_gpu * torch.cuda.device_count()} "
          f"( {cfg.data.samples_per_gpu} per GPU × {torch.cuda.device_count()} GPUs )")

    if args.device.startswith("cuda"):
        reset_cuda_mem()

    dummy_input = torch.randn(1, 3, args.input_size[1], args.input_size[0]).to(args.device)

    # 保存原始 forward 方法
    orig_forward = model.forward

    # 使用 forward_dummy 进行 FLOPs 计算
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        print("Warning: model has no forward_dummy(), FLOPs may not be accurate.")

    # FLOPs统计
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # 显存峰值
    if args.device.startswith("cuda"):
        print_cuda_mem("after FLOPs calc")

    print(f"\n=== Model FLOPs & Params ===")
    print(f"Input size: {args.input_size[0]} x {args.input_size[1]}")
    print(f"Total FLOPs: {flops / 1e9:.3f} GFLOPs")
    print(f"Total Params: {params / 1e6:.3f} M")
    print("============================\n")

    # 恢复原 forward 方法，继续正常推理
    model.forward = orig_forward
    # ==== FLOPs统计 End ====

    # print("*"*60)
    # print("model params")
    # print("*"*60)
    # print_model_parameters(model)

    # print("*"*60)
    # print("backbone params")
    # print("*"*60)
    # print_model_parameters(model.backbone)

    # print("*"*60)
    # print("head params")
    # print("*"*60)
    # print_model_parameters(model.decode_head)
    if args.device.startswith("cuda"):
        model = MMDataParallel(model, device_ids=[0])
        reset_cuda_mem()

    outputs = single_gpu_test(model, data_loader, show=False)

    if args.device.startswith("cuda"):
        print("")
        print_cuda_mem("after inference")

    metrics = dataset.evaluate(outputs, metric="mIoU")

    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
