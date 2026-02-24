# 微调检测头
需要修改一下配置文件中关于数据集的路径

用自己的数据集对ms_head进行微调(验证过)
```shell
python scripts/segmentation/train_head.py --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py
```
用自己的数据集对ms_head(540_960)进行微调(验证过)
```shell
python scripts/segmentation/train_head.py --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head_540_960.py
```
用自己的数据集对linear_head进行微调(未验证过)
```shell
python scripts/segmentation/train_head.py --config dinov2/configs/segmentation/dinov2_vits14_ctem_linear_head.py
```
用训练好的模型在测试集上进行可视化展示，会将预测的结果用0.5的透明度覆盖在原图片上
```shell
python scripts/segmentation/predict_overlay.py  --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py   --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms/latest.pth   --out-dir /home/tipriest/Documents/CVTask/CTEM/TestSet/PredOverlay   --alpha 0.5
```
使用微调后的模型对自己指定的测试集进行评估
```shell
python scripts/segmentation/eval_test.py   --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py   --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms/latest.pth
```

使用微调后的模型对自己指定的测试集(540_960)进行评估
```shell
python scripts/segmentation/eval_test.py   --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head_540_960.py   --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms_540_960/latest.pth
--input_size 640 640

python scripts/segmentation/eval_test.py \
    --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head_540_960.py \
    --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms_540_960/latest.pth \
    --input_size 960 540
```

# 评估参数量和计算的FLOPS
单尺度 FLOPs 用 --input-size 直接计算。
多尺度 FLOPs 会读取 config 里的 MultiScaleFlipAug 的 img_ratios，按 input-size 作为基准缩放并求和；如果 flip=True 会乘以 2。
输出会包含每个 ratio 对应的输入尺寸和 FLOPs。
如果你希望“多尺度 FLOPs”基于真实图片的长短边规则（而不是基于 input-size 缩放），告诉我你想用的实际基准尺寸或图片尺寸分布，我可以把计算逻辑改得更贴近实际推理。建议下一步你可以：

提供你实际推理时的图像尺寸规则（例如固定短边 640）
告诉我是否需要把模型移动到 GPU 并统计 GPU 上的 FLOPs

```shell
# CPU
python scripts/segmentation/compute_model_stats.py \
  --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py \
  --backbone dinov2_vits14 \
  --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms/latest.pth \
  --input-size 640 640 \
  --device cpu

# GPU
python scripts/segmentation/compute_model_stats.py \
  --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py \
  --backbone dinov2_vits14 \
  --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms/latest.pth \
  --input-size 640 640 \
  --device cuda
```

将模型导出成`torchscript`格式
```shell
python scripts/segmentation/export_netron_torchscript.py \
  --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py \
  --checkpoint /home/tipriest/Documents/CVTask/dinov2_segmentation_finetune/work_dirs/dinov2_vits14_ctem_ms/latest.pth \
  --input-size 640 640 \
  --patch-size 14 \
  --device cuda \
  --output work_dirs/dinov2_vits14_ctem_ms/head_torchscript.pt
```