# 微调检测头
需要修改一下配置文件中关于数据集的路径

用自己的数据集对ms_head进行微调(验证过)
```shell
python scripts/segmentation/train_head.py --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py
```
用自己的数据集对linear_head进行微调(未验证过)
```shell
python scripts/segmentation/train_head.py --config dinov2/configs/segmentation/dinov2_vits14_ctem_linear_head.py
```
用训练好的模型在测试集上进行可视化展示，会将预测的结果用0.5的透明度覆盖在原图片上
```shell
python scripts/segmentation/predict_overlay.py  --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py   --checkpoint /home/tipriest/Documents/CVTask/dinov2/work_dirs/dinov2_vits14_ctem_ms/latest.pth   --out-dir /home/tipriest/Documents/CVTask/CTEM/TestSet/PredOverlay   --alpha 0.5
```
使用微调后的模型对自己指定的测试集进行评估
```shell
python scripts/segmentation/eval_test.py   --config dinov2/configs/segmentation/dinov2_vits14_ctem_ms_head.py   --checkpoint /home/tipriest/Documents/CVTask/dinov2/work_dirs/dinov2_vits14_ctem_ms/latest.pth
```