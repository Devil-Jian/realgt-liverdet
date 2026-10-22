# Real-GT Liver Detection — 完全人工标注目标检测（独立项目）

本项目**只使用人工肿瘤真值**训练和评价 YOLOv8s，不接受教师伪标签、不读取旧项目的 `train_slices.json`，不使用肝脏Mask裁剪，不删除小病灶。AP/PVP分开训练；所有切片输出完整512×512 CT（具体像素尺寸与NIfTI一致），HU默认窗口 `[-160,240]`。所有 **Train/Val/Test 均保留经过完整人工标注的阳性与阴性切片**。Test仅在训练达到100 Epoch或早停之后，使用Val最佳权重做一次最终评价。

## 1. 输入数据和真实标签约束

严格扫描患者目录 `/dataset/dataset/gx_all_registered/<patient_id>/`，**只**读取：

```text
ap_registered.nii.gz
ap_tumor_mask_registered.nii.gz       # 人工肿瘤Mask
pvp_registered.nii.gz
pvp_tumor_mask_registered.nii.gz     # 人工肿瘤Mask
```

- 必须AP/PVP两期配对，且影像和Mask的shape+affine一致。没有人工肿瘤Mask的患者全部排除，只有一期标注的患者会导致明确报错，避免静默遗漏。
- 人工GT中所有前景 `>0` 视为肿瘤。逐层使用8连通域计算BBox；**1像素肿瘤也保留**。
- 人工Mask空层作为真正阴性使用，前提是人工Mask为全体积完整标注；若实际只标注了某个局部范围，**不能把范围外空层视为阴性**，需要先完善标注覆盖范围/另行修改采样策略。
- 此项目无任何伪标签回退。没有真实肿瘤Mask不能进入任何数据组。
- CT和GT NIfTI按Nibabel轴序 `[x,y,z]` 读取，PNG为 `[y,x]`，`slice_index`从0开始；读取每卷 `.nii.gz` 一次，避免逐层gzip解压。
- 使用患者级split，同患者AP/PVP以及全部层面始终归属于同一分组。

## 2. 安装（建议使用现有 det 环境）

将ZIP解压到服务器 `/workspace/realgt-liverdet`：

```bash
cd /workspace/realgt-liverdet
conda activate det
python -m pip install -e '.[train,test]'
python -m pytest -q
```

首次 `yolov8s.pt` 下载可能需要联网或提前放好权重。

## 3. 构建全人工GT数据集

```bash
realgt build \
  --root /dataset/dataset/gx_all_registered \
  --out /dataset/realgt_liverdet \
  --seed 42 \
  --train-fraction 0.70 \
  --val-fraction 0.15
```

如果正好108例符合要求，四舍五入后默认约 **76例Train、16例Val、16例Test**；实际数量以扫描结果为准。`manifests/splits.csv` 会保存明确划分，完全无患者交叉。

**重要：** 旧项目中108例人工GT先前被当作Val/Test。若重新划分训练，与旧项目原Test结果不再具有相同测试集，不能直接比较；已经看过的旧Test指标也会影响模型选择的独立性。如果想**保留旧Test 54例**，请提供覆盖所有GT患者的固定 `patient_id,split` CSV，然后用：

```bash
realgt build \
  --root /dataset/dataset/gx_all_registered \
  --out /dataset/realgt_liverdet_holdout \
  --splits-file /path/to/fixed_manual_gt_splits.csv
```

建议完整锁住原Test，原Val再划分真实Train/Val。**不要从旧项目的split文件直接导入全部病例**：旧文件包含大量伪标签患者；外部分组CSV必须精确包含新项目识别出的GT患者且只包含这些患者。

输出：

```text
/dataset/realgt_liverdet/
├── manifests/
│   ├── cases.csv                  # 每患者AP/PVP人工GT
│   ├── splits.csv                 # 固定按患者划分
│   ├── preprocess.json            # HU和Label规则
│   ├── slices.csv                 # 所有切片、BBox、split
│   └── train_slices.json          # Train全部阳性和人工阴性切片
├── data/real_gt/{ap,pvp}/         # 完整HU窗口PNG
└── exports/
    ├── real_gt_ap/
    │   ├── dataset.yaml
    │   ├── counts.csv
    │   ├── images/{train,val,test}/*.png
    │   ├── labels/{train,val,test}/*.txt
    │   └── annotations/{train,val,test}.json
    └── real_gt_pvp/               # 同上
```

导出文件通过软链接引用单份PNG，减少磁盘重复占用。`counts.csv` 给出每期每组患者、切片、阳性/阴性及BBox数量。

## 4. 训练前必须检查BBox可视化

```bash
realgt qc --dataset /dataset/realgt_liverdet/exports/real_gt_ap --split train
realgt qc --dataset /dataset/realgt_liverdet/exports/real_gt_pvp --split train
cat /dataset/realgt_liverdet/exports/real_gt_ap/counts.csv
cat /dataset/realgt_liverdet/exports/real_gt_pvp/counts.csv
```

查看 `qc_train.jpg`。如BBox错位、被标成阴性的切片存在实际肿瘤、遮盖区域并非完整标注，必须先修复数据，不能依靠继续训练补救。

## 5. 100 Epoch上限 + 每轮Val + 早停 + 最优权重最终Test

RTX5090 + YOLOv8s建议先用batch32、imgsz640。与原512PNG不冲突：网络内会缩放，标签同步缩放。

```bash
OUT=/dataset/realgt_liverdet
realgt train \
  --dataset "$OUT/exports/real_gt_ap" \
  --out "$OUT/runs/yolov8s_ap_seed42" \
  --epochs 100 --batch 32 --imgsz 640 --patience 15 --device 0 --seed 42

realgt train \
  --dataset "$OUT/exports/real_gt_pvp" \
  --out "$OUT/runs/yolov8s_pvp_seed42" \
  --epochs 100 --batch 32 --imgsz 640 --patience 15 --device 0 --seed 42
```

Ultralytics每Epoch在Val上验证，根据Val综合指标更新 `weights/best.pt`，若连续15轮未改善则提前停止。**Test不参与训练、早停、阈值选择或Checkpoint选择**；训练结束后自动用 `best.pt` 推断独立Test一次。

```text
runs/yolov8s_ap_seed42/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── best_checkpoint.txt
├── test_predictions.json
└── test_metrics.json
```

结果包括COCO `ap_50_95`、`ap50`、`ap75` 和**2D切片级** FROC（不是3D病灶级FROC）。生成 `test_metrics.json` 后不要反复用Test指标调整模型；后续调整应只参考Val。

## 6. 实验风险及公平比较

1. 原实验前14轮训练Loss显著下降，而Val mAP先高后低：可能是教师伪标签误差、Train/Val分布不一致、错位框、强类不平衡或过拟合；**仅凭日志不能断言唯一原因**。
2. 全GT样本量比伪标签样本少，最终泛化性能**不保证**更高。建议先比较训练和Val切片数及阳性率、可视化标签。
3. 固定100 Epoch不等于固定优化步数；REAL/SYN比较时需记录每组真实训练步数和增强策略。
4. 如果未来要与旧伪标签方案公平比较，两个方法必须使用**同一Test患者及同一人工GT**；旧Test患者不能被并入新训练集。
5. 患者病例ID和影像文件只保留在服务器；不要上传带患者ID的 `cases.csv`/`splits.csv`/NIfTI/PNG到公开GitHub。

## 7. 快速自检

```bash
python -m pytest -q
realgt --help
realgt build --help
realgt train --help
```

这是一个完全独立的Python项目，不会修改 `/workspace/detection`，也不会覆写 `/dataset/downstream_detection_cc4`。
