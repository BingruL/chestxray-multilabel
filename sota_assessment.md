# ChestXray14 项目策略评估与改进建议（21.8k 子集）

## 当前配置亮点
- Patient-wise 划分：避免同一病人泄漏，符合医学影像评估规范。
- 模型多样性：ConvNeXt / ViT / EfficientNet（ImageNet-21K/1K 预训练），利于集成互补。
- 损失与类不平衡：BCE + Focal + 可选 label smoothing，配合 per-class pos_weight。
- 指标与阈值：宏 AUC / 宏 F1，验证集逐类搜索阈值优于固定 0.5。
- 工程与可复现：保存验证预测 .npz 供集成；AMP + AdamW + Cosine 训练流程规范。

## 主要差距（对标 SOTA）
- 预训练：仅 ImageNet 预训练；缺少 CXR 专用或多源自监督/监督预训练（CheXpert/MIMIC/MAE 等）。
- 分辨率：多用 224/380/384，细小病灶可能不足；高分方法常用 512/640。
- 不平衡与噪声：虽有 pos_weight+Focal，但 ChestX-ray14 噪声/长尾严重，仍缺更强的不平衡与噪声鲁棒策略（如 ASL、class-balanced loss、logit adjustment、EMA、重加权/采样）。
- 训练配方：epoch 较短（15），缺少 warmup、梯度裁剪、EMA；ViT/ConvNeXt 往往受益于更长训练和更高正则。
- 集成与推理：目前仅简单概率平均；未做加权/学习权重的stacking，也未启用TTA；阈值应用到最终推理需明确（阈值虽搜了，但未描述在测试/提交阶段的固定应用）。

## 高优先级改进（按性价比排序）
1) 引入 CXR 专用或更强预训练
   - 直接尝试已有公开权重：torchxrayvision 的 DenseNet121/ResNet50；CheXpert/MIMIC 自监督 MAE/SimMIM/ConvNeXt/ViT 权重；或用 timm 的 EVA/SwinV2/ConvNeXtV2 21K/MAE 变体。
   - 若时间允许，用当前 21k 做轻量自监督（MAE/SimMIM）再微调；即使数据少，也可能带来 1–2 个点 AUC/F1。
2) 更强的不平衡/噪声处理
   - 损失：尝试 Asymmetric Loss (ASL) 或 BCE + Focal + label smoothing 轻量调优（gamma、smoothing）。
   - 采样/权重：类频率驱动的 reweight（class-balanced loss/logit adjustment）或正例过采样（保持 patient-wise 不重叠）。
   - EMA：在训练中维护 EMA 权重，常带来稳定提升。
   - 可选：Co-teaching / teacher-student 做噪声鲁棒，但实现成本高。
3) 分辨率与增强
   - 提升输入到 512（至少给 ConvNeXt/EfficientNet 试一版），必要时减小 batch 并开启 AMP。
   - 增强：在现有基础上小幅加 ColorJitter/AutoContrast/RandomAdjustSharpness；保持医学合理性，避免过重模糊与极端裁剪。
   - 轻量 TTA：水平翻转 + 多尺度（如 0.9/1.0/1.1 resize 后中心/五点裁剪）在推理时平均。
4) 训练配方细化
   - 学习率日程：加 5–10% 线性 warmup，再 Cosine；epoch 数拉到 25–40（注意早停或监控）。
   - 正则：Weight decay 适当调优（ConvNeXt/ViT 常用 0.05–0.1），Mixup（α 0.2–0.4）对多标签有效；CutMix 在胸片上需谨慎。
   - 梯度裁剪（如 1.0）防止不稳定。
5) 集成与阈值
   - 继续跑三大家族（ConvNeXt、ViT、EffNet）并保存验证预测；在验证集上用学习权重的线性融合或 logistic stacking，通常优于简单均值。
   - 在最终推理时固定“验证集搜到的 per-class 阈值”；如做集成，阈值基于集成概率重新搜索一次。
   - 可选：校准（温度缩放）提升概率质量，有助于 F1/PR-AUC。

## 预期性能区间（21.8k 子集，经验估计）
- 单模型宏 AUC ~0.80–0.84，宏 F1 ~0.25–0.35。  
- 良好集成 + 阈值调优：可再抬 1–2 个点 AUC/F1。  
- 引入 CXR 预训练 + 512 分辨率 + 加权集成：有机会逼近 ~0.85 AUC、F1 0.35–0.4；再高需更大数据或更强预训练。

## 下一步可执行清单
- 训练并保存验证预测：ConvNeXt (21K, 224/384 或 512)、ViT (21K/MAE)、EffNet (B0/B4，512 可选)。
- 对验证预测做：单模型评估 + 学习权重的集成 + 重新搜阈值。
- 在关键模型上启用：warmup+Cosine、EMA、梯度裁剪、ASL 或调优的 BCE+Focal，提升稳定性。
- 若可获取：替换/微调成 CXR 专用预训练权重；推理开启 TTA 并使用验证集确定的 per-class 阈值。

