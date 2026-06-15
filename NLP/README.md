# 文本情感分析：从传统机器学习到预训练模型

> AI组能力测试1 - 方向一：自然语言处理（NLP）

## 项目概述

本项目围绕**文本分类/情感分析**任务，系统梳理从传统机器学习方法到深度学习、再到预训练大模型的技术演进路线，并基于IMDB数据集完成BERT微调实战。

- **阶段一**：文献综述与技术演进分析（BoW/TF-IDF → Word2Vec/CNN/LSTM → BERT/GPT）
- **阶段二**：IMDB二分类情感分析实战（BERT-base-uncased微调，Accuracy 89.08%，F1 0.8921）

---

## 环境配置

| 项目           | 版本                            |
|:------------ |:----------------------------- |
| 操作系统         | Windows 11                    |
| Python       | 3.10.20 (Anaconda)            |
| PyTorch      | 2.7.1+cu118                   |
| CUDA         | 11.8                          |
| GPU          | NVIDIA GeForce RTX 4060 (8GB) |
| Transformers | 5.10.2                        |
| Datasets     | 4.8.5                         |
| scikit-learn | 1.7.2                         |
| matplotlib   | 3.10.9                        |
| numpy        | 2.2.6                         |
| pandas       | 2.3.3                         |
| tqdm         | 4.67.3                        |
| accelerate   | 1.13.0                        |
| safetensors  | 0.7.0                         |

### 环境安装

依赖文件 `requirements.txt` 位于项目根目录 `NLP/` 下

```bash
# 创建conda环境
conda create -n nlp python=3.10
conda activate nlp

# 安装依赖（推荐）
pip install -r requirements.txt
```

---

## 模型获取

本项目使用 **BERT-base-uncased** 作为预训练模型。

建议 : 本仓库已包含本地数据集和预训练模型，下载后可直接运行，无需额外下载。

BERT-base-uncased：约 440 MB

- **来源**：
  - 方式一：Hugging Face 自动下载（需联网，首次运行自动缓存）
  - **方式二（推荐）**：手动下载后放入 `./model/bert-base-uncased/`
- **手动下载地址**：https://huggingface.co/google-bert/bert-base-uncased/tree/main(国内访问可能需要网络代理)
- **需要下载的文件**：
  - `config.json`
  - `pytorch_model.bin`
  - `tokenizer_config.json`
  - `vocab.txt`
- **本地模型位置**：`./model/bert-base-uncased/`

---

## 数据获取

本项目使用 **IMDB 影评数据集**（50,000条，已划分train/test，正负平衡）。

 IMDB 数据集：约 80 MB

- **来源**：
  - 方式一：Hugging Face `datasets` 库自动下载（需联网）
  - **方式二（推荐）**：手动下载后放入 `./data/imdb_dataset/`
- **手动下载地址**：
  - Hugging Face：https://huggingface.co/datasets/stanfordnlp/imdb
  - 直接下载：https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
- **本地数据位置**：`./data/imdb_dataset/`
- **数据规模**：
  - 训练集：25,000条
  - 测试集：25,000条

---

## 项目结构

```
AI-Assessment/
├── NLP_华永航.pdf             # PDF报告（根目录）
├── Assessment.pdf             # 原考核题目
├── README.md                  # 总运行说明
├── submission_format.md       
├── CV/                        # 未选方向
├── MultiModal/                # 未选方向
└── NLP/                       # ← 选中的方向
    ├── src/
    │   ├── train.py           # 训练代码
    │   ├── utils.py           # 公用函数
    │   ├── evaluate.py        # 评估代码
    │   └── inference.py       # 推理代码
    ├── data/                  # ← 本地数据集（IMDB）
    │   └── imdb_dataset/      # 手动下载的IMDB数据
    │       ├── train/
    │       │   ├── pos/       # 正面评论
    │       │   └── neg/       # 负面评论
    │       └── test/
    │           ├── pos/
    │           └── neg/
    │   
    ├── model/                # ← 本地预训练模型
    │   └── bert-base-uncased/ # 手动下载的BERT模型
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       ├── tokenizer_config.json
    │       └── vocab.txt
    ├── results/               # 实验结果
    │   ├── checkpoint-*/      # 模型检查点
    │   ├── loss_curve.png
    │   └── confusion_matrix.png
    ├── requirements.txt
    └── README.md              # NLP方向详细说明
```

---

## 运行流程

### 1. 训练（BERT微调）

```bash
# 进入项目根目录
cd NLP
python src/train.py
```

**关键参数**（可在代码中修改）：

- `max_length=128`（序列长度）
- `batch_size=16`
- `learning_rate=2e-5`
- `num_epochs=3`
- `weight_decay=0.01`

**输出**：

- 模型检查点保存至 `./results/checkpoint-*/`
- 训练日志（loss/accuracy/F1）实时打印
- 损失曲线自动保存至 `./results/loss_curve.png`

**预期时间**：约20分钟（RTX 4060，全量25,000条数据）

---

### 2. 评估（测试集性能+错误分析）

```bash
python src/evaluate.py
```

**功能**：

- 加载最佳checkpoint（自动查找）
- 在25,000条测试集上计算Accuracy和F1-Score
- 生成混淆矩阵（`./results/confusion_matrix.png`）
- 打印错误样本（前10个），用于错误分析

**输出示例**：

```
最终结果: Accuracy = 0.8908, F1 = 0.8921
总错误样本数: 2729 / 25000
```

---

### 3. 推理（单条文本预测）

```bash
python src/inference.py --text "This movie is fantastic!"
```

或修改 `inference.py` 中的 `sample_text` 直接运行。

---

## 实验结果

### 核心指标

| 指标           | 数值         | 说明                 |
|:------------ |:---------- |:------------------ |
| **Accuracy** | **0.8908** | 测试集25,000条         |
| **F1-Score** | **0.8921** | 以positive类为关注类别    |
| Eval Loss    | 0.4533     | Epoch 3 checkpoint |

### 训练过程关键发现

| Epoch | 验证 Loss    | 验证 Accuracy | 验证 F1          |
|:----- |:---------- |:----------- |:-------------- |
| 1     | 0.3609     | 0.8722      | 0.8815         |
| **2** | **0.2996** | **0.8930**  | **0.8948**（最佳） |
| 3     | 0.4533     | 0.8908      | 0.8921（最终）     |

> **关键发现**：Epoch 2 的验证 F1（0.8948）为训练过程中最高值，但 `load_best_model_at_end=True` 默认按loss选择，最终加载了Epoch 3的checkpoint。验证loss在Epoch 2后大幅反弹（0.2996→0.4533），呈现典型过拟合曲线，提示**早停**的必要性。

### 可视化

- **训练/验证损失曲线**：`./results/loss_curve.png`
  
  - 训练loss持续下降，验证loss在Epoch 2后反弹，过拟合明显

- **测试集混淆矩阵**：`./results/confusion_matrix.png`
  
  - 假正例：1221，假负例：1508

---

## 关键结论

1. **BERT微调在IMDB任务上表现优异**：Accuracy 89.08%，接近90%目标，验证了预训练模型在下游情感分类任务中的有效性。

2. **模型对复杂语义结构仍有局限**：错误案例分析揭示，BERT对转折结构（but/although）、评分矛盾（"8/10" vs "5/10"）和领域特殊表达（cult film）的泛化能力不足。

3. **早停可进一步提升性能**：Epoch 2的F1（0.8948）高于最终值，采用早停策略有望稳定获得更高指标。

4. **训练效率较高**：消费级GPU（RTX 4060 8GB）上全量数据3 epoch仅需约20分钟，BERT微调在实际环境中可行。

---

## 改进方向

- **早停与正则化**：验证loss不再下降时停止，或增加weight_decay
- **数据增强**：对含转折词样本过采样，或回译生成变体
- **模型升级**：尝试RoBERTa/DeBERTa（通常高1-2个百分点）
- **更长上下文**：max_length提升至512（需降低batch_size或梯度累积）

---

## AI使用声明

本项目在撰写过程中使用了DeepSeek Kimi辅助：

- 整理文献综述的思路和关键点
- 调试代码中的Bug（损失曲线绘制、checkpoint路径查找、LayerNorm命名警告分析）
- 润色报告文字，使其更符合学术规范

所有核心技术内容（实验设计、超参数选择、错误分析、调参决策）均由作者独立完成，AI仅提供信息检索和语言支持。
