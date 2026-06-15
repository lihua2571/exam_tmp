import os
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer

# 可调参数集中在此（方便统一修改）
MAX_LENGTH = 128
TRAIN_SIZE = 2500
TEST_SIZE = 2500
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 2

def load_imdb_from_local(base_path="../data/imdb_dataset"):
    """从本地文件夹加载 IMDB 数据集"""
    data = {"train": {"text": [], "label": []}, "test": {"text": [], "label": []}}
    for split in ["train", "test"]:
        for label_name, label_id in [("pos", 1), ("neg", 0)]:
            dir_path = os.path.join(base_path, split, label_name)
            for filename in os.listdir(dir_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
                        text = f.read()
                    data[split]["text"].append(text)
                    data[split]["label"].append(label_id)
    return DatasetDict({
        "train": Dataset.from_dict(data["train"]),
        "test": Dataset.from_dict(data["test"]),
    })

def get_tokenized_datasets(tokenizer_path="../model/bert-base-uncased"):
    """加载、分词、采样（与训练和评估共用）"""
    dataset = load_imdb_from_local()
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    # 采样，保证可复现
    tokenized["train"] = tokenized["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    tokenized["test"]  = tokenized["test"].shuffle(seed=42).select(range(TEST_SIZE))
    return tokenized, tokenizer

def find_best_checkpoint(results_dir="../results"):
    """自动寻找 results 下数字最大的 checkpoint 路径"""
    if not os.path.exists(results_dir):
        return None
    checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # 按数字排序，取最大
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(results_dir, checkpoints[-1])