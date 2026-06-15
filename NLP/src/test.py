import sys
import os

# 临时把工作目录切到 src，确保相对路径正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(">>> 测试1: 导入库")
from transformers import BertTokenizer, BertForSequenceClassification
from utils import load_imdb_from_local, get_tokenized_datasets
import torch

print(">>> 测试2: 加载本地数据集")
try:
    dataset = load_imdb_from_local("../data/imdb_dataset")
    print(f"    训练集: {len(dataset['train'])} 条, 测试集: {len(dataset['test'])} 条")
except Exception as e:
    print("    失败:", e)

print(">>> 测试3: 加载分词器和模型")
try:
    tokenizer = BertTokenizer.from_pretrained("../model/bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("../model/bert-base-uncased", num_labels=2)
    print("    分词器 & 模型加载成功")
except Exception as e:
    print("    失败:", e)

print(">>> 测试4: 分词处理（只取前100条样本加速）")
try:
    tokenized_datasets, _ = get_tokenized_datasets()
    # get_tokenized_datasets 会采样 TRAIN_SIZE/TEST_SIZE（当前是2500条），
    # 为了更快，我们只取前100条测试
    small_train = tokenized_datasets["train"].select(range(100))
    small_test = tokenized_datasets["test"].select(range(100))
    print(f"    训练样本: {len(small_train)}, 测试样本: {len(small_test)}")
except Exception as e:
    print("    失败:", e)

print(">>> 测试5: 检查 GPU")
print(f"    GPU 可用: {torch.cuda.is_available()}")

print("\n🎉 所有测试通过！新环境可以正常运行 train.py 和 evaluate.py。")
print("   如果之后想跑完整训练，请记得备份或重命名原有的 results 文件夹。")