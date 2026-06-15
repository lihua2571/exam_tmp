import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from utils import get_tokenized_datasets, find_best_checkpoint, BATCH_SIZE

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

def main():
    print("加载测试数据...")
    tokenized_datasets, _ = get_tokenized_datasets()
    test_dataset = tokenized_datasets["test"]
    
    checkpoint_path = find_best_checkpoint()          # 现在自动指向 ../results
    if checkpoint_path is None:
        raise FileNotFoundError("未找到任何 checkpoint，请先运行 train.py")
    print(f"加载模型: {checkpoint_path}")
    
    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    
    training_args = TrainingArguments(
        output_dir="../results/temp_eval",    # 改这里（可保留为 temp_eval 避免覆盖）
        per_device_eval_batch_size=BATCH_SIZE,
        do_train=False,
        do_eval=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    results = trainer.evaluate()
    print(f"最终结果: Accuracy = {results['eval_accuracy']:.4f}, F1 = {results['eval_f1']:.4f}")
    # 保存结果到 results 文件夹
    with open("../results/evaluation_results.txt", "w") as f:   # 改这里
        f.write(f"Accuracy: {results['eval_accuracy']}\n")
        f.write(f"F1 Score: {results['eval_f1']}\n")
        f.write(f"Loss: {results['eval_loss']}\n")
    
    # 混淆矩阵
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.savefig("../results/confusion_matrix.png")    # 改这里
    print("混淆矩阵已保存至 ../results/confusion_matrix.png")
    plt.close()
    
    # 错误案例（加载原始文本）
    mis_idx = np.where(preds != labels)[0]
    print(f"总错误样本数: {len(mis_idx)} / {len(labels)}")
    print("前10个错误样例（真实 -> 预测）：")
    from utils import load_imdb_from_local
    raw_dataset = load_imdb_from_local()          # 默认路径已是 ../data/imdb_dataset
    for i in mis_idx[:10]:
        print(f"真实: {labels[i]}, 预测: {preds[i]}")
        print(raw_dataset["test"][i]["text"][:300])
        print("---")

if __name__ == "__main__":
    main()