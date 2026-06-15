import torch
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils import (
    get_tokenized_datasets, MAX_LENGTH, TRAIN_SIZE, TEST_SIZE,
    BATCH_SIZE, LEARNING_RATE, EPOCHS
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}

def main():
    print("加载数据...")
    tokenized_datasets, tokenizer = get_tokenized_datasets()
    
    print("加载模型...")
    model = BertForSequenceClassification.from_pretrained(
        "../model/bert-base-uncased", num_labels=2    # 改这里
    )
    
    training_args = TrainingArguments(
        output_dir="../results",          # 改这里
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir="../results/logs",    # 改这里
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("开始训练...")
    trainer.train()
    
    # 绘制损失曲线
    log_history = trainer.state.log_history
    train_steps, train_losses = [], []
    eval_epochs, eval_losses = [], []
    
    for log in log_history:
        if 'loss' in log and 'epoch' in log:
            train_steps.append(log['epoch'])
            train_losses.append(log['loss'])
        if 'eval_loss' in log:
            eval_epochs.append(log['epoch'])
            eval_losses.append(log['eval_loss'])
    
    plt.figure(figsize=(10,5))
    plt.plot(train_steps, train_losses, label='Train Loss', alpha=0.7)
    plt.plot(eval_epochs, eval_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/loss_curve.png')    # 改这里
    print("损失曲线已保存至 ../results/loss_curve.png")

if __name__ == "__main__":
    main()