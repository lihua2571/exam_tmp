import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils import find_best_checkpoint, MAX_LENGTH

def load_model_and_tokenizer():
    checkpoint = find_best_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("未找到训练好的模型，请先运行 train.py")
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    return "Positive" if pred == 1 else "Negative"

def main():
    tokenizer, model, device = load_model_and_tokenizer()
    print("模型加载完成，输入 'exit' 退出")
    while True:
        text = input("请输入评论: ")
        if text.lower() == 'exit':
            break
        sentiment = predict_sentiment(text, tokenizer, model, device)
        print(f"情感: {sentiment}\n")

if __name__ == "__main__":
    main()