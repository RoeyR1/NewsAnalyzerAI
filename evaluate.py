import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import json
import os
from tqdm import tqdm

from final_training import HeadlineScoreModel, HeadlineDataset
from app import predict_headline


def load_model_and_tokenizer(model_path, tokenizer_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeadlineScoreModel(n_regression_outputs=4, n_categories=5)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    return model, tokenizer, device


def evaluate_model(model, test_dataloader, device):
    model.eval()
    all_regression_preds = []
    all_regression_targets = []
    all_classification_preds = []
    all_classification_targets = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_regression = batch["labels_regression"].to(device)
            labels_multi_label = batch["labels_multi_label"].to(device)

            regression_outputs, classification_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            all_regression_preds.extend(regression_outputs.cpu().numpy())
            all_regression_targets.extend(labels_regression.cpu().numpy())
            all_classification_preds.extend(
                torch.sigmoid(classification_outputs).cpu().numpy())
            all_classification_targets.extend(labels_multi_label.cpu().numpy())

    regression_preds = np.array(all_regression_preds)
    regression_targets = np.array(all_regression_targets)
    classification_preds = np.array(all_classification_preds)
    classification_targets = np.array(all_classification_targets)

    metrics = {
        'mse_sentiment': mean_squared_error(regression_targets[:, 0], regression_preds[:, 0]),
        'mse_intensity': mean_squared_error(regression_targets[:, 1], regression_preds[:, 1]),
        'mse_urgency': mean_squared_error(regression_targets[:, 2], regression_preds[:, 2]),
        'mse_controversy': mean_squared_error(regression_targets[:, 3], regression_preds[:, 3]),
        'f1_score': f1_score(classification_targets, (classification_preds >= 0.5).astype(int), average='micro')
    }

    return metrics


def main():
    model_path = 'news_ai/distilbert_news_classifier_epoch16.pth'
    tokenizer_path = 'news_ai/distilbert_news_tokenizer'
    test_data_path = 'new_dataset_cleaned.csv'

    print("Loading model and tokenizer...")
    model, tokenizer, device = load_model_and_tokenizer(
        model_path, tokenizer_path)

    print("Loading test data...")
    test_data = pd.read_csv(test_data_path)

    test_dataset = HeadlineDataset(
        headlines=test_data['Headline'].tolist(),
        labels_regression=test_data[[
            'Sentiment', 'Intensity', 'Urgency', 'Controversy']].values,
        labels_multi_label=test_data[[
            'Health', 'Environment', 'Technology', 'Finance', 'Politics']].values,
        weights=np.ones(len(test_data)),
        tokenizer=tokenizer,
        max_len=183
    )

    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("Evaluating model...")
    metrics = evaluate_model(model, test_dataloader, device)

    print("\nModel Performance Metrics:")
    print("-" * 30)
    print(f"Sentiment MSE: {metrics['mse_sentiment']:.4f}")
    print(f"Intensity MSE: {metrics['mse_intensity']:.4f}")
    print(f"Urgency MSE: {metrics['mse_urgency']:.4f}")
    print(f"Controversy MSE: {metrics['mse_controversy']:.4f}")
    print(f"Classification F1 Score: {metrics['f1_score']:.4f}")

    metrics_path = 'news_ai/evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
