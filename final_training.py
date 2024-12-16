# final_training.py

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
from tqdm import tqdm
import time
import os
import pickle
import csv

# SETUP AND DATA LOADING


os.makedirs('news_ai', exist_ok=True)

data_path = 'new_dataset_cleaned.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"dataset: '{data_path}' does not exist")


expected_columns = ['Headline', 'Sentiment', 'Intensity', 'Urgency',
                    'Controversy', 'Health', 'Environment', 'Technology',
                    'Finance', 'Politics']

# List for our processed rows
processed_rows = []
skipped_rows = 0

# Essentially to get dataset in the right format, removes anything that is not followign specified format
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("empty file")

    header = [col.strip() for col in header]
    if header != expected_columns:
        print(f"Header does not match")

    # Starting at line 2 considering header is line 1
    for line_number, row in enumerate(reader, start=2):
        if len(row) < 10:
            print(f"Line {line_number}, Not enough columns.")
            skipped_rows += 1
            continue
        elif len(row) == 10:
            # properly formatted row
            processed_rows.append(row)
        else:
            # wrongly formatted row
            headline = ','.join(row[:len(row)-9]).strip()
            sentiment = row[-9].strip()
            intensity = row[-8].strip()
            urgency = row[-7].strip()
            controversy = row[-6].strip()
            health = row[-5].strip()
            environment = row[-4].strip()
            technology = row[-3].strip()
            finance = row[-2].strip()
            politics = row[-1].strip()

            processed_rows.append([headline, sentiment, intensity, urgency,
                                   controversy, health, environment, technology,
                                   finance, politics])

if skipped_rows > 0:
    print(f"Skipped {skipped_rows} broken rows during CSV processing.")

# Create DataFrame
columns = expected_columns
data = pd.DataFrame(processed_rows, columns=columns)

# DATA VALIDATION

# define data types for csv reading and usage
dtype_spec = {
    'Headline': str,
    'Sentiment': float,
    'Intensity': float,
    'Urgency': float,
    'Controversy': float,
    'Health': float,
    'Environment': float,
    'Technology': float,
    'Finance': float,
    'Politics': float
}

for col, dtype in dtype_spec.items():
    try:
        data[col] = data[col].astype(dtype)
    except ValueError as ve:
        print(f"Error converting column '{col}' to {dtype}: {ve}")
        data[col] = pd.to_numeric(
            data[col], errors='coerce') if dtype != str else data[col].astype(str)

data.columns = data.columns.str.strip()

required_columns = set(expected_columns)
missing = required_columns - set(data.columns)
if missing:
    raise ValueError(f"Missing the following column {missing}")
else:
    print("All required columns are present.")


def validate_regression(row):
    return all(0 <= row[col] <= 10 for col in ['Sentiment', 'Intensity', 'Urgency', 'Controversy'])


def validate_categories(row):
    return all(row[col] in [0, 1] for col in ['Health', 'Environment', 'Technology', 'Finance', 'Politics'])


Validated_regression = data.apply(validate_regression, axis=1)
Validated_categories = data.apply(validate_categories, axis=1)
valid_data = data[Validated_regression &
                  Validated_categories].reset_index(drop=True)


num_removed = len(data) - len(valid_data)
if num_removed > 0:
    print(f"Removed {num_removed} rows due to invalid data.")
else:
    print("Perfect, Nothing removed")

# PREPARING THE DATASET


def compute_sample_weight(regression_labels):
    weight = 1.0
    if any(label <= 0 or label >= 10 for label in regression_labels):
        weight = 3.0
    elif any(label == 1 or label == 9 for label in regression_labels):
        weight = 2.0
    elif any(label == 3 or label == 8 for label in regression_labels):
        weight = 1.5
    return weight


X = valid_data['Headline'].tolist()
y_regression = valid_data[['Sentiment',
                           'Intensity', 'Urgency', 'Controversy']].values
y_multi_label = valid_data[['Health', 'Environment',
                            'Technology', 'Finance', 'Politics']].values.astype(np.float32)

# prepares each of the weights for training
sample_weights = np.array([
    compute_sample_weight(label)
    for label in y_regression
])

# Split data into training and test sets


def weight_conversion(weight):
    if weight == 3.0:
        return 'weight_3'
    elif weight == 2.0:
        return 'weight_2'
    elif weight == 1.5:
        return 'weight_1.5'
    else:
        return 'weight_1'


weight_bins = np.array([weight_conversion(w) for w in sample_weights])

X_train, X_test, y_train_regression, y_test_regression, y_train_multi_label, y_test_multi_label, train_weights, test_weights = train_test_split(
    X,
    y_regression,
    y_multi_label,
    sample_weights,
    test_size=0.10,
    random_state=42,
    stratify=weight_bins  # Ensures the distribution of weights is maintained
)
# Number of training samples/tests
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# dataset and loader


class HeadlineDataset(Dataset):
    def __init__(self, headlines, labels_regression, labels_multi_label, weights, tokenizer, max_len):
        self.headlines = headlines
        self.labels_regression = labels_regression
        self.labels_multi_label = labels_multi_label
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline = self.headlines[idx]
        label_regression = self.labels_regression[idx]
        label_multi_label = self.labels_multi_label[idx]
        weight = self.weights[idx]

        encoding = self.tokenizer.encode_plus(
            headline,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'headline_text': headline,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_regression': torch.tensor(label_regression, dtype=torch.float),
            'labels_multi_label': torch.tensor(label_multi_label, dtype=torch.float),
            'weights': torch.tensor(weight, dtype=torch.float)
        }


# loading tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 183
BATCH_SIZE = 16  # Adjust based on your hardware capabilities
EPOCHS = 16  # Could expand to a larger number if you want to train more
# theoretically could be lower but for speed we kept it on higher end
LEARNING_RATE = 0.000125

train_dataset = HeadlineDataset(
    X_train, y_train_regression, y_train_multi_label, train_weights, tokenizer, MAX_LEN)
test_dataset = HeadlineDataset(
    X_test, y_test_regression, y_test_multi_label, test_weights, tokenizer, MAX_LEN)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model Classes and definition


class HeadlineScoreModel(nn.Module):
    def __init__(self, n_regression_outputs, n_categories):
        super(HeadlineScoreModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.regression_head = nn.Linear(
            self.bert.config.hidden_size, n_regression_outputs)
        self.classification_heads = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, 1) for _ in range(n_categories)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        regression_output = self.regression_head(output)
        classification_outputs = [head(output)
                                  for head in self.classification_heads]
        # Shape: (batch_size, n_categories)
        classification_output = torch.cat(classification_outputs, dim=1)
        return regression_output, classification_output


# SETUP FOR TRAINING
# utilizing CUda is not necessary do to the lack of gpu (trained on M1 macbook )
device = torch.device("cpu")

print(f"Using device: {device}")


model = HeadlineScoreModel(n_regression_outputs=4, n_categories=5).to(device)
# optimzer via learning rate
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn_regression = nn.MSELoss(reduction='none').to(device)
loss_fn_classification = nn.BCEWithLogitsLoss(reduction='none').to(device)

# Checkpoint Loading
# essential for being able to pause training between epochs without causing an error
# training took over 4 days so this had a HUGE role


def load_latest_checkpoint(save_dir, model, optimizer):
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith(
        'checkpoint_epoch') and f.endswith('.pth')]
    if not checkpoints:
        print("Starting from 0, none found")
        return 0  # Starting from 0

    # Checkpoint functionality and loading via torch
    checkpoints.sort(key=lambda x: int(
        x.split('checkpoint_epoch')[1].split('.pth')[0]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(save_dir, latest_checkpoint)
    print(f"loading epoch: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch


start_epoch = load_latest_checkpoint('news_ai', model, optimizer)

# Training and eval functions


def train_epoch(model, dataloader, loss_fn_regression, loss_fn_classification, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_regression = batch["labels_regression"].to(device)
        labels_multi_label = batch["labels_multi_label"].to(device)
        weights = batch["weights"].to(device).unsqueeze(
            1)  # Shape: (batch_size, 1)

        optimizer.zero_grad()

        regression_outputs, classification_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask)

        loss_regression = loss_fn_regression(
            regression_outputs, labels_regression)  # Shape: (batch_size, 4)
        loss_regression = (loss_regression.mean(dim=1)
                           * weights.squeeze()).mean()
        loss_classification = loss_fn_classification(
            classification_outputs, labels_multi_label)  # Shape: (batch_size, 5)
        loss_classification = (loss_classification.mean(
            dim=1) * weights.squeeze()).mean()
        loss = loss_regression + loss_classification
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(losses)
# Eval Function during trading


def eval_model(model, dataloader, loss_fn_regression, loss_fn_classification, device):
    model.eval()
    losses = []
    preds_regression = []
    preds_multi_label = []
    targets_regression = []
    targets_multi_label = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_regression = batch["labels_regression"].to(device)
            labels_multi_label = batch["labels_multi_label"].to(device)
            weights = batch["weights"].to(device).unsqueeze(
                1)  # Shape: (batch_size, 1)

            regression_outputs, classification_outputs = model(
                input_ids=input_ids, attention_mask=attention_mask)

            loss_regression = loss_fn_regression(
                regression_outputs, labels_regression)  # Shape: (batch_size, 4)
            loss_regression = (loss_regression.mean(dim=1)
                               * weights.squeeze()).mean()

            loss_classification = loss_fn_classification(
                classification_outputs, labels_multi_label)  # Shape: (batch_size, 5)
            loss_classification = (loss_classification.mean(
                dim=1) * weights.squeeze()).mean()

            loss = loss_regression + loss_classification
            losses.append(loss.item())

            preds_regression.extend(regression_outputs.cpu().numpy())
            preds_multi_label.extend(torch.sigmoid(
                classification_outputs).cpu().numpy())
            targets_regression.extend(labels_regression.cpu().numpy())
            targets_multi_label.extend(labels_multi_label.cpu().numpy())

    return np.mean(losses), preds_regression, targets_regression, preds_multi_label, targets_multi_label


# epoch loop training
for epoch in range(start_epoch, EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    start_time = time.time()

    train_loss = train_epoch(
        model, train_dataloader, loss_fn_regression, loss_fn_classification, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}")

    test_loss, preds_regression, targets_regression, preds_multi_label, targets_multi_label = eval_model(
        model, test_dataloader, loss_fn_regression, loss_fn_classification, device)
    print(f"Test Loss: {test_loss:.4f}")

    preds_regression = np.array(preds_regression)
    targets_regression = np.array(targets_regression)

    MSE_sentiment = mean_squared_error(
        targets_regression[:, 0], preds_regression[:, 0])
    MSE_intensity = mean_squared_error(
        targets_regression[:, 1], preds_regression[:, 1])
    MSE_urgency = mean_squared_error(
        targets_regression[:, 2], preds_regression[:, 2])
    MSE_controversy = mean_squared_error(
        targets_regression[:, 3], preds_regression[:, 3])

    print(f"MSE (Sentiment): {MSE_sentiment:.4f}")
    print(f"MSE (Intensity): {MSE_intensity:.4f}")
    print(f"MSE (Urgency): {MSE_urgency:.4f}")
    print(f"MSE (Controversy): {MSE_controversy:.4f}")
    preds_multi_label = np.array(preds_multi_label)
    targets_multi_label = np.array(targets_multi_label)

    preds_multi_label_binary = (preds_multi_label >= 0.5).astype(int)
    f1 = f1_score(targets_multi_label,
                  preds_multi_label_binary, average='micro')
    print(f"F1 Score (Category): {f1:.4f}")

    # Checkpoint after each epoch
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }
    checkpoint_save_path = f"news_ai/checkpoint_epoch{epoch + 1}.pth"
    torch.save(checkpoint, checkpoint_save_path)
    print(f"Checkpoint saved to {checkpoint_save_path}")

    model_save_path = f"news_ai/distilbert_news_classifier_epoch{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Time for epoch {epoch + 1}: {epoch_time:.2f} seconds")

# save The tokenizer
tokenizer_save_path = "news_ai/distilbert_news_tokenizer"
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")

categories = ['Health', 'Environment', 'Technology', 'Finance', 'Politics']
category_to_idx = {category: idx for idx, category in enumerate(categories)}
idx_to_category = {idx: category for category, idx in category_to_idx.items()}

category_to_idx_path = "news_ai/category_to_idx.pkl"
idx_to_category_path = "news_ai/idx_to_category.pkl"

with open(category_to_idx_path, "wb") as f:
    pickle.dump(category_to_idx, f)
with open(idx_to_category_path, "wb") as f:
    pickle.dump(idx_to_category, f)
