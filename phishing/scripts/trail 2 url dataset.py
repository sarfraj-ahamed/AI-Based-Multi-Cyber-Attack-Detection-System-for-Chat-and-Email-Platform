import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import re
import urllib.parse
from tldextract import extract
import math
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Constants
MAX_LEN = 128  # URL length
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 2e-5
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
LSTM_UNITS = 64
NUM_FEATURES = 17  # Number of handcrafted features

# Path configuration
DATASET_FOLDER = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\clean_dataset\url"  # Folder containing multiple CSV files
MODEL_SAVE_PATH = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\model_1\enhanced_url_phishing_model.pth"
RESULTS_SAVE_PATH = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\model_1\enhanced_url_results.csv"

# Suspicious keywords in URLs/parameters
SUSPICIOUS_KEYWORDS = [
    'login', 'account', 'bank', 'paypal', 'secure', 'verify', 'password',
    'confirm', 'update', 'personal', 'credit', 'card', 'ssn', 'click'
]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_datasets_from_folder(folder_path):
    """Load all CSV files from a folder and combine them"""
    dfs = []
    problematic_files = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)

                # Check for required columns
                if 'url' not in df.columns or 'label' not in df.columns:
                    problematic_files.append((file, "Missing required columns (url or label)"))
                    continue

                # Check for NaN values
                nan_check = df[['url', 'label']].isna().sum()
                if nan_check.any():
                    problematic_files.append((file, f"NaN values found: {nan_check.to_dict()}"))
                    df = df.dropna(subset=['url', 'label'])

                dfs.append(df)
            except Exception as e:
                problematic_files.append((file, str(e)))

    if problematic_files:
        print("\n=== Problematic Files Report ===")
        for file, issue in problematic_files:
            print(f"\nFile: {file}")
            print(f"Issue: {issue}")

    if not dfs:
        raise ValueError("No valid datasets found in the folder")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


class URLDataset(Dataset):
    def __init__(self, urls, features, labels, tokenizer, max_len):
        self.urls = urls
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = str(self.urls[idx])
        features = self.features[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            url,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.FloatTensor(features),
            'label': torch.tensor([label], dtype=torch.float)

        }


def extract_url_features(url):
    """Extract handcrafted features from URL"""
    if not isinstance(url, str) or pd.isna(url):
        return [0.0] * NUM_FEATURES

    features = []
    try:
        parsed = urllib.parse.urlparse(url)
        ext = extract(url)
    except Exception as e:
        # return zero features if URL is invalid
        return [0.0] * NUM_FEATURES

    subdomain = ext.subdomain
    domain = ext.domain
    suffix = ext.suffix

    # Basic URL features
    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('_'))
    features.append(url.count('/'))
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('&'))
    features.append(url.count('%'))

    # Domain features
    features.append(len(domain))
    features.append(len(subdomain))
    features.append(len(suffix))
    features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0)

    # Query parameter features
    num_params = 0
    param_length = 0
    suspicious_keywords = 0

    if parsed.query:
        params = urllib.parse.parse_qs(parsed.query)
        num_params = len(params)
        param_length = len(parsed.query)

        # Check for suspicious keywords in parameters
        for param, values in params.items():
            lower_param = param.lower()
            if any(keyword in lower_param for keyword in SUSPICIOUS_KEYWORDS):
                suspicious_keywords += 1
            for value in values:
                if any(keyword in value.lower() for keyword in SUSPICIOUS_KEYWORDS):
                    suspicious_keywords += 1

    features.append(num_params)
    features.append(param_length)
    features.append(suspicious_keywords)

    # Entropy calculation
    if len(url) > 0:
        prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
    else:
        entropy = 0
    features.append(entropy)

    return features


def preprocess_urls(urls):
    """Minimal URL preprocessing keeping query parameters"""
    processed_urls = []
    for url in urls:
        if not isinstance(url, str) or pd.isna(url):
            processed_urls.append("")
            continue
        processed_urls.append(url.lower())
    return processed_urls


class HybridURLModel(nn.Module):
    def __init__(self, lstm_units=LSTM_UNITS, num_features=NUM_FEATURES):
        super(HybridURLModel, self).__init__()

        # DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained(DISTILBERT_MODEL_NAME)
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.distilbert.config.hidden_size,
            hidden_size=lstm_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)

        # Feature processing
        self.feature_layer = nn.Linear(num_features, 32)

        # Combined layers
        self.combined = nn.Linear(lstm_units * 2 + 32, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask, features):
        # DistilBERT output
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

        # BiLSTM output
        lstm_output, _ = self.lstm(distilbert_output)
        lstm_output = self.dropout(lstm_output[:, -1, :])

        # Feature processing
        features_output = torch.relu(self.feature_layer(features))

        # Combine features
        combined = torch.cat([lstm_output, features_output], dim=1)
        combined = torch.relu(self.combined(combined))

        # Final output
        output = torch.sigmoid(self.output(combined))

        return output


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def main():
    # Load datasets from folder
    print("\nLoading datasets from folder...")
    df = load_datasets_from_folder(DATASET_FOLDER)

    # Convert labels if they're strings
    # Normalize label types (convert all to string)
    # Normalize labels to 0 and 1
    label_map = {
        'Phishing': 1, 'phishing': 1, '1': 1, 1: 1,
        'Legitimate': 0, 'legitimate': 0, '0': 0, 0: 0
    }
    df['label'] = df['label'].map(label_map)

    # Drop invalid or unrecognized labels
    df = df[df['label'].isin([0, 1])]
    print("✅ Cleaned label distribution:\n", df['label'].value_counts())

    # Extract features
    print("\nExtracting URL features...")
    df['features'] = df['url'].apply(extract_url_features)
    features = np.array(df['features'].tolist())

    # Preprocess URLs
    urls = preprocess_urls(df['url'].tolist())
    labels = df['label'].values

    # Split data
    train_urls, temp_urls, train_features, temp_features, train_labels, temp_labels = train_test_split(
        urls, features, labels, test_size=0.3, random_state=42
    )
    val_urls, test_urls, val_features, test_features, val_labels, test_labels = train_test_split(
        temp_urls, temp_features, temp_labels, test_size=0.5, random_state=42
    )

    print(f"\nFinal dataset sizes:")
    print(f"Train: {len(train_urls)}")
    print(f"Validation: {len(val_urls)}")
    print(f"Test: {len(test_urls)}")

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

    # Create datasets
    train_dataset = URLDataset(train_urls, train_features, train_labels, tokenizer, MAX_LEN)
    val_dataset = URLDataset(val_urls, val_features, val_labels, tokenizer, MAX_LEN)
    test_dataset = URLDataset(test_urls, test_features, test_labels, tokenizer, MAX_LEN)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = HybridURLModel().to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nTraining model...")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Train
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model!")

    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, y_pred, y_true = evaluate_model(model, test_loader, criterion, device)

    # Calculate metrics
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Save results
    results = {
        'Test Accuracy': test_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    pd.DataFrame.from_dict(results, orient='index').to_csv(RESULTS_SAVE_PATH, header=False)

    print("\n=== Final Metrics ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Plot results
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()