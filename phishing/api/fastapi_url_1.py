from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

import torch
import torch.nn as nn
import re
from urllib.parse import urlparse

# Define the same model class from your Streamlit code
class HybridPhishingModel(nn.Module):
    def __init__(self):
        super(HybridPhishingModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.feature_layer = nn.Linear(17, 32)
        self.combined = nn.Linear(64 * 2 + 32, 64)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        lstm_features = lstm_out[:, -1, :]
        manual_features = self.feature_layer(features)
        combined = torch.cat([lstm_features, manual_features], dim=1)
        combined = torch.relu(self.combined(combined))
        return self.sigmoid(self.output(combined))

# URL features extraction
def extract_url_features(url: str):
    features = []
    features.append(len(url))
    features.append(sum(c.isdigit() for c in url))
    special_chars = re.compile(r'[~!@#$%^&*()_+{}|:"<>?`\-=\[\]\\;\',./]')
    features.append(len(special_chars.findall(url)))
    features.append(1 if url.startswith('https://') else 0)
    features.append(1 if url.startswith('http://') else 0)
    parsed = urlparse(url)
    features.append(parsed.netloc.count('.'))
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    features.append(1 if ip_pattern.search(url) else 0)
    shorteners = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly']
    features.append(1 if any(shortener in url for shortener in shorteners) else 0)
    features.append(1 if '@' in url else 0)
    features.append(url.count('-'))
    features.append(len(url.split('/')))
    features.append(1 if 'login' in url.lower() else 0)
    features.append(1 if 'account' in url.lower() else 0)
    features.append(len(url.split('?')))
    features.append(1 if '=' in url else 0)
    features.append(1 if '-' in parsed.netloc else 0)
    features.append(len(parsed.path))
    return torch.FloatTensor(features).unsqueeze(0)  # add batch dim

# Pydantic model for request
class URLRequest(BaseModel):
    url: str

app = FastAPI(title="URL Phishing Detection API")

# Load model and tokenizer on startup
model_path = r"D:\program\python\ai for treat detection\phishing\phishing\url_model_1\enhanced_url_phishing_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = HybridPhishingModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.post("/predict")
def predict_phishing(request: URLRequest):
    url = request.url

    # Basic input validation
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        # Tokenize input
        inputs = tokenizer(
            url,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract handcrafted features
        features = extract_url_features(url)

        # Move tensors to device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        features = features.to(device)

        # Predict
        with torch.no_grad():
            prediction = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            prob = prediction.item()

        # Create risk label based on probability
        if prob > 0.7:
            label = "high risk"
        elif prob > 0.4:
            label = "suspicious"
        else:
            label = "likely safe"

        return {
            "url": url,
            "phishing_probability": prob,
            "risk_label": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

