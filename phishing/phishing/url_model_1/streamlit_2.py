import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np
import os
import re
from urllib.parse import urlparse
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up the app
st.set_page_config(page_title="URL Phishing Detector", layout="wide")
st.title("🔗 URL Phishing Detector")
st.write("This tool analyzes URLs for potential phishing attempts using AI")


class HybridPhishingModel(nn.Module):
    def __init__(self):
        super(HybridPhishingModel, self).__init__()
        # DistilBERT components
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # BiLSTM components
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Manual feature processing (17 features)
        self.feature_layer = nn.Linear(17, 32)

        # Combined layers
        self.combined = nn.Linear(64 * 2 + 32, 64)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, features):
        # DistilBERT processing
        bert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state

        # BiLSTM processing
        lstm_out, _ = self.lstm(sequence_output)
        lstm_features = lstm_out[:, -1, :]

        # Manual features processing
        manual_features = self.feature_layer(features)

        # Combine all features
        combined = torch.cat([lstm_features, manual_features], dim=1)
        combined = torch.relu(self.combined(combined))

        return self.sigmoid(self.output(combined))


@st.cache_resource
def load_model_and_tokenizer(model_path):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = HybridPhishingModel()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def extract_url_features(url):
    features = []

    # Basic URL features (1–10)
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

    # Additional features (11–17)
    features.append(len(url.split('/')))
    features.append(1 if 'login' in url.lower() else 0)
    features.append(1 if 'account' in url.lower() else 0)
    features.append(len(url.split('?')))
    features.append(1 if '=' in url else 0)
    features.append(1 if '-' in parsed.netloc else 0)
    features.append(len(parsed.path))

    return torch.FloatTensor(features)


def main():
    # Load model file
    default_model_path = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\url_model_1\enhanced_url_phishing_model.pth"
    model_path = default_model_path

    if not os.path.exists(model_path):
        st.warning("Default model not found. Please upload your `.pth` model file.")
        uploaded_model = st.file_uploader("Upload your .pth model file", type=['pth'])
        if uploaded_model:
            with open("temp_model.pth", "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = "temp_model.pth"
        else:
            st.stop()  # Stop execution until model is uploaded

    # Load tokenizer and model
    tokenizer, model = load_model_and_tokenizer(model_path)
    if tokenizer is None or model is None:
        st.error("Failed to load model. Please check your model file.")
        return

    with st.form("url_form"):
        url = st.text_input("Enter URL to analyze:", placeholder="https://example.com")
        submitted = st.form_submit_button("Analyze URL")

    if submitted and url:
        try:
            inputs = tokenizer(
                url,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            features = extract_url_features(url).unsqueeze(0)

            with torch.no_grad():
                prediction = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    features=features
                )
                prob = prediction.item()

            # Show results
            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Phishing Probability", f"{prob * 100:.2f}%")
                st.progress(prob)

            with col2:
                if prob > 0.7:
                    st.error("🚨 High Risk: Likely phishing URL")
                elif prob > 0.4:
                    st.warning("⚠️ Suspicious: Potential phishing elements")
                else:
                    st.success("✅ Likely Safe: No phishing detected")

            with st.expander("Feature Analysis"):
                feature_names = [
                    "URL Length", "Digit Count", "Special Chars", "HTTPS", "HTTP",
                    "Subdomains", "Contains IP", "URL Shortener", "Has @", "Hyphen Count",
                    "Path Segments", "Contains 'login'", "Contains 'account'",
                    "Has Query", "Has Key-Value", "Domain Hyphen", "Path Length"
                ]
                for name, val in zip(feature_names, features.squeeze(0).numpy()):
                    st.write(f"{name}: {val}")

        except Exception as e:
            st.error(f"Error analyzing URL: {str(e)}")


if __name__ == "__main__":
    main()
