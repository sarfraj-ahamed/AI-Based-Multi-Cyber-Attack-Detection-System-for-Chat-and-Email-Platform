import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import numpy as np
import os

# Configure the app
st.set_page_config(page_title="Phishing Detector", layout="wide")
st.title("🔍 Phishing Message Detector")
st.write("This tool analyzes text messages for phishing attempts using AI")


@st.cache_resource
def load_model_components():
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Load DistilBERT model
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

        # Model parameters (adjust based on your model)
        max_len = 128
        lstm_units = 64

        # Rebuild model architecture
        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

        # DistilBERT layer
        bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]

        # BiLSTM layers
        lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(bert_output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_out)

        # Create model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        # Load weights
        model_path = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\nlp_model\hybrid_distilbert_bilstm_phishing.h5"
        model.load_weights(model_path)

        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        st.stop()


# Load components
with st.spinner("Loading model (this may take a minute)..."):
    tokenizer, model = load_model_components()

# User interface
with st.form("detection_form"):
    message = st.text_area("Enter the suspicious message:",
                           placeholder="Paste the message you want to analyze here...",
                           height=200)

    submitted = st.form_submit_button("Analyze Message")

# Processing
if submitted:
    if not message.strip():
        st.warning("Please enter a message to analyze")
    else:
        with st.spinner("Analyzing message content..."):
            try:
                # Tokenize input
                inputs = tokenizer.encode_plus(
                    message,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='tf'
                )

                # Make prediction
                prediction = model.predict({
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                })

                phishing_prob = float(prediction[0][0])

                # Display results
                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Phishing Probability", f"{phishing_prob:.2%}")
                    st.progress(phishing_prob)

                with col2:
                    if phishing_prob > 0.7:
                        st.error("🚨 High Risk: Likely phishing attempt")
                    elif phishing_prob > 0.4:
                        st.warning("⚠️ Suspicious: Potential phishing elements")
                    else:
                        st.success("✅ Likely Safe: No phishing detected")

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

