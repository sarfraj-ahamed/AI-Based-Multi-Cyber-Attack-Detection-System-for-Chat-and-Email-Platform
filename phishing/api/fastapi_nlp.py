from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

# Globals
model = None
tokenizer = None
max_len = 128
lstm_units = 64

# Lifespan setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

        input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
        bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(bert_output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_out)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model_path = r"D:\program\python\ai for treat detection\phishing\phishing\nlp_model\hybrid_distilbert_bilstm_phishing.h5"
        model.load_weights(model_path)

        print("✅ Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
    yield  # app runs here
    # Cleanup logic if needed

# Initialize app
app = FastAPI(title="Phishing Detector API", lifespan=lifespan)

# Request schema
class MessageRequest(BaseModel):
    message: str

# Prediction endpoint
@app.post("/predict")
async def predict_phishing(request: MessageRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    try:
        inputs = tokenizer.encode_plus(
            request.message,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )

        preds = model.predict({
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        })

        phishing_prob = float(preds[0][0])

        if phishing_prob > 0.7:
            risk_label = "high"
        elif phishing_prob > 0.4:
            risk_label = "suspicious"
        else:
            risk_label = "safe"

        return {
            "url": request.message,
            "phishing_probability": round(phishing_prob, 4),
            "risk_label": risk_label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
