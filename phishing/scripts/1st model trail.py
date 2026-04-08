import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants (Reduced for memory)
MAX_LEN = 64  # Reduced from 128
BATCH_SIZE = 16  # Reduced from 32
EPOCHS = 15  # Reduced from 20
LEARNING_RATE = 2e-5
DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
LSTM_UNITS = 64  # Reduced from 128

# Path configuration
DATASET_PATH = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\clean_dataset\ds3"
MODEL_SAVE_PATH = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\model\hybrid_distilbert_bilstm_phishing.h5"
RESULTS_SAVE_PATH = r"C:\Users\matha\OneDrive\Desktop\sarf\phishing\model\model_results.csv"

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Perform NLP preprocessing and return cleaned text or None if invalid"""
    if not isinstance(text, str) or pd.isna(text):
        return None

    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens) if tokens else None
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return None


def load_datasets(folder_path):
    """Load all CSV files and identify problematic datasets"""
    dfs = []
    problematic_files = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                nan_check = df[['subject', 'body', 'label']].isna().any()
                if nan_check.any():
                    problematic_files.append((file, nan_check.to_dict()))
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                problematic_files.append((file, str(e)))

    combined_df = pd.concat(dfs, ignore_index=True)

    if problematic_files:
        print("\n=== Problematic Files Report ===")
        for file, issues in problematic_files:
            print(f"\nFile: {file}")
            if isinstance(issues, dict):
                print("NaN values found in:")
                for col, has_nan in issues.items():
                    if has_nan:
                        print(f"  - {col}")
            else:
                print(f"Error: {issues}")

    return combined_df


def preprocess_data(df):
    """Full preprocessing pipeline with strict NaN handling"""
    print(f"\nOriginal dataset size: {len(df)}")

    # 1. Initial NaN removal
    df = df.dropna(subset=['subject', 'body', 'label'])
    print(f"After initial NaN removal: {len(df)}")

    # 2. Combine text fields
    df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)

    # 3. Apply NLP cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 4. Remove rows where cleaning failed
    df = df[df['cleaned_text'].notna()]
    print(f"After NLP cleaning: {len(df)}")

    # 5. Convert labels to integers
    df['label'] = df['label'].astype(int)

    # 6. Final validation
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        print(f"Final NaN removal in columns: {nan_cols}")
        df = df.dropna()
        print(f"Final dataset size: {len(df)}")

    return df


def tokenize_texts(texts, max_len=MAX_LEN):
    """Tokenize texts with error handling"""
    input_ids = []
    attention_masks = []

    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            continue

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)


def create_hybrid_model(max_len=MAX_LEN, lstm_units=LSTM_UNITS):
    """Create memory-optimized hybrid model"""
    # Load DistilBERT
    distilbert = TFDistilBertModel.from_pretrained(DISTILBERT_MODEL_NAME, from_pt=True)

    # Freeze DistilBERT layers
    for layer in distilbert.layers:
        layer.trainable = False

    # Input layers
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

    # DistilBERT outputs
    distilbert_output = distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]

    # BiLSTM layer
    lstm_output = Bidirectional(LSTM(lstm_units))(distilbert_output)
    lstm_output = Dropout(0.3)(lstm_output)

    # Output layer
    output = Dense(1, activation='sigmoid')(lstm_output)

    # Create model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def main():
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(e)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_datasets(DATASET_PATH)
    df = preprocess_data(df)

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.35, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.4286, random_state=42)

    print(f"\nFinal dataset sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    # Initialize tokenizer
    global tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

    # Tokenize texts
    print("\nTokenizing texts...")
    train_input_ids, train_attention_masks = tokenize_texts(train_df['cleaned_text'].tolist())
    val_input_ids, val_attention_masks = tokenize_texts(val_df['cleaned_text'].tolist())
    test_input_ids, test_attention_masks = tokenize_texts(test_df['cleaned_text'].tolist())

    # Get labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Create model
    print("\nCreating model...")
    model = create_hybrid_model()
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    # Train model
    print("\nTraining model...")
    history = model.fit(
        [train_input_ids, train_attention_masks],
        y_train,
        validation_data=([val_input_ids, val_attention_masks], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Load best model
    model.load_weights(MODEL_SAVE_PATH)

    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate([test_input_ids, test_attention_masks], y_test)
    y_pred = (model.predict([test_input_ids, test_attention_masks]) > 0.5).astype(int)

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save results
    with open(RESULTS_SAVE_PATH, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))

    print("\n=== Final Metrics ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    main()