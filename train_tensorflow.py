import json
import pickle
import os
import numpy as np
import nltk
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# TensorFlow imports with error handling
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow not available: {e}")
    print("Please install TensorFlow: pip install tensorflow>=2.13.0")
    TENSORFLOW_AVAILABLE = False
    exit(1)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def read_intents_json(json_path: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """Read intents.json and return patterns, labels, and responses"""
    questions, answers, categories = [], [], []
    responses = {}
    
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data.get('intents', []):
        intent_name = item.get('intent', '').strip()
        texts = item.get('text', [])
        responses_list = item.get('responses', [])
        
        if not texts or not responses_list:
            continue
            
        # Store responses for this intent
        responses[intent_name] = responses_list
        
        # Add each text pattern with its intent
        for text in texts:
            questions.append(str(text).strip())
            answers.append(str(responses_list[0]).strip())  # Use first response as canonical
            categories.append(intent_name)
    
    return questions, answers, (categories if any(categories) else None), responses

def train_tensorflow_model(patterns: List[str], labels: List[str]):
    """Train TensorFlow neural network model"""
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Vectorize text using CountVectorizer with NLTK tokenizer
    vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, lowercase=True)
    X = vectorizer.fit_transform(patterns).toarray()
    y = labels_encoded
    
    # Build neural network
    model = Sequential([
        Dense(128, input_shape=(X.shape[1],), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(set(y)), activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    # Train model
    model.fit(X, y, epochs=200, batch_size=8, verbose=1)
    
    return model, vectorizer, label_encoder

def save_tensorflow_artifacts(model, vectorizer, label_encoder, responses, out_dir="."):
    """Save TensorFlow model and related artifacts"""
    # Save model
    model.save(os.path.join(out_dir, "tensorflow_model"))
    
    # Save vectorizer
    with open(os.path.join(out_dir, "tf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save label encoder
    with open(os.path.join(out_dir, "tf_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    # Save responses mapping
    with open(os.path.join(out_dir, "tf_responses.pkl"), "wb") as f:
        pickle.dump(responses, f)

if __name__ == "__main__":
    if not os.path.exists("intents.json"):
        raise FileNotFoundError("intents.json not found")
    
    print("Loading intents data...")
    questions, answers, categories, responses = read_intents_json("intents.json")
    
    print(f"Training TensorFlow model on {len(questions)} patterns...")
    model, vectorizer, label_encoder = train_tensorflow_model(questions, categories)
    
    print("Saving TensorFlow artifacts...")
    save_tensorflow_artifacts(model, vectorizer, label_encoder, responses)
    
    print("TensorFlow model training completed!")
    print(f"Model trained on {len(set(categories))} intents with {len(questions)} training patterns.")
