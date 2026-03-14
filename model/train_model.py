"""
model/train_model.py
---------------------
Trains a TF-IDF + Logistic Regression pipeline for intent classification.
Saves the trained model to model/intent_model.pkl
"""

import os
import sys
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessing import batch_preprocess

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'intents.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'intent_model.pkl')

def train():
    print("=" * 55)
    print("  Smart Campus Navigation — Model Training")
    print("=" * 55)

    # ── 1. Load dataset ──────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"      Loaded {len(df)} samples across {df['intent'].nunique()} intents.")
    print(f"      Intent distribution:\n{df['intent'].value_counts().to_string()}\n")

    # ── 2. Preprocess queries ────────────────────────────────
    print("[2/5] Preprocessing text with NLTK...")
    df['clean_query'] = batch_preprocess(df['query'].tolist())

    # ── 3. Split data ────────────────────────────────────────
    print("[3/5] Splitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_query'], df['intent'],
        test_size=0.2, random_state=42, stratify=df['intent']
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Build & train pipeline ────────────────────────────
    print("\n[4/5] Training TF-IDF + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams
            max_features=5000,        # top 5000 features
            sublinear_tf=True,        # apply log normalization
            min_df=1
        )),
        ('clf', LogisticRegression(
            C=5.0,                    # regularization strength
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # ── 5. Evaluate ──────────────────────────────────────────
    print("\n[5/5] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # ── Save model ───────────────────────────────────────────
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Model saved → {MODEL_PATH}")
    print("=" * 55)

    return pipeline


def load_model():
    """Load the saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_intent(query: str, model=None) -> dict:
    """
    Predict intent for a raw user query.
    Returns dict with intent label and confidence scores.
    """
    from utils.preprocessing import preprocess_text

    if model is None:
        model = load_model()

    clean = preprocess_text(query)
    intent = model.predict([clean])[0]
    proba  = model.predict_proba([clean])[0]
    labels = model.classes_

    confidence = dict(zip(labels, [round(float(p), 4) for p in proba]))
    top_conf = round(float(max(proba)) * 100, 1)

    return {
        "intent":     intent,
        "confidence": top_conf,
        "all_scores": confidence
    }


if __name__ == '__main__':
    train()

    # Quick sanity check
    print("\n── Quick Inference Test ─────────────────────────────")
    test_queries = [
        "I want a quiet place to study",
        "where can I get food",
        "I need a doctor urgently",
        "where is the admission office",
        "I need to use the computer lab",
        "where is my hostel block"
    ]
    model = load_model()
    for q in test_queries:
        result = predict_intent(q, model)
        print(f"  '{q}'")
        print(f"  → Intent: {result['intent']}  (Confidence: {result['confidence']}%)\n")
