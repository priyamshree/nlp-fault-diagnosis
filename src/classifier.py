import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# We save the trained model to disk so the app loads it instantly
# instead of retraining every time it starts.
MODEL_PATH = "models/classifier.joblib"


def train(df):
    # df must have columns: 'maintenance_log' and 'failure_type'
    # These come directly from text_generator.generate_all_logs()

    X = df["maintenance_log"].values
    y = df["failure_type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline = TF-IDF vectorizer + Logistic Regression in one object.
    # This means we can call pipeline.predict(["some log text"]) directly
    # without manually vectorizing first — the pipeline handles it.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams — "tool wear" as one feature
            max_features=500,     # top 500 most informative terms — keeps it lightweight
            sublinear_tf=True,    # dampens very frequent terms logarithmically
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles class imbalance (9661 normal vs 46 TWF)
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=== CLASSIFIER EVALUATION ===")
    print(f"Test Accuracy: {round(acc * 100, 2)}%\n")
    print(classification_report(y_test, y_pred))

    # Save to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return pipeline, acc


def load_or_train(df=None):
    # If a saved model exists, load it (fast).
    # If not, train from scratch using the provided dataframe.
    if os.path.exists(MODEL_PATH):
        print("Loading existing classifier from disk...")
        return joblib.load(MODEL_PATH)

    if df is None:
        raise ValueError("No saved model found and no dataframe provided for training.")

    print("No saved model found. Training classifier...")
    pipeline, _ = train(df)
    return pipeline


def predict(pipeline, log_text):
    # Returns:
    #   predicted_label: the fault type string
    #   confidence: float 0-100
    #   all_probs: dict of {fault_type: probability} for all classes

    probs = pipeline.predict_proba([log_text])[0]
    classes = pipeline.classes_
    predicted_label = classes[np.argmax(probs)]
    confidence = round(float(np.max(probs)) * 100, 1)
    all_probs = {
        cls: round(float(p) * 100, 1)
        for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1])
    }
    return predicted_label, confidence, all_probs


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.text_generator import generate_all_logs

    df = generate_all_logs("data/ai4i2020.csv")
    pipeline, acc = train(df)

    print("\n=== SAMPLE PREDICTIONS (one per failure type) ===\n")
    for fault in ["Tool Wear Failure", "Heat Dissipation Failure",
                  "Power Failure", "Overstrain Failure", "No Failure"]:
        row = df[df["failure_type"] == fault].iloc[0]
        log = row["maintenance_log"]
        label, conf, probs = predict(pipeline, log)
        correct = "✓" if label == fault else "✗"
        print(f"{correct} Actual  : {fault}")
        print(f"  Predict : {label} ({conf}%)")
        print(f"  Log     : {log[:80]}...")
        print(f"  Top 3   : {list(probs.items())[:3]}")
        print("-" * 70)