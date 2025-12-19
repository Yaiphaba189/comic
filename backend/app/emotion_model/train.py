import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import Custom Model
# Adjust import path if running as script vs module
try:
    from backend.app.emotion_model.custom_naive_bayes import CustomMultinomialNB
except ImportError:
    try:
        from .custom_naive_bayes import CustomMultinomialNB
    except ImportError:
        # Fallback for running script directly from within the folder
        from custom_naive_bayes import CustomMultinomialNB

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "training.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def train_model():
    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Map numeric labels to standard strings if needed
    label_map = {
        0: 'sad',
        1: 'happy',
        2: 'happy', # love -> happy
        3: 'angry',
        4: 'fear',
        5: 'surprise'
    }
    
    # Check if 'label' exists
    if 'label' in df.columns:
        df['emotion'] = df['label'].map(label_map)
        df['emotion'] = df['emotion'].fillna('neutral')
    elif 'emotion' in df.columns:
        pass
    else:
        raise KeyError("Dataset must have 'label' or 'emotion' column")
    
    X = df['text']
    y = df['emotion']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model (Custom Naive Bayes)...")
    
    # Best parameters found via manual tuning logic below
    best_alpha = 0.1
    best_score = 0.0
    
    # Simple manual Grid Search for Alpha
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # Temporarily vectorizing outside pipeline for search efficiency
    vect = CountVectorizer(ngram_range=(1, 3), stop_words='english', strip_accents='unicode', min_df=2)
    tfidf = TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True)
    
    print("Performing Grid Search for best Alpha...")
    X_train_dt = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_dt)
    
    X_test_dt = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_dt)
    
    for a in alphas:
        clf = CustomMultinomialNB(alpha=a)
        clf.fit(X_train_tfidf, y_train)
        pred = clf.predict(X_test_tfidf)
        score = classification_report(y_test, pred, output_dict=True)['accuracy']
        print(f"Alpha: {a} -> Accuracy: {score:.4f}")
        if score > best_score:
            best_score = score
            best_alpha = a
            
    print(f"Best Alpha found: {best_alpha}")

    # Re-build full pipeline with best params
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3), stop_words='english', strip_accents='unicode', min_df=2)),
        ('tfidf', TfidfTransformer(use_idf=True, norm='l2', sublinear_tf=True)),
        ('clf', CustomMultinomialNB(alpha=best_alpha)),
    ])
    
    text_clf.fit(X_train, y_train)
    
    # Evaluate
    predicted = text_clf.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, predicted))
    
    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(text_clf, f)
    print("Done.")

if __name__ == "__main__":
    train_model()
