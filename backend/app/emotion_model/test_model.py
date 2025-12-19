import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Hack to find CustomMultinomialNB class during unpickling
sys.path.append(BASE_DIR)

def test_model():
    print(f"Loading test dataset from {TEST_DATA_PATH}...")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

    df = pd.read_csv(TEST_DATA_PATH)
    
    # Map numeric labels to standard strings (Same as training)
    label_map = {
        0: 'sad',
        1: 'happy',
        2: 'happy', # love -> happy
        3: 'angry',
        4: 'fear',
        5: 'surprise'
    }
    
    if 'label' in df.columns:
        df['emotion'] = df['label'].map(label_map)
        df['emotion'] = df['emotion'].fillna('neutral')
    else:
        raise KeyError("Dataset must have 'label' column")
    
    X_test = df['text']
    y_test = df['emotion']
    
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please run train.py first.")
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except ModuleNotFoundError:
        # Fallback if pickle fails on class lookup
        from custom_naive_bayes import CustomMultinomialNB
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
    print("Running predictions on test set...")
    predictions = model.predict(X_test)
    
    print("\nXXX Independent Test Set Results XXX")
    print(classification_report(y_test, predictions))
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    test_model()
