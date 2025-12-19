import pickle
import os
import sys

# Hack to allow unpickling of custom class trained as script
# It expects 'custom_naive_bayes' to be in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from custom_naive_bayes import CustomMultinomialNB
except ImportError:
    pass # It will be loaded by pickle

class EmotionModel:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}. Please run train.py first.")
            return

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, text):
        if not self.model:
            raise ValueError("Model is not loaded.")
        
        # Predict
        prediction = self.model.predict([text])[0]
        # Get probability (optional, but good for confidence)
        # proba = self.model.predict_proba([text])
        return prediction

# Singleton instance
emotion_predictor = EmotionModel()
