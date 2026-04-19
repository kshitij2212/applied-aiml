import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_pipeline import NoShowPredictor

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'KaggleV2-May-2016.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'user_trained_predictor.pkl')

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

print(f"Loading CSV from: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"CSV loaded, shape={df.shape}")

predictor = NoShowPredictor()
print("Training predictor on provided CSV (this may take a short while)...")
results = predictor.train(df, model_name='Logistic Regression', test_size=0.2, threshold=0.35)
print("Training complete. Results:")
for k, v in results.items():
    print(f"{k}: {v}")

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(predictor, f)

print(f"Trained predictor saved to: {MODEL_OUT}")
