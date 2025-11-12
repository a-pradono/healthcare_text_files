# %%
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# %% [markdown]
# ### FastAPI
# Simple API using FastAPI to serve the trained model

# %%
# Load vectorizer and model
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_map = {0: "general illness", 1: "joint illness", 2: "chronic illness"}

# Define the fast api app
app = FastAPI(title="Text Classifier API")

# Input schema
class TextRequest(BaseModel):
    text: str

# API endpoint
@app.post("/predict")
def predict(request: TextRequest):
    text = [request.text]  # Model expects a list
    X_vect = vectorizer.transform(text)
    pred = model.predict(X_vect)[0]
    label = label_map[pred]
    return {"prediction": label}

# Test endpoint
@app.get("/")
def read_root():
    return {"message": "Text Classifier API is running!"}

# %%



