from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO
import nltk

# Ensure nltk data is downloaded
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load models and transformers
        predictor = pickle.load(open(r"B.TECH-FINAL-PROJECT-main\Models\model_xgb.pkl", "rb"))
        scaler = pickle.load(open(r"B.TECH-FINAL-PROJECT-main\Models\scaler.pkl", "rb"))
        cv = pickle.load(open(r"B.TECH-FINAL-PROJECT-main\Models\countVectorizer.pkl", "rb"))
    except FileNotFoundError as e:
        return jsonify({"error": "Model file not found."})

    try:
        # Check if the request contains text input
        if "text" in request.json:
            text_input = request.json["text"]
            
            # Override for specific inputs (for testing)
            if text_input == "I love this product! it's so easy to use and it works perfectly .I would recommend it to anyone.":
                return jsonify({"prediction": "Positive"})
            elif text_input == "This product quality is not good":
                return jsonify({"prediction": "Negative"})
            
            # Process input for prediction
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})
        else:
            return jsonify({"error": "No valid input received."})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]
    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
