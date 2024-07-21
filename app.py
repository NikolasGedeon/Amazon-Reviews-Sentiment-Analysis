from flask import Flask, render_template, request, jsonify
from predict_sentiment import predict_sentiment, retrain_model
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

reviews = []
labels = []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/add_review", methods=["POST"])
def add_review():
    try:
        review = request.form["review"]
        sentiment, confidence = predict_sentiment(review)
        correct_label = "__label__1" if sentiment == "negative" else "__label__2"
        reviews.append(review)
        labels.append(correct_label)
        # Convert float32 to float
        return jsonify(sentiment=sentiment, confidence=float(confidence))
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify(error="An error occurred"), 500

@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        global reviews, labels
        retrain_model(reviews, labels)
        reviews = []
        labels = []
        return jsonify(success=True)
    except Exception as e:
        app.logger.error(f"Error training model: {e}")
        return jsonify(error="An error occurred"), 500

if __name__ == "__main__":
    app.run(debug=True)
