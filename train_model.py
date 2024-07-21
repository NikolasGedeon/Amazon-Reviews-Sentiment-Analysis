import re
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  # Ensure matplotlib is imported
from tensorflow.python.keras.models import load_model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib

CONFIDENCE_THRESHOLD = 0.67  # Define confidence threshold for classification

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def load_model_if_exists():
    try:
        model = load_model('sentiment_model.h5')
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, vectorizer, le
    except (OSError, IOError):
        return None, None, None

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vector)
    confidence = np.max(prediction)  # Get the highest confidence score
    predicted_label = prediction.argmax()
    sentiment = le.inverse_transform([predicted_label])[0]

    if confidence >= CONFIDENCE_THRESHOLD:
        if sentiment == '__label__1':
            return 'negative', confidence
        elif sentiment == '__label__2':
            return 'positive', confidence
    else:
        return 'neutral', confidence

def retrain_model(new_texts, new_labels):
    # Load existing data
    labels, texts = load_data('test.ft.txt')

    # Add new data
    texts.extend(new_texts)
    labels.extend(new_labels)

    # Save updated data back to file
    with open('test.ft.txt', 'w', encoding='utf-8') as file:
        for label, text in zip(labels, texts):
            file.write(f"{label} {text}\n")

    # Retrain model
    model, vectorizer, le = train_model()
    model.save('sentiment_model.h5')
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())
    labels, texts = zip(*[line.split(' ', 1) for line in data])
    return list(labels), list(texts)

def train_model():
    labels, texts = load_data('test.ft.txt')
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    num_classes = len(le.classes_)

    texts = [preprocess_text(text) for text in texts]
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig('static/training_plot.png')  # Save the plot as a file
    plt.close()  # Close the plot

    model.save('sentiment_model.h5')  # Save the trained model
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    return model, vectorizer, le

if __name__ == "__main__":
    model, vectorizer, le = load_model_if_exists()
    if model is None:
        model, vectorizer, le = train_model()
        print("Model training complete and saved.")
    else:
        print("Model loaded successfully.")
