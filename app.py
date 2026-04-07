import random
import re
import json
import numpy as np
import tensorflow as tf
import pickle

from flask import Flask, request, jsonify, render_template

# ===== Register AttentionLayer before loading model =====
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="normal", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# === Load model and tokenizers ===
model = tf.keras.models.load_model('best_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# === Load and merge intents from multiple JSON files ===
json_files = [
    "Books.json",
    "Computer Science Theory QA Dataset.json",
    "intentss.json",
    "IT Helpdesk Chatbot Dataset.json",
    "starwarsintents.json",
    "University Chatbot Dataset.json",
]

tag_to_responses = {}

def flatten_responses(responses):
    """Flatten nested response lists (for e.g. Books.json)"""
    flat = []
    for item in responses:
        if isinstance(item, list):
            flat.extend(flatten_responses(item))
        else:
            flat.append(item)
    return flat

for file in json_files:
    with open(file, encoding='utf8') as f:
        data = json.load(f)
        # Some files have 'intents' key, some do not (e.g. Books.json is just a dict with intents as values)
        intents = data.get('intents', data)
        # If intents is a dict, get its values (for some book/style files)
        if isinstance(intents, dict):
            intents = intents.values()
        for intent in intents:
            tag = intent.get('tag')
            responses = intent.get('responses', [])
            if not tag:
                continue
            # Flatten lists if needed
            responses = flatten_responses(responses)
            # Only add if not empty
            if not responses:
                continue
            if tag in tag_to_responses:
                tag_to_responses[tag].extend(responses)
            else:
                tag_to_responses[tag] = responses

max_seq_len = model.input_shape[1]

# === Preprocessing function ===
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def get_response_for_tag(tag):
    responses = tag_to_responses.get(tag)
    if not responses:
        return "Sorry, I don't understand."
    response = random.choice(responses)
    if isinstance(response, dict):
        # Pretty format for book responses
        book = response.get("Book", "Unknown Book")
        feedback = response.get("Feedback", "")
        rate = response.get("Rate", "")
        return f"Book Recommendation:\nTitle: {book}\nRating: {rate}\nDescription: {feedback}"
    return response

# === Flask app ===
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['message']
    processed = preprocess_text(user_input)
    seq = tokenizer.texts_to_sequences([processed])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_seq_len, padding='post')
    pred = model.predict(pad)
    idx = np.argmax(pred)
    tag = label_encoder.inverse_transform([idx])[0]
    confidence = float(np.max(pred))
    if confidence > 0.7:
        response = get_response_for_tag(tag)
    else:
        response = "Sorry, I didn't understand that. Can you rephrase?"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)