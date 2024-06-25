import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import re

# Load the dataset
with open('model.json', 'r') as file:
    data = json.load(file)

# Prepare the data
def preprocess_text(text):
    # Lowercase and remove special characters
    return re.sub(r'\W+', ' ', text.lower())

intents = data['intents']
texts = []
labels = []

for intent in intents:
    for pattern in intent['patterns']:
        texts.append(preprocess_text(pattern))
        labels.append(intent['tag'])

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Create and train the model pipeline
model = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel='linear', probability=True)
)

# Train the model
model.fit(texts, encoded_labels)

# Function to predict the intent
def predict_intent(text):
    text = preprocess_text(text)
    pred = model.predict([text])
    return label_encoder.inverse_transform(pred)[0]

# Function to get a response
def get_response(intent):
    for item in intents:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    return "I'm not sure how to respond to that."

# Chatbot interaction
def chatbot_response(user_input):
    intent = predict_intent(user_input)
    response = get_response(intent)
    return response

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
