# Chatbot with Intent Recognition

This repository contains code for a simple chatbot that uses a Support Vector Machine (SVM) to recognize user intents from text input and provide appropriate responses. The chatbot is trained on a set of predefined intents and patterns stored in a JSON file.

## Features

- Preprocesses text input by converting to lowercase and removing special characters.
- Uses TF-IDF Vectorization to convert text input into numerical features.
- Employs an SVM model with a linear kernel for intent classification.
- Encodes and decodes labels using `LabelEncoder` for model training and prediction.
- Generates responses based on recognized intents from a predefined set of responses.

## Installation

To run the chatbot, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/umer066/Chatbot_intent_recognition.git
cd chatbot_intent_recognition
```

2. Ensure you have Python 3.x installed. Create a virtual environment (optional but recommended):

```bash
python -m venv chatbot-env
source chatbot-env/bin/activate  # On Windows use `chatbot-env\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have your dataset file (`model.json`) in the same directory as your script. The `model.json` file should have the following structure:

```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hello", "Hi", "Greetings", "What's up?", "Hey"],
            "responses": ["Hello!", "Hi there!", "Greetings!", "How can I assist you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Farewell"],
            "responses": ["Goodbye!", "See you later!", "Farewell!", "Have a great day!"]
        }
        // Add more intents as needed
    ]
}
```

2. Run the chatbot script:

```bash
python chat.py
```

3. Interact with the chatbot in the terminal. Type your messages and get responses from the chatbot. To exit the chat, type "exit", "quit", or "bye".

## Code Explanation

### Data Preparation

The JSON file is loaded, and the text patterns are preprocessed by converting to lowercase and removing special characters. The corresponding tags are also collected.

### Model Training

A pipeline is created with `TfidfVectorizer` for text vectorization and `SVC` (Support Vector Classifier) for classification. The model is trained on the preprocessed text patterns and their corresponding encoded tags.

### Prediction and Response

The chatbot predicts the intent of the user's input text and generates a response based on the predicted intent. If the intent is not recognized, it responds with a default message.

### Chatbot Interaction

The chatbot runs in a loop, continuously taking user input and providing responses until the user decides to exit.

## Example

```bash
You: Hello
Bot: Hi there!

You: How are you?
Bot: I'm not sure how to respond to that.

You: Bye
Bot: Goodbye!
```

## Contributing

Feel free to contribute to this project by creating pull requests or opening issues.
