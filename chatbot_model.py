import json
import random
import nltk
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk  # For adding and displaying the logo
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize Lemmatizer and other variables
lemmatizer = WordNetLemmatizer()

# Load intents (training data)
with open('F:/chatbot_health_care/intents.json') as file:
    intents = json.load(file)

# Initialize lists to store data
training_sentences = []
training_labels = []
classes = []
words = []

# Process the intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    
    # Add the intent tag to the classes if it's not already present
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize and remove duplicates from words
words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = sorted(list(set(words)))

# Sort the classes
classes = sorted(list(set(classes)))

# Function to create a bag of words for each sentence
def bow(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Prepare training data
training_data = []
for i, sentence in enumerate(training_sentences):
    bag_of_words = bow(sentence, words)
    training_data.append(bag_of_words)

# Convert to numpy array
training_data = np.array(training_data)

# Convert the labels into a one-hot encoding format
encoder = LabelEncoder()
training_labels = encoder.fit_transform(training_labels)

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_data[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])

# Train the model
model.fit(training_data, training_labels, epochs=200, batch_size=5, verbose=1)

# Save the model to an h5 file
model.save('F:/chatbot_health_care/model.h5')

# Save the words and classes used in training for later use
pickle.dump(words, open('F:/chatbot_health_care/words.pkl', 'wb'))
pickle.dump(classes, open('F:/chatbot_health_care/classes.pkl', 'wb'))

print("Model has been trained and saved as 'model.h5'")

# Load the trained model and other resources
model = load_model('F:/chatbot_health_care/model.h5')

# Load intents
with open('F:/chatbot_health_care/intents.json') as file:
    intents = json.load(file)

# Load words and classes
words = pickle.load(open('F:/chatbot_health_care/words.pkl', 'rb'))
classes = pickle.load(open('F:/chatbot_health_care/classes.pkl', 'rb'))

# Function to preprocess input for prediction
def bow(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Function to get chatbot response
def chatbot_response(message):
    bag_of_words = bow(message, words)
    res = model.predict(np.array([bag_of_words]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    return "Sorry, I don't understand that."

# GUI for the chatbot
def send_message():
    user_message = entry_box.get("1.0", tk.END).strip()
    if user_message:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"You: {user_message}\n", "user_message")
        entry_box.delete("1.0", tk.END)

        response = chatbot_response(user_message)
        chat_window.insert(tk.END, f"Bot: {response}\n", "bot_response")
        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

# Initialize Tkinter
root = tk.Tk()
root.title("Healthcare Chatbot")

# Add a logo
logo_frame = tk.Frame(root)
logo_frame.grid(row=0, column=0, columnspan=2, pady=10)

# Load and display the logo
logo_image = Image.open("F:/chatbot_health_care/logo.jpg")  # Replace 'logo.png' with the path to your logo
logo_image = logo_image.resize((100, 100))  # Resize the image to fit
logo_photo = ImageTk.PhotoImage(logo_image)

logo_label = tk.Label(logo_frame, image=logo_photo)
logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
logo_label.pack()

# Chat window
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=60, height=20, font=("Arial", 12))
chat_window.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Entry box
entry_box = tk.Text(root, height=2, width=50, font=("Arial", 12))
entry_box.grid(row=2, column=0, padx=10, pady=10)

# Send button
send_button = tk.Button(root, text="Send", width=10, command=send_message)
send_button.grid(row=2, column=1, padx=10, pady=10)

# Tag styles for chat
chat_window.tag_config("user_message", foreground="blue")
chat_window.tag_config("bot_response", foreground="green")

# Start the chatbot GUI
root.mainloop()
