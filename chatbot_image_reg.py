import json
import random
import nltk
import numpy as np
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from nltk.stem import WordNetLemmatizer
import pickle
import os

# Initialize Lemmatizer and variables
lemmatizer = WordNetLemmatizer()

# Paths
chatbot_model_path = "F:/chatbot_health_care/model.h5"
skin_model_path = "F:/chatbot_health_care/skin_disease_model.h5"
chatbot_words_path = "F:/chatbot_health_care/words.pkl"
chatbot_classes_path = "F:/chatbot_health_care/classes.pkl"
intents_path = "F:/chatbot_health_care/intents.json"
logo_image_path = "F:/chatbot_health_care/logo.jpg"  # Add your logo path here

# Load Chatbot resources
with open(intents_path) as file:
    intents = json.load(file)

words = pickle.load(open(chatbot_words_path, 'rb'))
classes = pickle.load(open(chatbot_classes_path, 'rb'))
chatbot_model = load_model(chatbot_model_path)

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
    res = chatbot_model.predict(np.array([bag_of_words]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

# Load Skin Disease Model
if os.path.exists(skin_model_path):
    skin_model = load_model(skin_model_path)
else:
    print("Skin disease model not found!")

# Function to classify skin disease
def classify_skin_image(image_path):
    img_size = (224, 224)
    image = load_img(image_path, target_size=img_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = skin_model.predict(image)
    return predictions

# Class Labels for Skin Disease (modify this with your actual classes)
skin_disease_classes = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']

# GUI
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

def upload_and_classify_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            predictions = classify_skin_image(file_path)
            predicted_class = np.argmax(predictions)
            predicted_label = skin_disease_classes[predicted_class]
            confidence = predictions[0][predicted_class] * 100

            chat_window.config(state=tk.NORMAL)
            chat_window.insert(tk.END, f"\n[Image Uploaded]\nPredicted Class: {predicted_label}, Confidence: {confidence:.2f}%\n", "image_prediction")
            chat_window.config(state=tk.DISABLED)
            chat_window.yview(tk.END)
    except Exception as e:
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, f"\n[Error] Unable to classify image: {str(e)}\n", "error_message")
        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

def show_image_preview(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize to fit the preview area
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference
    image_label.grid(row=3, column=0, padx=10, pady=10)

# Main Tkinter window
root = tk.Tk()
root.title("alphasAi Healthcare System")

# Load and display the logo at the top of the window
logo_image = Image.open(logo_image_path)
logo_image = logo_image.resize((100, 100), Image.Resampling.LANCZOS)  # Resize if needed
logo_photo = ImageTk.PhotoImage(logo_image)

# Add a Label for the logo
logo_label = tk.Label(root, image=logo_photo)
logo_label.grid(row=0, column=0, padx=10, pady=10)

# Chatbot GUI
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, width=60, height=20, font=("Arial", 12))
chat_window.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

entry_box = tk.Text(root, height=2, width=50, font=("Arial", 12))
entry_box.grid(row=2, column=0, padx=10, pady=10)

send_button = tk.Button(root, text="Send", width=10, command=send_message)
send_button.grid(row=2, column=1, padx=10, pady=10)

upload_button = tk.Button(root, text="Upload Image", width=15, command=upload_and_classify_image)
upload_button.grid(row=3, column=0, columnspan=2, pady=10)

clear_button = tk.Button(root, text="Clear", width=10, command=lambda: chat_window.delete(1.0, tk.END))
clear_button.grid(row=4, column=1, padx=10, pady=10)

# Add an Image Label for preview
image_label = tk.Label(root)
image_label.grid(row=4, column=0, padx=10, pady=10)

# Chat window tag styles
chat_window.tag_config("user_message", foreground="blue")
chat_window.tag_config("bot_response", foreground="black")
chat_window.tag_config("image_prediction", foreground="purple")
chat_window.tag_config("error_message", foreground="red")

# Start GUI loop
root.mainloop()
