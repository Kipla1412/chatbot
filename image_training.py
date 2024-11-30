from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

dataset_path = "F:/chatbot_health_care/skin-disease-dataset"  # Make sure the path is correct
model_save_path = "F:/chatbot_health_care/skin_disease_model.h5"

def preprocess_dataset():
    images = []
    labels = []
    img_size = (224, 224)

    # Ensure dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist!")
        return None, None, None, None, None

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                try:
                    img_path = os.path.join(class_dir, img_name)
                    img = load_img(img_path, target_size=img_size)
                    img = img_to_array(img) / 255.0  # Normalize pixel values
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Encode labels to categorical values
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = to_categorical(encoded_labels)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_skin_disease_model():
    X_train, X_test, y_train, y_test, label_encoder = preprocess_dataset()

    if X_train is None:
        return None  # Exit if dataset loading failed

    # Load MobileNetV2 base model with pre-trained weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model layers

    # Build the model
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation using ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)

    # Train the model
    model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=10)

    # Save the model after training
    model.save(model_save_path)
    print("Skin disease model trained and saved successfully!")

    return label_encoder

# Call the function to start training
label_encoder = train_skin_disease_model()

if label_encoder:
    print(f"Model training complete. Label encoder classes: {label_encoder.classes_}")
else:
    print("Model training failed.")
