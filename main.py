import streamlit as st
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tempfile

# Load class labels
class_file = st.file_uploader("Choose a .json file...", type=["json"])
class_labels = {}

# Preprocessing function
def preprocess_image(img, img_size=(224, 224)):
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if class_file is not None:
    class_indices = json.load(class_file)
    class_labels = {v: k for k, v in class_indices.items()}

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(class_labels), activation='softmax'))

    # Load weights
    model_file = st.file_uploader("Choose an .h5 file...", type=["h5"])

    if model_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(model_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load the model weights
        model.load_weights(tmp_file_path)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Streamlit UI
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed_img = preprocess_image(image)
        
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_disease = class_labels[predicted_class]

        st.write(f"🩺 Predicted Disease: {predicted_disease}")
