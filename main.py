import shutil
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gdown

app = FastAPI()

# Templates for HTML rendering
templates = Jinja2Templates(directory="templates")  # Directory for HTML templates (already exists)

# Directory to store uploaded images
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Directory to store models
MODEL_DIR = "."  # Models are in the current directory with main.py

def download_models():
    model1_url = 'https://drive.google.com/uc?id=1SVzcoWXlVO-J8z-zusC7C4LouHDIqic1'
    model2_url = 'https://drive.google.com/uc?id=1t5L-Od5WnETF4VHlWcfY5QjWMtNxZVCW'
    model3_url = 'https://drive.google.com/uc?id=1KgQyE7-sDnOhMQIlLpjPr63oUfU6W9SX'

    model1_path = os.path.join(MODEL_DIR, "braintumor.h5")
    model2_path = os.path.join(MODEL_DIR, "Tuberculosis_model.h5")
    model3_path = os.path.join(MODEL_DIR, "pneumonia_model.h5")

    try:
        if not os.path.exists(model1_path):
            gdown.download(model1_url, model1_path, quiet=False)
        if not os.path.exists(model2_path):
            gdown.download(model2_url, model2_path, quiet=False)
        if not os.path.exists(model3_path):
            gdown.download(model3_url, model3_path, quiet=False)
    except AttributeError as e:
        print(f"Error downloading model: {e}")

# Download models if not already downloaded
download_models()

# Load your machine learning models outside the route functions
model1 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "braintumor.h5"))
model2 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "Tuberculosis_model.h5"))
model3 = tf.keras.models.load_model(os.path.join(MODEL_DIR, "pneumonia_model.h5"))

# Print model input shapes for debugging
print(f"Model1 expected input shape: {model1.input_shape}")
print(f"Model2 expected input shape: {model2.input_shape}")
print(f"Model3 expected input shape: {model3.input_shape}")

# Define a function to save uploaded files
def save_uploaded_file(file, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def preprocess_image(image_path, target_size=(224, 224), color_mode='rgb'):
    print(f"Loading image {image_path} with color_mode={color_mode}")
    img = image.load_img(image_path, target_size=target_size, color_mode=color_mode)
    print(f"Image mode after loading: {img.mode}")
    img = image.img_to_array(img)
    print(f"Image shape after img_to_array: {img.shape}")
    img = np.expand_dims(img, axis=0)
    print(f"Image shape after expand_dims: {img.shape}")
    img = img / 255.0  # Normalize the image
    return img


# Define the main route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the route to handle image uploads
@app.post("/upload/")
async def upload_file(
    request: Request,
    choice: int = Form(...),
    data: UploadFile = Form(...),):
    file_location = os.path.join(UPLOAD_DIR, 'internet.jpg')
    save_uploaded_file(data, file_location)

    # Modify the target_size for model1 to (224, 224)
    if choice == 1:
        img = preprocess_image(file_location, target_size=(224, 224), color_mode='rgb')
        print(f"Preprocessed image shape: {img.shape}")
        predictions = model1.predict(img)
        predicted_class = np.argmax(predictions)
        print(predicted_class)
        class_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
        result = class_labels[predicted_class]
        return result
    elif choice == 2:
        img = preprocess_image(file_location, target_size=(64, 64), color_mode='rgb')  # Adjust target_size accordingly
        print(f"Preprocessed image shape: {img.shape}")

        # Check if the model expects a flattened input
        if len(model2.input_shape) == 2 and model2.input_shape[1]:
            img = img.reshape(1, -1)
            print(f"Image shape after flattening: {img.shape}")
        prediction = model2.predict(img)
        print(f"Raw prediction output: {prediction}")
        if prediction >= 0.5:
            result = "Tuberculosis"
        else:
            result = "Normal"
        return result

    elif choice == 3:
        # Model3 (Pneumonia Detection)
        img = preprocess_image(file_location, target_size=(150, 150), color_mode='grayscale')
        print(f"Preprocessed image shape: {img.shape}")
        prediction = model3.predict(img)
        prediction = prediction[0][0]  # Assuming prediction is [[probability]]
        print(f"Prediction value: {prediction}")
        if prediction >= 0.5:
            result = "Pneumonia"
        else:
            result = "Normal"
        return result

    # If none of the above conditions match, return a response
    return "Invalid choice or prediction failed."

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
