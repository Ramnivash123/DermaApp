import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load the trained model
model = load_model("derm_model.h5")

# Define class labels
class_labels = [
    "Acne and Rosacea Photos",
    "Eczema Photos",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Psoriasis Pictures Lichen Planus and related Diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos"
]

# Define the prediction function
def predict_image(image):
    # Resize the image to 224x224 pixels
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])  # Get the index of the highest probability
    confidence = predictions[0][class_index]  # Get the confidence of the prediction
    label = class_labels[class_index]  # Map the index to the class label

    return {label: float(confidence)}

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Skin Disease Classification",
    description="Upload an image of skin, and the model will classify it into one of 12 skin disease categories.",
    live=True
)

# Launch the Gradio app
interface.launch()
