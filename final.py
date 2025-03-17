import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import requests
from bs4 import BeautifulSoup

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

# Create the Gradio interface for skin disease classification
skin_disease_interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Identify Disease",
    description="Upload an image of skin, and the model will classify it into one of 12 skin disease categories.",
    live=True
)

# Function to scrape Wikipedia for disease information
def scrape_wikipedia_info(disease_name):
    formatted_name = disease_name.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{formatted_name}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Wipn64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        def extract_info(section_name):
            section_label = soup.find('th', text=section_name)
            if section_label:
                section_data = section_label.find_next_sibling('td')
                if section_data:
                    return section_data.get_text(separator=" ", strip=True)
            return f"{section_name} section not found for '{disease_name}'."
        
        causes = extract_info("Causes")
        symptoms = extract_info("Symptoms")
        medication = extract_info("Medication")
        
        return f"Causes of {disease_name}:\n{causes}\n\nSymptoms of {disease_name}:\n{symptoms}\n\nMedication for {disease_name}:\n{medication}"
    
    elif response.status_code == 404:
        return f"Page not found for '{disease_name}'. Please check the spelling or try another term."
    else:
        return f"Failed to fetch data from Wikipedia (status code: {response.status_code})."

# Create the Gradio interface for Wikipedia disease info scraper
wikipedia_interface = gr.Interface(
    fn=scrape_wikipedia_info,
    inputs=gr.Textbox(label="Enter Disease Name", placeholder="e.g., Scabies"),
    outputs=gr.Textbox(label="Wikipedia Information"),
    title="Disease Info",
    description="Enter a disease name to scrape 'Causes', 'Symptoms', and 'Medication' sections from Wikipedia."
)

# Function to scrape Practo for dermatologist information
def scrape_practo(city_name):
    # Construct the URL dynamically
    url = f"https://www.practo.com/{city_name}/dermatologist"
    
    # Send a GET request to the website
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all doctor listing containers
        doctor_listings = soup.find_all('div', {'data-qa-id': 'doctor_card'})
        
        # Check if doctors are found
        if not doctor_listings:
            return f"No dermatologists found for {city_name}."
        
        # Prepare a list to store details
        results = []
        for doctor in doctor_listings:
            # Extract doctor's name
            doctor_name = doctor.find('h2', {'data-qa-id': 'doctor_name'}).get_text(strip=True) if doctor.find('h2', {'data-qa-id': 'doctor_name'}) else 'N/A'
            
            # Extract practice location
            practice_location = doctor.find('span', {'data-qa-id': 'practice_locality'}).get_text(strip=True) if doctor.find('span', {'data-qa-id': 'practice_locality'}) else 'N/A'
            practice_city = doctor.find('span', {'data-qa-id': 'practice_city'}).get_text(strip=True) if doctor.find('span', {'data-qa-id': 'practice_city'}) else 'N/A'
            
            # Extract clinic name
            clinic_name = doctor.find('span', {'data-qa-id': 'doctor_clinic_name'}).get_text(strip=True) if doctor.find('span', {'data-qa-id': 'doctor_clinic_name'}) else 'N/A'
            
            # Append the details as a dictionary
            results.append(f"Doctor Name: {doctor_name}\nPractice Location: {practice_location}, {practice_city}\nClinic Name: {clinic_name}\n{'-' * 40}")
        
        return "\n".join(results)
    else:
        return f"Failed to retrieve the page for {city_name}. Status code: {response.status_code}"
    
 

# Create the Gradio interface for Practo dermatologist scraper
practo_interface = gr.Interface(
    fn=scrape_practo,
    inputs=gr.Textbox(label="Enter City Name", placeholder="e.g., bangalore"),
    outputs=gr.Textbox(label="Scraped Data", lines=20),
    title="Find Doctors Nearby",
    description="Enter a city name to scrape dermatologist details from Practo.",
)





# Create a single page interface using gr.Column
with gr.Blocks() as combined_interface:
    gr.Markdown("# Derma App")  # Use Markdown for a large heading

    skin_disease_interface.render()

    wikipedia_interface.render()

    practo_interface.render()

# Launch the combined Gradio app
combined_interface.launch()