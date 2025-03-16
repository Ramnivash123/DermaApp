import requests
from bs4 import BeautifulSoup
import gradio as gr

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

# Gradio Interface
def gradio_interface(city_name):
    return scrape_practo(city_name)

# Create Gradio app
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter City Name", placeholder="e.g., bangalore"),
    outputs=gr.Textbox(label="Scraped Data", lines=20),
    title="Practo Dermatologist Scraper",
    description="Enter a city name to scrape dermatologist details from Practo.",
)

# Launch the Gradio app
iface.launch()
