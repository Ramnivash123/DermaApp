import requests
from bs4 import BeautifulSoup
import gradio as gr

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

def fetch_info(disease_name):
    return scrape_wikipedia_info(disease_name)

interface = gr.Interface(
    fn=fetch_info,
    inputs=gr.Textbox(label="Enter Disease Name", placeholder="e.g., Scabies"),
    outputs=gr.Textbox(label="Wikipedia Information"),
    title="Wikipedia Disease Info Scraper",
    description="Enter a disease name to scrape 'Causes', 'Symptoms', and 'Medication' sections from Wikipedia."
)

interface.launch()
