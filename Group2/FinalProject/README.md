# SEG Web Application Project

## Overview
This project is a web-based application that leverages Natural Language Processing (NLP) techniques to analyze and process e-commerce data from Shopee. The system uses machine learning models to provide analysis or predictions based on textual inputs.

## Features
- Web interface for interacting with the NLP models
- Data crawling capabilities for Shopee e-commerce platform
- Text processing and analysis using NLTK
- Multiple trained machine learning models for different versions/approaches

## Project Structure
- **Models**: Various trained models (`best_model.pkl`, `best_model_v2.pkl`, etc.) and their corresponding vectorizers
- **Web Application**: Flask-based web interface in [SEG_on_web.py](SEG_on_web.py)
- **Templates**: HTML templates for the web interface in the [templates](templates/) directory
- **Data Crawler**: Shopee data crawler in the [shopee_crawler](shopee_crawler/) directory
- **NLTK Data**: Required NLTK resources for text processing

## Setup and Installation
1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure the NLTK data is correctly set up.
4. Run the web application:
   ```bash
   python SEG_on_web.py
   ```

## Usage
1. Access the web interface through your browser.
2. Input the text you want to analyze.
3. The system will process your input and return results based on the trained models.

## Data Crawling
The [shopee_crawler](shopee_crawler/) module allows you to gather additional data from Shopee. See the crawler's [README.md](shopee_crawler/README.md) for detailed instructions.

## Configuration
Application settings can be modified in [config.json](config.json).

## External Access
The project includes [ngrok.yml](ngrok.yml) for exposing the local web server to the internet for testing or demonstration purposes.

## Team
Our project team members are represented in the images directory.

## License
[Specify your license here]
