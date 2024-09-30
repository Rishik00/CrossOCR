# CrossOCR: A Multi-Lingual Image Analysis

## Introduction
CrossOCR is my attempt to tackle IIT Roorkee's challenging assignment on multi-lingual image analysis and text extraction. This project showcases a combination of state-of-the-art models to extract and analyze text from images containing both Hindi and English language content.

#### Developed by: Rishikesh M
#### Demo: [Google Colab Notebook](https://colab.research.google.com/drive/14rgj1Q8hmTUDlH7c8CAz99X399tzhkKF?usp=sharing)

### Project Requirements
1. Develop a Streamlit/Gradio application capable of processing images with cross-lingual content (Hindi and English).
2. Extract and display text from the images based on specific word searches.
3. Experiment with various Optical Character Recognition (OCR) and Vision Language Models (VLM) to create an efficient solution.

### Solution
The following is an image Retrieval-Augmented Generation (RAG) system that uses the following models:

1. CLiP (OpenAI): For visual understanding and analysis
2. An ensemble of PaddleOCR, SentenceTransformers, and FAISS: For efficient text extraction and retrieval

This innovative approach allows us to:
- Answer simple questions related to image content
- Retrieve all text present in the image with high accuracy
- Handle multi-lingual content seamlessly

## Project Architecture 
<img width="606" alt="Screenshot 2024-09-30 201452" src="https://github.com/user-attachments/assets/31500331-f196-4cf6-8891-3d06f502f559">

## Getting Started

### Option 1: Google Colab (Recommended)
1. Open the [Colab notebook](https://colab.research.google.com/drive/14rgj1Q8hmTUDlH7c8CAz99X399tzhkKF?usp=sharing)
2. Execute the cells sequentially to launch the Gradio demo

### Option 2: Local Installation
1. Clone the GitHub repository:
   ```
   git clone https://github.com/your-username/CrossOCR.git
   ```
2. Navigate to the project directory and run the setup script:
   ```
   cd CrossOCR
   python setup.py
   ```
3. run the cells in `demo.ipynb`

### Project Future Scope
I was given a week to complete this task, and I think the system I proposed is only the start of something more complex and sophisticated that can answer more complex questions. Here are some of the features I intend to add in the future to the backend: 
1. Use APIs provided by Groq to decrease latency while having the chance to experiment with lot many more models. I am suggesting to use LLMs to explain the outputs.
2. PDF Integration to allow visual question answering for PDFs using models like layoutMV3
3. Exploring and integrating with more models that can provide better outputs while optimizing for latency.
4. Integrate more Indian languages like Marathi, Tamil, and Kannada.
5. Improve text handling using tokenizers.

## Testing
For optimal results, I recommend using the sample images provided in the `/sampleimages` directory. These images have been extensively tested and yield reliable outcomes.
