# CrossOCR: A Multi-Lingual Image Analysis Solution

## Introduction
CrossOCR is an innovative attempt to tackle IIT Roorkee's challenging assignment on multi-lingual image analysis. This project showcases a powerful combination of state-of-the-art models to extract and analyze text from images containing both Hindi and English content.

**Developer:** Rishikesh M

**Demo:** [Google Colab Notebook](https://colab.research.google.com/drive/14rgj1Q8hmTUDlH7c8CAz99X399tzhkKF?usp=sharing)

## Project Requirements
1. Develop a Streamlit/Gradio application capable of processing images with cross-lingual content (Hindi and English).
2. Extract and display text from the given images based on specific word searches.
3. Experiment with various Optical Character Recognition (OCR) and Vision Language Models (VLM) to create an efficient solution.

## Our Solution
We've engineered an advanced image Retrieval-Augmented Generation (RAG) system that leverages three powerful models:

1. CLiP (OpenAI): For visual understanding and analysis
2. LLama (Meta): For natural language processing and generation
3. A custom ensemble of PaddleOCR, SentenceTransformers, and FAISS: For efficient text extraction and retrieval

This innovative approach allows us to:
- Answer complex questions related to image content
- Retrieve all text present in the image with high accuracy
- Handle multi-lingual content seamlessly

### Key Features
- Lightweight architecture suitable for running in a standard Google Colab environment
- High accuracy in cross-lingual text recognition
- Efficient text retrieval based on user queries

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
3. Launch the demo:
   ```
   python demo.py
   ```

## Testing
For optimal results, we recommend using the sample images provided in the `/sampleimages` directory. These images have been extensively tested and yield reliable outcomes.

## Conclusion
CrossOCR represents a significant step forward in multi-lingual image analysis, offering a robust solution that balances efficiency with accuracy. We welcome feedback and contributions to enhance this project further.
