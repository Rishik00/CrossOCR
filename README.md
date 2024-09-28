## Introducing CrossOCR, my attempt at IIT Roorkee's assignment. 
Done by: Rishikesh M

## Interpreted requirements: 
1. To construct a streamlit/gradio application that can take images containing cross-linguals (Hindi and English).
2. The given image text has to be extracted and displayed on the screen upon searching for a particular word.
3. The task was to experiment with different OCR (Optical Character Recognition)/VLM (Vision Language Models) models and develop an efficient solution.

## My Approach: 
1. I like constructing apps that use simple and fast models in the backend and that can get the job done. Highlighted in the task was a combination of the byaldi library and ColPali (A combination of Colbert and Pali Gemma) for document retrieval. I experimented using a colab notebook with a T4 session (15GB VRAM and 12GB RAM) and it barely fit RAM and I felt the model was too big and complex to implement within the given timeframe. However, I have added a notebook in the repo that uses the same model for the task.
  a. Some of the key challenges I faced here were just understanding the architecture and the workings of the byaldi library, even though it seemed straightforward I faced a lot of dependency errors while experimenting.
  b. But It was very exciting to see that such models exist and do a really good job of addressing the task of OCR/Visual Document Answering.

2. My second approach was to use popular OCR libraries to extract text from the images from /sample_images. A few of them include:
  a. tesseract (pytesseract-hi and pytesseract-en) 
  b. TrOCR
  c. PaddleOCR
  d. General OCR Theory
One of my key criteria for considering any OCR model was the availability of multilingual text OCR, which was not a problem I faced because of the wide variety in the training data. In my opinion, a combination of Tesseract and PaddleOCR was not only simple to use but also very efficient, which was my second criterion. Unfortunately, I couldn't get General OCR theory to work in my colab environment, and even when I could the results were not on par with Tesseract and PaddleOCR. I have included all the model cards below for your reference.

3. For the demo, I have two solid options, either streamlit or gradio. But this time I went with gradio because I never used it for demos before and it works well with jupyter notebook environments such as colab, which was my go-to development place for tasks like these. 
