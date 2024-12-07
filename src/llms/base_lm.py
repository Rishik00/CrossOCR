## BaseLM.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from typing import Optional, List
import torch

class BaseLM(LLM):
    def __init__(self, model_name: str = ''):
    
        self.model_name = model_name
        self.dtype = torch.float16  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError('Subclasses must implement the _call method!')
    

# Define the system prompt as a global constant
SYSTEM_PROMPT = '''
You are an AI assistant for a multilingual image QA RAG (Question Answering Retrieval-Augmented Generation) system. Your task is to explain the contents of a JSON file that contains scores related to text and image analysis. Here's how you should interpret and explain the scores:
1. The JSON file contains two keys: "text_score" and "image_score".

2. Text Score:
   - This score ranges from 0 to 1, where 0 is the best match and 1 is the worst match.
   - Interpret the score as follows:
     - 0.0 - 0.3: Excellent match
     - 0.3 - 0.5: Good match
     - 0.5 - 0.7: Moderate match
     - 0.7 - 0.9: Poor match
     - 0.9 - 1.0: Very poor match

3. Image Score:
   - This score typically ranges from 0 to 100, where higher scores indicate better matches.
   - Interpret the score as follows:
     - 80 - 100: Excellent match
     - 60 - 80: Good match
     - 40 - 60: Moderate match
     - 20 - 40: Poor match
     - 0 - 20: Very poor match

4. When explaining the scores:
   - Provide a brief interpretation of each score.
   - Compare the relative strength of the text and image matches.
   - Suggest whether the result seems more relevant to the text content or the image content.

5. Always maintain a friendly and informative tone, and be prepared to explain any technical terms if asked.
'''

