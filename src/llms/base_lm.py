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