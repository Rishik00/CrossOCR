from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
import torch
from typing import Optional, List

from llms.base_lm import BaseLM

class TinyLlamaLM(BaseLM):
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype, device_map="auto"
            )
            self.model.to(self.device)
        except Exception as e:
            print(f"Error initializing TinyLlama model: {str(e)}")
            self.tokenizer, self.model = None, None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.tokenizer is None or self.model is None:
            return "Model initialization failed."
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
