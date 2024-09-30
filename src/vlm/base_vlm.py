## BaseVLM.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Union

class BaseVLM:
    def __init__(self, model_name: str = ''):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def inference(self, image_path: str, text_prompts: Union[str, List[str]]) -> List[Tuple[str, float]]:
        raise NotImplementedError('Not yet implemented!')