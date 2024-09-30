## clip.py
import torch
from PIL import Image
from typing import Union, List, Tuple
from transformers import CLIPProcessor, CLIPModel

from vlm.base_vlm import BaseVLM

class CLiP(BaseVLM):
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14'):
        super().__init__(model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device).to(self.dtype)
        self.model.eval()

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt', padding=True)
        return inputs.to(self.device)

    def preprocess_text(self, text_prompts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        text_inputs = self.processor(text=text_prompts, return_tensors='pt', padding=True)
        return text_inputs.to(self.device)

    def get_image_features(self, image_inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)

        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_text_features(self, text_inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)

        return text_features / text_features.norm(dim=-1, keepdim=True)

    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor, temperature: float = 150.0) -> torch.Tensor:
        return (image_features @ text_features.T).squeeze(0) * temperature

    def inference(self, image_path: str, text_prompts: Union[str, List[str]], top_k: int = 5) -> List[Tuple[str, float]]:
        image_inputs = self.preprocess_image(image_path)
        text_inputs = self.preprocess_text(text_prompts)

        image_features = self.get_image_features(image_inputs)
        text_features = self.get_text_features(text_inputs)

        similarity_scores = self.compute_similarity(image_features, text_features)
        
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        scores = similarity_scores.cpu().numpy()
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = [(text_prompts[i], scores[i]) for i in top_indices]
        
        return results
