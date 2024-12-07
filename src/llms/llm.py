import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
import torch
import json
from typing import Optional, List
from dotenv import load_dotenv

from base_lm import SYSTEM_PROMPT

load_dotenv()

class TinyLlamaLM:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map="auto"
        )
        self.model.to(self.device)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    

class GemmaGroqForJSONExplanation:
    def __init__(self, model_name: str = "gemma-7b-it"):
        # Initialize the Groq client with the API key
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name  # Model name can be passed dynamically

    def explain(self, content: dict):
        try:
            # Serialize the content dictionary to a JSON string
            content_str = json.dumps(content, indent=2)
            
            # Request explanation from the Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Here's the JSON data:\n{content_str}"}
                ],
                model=self.model_name,
            )
            # Get the model's response
            model_response = chat_completion.choices[0].message.content
            
            # Format the response in Markdown
            markdown_response = f"### Explanation\n\n{model_response}\n"
            
            return markdown_response
        except Exception as e:
            # Return a Markdown-formatted error message
            return f"### Error\n\nAn error occurred: `{str(e)}`\n"

# if __name__ == "__main__":
#     # Example usage
#     gemma = GemmaGroqForJSONExplanation()
#     response = gemma.explain(
#         content={
#             'text_score': 0.0432,
#             'image_score': 0.88
#         }
#     )
#     print(response)  # Display the model's final response
