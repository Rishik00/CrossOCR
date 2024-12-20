from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel


class OllamaInput(BaseModel):
    query: str
    image_path: str


def get_ollama_output(model_name: str, inputs: OllamaInput) -> str:
    """
    Sends the query and image to the Ollama model and returns the response message content.
    
    Args:
        model_name (str): The name of the Ollama model to use.
        inputs (OllamaInput): The inputs containing the query and image path.

    Returns:
        str: The response message content from the model.
    """
    print(f"Model Name: {model_name}")
    print(f"Query: {inputs.query}")
    print(f"Image Path: {inputs.image_path}")

    try:
        # Sending data to the Ollama model
        response: ChatResponse = chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": inputs.query,
                    "images": [inputs.image_path],
                }
            ],
        )
        return response.message.content  # Return the model's response
    except Exception as e:
        raise RuntimeError(f"Failed to communicate with Ollama: {str(e)}")
