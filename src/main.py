import json
import numpy as np
from typing import List, Dict

## Local importa
from llms.vectorstore import FAISSEmbeddingsSearch
from ocr.paddle_ocr import PaddleOCRutil
from vlm.clip import CLiP

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_image(image_path: str, text_prompts: Dict[str, List[str]]) -> Dict:
    final_json_output = {
        'text_score': None,
        'image_score': None,
    }

    # OCR processing
    ocr = PaddleOCRutil(image_path)
    ocr_results, _ = ocr.parse_ocr_results()
    
    # CLIP processing
    vlm = CLiP()
    clip_res = vlm.inference(image_path, text_prompts['CliPPrompts'])
    
    print('CLIP Matches:')
    for prompt, score in clip_res:
        print(f"{prompt}: {numpy_to_python(score):.4f}")
    
    # Store the best CLIP score
    final_json_output['image_score'] = numpy_to_python(clip_res[0][1]) if clip_res else None

    print(ocr_results)
    # FAISS processing
    vems = FAISSEmbeddingsSearch()
    corpus = list(ocr_results.keys())  
    corpus_embeddings = vems.get_embeddings(corpus)
    vems.create_faiss_index(corpus_embeddings)

    query = text_prompts['FAISSPrompt'][0]
    dist, ind = vems.get_query_result(query, k=1)

    print('FAISS Results:')
    for i, idx in enumerate(ind[0]):
        print(f"Index: {idx}, Text: {corpus[idx]}, Distance: {numpy_to_python(dist[0][i])}")
    
    # Store the best FAISS score
    final_json_output['text_score'] = numpy_to_python(dist[0][0]) if len(dist) > 0 and len(dist[0]) > 0 else None

    # Save FAISS index
    vems.save_index("faiss_index.bin")

    return final_json_output

