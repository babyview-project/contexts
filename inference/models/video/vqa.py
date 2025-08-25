import os
import numpy as np
import torch
import random
import shutil
import ray
import time
from typing import Dict, List, Optional, Union, Any

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

class VQA:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._setup_model()
    
    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    def _setup_model(self) -> None:
        raise NotImplementedError
        
    def get_response(self, video_path: str, question: str) -> str:
        raise NotImplementedError
        
    def convert_model_response_to_dict(self, 
                                        response: str, 
                                        key_list: List[str],
                                        value_constraints: Optional[Dict[str, List[str]]] = None) -> Optional[Dict[str, str]]:
        if len(response) < 10:
            print(f"Response too short, skipping..., response: {response}")
            return None
        
        lines = response.split('||')
        lines = [line.strip() for line in lines if line.strip()]
        lines = [line.replace('<', '').replace('>', '') for line in lines]
        print(f"raw response: {lines}")
        response_dict = {}
        for key in key_list:
            found = False
            for line in lines:
                line_split = line.split(":")
                if len(line_split) <= 1:
                    continue
                answer = line_split[-1].strip()
                if len(answer) < 2:
                    continue
                if line.startswith(key):
                    if value_constraints and key in value_constraints:
                        if not any(val.lower() == answer.lower() for val in value_constraints[key]):
                            print(f"Invalid value for {key}: {answer}")
                            return None
                        # Ensure correct case
                        answer = next(val for val in value_constraints[key] 
                                    if val.lower() == answer.lower())
                    found = True
                    response_dict[key] = answer
                    break
            if not found:
                print(f"Missing key: {key}, response: {response}")
                return None
        
        return response_dict
