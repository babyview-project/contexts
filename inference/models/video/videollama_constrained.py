import logging
import time
import torch
import re
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from models.video.vqa import VQA
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# TODO: condense
class ConstrainedVideoLLaMALogitsProcessor:
    """Custom logits processor for constrained decoding in VideoLLaMA"""
    
    def __init__(self, processor, prompt_key_values: Dict[str, List[str]], tokenizer):
        self.processor = processor
        self.prompt_key_values = prompt_key_values
        self.tokenizer = tokenizer
        
        # Build vocabulary for allowed values with position tracking
        self.key_token_sequences = {}
        self.value_token_sequences = {}
        self.common_token_sequences = {}
        
        # Precompute key token sequences
        for key in prompt_key_values.keys():
            key_tokens = self.tokenizer.encode(key, add_special_tokens=False)
            self.key_token_sequences[key] = key_tokens
        
        # Precompute value token sequences for constrained keys
        for key, values in prompt_key_values.items():
            if values is not None:  # Only for constrained keys
                self.value_token_sequences[key] = {}
                for value in values:
                    value_tokens = self.tokenizer.encode(value, add_special_tokens=False)
                    self.value_token_sequences[key][value] = value_tokens
        
        # Precompute common token sequences
        common_phrases = [
            ":", "||", "/", " ", ".", ",", "\n", 
            " || ", "Location", "Activity", "Video", "description"
        ]
        for phrase in common_phrases:
            tokens = self.tokenizer.encode(phrase, add_special_tokens=False)
            self.common_token_sequences[phrase] = tokens
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply constraints to the logits"""
        batch_size, vocab_size = scores.shape
        
        for batch_idx in range(batch_size):
            current_sequence = input_ids[batch_idx]
            
            # Apply constraints based on current context
            constrained_scores = self._apply_constraints(current_sequence, scores[batch_idx])
            scores[batch_idx] = constrained_scores
            
        return scores
    
    def _apply_constraints(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply specific constraints based on the current token sequence"""
        # Create a copy to modify
        constrained_scores = scores.clone()
        
        # Decode the current sequence for state analysis
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Get current state and valid next tokens
        current_state, context = self._get_current_state(input_ids, decoded_text)
        valid_next_tokens = self._get_valid_next_tokens(current_state, context)
        
        if valid_next_tokens is not None:
            # Create mask that blocks all tokens except valid ones
            mask = torch.full_like(scores, float('-inf'))
            for token_id in valid_next_tokens:
                if token_id < len(mask):
                    mask[token_id] = 0
            constrained_scores = scores + mask
        
        return constrained_scores
    
    def _get_current_state(self, input_ids: torch.Tensor, decoded_text: str) -> Tuple[str, dict]:
        """Determine what we should be generating next based on current token sequence"""
        context = {}
        
        # Check if we're in the middle of generating a key
        for key, key_tokens in self.key_token_sequences.items():
            if self._is_partial_match(input_ids, key_tokens):
                remaining_position = self._get_partial_match_position(input_ids, key_tokens)
                return "generating_key", {"key": key, "position": remaining_position}
        
        # Check if we're in the middle of generating a value
        current_key = self._get_current_expecting_key(decoded_text)
        if current_key and current_key in self.value_token_sequences:
            for value, value_tokens in self.value_token_sequences[current_key].items():
                if self._is_partial_match(input_ids, value_tokens):
                    remaining_position = self._get_partial_match_position(input_ids, value_tokens)
                    return "generating_value", {
                        "key": current_key, 
                        "value": value, 
                        "position": remaining_position
                    }
        
        # Check if we're in the middle of generating common tokens (separators, etc.)
        for phrase, phrase_tokens in self.common_token_sequences.items():
            if self._is_partial_match(input_ids, phrase_tokens):
                remaining_position = self._get_partial_match_position(input_ids, phrase_tokens)
                return "generating_common", {"phrase": phrase, "position": remaining_position}
        
        # Determine state based on complete decoded text
        if decoded_text.strip() == "" or decoded_text.endswith("|| "):
            return "expecting_key", {}
        
        # Check if we just finished a key
        for key in self.prompt_key_values.keys():
            if decoded_text.endswith(key):
                return "expecting_colon", {}
        
        # Check if we just finished a colon after a key
        for key in self.prompt_key_values.keys():
            if f"{key}:" in decoded_text and not self._has_value_for_key(decoded_text, key):
                return "expecting_value", {"key": key}
        
        # Check if we need a separator
        if self._should_add_separator(decoded_text):
            return "expecting_separator", {}
        
        return "free_generation", {}
    
    def _is_partial_match(self, input_ids: torch.Tensor, target_tokens: List[int]) -> bool:
        """Check if the end of input_ids partially matches the beginning of target_tokens"""
        if len(target_tokens) == 0:
            return False
            
        input_list = input_ids.tolist()
        
        # Check all possible partial matches at the end of input
        for i in range(1, min(len(target_tokens) + 1, len(input_list) + 1)):
            if input_list[-i:] == target_tokens[:i]:
                return True
        return False
    
    def _get_partial_match_position(self, input_ids: torch.Tensor, target_tokens: List[int]) -> int:
        """Get the position in target_tokens where we should continue"""
        input_list = input_ids.tolist()
        
        # Find the longest partial match at the end
        for i in range(min(len(target_tokens), len(input_list)), 0, -1):
            if input_list[-i:] == target_tokens[:i]:
                return i
        return 0
    
    def _get_valid_next_tokens(self, state: str, context: dict) -> Optional[Set[int]]:
        """Get the set of valid next tokens based on current state"""
        if state == "generating_key":
            key = context["key"]
            position = context["position"]
            key_tokens = self.key_token_sequences[key]
            if position < len(key_tokens):
                return {key_tokens[position]}
            
        elif state == "generating_value":
            key = context["key"]
            value = context["value"]
            position = context["position"]
            value_tokens = self.value_token_sequences[key][value]
            if position < len(value_tokens):
                return {value_tokens[position]}
                
        elif state == "generating_common":
            phrase = context["phrase"]
            position = context["position"]
            phrase_tokens = self.common_token_sequences[phrase]
            if position < len(phrase_tokens):
                return {phrase_tokens[position]}
        
        elif state == "expecting_key":
            # Return first tokens of all unused keys
            valid_tokens = set()
            for key in self.prompt_key_values.keys():
                key_tokens = self.key_token_sequences[key]
                if key_tokens:
                    valid_tokens.add(key_tokens[0])
            return valid_tokens
            
        elif state == "expecting_colon":
            colon_tokens = self.common_token_sequences.get(":", [])
            return set(colon_tokens) if colon_tokens else None
            
        elif state == "expecting_value":
            key = context["key"]
            if key in self.value_token_sequences:
                # Return first tokens of all valid values for this key
                valid_tokens = set()
                for value_tokens in self.value_token_sequences[key].values():
                    if value_tokens:
                        valid_tokens.add(value_tokens[0])
                return valid_tokens
            
        elif state == "expecting_separator":
            separator_tokens = self.common_token_sequences.get("||", [])
            space_separator_tokens = self.common_token_sequences.get(" || ", [])
            valid_tokens = set()
            if separator_tokens:
                valid_tokens.update(separator_tokens)
            if space_separator_tokens:
                valid_tokens.add(space_separator_tokens[0])  # First token of " || "
            return valid_tokens
        
        return None  # No constraints (free generation)
    
    def _get_current_expecting_key(self, decoded_text: str) -> Optional[str]:
        """Find which key we're currently expecting a value for"""
        for key in self.prompt_key_values.keys():
            if f"{key}:" in decoded_text and not self._has_value_for_key(decoded_text, key):
                return key
        return None
    
    def _has_value_for_key(self, text: str, key: str) -> bool:
        """Check if the key already has a value assigned"""
        pattern = f"{key}:\\s*([^|]+)"
        match = re.search(pattern, text)
        return match is not None and match.group(1).strip() != ""
    
    def _should_add_separator(self, text: str) -> bool:
        """Check if we should add a separator"""
        # Count completed key-value pairs
        completed_pairs = 0
        for key in self.prompt_key_values.keys():
            if self._has_value_for_key(text, key):
                completed_pairs += 1
        
        # If we have completed pairs but not all, and the last character isn't a separator
        return completed_pairs > 0 and completed_pairs < len(self.prompt_key_values) - 1 and not text.endswith("|| ")

class VideoLLaMA(VQA):
    def __init__(self, model_name: str = "DAMO-NLP-SG/VideoLLaMA3-7B", device: str = "cuda", use_constrained_decoding: bool = True):
        self.use_constrained_decoding = use_constrained_decoding
        super().__init__(model_name, device)

    def _setup_model(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.max_frames = 100
        self.fps = 2
    
    def get_response(self, video_path: str, question: str, prompt_key_values: Optional[Dict[str, List[str]]] = None) -> str:
        start_time = time.time()
        fps = self.fps if self.fps is not None else 2
        max_frames = self.max_frames if self.max_frames is not None else 100

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    {"type": "text", "text": question},
                ]
            },
        ]

        inputs = self.processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Prepare generation kwargs
        generation_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        
        # Add constrained decoding if enabled and constraints are provided
        if self.use_constrained_decoding and prompt_key_values is not None:
            try:
                from transformers import LogitsProcessorList
                
                # Create constrained logits processor
                constrained_processor = ConstrainedVideoLLaMALogitsProcessor(
                    self.processor, 
                    prompt_key_values, 
                    self.processor.tokenizer
                )
                
                # Add to logits processors
                generation_kwargs["logits_processor"] = LogitsProcessorList([constrained_processor])
                
                # Use lower temperature for more focused generation
                generation_kwargs["temperature"] = 0.3
                generation_kwargs["do_sample"] = True
                
                logging.info("Using constrained decoding")
                
            except Exception as e:
                logging.warning(f"Failed to setup constrained decoding: {e}. Falling back to regular generation.")
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode response
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # Post-process to extract just the assistant's response
        if "assistant" in response.lower():
            # Split on assistant and take the last part
            parts = response.lower().split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Clean up the response
        response = self._clean_response(response, prompt_key_values)
        
        logging.info(f"Model response time: {time.time() - start_time:.1f} seconds")
        return response
    
    def _clean_response(self, response: str, prompt_key_values: Optional[Dict[str, List[str]]] = None) -> str:
        """Clean and format the response to match expected format"""
        if prompt_key_values is None:
            return response
        
        # Remove any system/user/assistant prefixes
        response = re.sub(r'^(system|user|assistant):\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # If the response doesn't follow the expected format, try to extract information
        #if not self._is_properly_formatted(response, prompt_key_values):
        #    response = self._reformat_response(response, prompt_key_values)
        
        return response
    
    def _is_properly_formatted(self, response: str, prompt_key_values: Dict[str, List[str]]) -> bool:
        """Check if response follows the expected key: value || format"""
        for key in prompt_key_values.keys():
            if f"{key}:" not in response:
                return False
        return True
    
    def _reformat_response(self, response: str, prompt_key_values: Dict[str, List[str]]) -> str:
        """Attempt to reformat response to match expected format"""
        formatted_parts = []
        
        for key in prompt_key_values.keys():
            value = "..."  # Default value
            
            # Try to extract value for this key from the response
            if prompt_key_values[key] is not None:  # Constrained key
                for allowed_value in prompt_key_values[key]:
                    if allowed_value.lower() in response.lower():
                        value = allowed_value
                        break
            else:  # Free-form key (like Video description)
                # Try to find a relevant part of the response
                # This is a simple heuristic - you might want to improve this
                sentences = response.split('.')
                if sentences and len(sentences[0]) > 10:
                    value = sentences[0].strip()[:100] + "..."
            
            formatted_parts.append(f"{key}: {value}")
        
        return " || ".join(formatted_parts)

    def get_response_with_constraints(self, video_path: str, question: str, prompt_key_values: Dict[str, List[str]]) -> str:
        """Convenience method that explicitly passes constraints"""
        return self.get_response(video_path, question, prompt_key_values)