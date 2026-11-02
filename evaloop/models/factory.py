"""Factory for creating model instances."""

import logging
from typing import Dict, Any

from .base import BaseLLM

# Backends are imported lazily: open_source_vllm raises ImportError when vLLM is missing
# and open_source needs torch, so eager imports broke API-only installs.


class ModelFactory:
    """Factory class for creating model instances."""
    
    def __init__(self):
        """Initialize the model factory."""
        self.logger = logging.getLogger(__name__)
    
    def create_model(self, model_config: Dict[str, Any]) -> BaseLLM:
        """
        Create a model instance based on configuration.
        
        Args:
            model_config: Configuration dictionary for the model.
            
        Returns:
            BaseLLM instance.
            
        Raises:
            ValueError: If model type is not supported.
        """
        model_type = model_config.get("type", "").lower()
        
        # Log configs without revealing full API keys
        sanitized_config = model_config.copy()
        if 'api_key' in sanitized_config:
            api_key = sanitized_config['api_key']
            if api_key:
                sanitized_config['api_key'] = f"{api_key[:5]}...{api_key[-3:]}" if len(api_key) > 8 else "***"
            else:
                self.logger.warning(f"API key for model {sanitized_config.get('name', 'unknown')} is empty!")
        
        self.logger.debug(f"Creating model with config: {sanitized_config}")
        
        if model_type in ["llama", "deepseek", "huggingface", "open_source"]:
            from .open_source import HuggingFaceLLM
            return HuggingFaceLLM(model_config)
        elif model_type in ["vllm"]:
            from .open_source_vllm import VllmLLM
            return VllmLLM(model_config)
        elif model_type in ["openai", "chatgpt", "closed_source"]:
            from .closed_source import OpenAILLM
            return OpenAILLM(model_config)
        elif model_type == "local_chat":
            from .local_chat import LocalChatLLM
            return LocalChatLLM(model_config)
        elif model_type == "gemini":
            from .gemini import GeminiLLM
            return GeminiLLM(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def list_supported_types(self) -> list:
        """
        List all supported model types.
        
        Returns:
            List of supported model type strings.
        """
        return [
            "huggingface",
            "open_source", 
            "llama",
            "deepseek",
            "vllm",
            "openai",
            "chatgpt",
            "closed_source",
            "gemini",
            "local_chat"
        ]
