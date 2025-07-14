
from langchain_groq import ChatGroq
from langchain_qwq import ChatQwen
from langchain_community.llms.llamacpp import LlamaCpp
from enum import Enum
import logging
import os
from ai_service.utils.constants import DEST
logging.basicConfig(level=logging.INFO)

class LLMProvider(Enum):
    GROQ = "groq"
    QWEN_REMOTE = "qwen_remote"
    QWEN_LOCAL = "qwen_local"


class LLM():
    """A class to manage the LLM instance."""
    
    def __init__(self, provider: LLMProvider = LLMProvider.QWEN_REMOTE, model_name: str | None = None):
        self.provider = provider
        self.model_name = model_name

    def get_llm(self):
        """Create and return an LLM instance based on the provider."""
        if self.provider == LLMProvider.GROQ:
            return self.get_groq_llm()
        elif self.provider == LLMProvider.QWEN_REMOTE:
            return self.get_qwen_llm()
        elif self.provider == LLMProvider.QWEN_LOCAL:
            return self.get_llamacpp_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        

    def get_groq_llm(self):
        """Create and return a Groq LLM instance with specified parameters.
        """
        self.model_name = "meta-llama/llama-4-maverick-17b-128e-instruct" if self.model_name is None else self.model_name
        logging.info("Using Groq LLM with model: %s", self.model_name)
        return ChatGroq(model=self.model_name,
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=5)

    def get_qwen_llm(self):
        """Create and return a Qwen LLM instance with specified parameters.
        """
        self.model_name = "qwen3-32b" if self.model_name is None else self.model_name
        logging.info("Using Qwen LLM with model: %s", self.model_name)
        return ChatQwen(model=self.model_name)
    
    def get_llamacpp_llm(self):
        """Create and return a LlamaCpp LLM instance with specified parameters.
        """
        if not DEST.exists():
            raise FileNotFoundError(f"Model file {DEST} does not exist. Please run the correct docker profile (docker compose --profile local-gpu up -d) to download the model.")
        logging.info("Using LlamaCpp LLM with model: %s", self.model_name)

        n_gpu_layers = 28
        # Optimized for RTX 3080 10GB
        return LlamaCpp(model_path=DEST,  # type: ignore[call-arg]
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            n_batch=256,
            temperature=0.0,
            max_tokens=1024,
            n_threads=os.cpu_count(),
            use_mlock=True,
            use_mmap=True,
            verbose=False,
        )