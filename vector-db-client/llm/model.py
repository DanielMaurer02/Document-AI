
from langchain_groq import ChatGroq
from langchain_qwq import ChatQwen
from enum import Enum

class LLMProvider(Enum):
    GROQ = "groq"
    QWEN = "qwen"


class LLM():
    """A class to manage the LLM instance."""
    
    def __init__(self, provider: LLMProvider = LLMProvider.QWEN, model_name: str | None = None):
        self.provider = provider
        self.model_name = model_name

    def get_llm(self):
        """Create and return an LLM instance based on the provider."""
        if self.provider == LLMProvider.GROQ:
            return self.get_groq_llm()
        elif self.provider == LLMProvider.QWEN:
            return self.get_qwen_llm()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        

    def get_groq_llm(self):
        """Create and return a Groq LLM instance with specified parameters.
        """
        self.model_name = "meta-llama/llama-4-maverick-17b-128e-instruct" if self.model_name is None else self.model_name
        print("Using Groq LLM with model:", self.model_name)
        return ChatGroq(model=self.model_name,
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=5)

    def get_qwen_llm(self):
        """Create and return a Qwen LLM instance with specified parameters.
        """
        self.model_name = "qwen3-30b-a3b" if self.model_name is None else self.model_name
        print("Using Qwen LLM with model:", self.model_name)
        return ChatQwen(model=self.model_name)