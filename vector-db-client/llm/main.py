import  os
from dotenv import load_dotenv

load_dotenv()
MODEL = os.environ.get('GROQ_MODEL', 'meta-llama/llama-4-scout-17b-16e-instruct')

## Groq LLM
from langchain_groq import ChatGroq
def get_llm():
    """Create and return a Groq LLM instance with specified parameters.
    """
    return get_groq_llm()

def get_groq_llm():
    """Create and return a Groq LLM instance with specified parameters.
    """
    print("Using Groq LLM with model:", MODEL)
    return ChatGroq(model=MODEL,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=5,)