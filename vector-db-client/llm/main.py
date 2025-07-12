import  os
from getpass import getpass
#os.environ['GROQ_API_KEY'] = getpass('GROQ_API_KEY')

## Groq LLM
from langchain_groq import ChatGroq
def get_groq_llm():
    """
    Returns a Groq LLM instance with specified parameters.
    """
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=5,)
