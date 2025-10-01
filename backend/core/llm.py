import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def make_llm(provider: str | None = None,
             model: str | None = None,
             temperature: float = 0.2):
    """Fabrique un LLM interchangeable OpenAI/Gemini selon .env."""
    p = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")).lower()
    m = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    if p == "gemini":
        return ChatGoogleGenerativeAI(model=m, temperature=temperature)
    return ChatOpenAI(model=m, temperature=temperature)

import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def make_llm(provider: str | None = None,
             model: str | None = None,
             temperature: float = 0.2):
    p = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")).lower()
    m = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    if p == "gemini":
        return ChatGoogleGenerativeAI(model=m, temperature=temperature)
    return ChatOpenAI(model=m, temperature=temperature)
