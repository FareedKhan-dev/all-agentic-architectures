import os

def make_llm(provider: str | None = None,
             model: str | None = None,
             temperature: float = 0.2):
    """
    Fabrique un LLM interchangeable OpenAI/Gemini selon .env.
    Évite de passer des kwargs non supportés (ex: max_retries) au SDK Gemini.
    """
    p = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "openai")).lower()
    m = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=m, temperature=temperature)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m, temperature=temperature)