import os
from langchain_core.tools import Tool
try:
    # Nouveau chemin (community)
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception:  # fallback si ancienne version installée
    from langchain_tavily import TavilySearchResults


def web_search_tool(max_results=5):
    """Outil de recherche web (Tavily)."""
    api_key = os.getenv("TAVILY_API_KEY")
    # API community attend généralement `tavily_api_key=...`
    try:
        t = TavilySearchResults(max_results=max_results, tavily_api_key=api_key)
    except TypeError:
        # Compat: certaines versions acceptent api_key=
        t = TavilySearchResults(max_results=max_results, api_key=api_key)
    return Tool.from_function(
        func=lambda q: t.invoke({"query": q}),
        name="web_search",
        description="Recherche web factuelle et récente (news, événements, infos actualisées)."
    )
