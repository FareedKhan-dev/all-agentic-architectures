import os
from langchain_core.tools import Tool


def web_search_tool(max_results=5):
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return Tool.from_function(
                func=lambda q: [
                    {"title": "Tavily non configuré", "url": "", "content": ""}
                ],
                name="web_search",
                description="Recherche web (désactivée: TAVILY_API_KEY manquante)",
            )
        t = TavilySearchResults(max_results=max_results, api_key=api_key)
        return Tool.from_function(
            func=lambda q: t.invoke({"query": q}),
            name="web_search",
            description="Recherche web factuelle et récente.",
        )
    except Exception as exc:
        err_msg = str(exc)
        return Tool.from_function(
            func=lambda q: [{"title": "Erreur outil", "url": "", "content": err_msg}],
            name="web_search",
            description="Erreur d'initialisation Tavily",
        )
