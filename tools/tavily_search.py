from langchain_tavily import TavilySearch

 
def get_tavily_tools():
    return [TavilySearch(max_results=3)] 