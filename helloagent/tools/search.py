from serpapi import SerpApiClient
import os
import 

def searhc(query: str) -> str:
    """
    A web search engine tool based on SerpApi.
    It intelligently parses search results, prioritizing direct answers or knowledge graph information.

    Arg:
    query: The search query as a string
    """
    print(f"Running SerpApi search for {query}")

    try:
        api_key = os.getenv("SERPAPI_API_KEY")

        if not api_key:
            return "Error no Serpapi api key found in .env"
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "au",
            "hl": "en-us"
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        #Smart interpretation
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"