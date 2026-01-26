import os
from mcp.server.fastmcp import FastMCP
import requests
from typing import List, Dict, Any, Optional

# Initialize FastMCP server
mcp = FastMCP("Semantic Scholar")

BASE_URL = "https://api.semanticscholar.org/graph/v1"

def get_session():
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    session = requests.Session()
    if api_key:
        session.headers.update({"x-api-key": api_key})
    return session

@mcp.tool()
def search_papers(
    query: str,
    limit: int = 10,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Search for papers using Semantic Scholar API.
    """
    fields = ['paperId', 'title', 'abstract', 'year', 'venue', 
             'authors', 'citationCount', 'influentialCitationCount']
    
    params = {
        'query': query,
        'fields': ','.join(fields),
        'limit': limit
    }
    
    if year_min:
        params['year'] = f"{year_min}-{year_max or ''}"
    
    session = get_session()
    try:
        response = session.get(
            f"{BASE_URL}/paper/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        return [{"error": str(e)}]

@mcp.tool()
def get_paper_details(paper_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific paper."""
    fields = ['title', 'abstract', 'year', 'venue', 'authors', 'citationCount']
    
    session = get_session()
    try:
        response = session.get(
            f"{BASE_URL}/paper/{paper_id}",
            params={'fields': ','.join(fields)},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()
