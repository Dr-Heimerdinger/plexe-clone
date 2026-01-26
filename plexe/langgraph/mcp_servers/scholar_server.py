import sys
from mcp.server.fastmcp import FastMCP
from scholarly import scholarly

# Initialize FastMCP server
mcp = FastMCP("Google Scholar")

@mcp.tool()
def search_scholar(query: str, limit: int = 5):
    """
    Search for academic papers on Google Scholar.
    
    Args:
        query: Search query
        limit: Maximum number of results
    """
    search_query = scholarly.search_pubs(query)
    results = []
    for i, pub in enumerate(search_query):
        if i >= limit:
            break
        results.append({
            "title": pub.get('bib', {}).get('title'),
            "author": pub.get('bib', {}).get('author'),
            "pub_year": pub.get('bib', {}).get('pub_year'),
            "venue": pub.get('bib', {}).get('venue'),
            "abstract": pub.get('bib', {}).get('abstract'),
            "url": pub.get('pub_url'),
            "num_citations": pub.get('num_citations')
        })
    return results

@mcp.tool()
def get_author_info(name: str):
    """
    Get information about an academic author on Google Scholar.
    """
    search_query = scholarly.search_author(name)
    author = next(search_query, None)
    if author:
        author = scholarly.fill(author)
        return {
            "name": author.get('name'),
            "affiliation": author.get('affiliation'),
            "interests": author.get('interests'),
            "citedby": author.get('citedby'),
            "hindex": author.get('hindex'),
            "publications_count": len(author.get('publications', []))
        }
    return "Author not found"

if __name__ == "__main__":
    mcp.run()
