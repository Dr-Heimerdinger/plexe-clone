import os
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Kaggle")

@mcp.tool()
def search_kaggle_datasets(query: str, limit: int = 5):
    """
    Search for datasets on Kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    datasets = api.dataset_list(search=query)
    results = []
    for i, ds in enumerate(datasets):
        if i >= limit:
            break
        results.append({
            "ref": ds.ref,
            "title": ds.title,
            "size": ds.size,
            "lastUpdated": str(ds.lastUpdated),
            "downloadCount": ds.downloadCount,
            "voteCount": ds.voteCount
        })
    return results

@mcp.tool()
def download_kaggle_dataset(dataset_ref: str, path: str = "data"):
    """
    Download a Kaggle dataset.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(path, exist_ok=True)
    api.dataset_download_files(dataset_ref, path=path, unzip=True)
    return f"Dataset {dataset_ref} downloaded to {path}"

if __name__ == "__main__":
    mcp.run()
