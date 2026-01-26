"""
External API clients for accessing academic databases and ML benchmarks.

This module provides clients for:
- Semantic Scholar API: Academic paper search and metadata (DEPRECATED - Use MCP instead)
- arXiv API: Preprint paper search (DEPRECATED - Use MCP instead)
- Papers With Code API: Benchmarks and SOTA results
- OpenML API: Dataset and model benchmarks
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import requests
from urllib.parse import urlencode
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)


@dataclass
class APIRateLimiter:
    """Simple rate limiter for API requests."""
    max_requests_per_minute: int = 10
    last_request_times: List[float] = None
    
    def __post_init__(self):
        if self.last_request_times is None:
            self.last_request_times = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove requests older than 1 minute
        self.last_request_times = [t for t in self.last_request_times if now - t < 60]
        
        if len(self.last_request_times) >= self.max_requests_per_minute:
            # Need to wait
            oldest = min(self.last_request_times)
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self.last_request_times.append(time.time())


class SemanticScholarClient:
    """
    Client for Semantic Scholar API.
    
    API Documentation: https://api.semanticscholar.org/
    Get API Key: https://www.semanticscholar.org/product/api#Partner-Form
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.rate_limiter = APIRateLimiter(max_requests_per_minute=100 if self.api_key else 1)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
    
    def search_papers(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        limit: int = 10,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers using Semantic Scholar API.
        
        Args:
            query: Search query string
            fields: List of fields to return (e.g., ['title', 'year', 'authors'])
            limit: Maximum number of results
            year_min: Minimum publication year
            year_max: Maximum publication year
        
        Returns:
            List of paper dictionaries
        """
        if fields is None:
            fields = ['paperId', 'title', 'abstract', 'year', 'venue', 
                     'authors', 'citationCount', 'influentialCitationCount']
        
        self.rate_limiter.wait_if_needed()
        
        params = {
            'query': query,
            'fields': ','.join(fields),
            'limit': limit
        }
        
        if year_min:
            params['year'] = f"{year_min}-{year_max or ''}"
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar API error: {e}")
            return []
    
    def get_paper_details(self, paper_id: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific paper."""
        if fields is None:
            fields = ['title', 'abstract', 'year', 'venue', 'authors', 'citationCount']
        
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={'fields': ','.join(fields)},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None


class ArxivClient:
    """
    Client for arXiv API.
    
    API Documentation: https://info.arxiv.org/help/api/index.html
    No API key required, but rate limits apply (1 request per 3 seconds recommended)
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter(max_requests_per_minute=20)
        self.session = requests.Session()
    
    def search_papers(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers.
        
        Args:
            query: Search query (can use arXiv query syntax)
            max_results: Maximum number of results
            sort_by: Sort by 'relevance', 'lastUpdatedDate', or 'submittedDate'
            sort_order: 'ascending' or 'descending'
        
        Returns:
            List of paper dictionaries
        """
        self.rate_limiter.wait_if_needed()
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            response = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_arxiv_response(response.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"arXiv API error: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse arXiv XML response into list of paper dicts."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'id': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                    'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else '',
                    'summary': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                    'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                    'updated': entry.find('atom:updated', ns).text if entry.find('atom:updated', ns) is not None else '',
                    'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                    'categories': [cat.get('term') for cat in entry.findall('atom:category', ns)]
                }
                papers.append(paper)
        except ET.ParseError as e:
            logger.error(f"Error parsing arXiv response: {e}")
        
        return papers


class PapersWithCodeClient:
    """
    Client for Papers With Code API.
    
    API Documentation: https://paperswithcode.com/api/v1/docs/
    No API key required
    """
    
    BASE_URL = "https://paperswithcode.com/api/v1"
    
    def __init__(self):
        self.rate_limiter = APIRateLimiter(max_requests_per_minute=20)
        self.session = requests.Session()
    
    def search_papers(self, query: str, items_per_page: int = 10) -> List[Dict[str, Any]]:
        """Search for papers on Papers With Code."""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/papers/",
                params={'q': query, 'items_per_page': items_per_page},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Papers With Code API error: {e}")
            return []
    
    def get_benchmarks(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get benchmark datasets and results."""
        self.rate_limiter.wait_if_needed()
        
        params = {}
        if task:
            params['task'] = task
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/benchmarks/",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Papers With Code benchmarks API error: {e}")
            return []
    
    def get_sota_results(self, benchmark: str) -> List[Dict[str, Any]]:
        """Get state-of-the-art results for a specific benchmark."""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/benchmarks/{benchmark}/results/",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Papers With Code SOTA API error: {e}")
            return []


class OpenMLClient:
    """
    Client for OpenML API.
    
    API Documentation: https://www.openml.org/apis
    API Key: https://www.openml.org/auth/sign-in (get from user profile)
    """
    
    BASE_URL = "https://www.openml.org/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENML_API_KEY")
        self.rate_limiter = APIRateLimiter(max_requests_per_minute=20)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"api_key": self.api_key})
    
    def search_datasets(
        self,
        task_type: Optional[str] = None,
        limit: int = 10,
        status: str = "active"
    ) -> List[Dict[str, Any]]:
        """Search for datasets on OpenML."""
        self.rate_limiter.wait_if_needed()
        
        params = {'limit': limit, 'status': status}
        if task_type:
            params['type'] = task_type
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/json/data/list",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('data', {}).get('dataset', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenML API error: {e}")
            return []
    
    def get_flow_runs(
        self,
        flow_id: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get flow runs (model evaluations) from OpenML.
        This can be used to find hyperparameters used in successful runs.
        """
        self.rate_limiter.wait_if_needed()
        
        params = {'limit': limit}
        if flow_id:
            params['flow'] = flow_id
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/json/run/list",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get('runs', {}).get('run', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenML flow runs API error: {e}")
            return []
