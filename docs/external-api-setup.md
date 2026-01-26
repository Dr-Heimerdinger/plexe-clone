# External API Configuration Guide
## Overview

Plexe's hyperparameter optimization tools use external APIs to search academic literature and benchmarks for optimal hyperparameters. This eliminates the need for expensive trial-and-error training runs.

## Required API Keys

### 1. Semantic Scholar API

**Purpose**: Search academic papers for hyperparameter recommendations  
**Cost**: Free for academic use (up to 100 requests/minute with API key, 1/minute without)

**How to get it:**
1. Visit https://www.semanticscholar.org/product/api
2. Click "Apply for an API Key" or visit the Partner Form
3. Fill out the application form (usually approved within 1-2 business days)
4. You'll receive an API key via email

**Configuration:**
```bash
export SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
```

Or add to `.env` file:
```
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
```

### 2. OpenML API Key

**Purpose**: Access benchmark datasets and model performance data  
**Cost**: Free

**How to get it:**
1. Visit https://www.openml.org/auth/sign-in
2. Create an account (or sign in with GitHub)
3. Go to your profile settings
4. Click on "API authentication" section
5. Your API key will be displayed there

**Configuration:**
```bash
export OPENML_API_KEY=your_api_key_here
```

Or add to `.env` file:
```
OPENML_API_KEY=your_api_key_here
```

### 3. Hugging Face Token (Optional)

**Purpose**: Access Hugging Face datasets and models  
**Cost**: Free

**How to get it:**
1. Visit https://huggingface.co/settings/tokens
2. Create an account if you don't have one
3. Click "New token"
4. Give it a name (e.g., "plexe-hpo")
5. Select "Read" permissions
6. Copy the generated token

**Configuration:**
```bash
export HF_TOKEN=your_token_here
```

Or add to `.env` file:
```
HF_TOKEN=your_token_here
```

## APIs That Don't Require Keys

### arXiv API

**Purpose**: Search preprint papers  
**No API key required** - Rate limited to 1 request per 3 seconds  
**Documentation**: https://info.arxiv.org/help/api/index.html

### Papers With Code API

**Purpose**: Access benchmark leaderboards and SOTA results  
**No API key required** - Rate limited  
**Documentation**: https://paperswithcode.com/api/v1/docs/

## Configuration File

Create a `.env` file in the project root:

```bash
# Copy from .env.example
cp .env.example .env
```

Then edit `.env` and add your API keys:

```bash
# External API Keys for HPO Search
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
OPENML_API_KEY=your_openml_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## Rate Limits

Be aware of rate limits for each service:

| Service | With API Key | Without API Key |
|---------|-------------|-----------------|
| Semantic Scholar | 100 req/min | 1 req/min |
| arXiv | N/A | 1 req/3s (~20/min) |
| Papers With Code | N/A | ~20 req/min |
| OpenML | 20 req/min | 10 req/min |

The clients include automatic rate limiting to respect these limits.

## Troubleshooting

### "API key not found" error

Make sure your `.env` file is in the project root and the key names match exactly:
- `SEMANTIC_SCHOLAR_API_KEY`
- `OPENML_API_KEY`
- `HF_TOKEN`

### "Rate limit exceeded" error

The clients automatically handle rate limiting, but if you see this error:
1. Wait a few minutes
2. Reduce the number of concurrent requests
3. Consider getting an API key if using a service without one

### "Connection error" or "Timeout"

1. Check your internet connection
2. Verify the API service is online (check status pages)
3. Try increasing the timeout in the client configuration

## Using MCP (Model Context Protocol)

**Note**: The current implementation uses direct REST API calls. To use MCP:

1. Install MCP client library:
```bash
pip install model-context-protocol
```

2. Configure MCP servers in your MCP configuration file

3. Update the API clients to use MCP instead of direct HTTP requests

MCP provides a standardized way to access these services with better error handling, caching, and composability. See the MCP documentation for more details: https://modelcontextprotocol.io/

## Security Best Practices

1. **Never commit API keys** to version control
2. Add `.env` to `.gitignore` (already done in this project)
3. Use environment variables in production
4. Rotate API keys periodically
5. Use minimal permissions (e.g., read-only tokens for Hugging Face)