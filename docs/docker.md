# Docker Integration Guide

This project includes complete Docker support for development and production deployment.

## Services

The `docker-compose.yml` orchestrates:

| Service | Image/Build | Port | Purpose |
|---------|------------|------|---------|
| **frontend** | Custom (Vite + nginx) | 3000 | React UI for Plexe Assistant |
| **backend** | Python 3.11 | 8000 | FastAPI server + WebSocket |
| **postgres** | postgres:15 | 5432 | Database (MLflow backend) |
| **mlflow** | Python 3.11 | 5000 | Experiment tracking server |
| **pgadmin** | pgadmin4:7 | 8080 | Database management UI |

## Quick Start

### Prerequisites

- Docker & Docker Compose installed
- Node.js 18+ (for local frontend dev; not needed if using Docker)

### Option 1: Full Stack with Docker (Recommended for Development)

```bash
# 1. Copy and edit environment variables
cp .env.example .env
# Edit .env if you want to change credentials

# 2. Start all services
docker compose up -d

# 3. Watch logs (optional)
docker compose logs -f frontend
docker compose logs -f backend

# 4. Access services
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# MLflow UI: http://localhost:5000
# pgAdmin: http://localhost:8080 (admin@example.com / admin)
# Postgres: localhost:5432 (mlflow / mlflow)

# 5. Stop services
docker compose down
```

### Option 2: Local Frontend Dev + Docker Backend

Useful if you want hot-reload for React without rebuilding Docker image:

```bash
# 1. Start backend services only
docker compose up -d postgres mlflow pgadmin

# 2. In a separate terminal, run frontend dev server
cd plexe/ui/frontend
npm install
npm run dev
# Opens http://localhost:5173

# 3. The frontend dev server will connect to backend at http://localhost:8000
# (because the React code uses relative URLs and defaults to current host)
```

### Option 3: Build & Run Frontend Image Separately

For testing the Dockerfile locally:

```bash
cd plexe/ui/frontend

# Build the image
docker build -t plexe-frontend:latest .

# Run the container
docker run -p 3000:80 plexe-frontend:latest

# Open http://localhost:3000
```

## Environment Variables

Key variables in `.env.example` (copy to `.env` to customize):

```env
# Database credentials
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow_db

# MLflow backend store URI
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow_db

# MLflow tracking URI (where server is running)
MLFLOW_TRACKING_URI=http://mlflow:5000

# Frontend API URL (for backend communication)
REACT_APP_API_URL=http://localhost:8000

# pgAdmin credentials
PGADMIN_EMAIL=admin@example.com
PGADMIN_PASSWORD=admin
```

## Architecture

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌──────────────────────────┐
│  Frontend (nginx)        │ :3000
│  Serves React SPA        │
└──────────────────────────┘
       │ API calls & WS
       ▼
┌──────────────────────────┐
│  Backend (FastAPI)       │ :8000
│  /ws for chat            │
│  / serves dist/index.html│
└──────────────────────────┘
       │
  ┌────┴─────┐
  ▼          ▼
┌──────────┐ ┌────────────┐
│Postgres  │ │  MLflow    │
│  :5432   │ │   :5000    │
└──────────┘ └────────────┘
```

In Docker Compose, services communicate via:
- Service name (within containers): `postgres`, `mlflow`, `frontend`, `backend`
- Localhost + port (from host): `localhost:5432`, `localhost:5000`, etc.

## Frontend Docker Details

### Dockerfile Structure

The `Dockerfile` uses a **multi-stage build**:

1. **Build stage** (Node): Installs npm deps and runs `npm run build` → outputs `dist/`
2. **Serve stage** (nginx): Copies `dist/` and serves with nginx on port 80

### nginx Configuration

- `nginx.conf` routes all requests to `index.html` (SPA routing)
- Assets in `/assets/` are cached long-term
- `/health` endpoint for container health checks

### Building the Frontend Image

When you run `docker compose up frontend`, Docker:
1. Reads `plexe/ui/frontend/Dockerfile`
2. Installs Node deps
3. Runs `npm run build`
4. Copies the built `dist/` folder into nginx container
5. Exposes port 80 (mapped to 3000 on host)

No rebuilds needed unless source code changes (Docker uses cache).

## Backend Docker Details

The backend runs:
- Python 3.11-slim image
- Installs the repo as editable package (`pip install -e .`)
- Runs `uvicorn plexe.server:app --reload`

Volume mapping (`- .:/app`) allows live code reloads during development.

## Common Tasks

### View service logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f frontend
docker compose logs -f backend
docker compose logs -f postgres

# Last 50 lines of a service
docker compose logs -n 50 frontend
```

### Access a container shell

```bash
# Frontend (nginx)
docker compose exec frontend sh

# Backend (Python)
docker compose exec backend bash

# Database (psql)
docker compose exec postgres psql -U mlflow -d mlflow_db
```

### Rebuild a service (if Dockerfile changed)

```bash
# Rebuild frontend
docker compose build frontend

# Or rebuild and restart
docker compose up -d --build frontend
```

### Clean up volumes and containers

```bash
# Stop and remove containers, networks (keeps volumes by default)
docker compose down

# Also remove volumes (⚠️ deletes data)
docker compose down -v
```

## Troubleshooting

### Frontend won't connect to backend

- Check that backend is running: `docker compose ps` should show backend as `Up`
- Check backend logs: `docker compose logs -f backend`
- Verify WebSocket URL in React code matches backend port (default `:8000`)
- If using dev server, ensure CORS is enabled or use a proxy

### Port already in use

If port 3000 or 8000 is already in use:
```bash
# Change in docker-compose.yml or .env
# Edit ports:
#   - "3001:80"  # frontend on 3001 instead of 3000
#   - "8001:8000" # backend on 8001
```

### Database connection errors

- Verify `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` match in all services
- Check postgres is running: `docker compose logs postgres`
- Ensure MLflow can reach postgres: `docker compose exec mlflow psql -h postgres -U mlflow -d mlflow_db`

### React build fails in Docker

- Check `npm run build` works locally: `cd plexe/ui/frontend && npm run build`
- Verify `package.json` has all required scripts
- Check Node version in Dockerfile (currently 18-alpine)

## Production Deployment

For production:

1. **Build images with specific versions** (don't use `latest`)
   ```bash
   docker compose build --tag plexe:v1.0.0
   ```

2. **Use a Kubernetes or Swarm orchestrator** for scaling

3. **Use environment-specific `.env` files**:
   ```bash
   docker compose --env-file .env.prod up -d
   ```

4. **Use a reverse proxy** (nginx/traefik) in front of the services

5. **Enable HTTPS/TLS** for WebSocket (WSS)

6. **Mount volumes for persistence**:
   - Database: `postgres_data` volume (already configured)
   - MLflow artifacts: `mlruns` directory (already configured)

## Next Steps

- Customize `nginx.conf` for your needs
- Add authentication/authorization layers
- Set up CI/CD to build and push Docker images
- Deploy to cloud (AWS, GCP, Azure, DigitalOcean, etc.)
