# Plexe Frontend

This folder contains a minimal Vite + React frontend for the Plexe Assistant.

## Development

Quick commands (from this folder):

- Install dependencies:

```bash
npm install
```

- Run dev server:

```bash
npm run dev
```

- Build production assets (output to `dist`):

```bash
npm run build
```

The FastAPI server at `plexe/server.py` will serve `ui/frontend/dist/index.html` if present. During development you can run `npm run dev` and open the Vite dev URL (usually http://localhost:5173) to work on the frontend.

## Docker

The frontend includes a `Dockerfile` for containerized builds:

- **Build stage**: Uses Node 18 to build the React app
- **Serve stage**: Uses nginx to serve the built assets

The `docker-compose.yml` at the repo root includes a `frontend` service that:
- Builds the frontend automatically using the `Dockerfile`
- Serves the frontend on `http://localhost:3000`
- Is managed alongside PostgreSQL, MLflow, and the backend

### Build & Run with Docker

From the repo root:

```bash
# Start all services (frontend, backend, postgres, mlflow, pgadmin)
docker compose up -d

# View logs
docker compose logs -f frontend
docker compose logs -f backend

# Stop all services
docker compose down
```

The frontend will be available at `http://localhost:3000` and the backend API at `http://localhost:8000`.

## Architecture

- **Frontend container**: nginx serving static React SPA
- **Backend container**: Python FastAPI serving the Plexe API and WebSocket at `/ws`
- **Database container**: PostgreSQL
- **MLflow container**: Experiment tracking
- **pgAdmin container**: Database UI

The frontend connects to the backend via:
- REST API endpoints (if needed)
- WebSocket at `/ws` (for real-time chat)

Both use relative URLs by default, so they work when served from the same origin (as with Docker Compose).

