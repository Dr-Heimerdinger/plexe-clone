# Quick Start Guide - Plexe with Docker

## 🚀 Fastest Way to Run Plexe

### Option A: Development Build (Recommended for Quick Start)

**Faster (2-3 min), but no torch/transformers:**

```bash
cd /home/admin1/plexe-clone

# Build with dev config (fast, no torch)
docker compose -f docker-compose.dev.yml build

# Start services
docker compose -f docker-compose.dev.yml up -d

# Access services (same ports as production)
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### Option B: Production Build (Full Features)

**Slower (5-10 min), but includes torch/transformers:**

```bash
cd /home/admin1/plexe-clone

# Build with prod config (full features)
docker compose build

# Start services
docker compose up -d
```

### Step 3: Access

Open in browser:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **MLflow**: http://localhost:5000
- **pgAdmin**: http://localhost:8080 (admin@example.com / admin)

### Step 4: Check Status

```bash
# See all containers
docker compose ps
# or for dev:
docker compose -f docker-compose.dev.yml ps

# See logs
docker compose logs -f frontend
# or for dev:
docker compose -f docker-compose.dev.yml logs -f frontend

# Check if backend is healthy
curl http://localhost:8000/health
```

## 🛑 Stop Services

```bash
# Stop all (data persists)
docker compose down

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v
```

## 🔄 Development Workflow

### When You Change Python Code

```bash
# Rebuild backend (fast, uses cache)
docker compose build backend
docker compose up -d
```

### When You Change Frontend Code

```bash
# Rebuild frontend
docker compose build frontend
docker compose up -d
```

### When You Change pyproject.toml

```bash
# Rebuild backend (includes dependency install)
docker compose build --no-cache backend
docker compose up -d
```

### Local Frontend Dev (Hot Reload)

```bash
# In terminal 1: Start backend + database only
docker compose up postgres mlflow pgadmin

# In terminal 2: Run frontend with hot reload
cd plexe/ui/frontend
npm install
npm run dev
# Opens http://localhost:5173
```

## 📦 What's Running

| Service | URL | Port | Purpose |
|---------|-----|------|---------|
| Frontend | http://localhost:3000 | 3000 | React UI |
| Backend | http://localhost:8000 | 8000 | FastAPI + /ws |
| MLflow | http://localhost:5000 | 5000 | Experiment tracking |
| pgAdmin | http://localhost:8080 | 8080 | Database UI |
| Postgres | localhost:5432 | 5432 | Database |

## 🔗 How Services Connect

```
Browser (localhost:3000)
    ↓
Frontend (nginx, port 3000)
    ├─→ /ws ──→ nginx proxy ──→ Backend (port 8000)
    ├─→ /api ──→ nginx proxy ──→ Backend (port 8000)
    └─→ /static ──→ Serve React files
         ↓
Backend (FastAPI, port 8000)
    ├─→ Postgres (port 5432)
    └─→ MLflow (port 5000)
```

## ❓ FAQ

### Q: Why is it slow the first time?

A: Docker builds images from scratch. Subsequent runs are 10x faster (cache).

### Q: Status says "Disconnected"?

A: Usually means:
1. Backend still starting (wait 10s)
2. Check logs: `docker compose logs -f backend`
3. Verify backend is healthy: `curl http://localhost:8000/health`

### Q: Can't connect to database?

A: Postgres needs time to start. Check logs:
```bash
docker compose logs postgres
```

### Q: Want to use my own database?

Edit `.env`:
```env
POSTGRES_HOST=my-postgres-server
POSTGRES_PORT=5432
```

### Q: How to debug WebSocket issues?

```bash
# Check backend logs
docker compose logs -f backend

# Test WebSocket directly
wscat -c ws://localhost:3000/ws
```

### Q: Can I run frontend without Docker?

Yes! Dev mode:
```bash
cd plexe/ui/frontend
npm install
npm run dev
# Then also run: docker compose up postgres mlflow pgadmin backend
```

## 🎯 Common Commands

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f frontend

# Access container shell
docker compose exec backend bash

# Rebuild everything
docker compose build
docker compose up -d

# Clean up everything (keeps data)
docker compose down

# Clean up everything (removes data too!)
docker compose down -v

# Check service status
docker compose ps

# Restart a service
docker compose restart backend

# View service details
docker compose config
```

## 📚 More Info

- Docker guide: `docs/docker.md`
- Docker performance tips: `docs/docker-performance.md`
- PostgreSQL setup: `docs/postgres.md`
- Frontend dev: `plexe/ui/frontend/README.md`

---

**Next**: Open http://localhost:3000 and start building ML models! 🚀
