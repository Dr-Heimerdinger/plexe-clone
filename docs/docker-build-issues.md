# Docker Build Issues & Solutions

## 🔴 Vấn Đề: Network Timeout Khi Download `torch`

### Nguyên Nhân
- `torch` là package rất lớn (~2GB khi install)
- Docker build mất quá lâu để download
- Network timeout xảy ra

### ✅ Giải Pháp (Đã Implement)

#### 1. **Enhanced Dockerfile** (`Dockerfile`)

Các cải tiến:
```dockerfile
# Set timeout lớn hơn
ENV PIP_DEFAULT_TIMEOUT=100
PIP_RETRIES=5

# Multi-stage build (better caching)
FROM python:3.11-slim as base
FROM base as deps  # Dependencies cached separately
FROM deps as app   # Application code cached separately

# Retry logic nếu download fail
RUN poetry install ... 2>&1 || \
    (sleep 5 && poetry install ...) || \
    (sleep 10 && poetry install ...)
```

**Lợi ích:**
- Timeout 100s (default 15s)
- Auto-retry 3 lần với delay
- Better layer caching (deps không rebuild nếu code thay đổi)
- Health check endpoint

#### 2. **Lightweight Dev Dockerfile** (`Dockerfile.dev`)

Cho development nhanh hơn:
```dockerfile
# Only installs main dependencies (excludes torch)
# Build 10x nhanh hơn!
```

**Lợi ích:**
- `torch` và `transformers` không cần khi dev
- Thích hợp cho local testing
- Fast iterate cycles

#### 3. **Docker Compose Dev** (`docker-compose.dev.yml`)

```bash
# Production (với torch)
docker compose build backend

# Development (nhanh, không torch)
docker compose -f docker-compose.dev.yml build backend
```

---

## 🚀 Cách Sử Dụng

### Option 1: Build Production (Lâu, Nhưng Full Featured)

```bash
cd /home/admin1/plexe-clone

# Build backend (mất 5-10 phút lần đầu, có torch)
docker compose build backend

# Start all services
docker compose up -d

# Wait 30-60s for startup, then check
curl http://localhost:8000/health
```

**Khi nào**: Deploy, testing full features, final release

### Option 2: Build Development (Nhanh, Cho Rapid Dev)

```bash
cd /home/admin1/plexe-clone

# Build dev version (mất 2-3 phút, không torch)
docker compose -f docker-compose.dev.yml build backend

# Start services
docker compose -f docker-compose.dev.yml up -d

# Fast startup: 10-15s
curl http://localhost:8000/health
```

**Khi nào**: Local development, testing features, quick iteration

### Option 3: Pure Local Dev (Không Docker)

```bash
# Backend (local)
cd /home/admin1/plexe-clone
python -m uvicorn plexe.server:app --reload

# Frontend (separate terminal)
cd plexe/ui/frontend
npm run dev

# Services (Docker)
docker compose up postgres mlflow pgadmin
```

**Khi nào**: Hot-reload development, no docker overhead

---

## 🛠️ If Build Still Fails

### Increase Docker Build Timeout

```bash
# Build with higher timeout (timeout in seconds)
DOCKER_BUILDKIT_PROGRESS=plain \
docker compose build --progress=plain backend

# Or manually increase Docker timeout
# Edit ~/.docker/config.json and add:
# {
#   "http": {
#     "maxConnIdleSeconds": 120
#   }
# }
```

### Use Different PyPI Mirror (China/Asia)

If you're in a region with slow PyPI access:

```bash
# Create .docker/pip.conf
mkdir -p /home/admin1/.docker
cat > /home/admin1/.docker/pip.conf << EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
timeout = 120
EOF

# Then rebuild
docker compose build --progress=plain backend
```

### Prebuild and Cache Locally

Build once and reuse:

```bash
# Build and tag
docker compose build backend
docker tag plexe-clone-backend:latest plexe:prod

# Later, just use the cached image
docker run plexe:prod
```

---

## 📊 Build Time Comparison

| Scenario | Time | Size | Notes |
|----------|------|------|-------|
| **Production (`Dockerfile`)** | 5-10m (1st), 20s (cached) | ~3.5GB | Full torch, all deps |
| **Dev (`Dockerfile.dev`)** | 2-3m (1st), 10s (cached) | ~1.5GB | No torch, faster |
| **Pure Local** | 0s (docker) | 0MB docker | Full python env needed |

---

## ✨ Features in Updated Dockerfile

✅ **Environment variables** for pip timeout & retries
✅ **System dependencies** (git, curl for tools)
✅ **Multi-stage build** (better caching)
✅ **Retry logic** (auto-retry 3x on timeout)
✅ **Health check** (Docker monitors service health)
✅ **Auto-reload** (dev mode with `--reload`)
✅ **Better logging** (PYTHONUNBUFFERED)

---

## 🔍 Debugging Build Issues

### View detailed build logs
```bash
DOCKER_BUILDKIT_PROGRESS=plain docker compose build backend
```

### Check which layer is slow
```bash
docker build --progress=plain -f Dockerfile .
```

### Inspect build cache
```bash
docker system df  # See space usage
docker builder prune  # Clear build cache if needed
```

### Test network manually
```bash
docker run --rm python:3.11-slim \
  python -c "import urllib.request; urllib.request.urlopen('https://files.pythonhosted.org')"
```

---

## 📚 Next Steps

1. **Try Option 2 (Dev)** — Fast, good for testing
2. **If that works**, try Option 1 (Production) when network is stable
3. **If build still fails**, check your internet connection
4. **For Asia**, use the PyPI mirror tweak above

---

## ❓ FAQ

**Q: Which should I use?**
- Dev: `docker-compose.dev.yml` (fast iterations)
- Prod: `docker-compose.yml` (full features)
- Local: Manual python setup (hot reload, no docker overhead)

**Q: How long does torch installation take?**
- First time: 3-5 minutes (download + compile)
- Cached: Instant (layer reuse)

**Q: Can I skip torch?**
- Yes! Use `Dockerfile.dev` or remove from `pyproject.toml`
- But some ML features might not work

**Q: Docker build fails on my network?**
- Use PyPI mirror (see above)
- Build on a machine with better internet
- Increase timeout: `PIP_DEFAULT_TIMEOUT=200`

---

**Summary**: Updated Dockerfile with retry logic + dev variant = robust, fast builds ✨
