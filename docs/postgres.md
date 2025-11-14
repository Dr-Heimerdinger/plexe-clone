## Running PostgreSQL (and MLflow) with Docker

This project uses MLflow and optionally can use PostgreSQL as a backing store. The repository includes a ready-to-run `docker-compose.yml` that starts:

- PostgreSQL 15
- MLflow server (configured to use Postgres for the backend store)
- pgAdmin (optional UI for managing the database)

Files added:

- `docker-compose.yml` — launches the services
- `.env.example` — example environment variables (copy to `.env` and edit)

Quick start

1. Copy the example env file and edit if you want different credentials:

```bash
cp .env.example .env
# edit .env if necessary
```

2. Start the stack:

```bash
docker compose up -d
```

3. Confirm services are running:

```bash
docker compose ps
```

4. MLflow UI will be available at http://localhost:5000

5. pgAdmin will be available at http://localhost:8080 (use credentials from `.env`).

How MLflow is configured

The compose uses an environment variable `MLFLOW_BACKEND_STORE_URI` pointing to Postgres:

```
postgresql+psycopg2://<user>:<password>@postgres:5432/<db>
```

MLflow stores metadata (experiments, runs) in Postgres and artifacts under the `./mlruns` folder mounted from the host.

Pointing Plexe to the MLflow server

The repo contains an MLflow callback at `plexe/internal/models/callbacks/mlflow.py` that sets the MLflow tracking URI via `mlflow.set_tracking_uri(...)` when the callback is initialized. You can either:

- Set the environment variable `MLFLOW_TRACKING_URI` before running your code, e.g. `export MLFLOW_TRACKING_URI=http://localhost:5000`.
- Or in Python set `mlflow.set_tracking_uri('http://localhost:5000')` before creating the `MLFlowCallback`.

Example — run a script with MLflow tracking enabled:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python examples/test_saved_model.py
```

Notes & tips

- The compose installs `psycopg2-binary` inside the `mlflow` container at startup for convenience. For production, build a small image with pinned deps.
- If you already have Postgres running locally, you can point `MLFLOW_BACKEND_STORE_URI` to it and remove the `postgres` service.
- To persist the DB data, `postgres_data` volume is created by compose.
