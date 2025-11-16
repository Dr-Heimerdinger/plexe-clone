"""
Dataset API endpoints for Plexe backend
Handles file uploads, PostgreSQL connections, and dataset management
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["datasets"])

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("./data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class PostgresConnection(BaseModel):
    """PostgreSQL connection configuration"""

    host: str
    port: int
    database: str
    username: str
    password: str


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more dataset files

    Supported formats: CSV, XLSX, JSON, Parquet
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []
    errors = []

    # Allowed file extensions
    allowed_extensions = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}

    for file in files:
        try:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                errors.append(f"{file.filename}: Unsupported file format")
                continue

            # Save file
            file_path = UPLOADS_DIR / file.filename

            # Read and save file content
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)

            uploaded_files.append({"filename": file.filename, "size": len(contents), "path": str(file_path)})

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    if not uploaded_files and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return {"uploaded": uploaded_files, "errors": errors, "total": len(uploaded_files), "failed": len(errors)}


@router.post("/postgres/test")
async def test_postgres_connection(config: PostgresConnection):
    """
    Test PostgreSQL connection without saving
    """
    try:
        import psycopg2

        # Build connection string
        conn_str = (
            f"dbname={config.database} "
            f"user={config.username} "
            f"password={config.password} "
            f"host={config.host} "
            f"port={config.port}"
        )

        # Test connection
        conn = psycopg2.connect(conn_str)
        conn.close()

        return {
            "success": True,
            "message": "Connection successful",
            "host": config.host,
            "port": config.port,
            "database": config.database,
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="psycopg2 not installed. Install with: pip install psycopg2-binary")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


@router.post("/postgres/save")
async def save_postgres_connection(config: PostgresConnection):
    """
    Save PostgreSQL connection configuration
    TODO: Store in database or secure config file
    """
    try:
        import psycopg2

        # Build connection string
        conn_str = (
            f"dbname={config.database} "
            f"user={config.username} "
            f"password={config.password} "
            f"host={config.host} "
            f"port={config.port}"
        )

        # Test connection first
        conn = psycopg2.connect(conn_str)
        conn.close()

        # TODO: Save to secure config file or database
        # For now, just return success
        return {
            "success": True,
            "message": "Connection saved successfully",
            "host": config.host,
            "port": config.port,
            "database": config.database,
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="psycopg2 not installed. Install with: pip install psycopg2-binary")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save connection: {str(e)}")


@router.get("/datasets")
async def get_datasets():
    """
    Get list of uploaded datasets
    """
    try:
        datasets = []

        if UPLOADS_DIR.exists():
            for file_path in UPLOADS_DIR.iterdir():
                if file_path.is_file():
                    datasets.append(
                        {
                            "id": file_path.stem,
                            "filename": file_path.name,
                            "size": file_path.stat().st_size,
                            "created_at": file_path.stat().st_ctime,
                            "path": str(file_path),
                        }
                    )

        return {"datasets": datasets, "total": len(datasets)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset by ID (filename without extension)
    """
    try:
        # Find and delete the file
        for file_path in UPLOADS_DIR.iterdir():
            if file_path.stem == dataset_id and file_path.is_file():
                file_path.unlink()
                return {"success": True, "message": f"Dataset '{dataset_id}' deleted successfully"}

        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
