#!/bin/bash
set -e

CONTAINER_NAME="plexe-clone-postgres-1"
DB_USER="mlflow"

print_header() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
    echo ""
}

print_error() {
    echo "ERROR: $1" >&2
}

print_info() {
    echo "INFO: $1"
}

cleanup() {
    if [ "$KEEP_DATA" != "true" ]; then
        print_header "Cleanup"
        
        print_info "Removing CSV files from container..."
        docker exec $CONTAINER_NAME bash -c "rm -f /tmp/*.csv /tmp/import_*.sql" 2>/dev/null || true
        
        if [ -d "$DATA_DIR" ]; then
            print_info "Removing local data directory: $DATA_DIR"
            rm -rf "$DATA_DIR"
            echo "Data directory removed"
        fi
        
        if [ "$REMOVE_RELBENCH_CACHE" = "true" ]; then
            RELBENCH_CACHE="$HOME/.cache/relbench"
            if [ -d "$RELBENCH_CACHE" ]; then
                print_info "Removing RelBench cache: $RELBENCH_CACHE"
                rm -rf "$RELBENCH_CACHE"
                echo "RelBench cache removed"
            fi
        fi
        
        echo "Cleanup complete"
    else
        print_info "Keeping data files (KEEP_DATA=true)"
    fi
}

trap 'print_error "Script failed on line $LINENO"; cleanup; exit 1' ERR
trap 'cleanup' EXIT

show_usage() {
    cat << EOF
Usage: $0 <dataset-name> [options]

Arguments:
  dataset-name          RelBench dataset name (e.g., rel-f1, rel-amazon, rel-hm)

Options:
  --db-name NAME        Database name (default: derived from dataset name)
  --output-dir DIR      Output directory for data files
  --keep-data           Keep CSV and SQL files after import
  --remove-cache        Remove RelBench cache after import
  --help                Show this help message

Examples:
  $0 rel-f1
  $0 rel-amazon --db-name amazon
  $0 rel-hm --keep-data
  $0 rel-stack --db-name stackoverflow --output-dir ./stack_data

Supported datasets:
  rel-f1, rel-amazon, rel-hm, rel-stack, rel-trial,
  rel-event, rel-avito, rel-salt, rel-arxiv, rel-ratebeer

EOF
    exit 0
}

if [ $# -eq 0 ]; then
    show_usage
fi

DATASET_NAME=""
DB_NAME=""
OUTPUT_DIR=""
KEEP_DATA="false"
REMOVE_RELBENCH_CACHE="false"

while [ $# -gt 0 ]; do
    case "$1" in
        --db-name)
            DB_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --keep-data)
            KEEP_DATA="true"
            shift
            ;;
        --remove-cache)
            REMOVE_RELBENCH_CACHE="true"
            shift
            ;;
        --help)
            show_usage
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            ;;
        *)
            if [ -z "$DATASET_NAME" ]; then
                DATASET_NAME="$1"
            else
                print_error "Multiple dataset names provided"
                show_usage
            fi
            shift
            ;;
    esac
done

if [ -z "$DATASET_NAME" ]; then
    print_error "Dataset name is required"
    show_usage
fi

DATASET_SHORT=$(echo "$DATASET_NAME" | sed 's/^rel-//')

if [ -z "$DB_NAME" ]; then
    DB_NAME="$DATASET_SHORT"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./${DATASET_SHORT}_data"
fi

DATA_DIR="$OUTPUT_DIR"
SQL_FILE="$DATA_DIR/import_${DATASET_SHORT}.sql"

print_header "RelBench Import - $DATASET_NAME"

echo "Configuration:"
echo "  Dataset:   $DATASET_NAME"
echo "  Database:  $DB_NAME"
echo "  Container: $CONTAINER_NAME"
echo "  Data dir:  $DATA_DIR"
echo ""

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
echo "Python 3 is available"

print_info "Checking Python dependencies..."
if ! python3 -c "import relbench" 2>/dev/null; then
    print_info "Installing relbench..."
    pip install relbench
fi
if ! python3 -c "import pandas" 2>/dev/null; then
    print_info "Installing pandas..."
    pip install pandas
fi
echo "Python dependencies OK"

print_header "Step 1: Verify PostgreSQL Container"
print_info "Checking container: $CONTAINER_NAME"
if ! docker ps | grep -q $CONTAINER_NAME; then
    print_error "Container '$CONTAINER_NAME' is not running"
    print_info "Start it with: docker start $CONTAINER_NAME"
    exit 1
fi
echo "Container is running"

print_header "Step 2: Prepare Database"
print_info "Checking database: $DB_NAME"
if ! docker exec $CONTAINER_NAME psql -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    print_info "Database '$DB_NAME' does not exist, creating..."
    docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;"
    echo "Database '$DB_NAME' created"
else
    echo "Database '$DB_NAME' already exists"
    print_info "Existing tables will be dropped and recreated"
fi

print_header "Step 3: Download Dataset and Generate SQL"
print_info "Running Python script to download $DATASET_NAME dataset..."

if [ ! -f "generate_relbench_sql.py" ]; then
    print_error "generate_relbench_sql.py not found in current directory"
    exit 1
fi

python3 generate_relbench_sql.py "$DATASET_NAME" --output-dir "$DATA_DIR"

if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not created, Python script may have failed"
    exit 1
fi

if [ ! -f "$SQL_FILE" ]; then
    print_error "SQL script not generated, Python script may have failed"
    exit 1
fi

echo "Dataset downloaded and SQL generated"

CSV_COUNT=$(ls -1 $DATA_DIR/*.csv 2>/dev/null | wc -l)
print_info "Found $CSV_COUNT CSV files"

print_header "Step 4: Copy Files to Container"
print_info "Copying CSV files to container..."
for csv_file in $DATA_DIR/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        docker cp "$csv_file" $CONTAINER_NAME:/tmp/
        echo "Copied $filename"
    fi
done

print_info "Copying SQL script to container..."
docker cp "$SQL_FILE" $CONTAINER_NAME:/tmp/
echo "SQL script copied"

print_header "Step 5: Import Data into PostgreSQL"
print_info "Running SQL import script..."
echo "========================================="

docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/import_${DATASET_SHORT}.sql

echo "========================================="
echo "Data import completed"

print_header "Step 6: Verify Import"
print_info "Checking table counts..."

docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c "
SELECT 
    schemaname,
    tablename,
    to_char(n_live_tup, 'FM999,999,999') as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
"

print_header "Import Complete"

echo "Database Details:"
echo "  Container: $CONTAINER_NAME"
echo "  Database:  $DB_NAME"
echo "  User:      $DB_USER"
echo ""
echo "Connect to database:"
echo "  docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME"
echo ""
echo "Example queries:"
echo "  \\dt                    -- List all tables"
echo "  \\d table_name          -- Show table structure"
echo "  SELECT COUNT(*) FROM table_name;"
echo ""

if [ "$KEEP_DATA" = "false" ]; then
    read -p "Keep data files? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        export KEEP_DATA=true
        print_info "Data files will be kept in: $DATA_DIR"
    fi
fi

if [ "$KEEP_DATA" = "false" ] && [ "$REMOVE_RELBENCH_CACHE" = "false" ]; then
    read -p "Remove RelBench cache? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        export REMOVE_RELBENCH_CACHE=true
    fi
fi

print_header "Done"