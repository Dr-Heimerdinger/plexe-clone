#!/bin/bash
set -e

CONTAINER_NAME="plexe-clone-postgres-1"
DB_USER="mlflow"

# Thiết lập file log
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/import_$(date +%Y%m%d_%H%M%S).log"

# Hàm log - ghi cả ra console và file
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" | tee -a "$LOG_FILE" >&2
}

log_step() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $1"
    echo "$message" | tee -a "$LOG_FILE"
}

print_header() {
    local header="========================================="
    echo "" | tee -a "$LOG_FILE"
    echo "$header" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "$header" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

print_error() {
    log_error "$1"
}

print_info() {
    log "INFO: $1"
}

cleanup() {
    log_step "Starting cleanup process"
    
    if [ "$KEEP_DATA" != "true" ]; then
        print_header "Cleanup"
        
        print_info "Removing CSV files from container..."
        if docker exec $CONTAINER_NAME bash -c "rm -f /tmp/*.csv /tmp/import_*.sql" 2>/dev/null; then
            log "CSV files removed from container successfully"
        else
            log "Failed to remove CSV files from container (may not exist)"
        fi
        
        if [ -d "$DATA_DIR" ]; then
            print_info "Removing local data directory: $DATA_DIR"
            rm -rf "$DATA_DIR"
            log "Data directory removed: $DATA_DIR"
        fi
        
        if [ "$REMOVE_RELBENCH_CACHE" = "true" ]; then
            RELBENCH_CACHE="$HOME/.cache/relbench"
            if [ -d "$RELBENCH_CACHE" ]; then
                print_info "Removing RelBench cache: $RELBENCH_CACHE"
                rm -rf "$RELBENCH_CACHE"
                log "RelBench cache removed"
            fi
        fi
        
        echo "Cleanup complete" | tee -a "$LOG_FILE"
    else
        print_info "Keeping data files (KEEP_DATA=true)"
    fi
    
    log_step "Cleanup completed"
}

trap 'log_error "Script failed on line $LINENO with exit code $?"; cleanup; exit 1' ERR
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

Logs are saved to: $LOG_DIR/

EOF
    exit 0
}

# Bắt đầu logging
log "=========================================="
log "Script started: $0"
log "Arguments: $*"
log "=========================================="

if [ $# -eq 0 ]; then
    show_usage
fi

DATASET_NAME=""
DB_NAME=""
OUTPUT_DIR=""
KEEP_DATA="false"
REMOVE_RELBENCH_CACHE="false"

log_step "Parsing command line arguments"

while [ $# -gt 0 ]; do
    case "$1" in
        --db-name)
            DB_NAME="$2"
            log "Setting DB_NAME=$DB_NAME"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            log "Setting OUTPUT_DIR=$OUTPUT_DIR"
            shift 2
            ;;
        --keep-data)
            KEEP_DATA="true"
            log "Setting KEEP_DATA=true"
            shift
            ;;
        --remove-cache)
            REMOVE_RELBENCH_CACHE="true"
            log "Setting REMOVE_RELBENCH_CACHE=true"
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
                log "Setting DATASET_NAME=$DATASET_NAME"
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
log "DATASET_SHORT=$DATASET_SHORT"

if [ -z "$DB_NAME" ]; then
    DB_NAME="$DATASET_SHORT"
    log "DB_NAME set to default: $DB_NAME"
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./${DATASET_SHORT}_data"
    log "OUTPUT_DIR set to default: $OUTPUT_DIR"
fi

DATA_DIR="$OUTPUT_DIR"
SQL_FILE="$DATA_DIR/import_${DATASET_SHORT}.sql"

print_header "RelBench Import - $DATASET_NAME"

echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Dataset:   $DATASET_NAME" | tee -a "$LOG_FILE"
echo "  Database:  $DB_NAME" | tee -a "$LOG_FILE"
echo "  Container: $CONTAINER_NAME" | tee -a "$LOG_FILE"
echo "  Data dir:  $DATA_DIR" | tee -a "$LOG_FILE"
echo "  Log file:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log_step "Checking Python installation"
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi
log "Python 3 is available: $(python3 --version)"

log_step "Checking Python dependencies"
print_info "Checking Python dependencies..."

if ! python3 -c "import relbench" 2>/dev/null; then
    print_info "Installing relbench..."
    if pip install relbench >> "$LOG_FILE" 2>&1; then
        log "relbench installed successfully"
    else
        log_error "Failed to install relbench"
        exit 1
    fi
else
    log "relbench is already installed"
fi

if ! python3 -c "import pandas" 2>/dev/null; then
    print_info "Installing pandas..."
    if pip install pandas >> "$LOG_FILE" 2>&1; then
        log "pandas installed successfully"
    else
        log_error "Failed to install pandas"
        exit 1
    fi
else
    log "pandas is already installed"
fi
echo "Python dependencies OK" | tee -a "$LOG_FILE"

print_header "Step 1: Verify PostgreSQL Container"
log_step "Verifying PostgreSQL container"
print_info "Checking container: $CONTAINER_NAME"

if ! docker ps | grep -q $CONTAINER_NAME; then
    print_error "Container '$CONTAINER_NAME' is not running"
    print_info "Start it with: docker start $CONTAINER_NAME"
    exit 1
fi
log "Container $CONTAINER_NAME is running"

print_header "Step 2: Prepare Database"
log_step "Preparing database"
print_info "Checking database: $DB_NAME"

if ! docker exec $CONTAINER_NAME psql -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    print_info "Database '$DB_NAME' does not exist, creating..."
    if docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" >> "$LOG_FILE" 2>&1; then
        log "Database '$DB_NAME' created successfully"
    else
        log_error "Failed to create database '$DB_NAME'"
        exit 1
    fi
else
    log "Database '$DB_NAME' already exists"
    print_info "Existing tables will be dropped and recreated"
fi

print_header "Step 3: Download Dataset and Generate SQL"
log_step "Downloading dataset and generating SQL"
print_info "Running Python script to download $DATASET_NAME dataset..."

if [ ! -f "generate_relbench_sql.py" ]; then
    print_error "generate_relbench_sql.py not found in current directory"
    exit 1
fi

log "Executing: python3 generate_relbench_sql.py $DATASET_NAME --output-dir $DATA_DIR"
if python3 generate_relbench_sql.py "$DATASET_NAME" --output-dir "$DATA_DIR" >> "$LOG_FILE" 2>&1; then
    log "Python script executed successfully"
else
    log_error "Python script failed with exit code $?"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not created, Python script may have failed"
    exit 1
fi

if [ ! -f "$SQL_FILE" ]; then
    print_error "SQL script not generated, Python script may have failed"
    exit 1
fi

log "Dataset downloaded and SQL generated successfully"

CSV_COUNT=$(ls -1 $DATA_DIR/*.csv 2>/dev/null | wc -l)
print_info "Found $CSV_COUNT CSV files"
log "CSV files count: $CSV_COUNT"

print_header "Step 4: Copy Files to Container"
log_step "Copying files to container"
print_info "Copying CSV files to container..."

for csv_file in $DATA_DIR/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        if docker cp "$csv_file" $CONTAINER_NAME:/tmp/ >> "$LOG_FILE" 2>&1; then
            log "Copied $filename to container"
            echo "Copied $filename"
        else
            log_error "Failed to copy $filename"
            exit 1
        fi
    fi
done

print_info "Copying SQL script to container..."
if docker cp "$SQL_FILE" $CONTAINER_NAME:/tmp/ >> "$LOG_FILE" 2>&1; then
    log "SQL script copied to container successfully"
    echo "SQL script copied"
else
    log_error "Failed to copy SQL script to container"
    exit 1
fi

print_header "Step 5: Import Data into PostgreSQL"
log_step "Importing data into PostgreSQL"
print_info "Running SQL import script..."
echo "=========================================" | tee -a "$LOG_FILE"

if docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/import_${DATASET_SHORT}.sql 2>&1 | tee -a "$LOG_FILE"; then
    log "SQL import completed successfully"
else
    log_error "SQL import failed with exit code $?"
    exit 1
fi

echo "=========================================" | tee -a "$LOG_FILE"
echo "Data import completed" | tee -a "$LOG_FILE"

print_header "Step 6: Verify Import"
log_step "Verifying import"
print_info "Checking table counts..."

docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c "
SELECT 
    schemaname,
    tablename,
    to_char(n_live_tup, 'FM999,999,999') as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
" 2>&1 | tee -a "$LOG_FILE"

print_header "Import Complete"

echo "Database Details:" | tee -a "$LOG_FILE"
echo "  Container: $CONTAINER_NAME" | tee -a "$LOG_FILE"
echo "  Database:  $DB_NAME" | tee -a "$LOG_FILE"
echo "  User:      $DB_USER" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Connect to database:" | tee -a "$LOG_FILE"
echo "  docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Example queries:" | tee -a "$LOG_FILE"
echo "  \\dt                    -- List all tables" | tee -a "$LOG_FILE"
echo "  \\d table_name          -- Show table structure" | tee -a "$LOG_FILE"
echo "  SELECT COUNT(*) FROM table_name;" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

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

log "=========================================="
log "Script completed successfully"
log "Log file: $LOG_FILE"
log "=========================================="

print_header "Done"