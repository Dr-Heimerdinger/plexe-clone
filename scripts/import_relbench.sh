#!/bin/bash
set -e

CONTAINER_NAME="plexe-clone-postgres-1"
DB_USER="mlflow"
DB_PASSWORD="mlflow"  # Thêm password

# Thiết lập file log
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/import_$(date +%Y%m%d_%H%M%S).log"

# Cấu hình sample size
SAMPLE_SIZE=5000  # Mục tiêu số bản ghi tổng
MAX_SEED_RECORDS=5000  # Số bản ghi seed ban đầu từ bảng chính

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
        if docker exec $CONTAINER_NAME bash -c "rm -f /tmp/*.csv /tmp/import_*.sql /tmp/sample_*.sql" 2>/dev/null; then
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
  --db-password PASS    Database password (default: mlflow)
  --output-dir DIR      Output directory for data files
  --sample-size N       Target total records to import (default: 3000)
  --seed-records N      Initial seed records from main table (default: 500)
  --keep-data           Keep CSV and SQL files after import
  --remove-cache        Remove RelBench cache after import
  --help                Show this help message

Examples:
  $0 rel-f1
  $0 rel-amazon --db-name amazon --sample-size 5000
  $0 rel-hm --keep-data --seed-records 1000
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
        --db-password)
            DB_PASSWORD="$2"
            log "Setting DB_PASSWORD=[HIDDEN]"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            log "Setting OUTPUT_DIR=$OUTPUT_DIR"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            log "Setting SAMPLE_SIZE=$SAMPLE_SIZE"
            shift 2
            ;;
        --seed-records)
            MAX_SEED_RECORDS="$2"
            log "Setting MAX_SEED_RECORDS=$MAX_SEED_RECORDS"
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
SAMPLE_SQL_FILE="$DATA_DIR/sample_${DATASET_SHORT}.sql"

print_header "RelBench Partial Import - $DATASET_NAME"

echo "Configuration:" | tee -a "$LOG_FILE"
echo "  Dataset:       $DATASET_NAME" | tee -a "$LOG_FILE"
echo "  Database:      $DB_NAME" | tee -a "$LOG_FILE"
echo "  Container:     $CONTAINER_NAME" | tee -a "$LOG_FILE"
echo "  Data dir:      $DATA_DIR" | tee -a "$LOG_FILE"
echo "  Sample size:   $SAMPLE_SIZE records" | tee -a "$LOG_FILE"
echo "  Seed records:  $MAX_SEED_RECORDS" | tee -a "$LOG_FILE"
echo "  Log file:      $LOG_FILE" | tee -a "$LOG_FILE"
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

print_header "Step 4: Create Schema and Load Full Data to Temporary Database"
log_step "Creating temporary database for sampling"

TEMP_DB="${DB_NAME}_temp"
print_info "Creating temporary database: $TEMP_DB"

docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $TEMP_DB;" >> "$LOG_FILE" 2>&1
if docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "CREATE DATABASE $TEMP_DB;" >> "$LOG_FILE" 2>&1; then
    log "Temporary database created: $TEMP_DB"
else
    log_error "Failed to create temporary database"
    exit 1
fi

print_info "Copying CSV files to container..."
for csv_file in $DATA_DIR/*.csv; do
    if [ -f "$csv_file" ]; then
        filename=$(basename "$csv_file")
        if docker cp "$csv_file" $CONTAINER_NAME:/tmp/ >> "$LOG_FILE" 2>&1; then
            log "Copied $filename to container"
        else
            log_error "Failed to copy $filename"
            exit 1
        fi
    fi
done

print_info "Copying SQL script to container..."
if docker cp "$SQL_FILE" $CONTAINER_NAME:/tmp/ >> "$LOG_FILE" 2>&1; then
    log "SQL script copied to container successfully"
else
    log_error "Failed to copy SQL script to container"
    exit 1
fi

print_info "Loading full data into temporary database..."
if docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $TEMP_DB -f /tmp/import_${DATASET_SHORT}.sql >> "$LOG_FILE" 2>&1; then
    log "Full data loaded into temporary database"
else
    log_error "Failed to load data into temporary database"
    exit 1
fi

print_header "Step 5: Analyze Schema and Generate Sample"
log_step "Analyzing database schema and relationships"

print_info "Generating intelligent sample with foreign key relationships..."

# Tạo script Python để phân tích và tạo sample
cat > "$DATA_DIR/create_sample.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import sys
import psycopg2
from collections import defaultdict, deque
import json

def get_table_info(conn):
    """Lấy thông tin về các bảng và số lượng bản ghi"""
    cur = conn.cursor()
    cur.execute("""
        SELECT schemaname, relname, n_live_tup 
        FROM pg_stat_user_tables 
        WHERE schemaname = 'public'
        ORDER BY n_live_tup DESC
    """)
    tables = {}
    for row in cur.fetchall():
        tables[row[1]] = int(row[2])  # row[1] is relname, row[2] is n_live_tup
    cur.close()
    return tables

def format_ids_for_sql(ids):
    """Format list of IDs for SQL IN clause"""
    formatted = []
    for id_val in ids:
        if id_val is None:
            continue
        elif isinstance(id_val, str):
            escaped = id_val.replace("'", "''")
            formatted.append(f"'{escaped}'")
        elif isinstance(id_val, (int, float)):
            formatted.append(str(id_val))
        else:
            escaped = str(id_val).replace("'", "''")
            formatted.append(f"'{escaped}'")
    return ','.join(formatted) if formatted else "''"

def get_foreign_keys(conn):
    """Lấy thông tin về foreign keys"""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            tc.table_name as from_table,
            kcu.column_name as from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
          AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
    """)
    
    fks = defaultdict(list)
    reverse_fks = defaultdict(list)
    
    for row in cur.fetchall():
        from_table, from_col, to_table, to_col = row
        fks[from_table].append({
            'from_col': from_col,
            'to_table': to_table,
            'to_col': to_col
        })
        reverse_fks[to_table].append({
            'to_col': to_col,
            'from_table': from_table,
            'from_col': from_col
        })
    
    cur.close()
    return dict(fks), dict(reverse_fks)

def find_central_table(tables, fks, reverse_fks, pk_columns):
    """Find central table - prioritize fact tables with multiple FKs for better sampling"""
    import math
    scores = {}
    
    print("\nAnalyzing table importance for sampling...")
    for table in tables:
        # Điểm = số FK đi ra + số FK đi vào
        outgoing = len(fks.get(table, []))
        incoming = len(reverse_fks.get(table, []))
        
        # PRIORITIZE FACT TABLES (tables with 2+ FKs) - these connect dimensions
        # Example: review table connects customer + product
        if outgoing >= 2:
            fk_score = 10000 + (outgoing * 1000)  # Huge bonus for fact tables
        else:
            fk_score = outgoing * 100
        
        # Add points for being referenced by other tables
        fk_score += incoming * 50
        
        # Prefer tables with primary key (easier to track)
        has_pk = pk_columns.get(table, 'ctid') != 'ctid'
        pk_bonus = 500 if has_pk else 0
        
        # Consider table size - prefer tables with data
        size_score = math.log10(tables.get(table, 0) + 1) * 10 if tables.get(table, 0) > 0 else 0
        
        total_score = fk_score + pk_bonus + size_score
        scores[table] = total_score
        
        print(f"  {table:20s}: FKs={outgoing}, reverse={incoming}, PK={has_pk}, rows={tables.get(table, 0):,}, score={total_score:.0f}")
    
    # Sắp xếp theo điểm
    sorted_tables = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_tables:
        selected = sorted_tables[0][0]
        print(f"\nSelected central table: {selected} (score={sorted_tables[0][1]:.0f})")
        return selected
    return list(tables.keys())[0] if tables else None

def create_sample_sql(conn, db_name, target_db, sample_size, seed_records):
    """Tạo SQL script để sample dữ liệu có liên kết"""
    
    print("Analyzing database schema...")
    tables = get_table_info(conn)
    fks, reverse_fks = get_foreign_keys(conn)
    
    print(f"\nFound {len(tables)} tables:")
    for table, count in sorted(tables.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {table}: {count:,} records")
    
    print(f"\nFound {sum(len(v) for v in fks.values())} foreign key relationships")
    
    # Get primary keys for all tables FIRST (needed for central table selection)
    cur = conn.cursor()
    pk_columns = {}
    for table in tables:
        cur.execute(f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{table}'::regclass AND i.indisprimary
        """)
        pk_result = cur.fetchone()
        if pk_result:
            pk_columns[table] = pk_result[0]
        else:
            # Fallback: use ctid (system column) for tables without primary key
            pk_columns[table] = 'ctid'
    
    central_table = find_central_table(tables, fks, reverse_fks, pk_columns)
    print(f"\nCentral table identified: {central_table} (PK: {pk_columns.get(central_table, 'none')})")
    
    # Tạo temporary tables để lưu IDs đã chọn
    sampled_ids = defaultdict(set)
    visited_tables = set()
    
    cur = conn.cursor()
    
    # Bước 1: Lấy seed records từ bảng trung tâm
    print(f"\nStep 1: Sampling {seed_records} seed records from {central_table}...")
    
    pk_col = pk_columns[central_table]
    
    # Sample random records - for tables without PK, use ctid
    if pk_col == 'ctid':
        # For tables without PK (fact tables), sample and immediately get referenced dimension records
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{central_table}' ORDER BY ordinal_position")
        all_columns = [row[0] for row in cur.fetchall()]
        columns_str = ', '.join(all_columns)
        
        # Use TABLESAMPLE for better performance on large tables
        cur.execute(f"""
            SELECT {columns_str} FROM {central_table}
            TABLESAMPLE SYSTEM (1)  -- Sample ~1% of pages
            LIMIT {seed_records}
        """)
        # Store tuples of all column values
        seed_ids = [row for row in cur.fetchall()]
        sampled_ids[central_table] = set(seed_ids)
        print(f"  Sampled {len(seed_ids)} records from {central_table} (no PK - using full row)")
        
        # CRITICAL: Extract FK values to get referenced dimension records
        col_name_to_idx = {col: idx for idx, col in enumerate(all_columns)}
        
        for fk in fks.get(central_table, []):
            to_table = fk['to_table']
            from_col = fk['from_col']
            to_col = fk['to_col']
            
            # Extract FK values from sampled rows
            fk_col_idx = col_name_to_idx.get(from_col)
            if fk_col_idx is not None:
                fk_values = set()
                for row in seed_ids:
                    if row[fk_col_idx] is not None:
                        fk_values.add(row[fk_col_idx])
                
                if fk_values:
                    sampled_ids[to_table] = fk_values
                    visited_tables.add(to_table)
                    total_records += len(fk_values)
                    print(f"    -> Collected {len(fk_values)} {to_table} records (via {from_col})")
        
        total_records = len(seed_ids)
    else:
        cur.execute(f"""
            SELECT {pk_col} FROM {central_table}
            TABLESAMPLE SYSTEM (1)
            LIMIT {seed_records}
        """)
        seed_ids = [row[0] for row in cur.fetchall()]
        sampled_ids[central_table] = set(seed_ids)
        print(f"  Sampled {len(seed_ids)} records from {central_table}")
        total_records = len(seed_ids)
    
    # Bước 2: BFS để lấy các bản ghi liên quan
    queue = deque([(central_table, seed_ids)])
    visited_tables.add(central_table)
    total_records = len(seed_ids)
    
    print(f"\nStep 2: Cascading through foreign key relationships...")
    iteration = 0
    
    while queue and total_records < sample_size:
        iteration += 1
        current_table, current_ids = queue.popleft()
        
        if not current_ids:
            continue
        
        print(f"\n  Iteration {iteration}: Processing {current_table} ({len(current_ids)} IDs)")
        
        # Lấy bảng cha (through FK)
        for fk in fks.get(current_table, []):
            to_table = fk['to_table']
            from_col = fk['from_col']
            to_col = fk['to_col']
            
            if total_records >= sample_size:
                break
            
            # Lấy parent IDs - sử dụng subquery an toàn hơn
            ids_str = format_ids_for_sql(current_ids)
            if not ids_str or ids_str == "''":
                continue
                
            try:
                cur.execute(f"""
                    SELECT DISTINCT {to_col}
                    FROM {to_table}
                    WHERE {to_col} IS NOT NULL
                    AND {to_col} IN (
                        SELECT DISTINCT {from_col}
                        FROM {current_table}
                        WHERE {from_col} IS NOT NULL
                        AND {from_col} IN ({ids_str})
                    )
                """)
                
                parent_ids = [row[0] for row in cur.fetchall()]
                new_ids = set(parent_ids) - sampled_ids[to_table]
                
                if new_ids:
                    sampled_ids[to_table].update(new_ids)
                    total_records += len(new_ids)
                    print(f"    -> {to_table}: +{len(new_ids)} records (total: {len(sampled_ids[to_table])})")
                    
                    if to_table not in visited_tables:
                        queue.append((to_table, list(new_ids)))
                        visited_tables.add(to_table)
            except Exception as e:
                print(f"    -> {to_table}: Error - {e}")
                conn.rollback()  # Rollback transaction on error
                continue
        
        # Lấy bảng con (reverse FK) - giới hạn số lượng để tránh quá nhiều
        for rfk in reverse_fks.get(current_table, []):
            from_table = rfk['from_table']
            from_col = rfk['from_col']
            to_col = rfk['to_col']
            
            if total_records >= sample_size:
                break
            
            # Lấy child IDs với giới hạn
            ids_str = format_ids_for_sql(current_ids)
            if not ids_str or ids_str == "''":
                continue
                
            limit = min(500, sample_size - total_records)  # Giới hạn mỗi lần lấy 500 records
            
            try:
                child_pk_col = pk_columns.get(from_table)
                
                # For tables without PK, we can't track by ID - skip them in reverse FK traversal
                # They will be sampled based on FK constraints later
                if child_pk_col == 'ctid':
                    print(f"    <- {from_table}: Skipped (no PK - will sample by FK constraint)")
                    continue
                
                cur.execute(f"""
                    SELECT DISTINCT {child_pk_col}
                    FROM {from_table}
                    WHERE {from_col} IS NOT NULL
                    AND {from_col} IN ({ids_str})
                    LIMIT {limit}
                """)
                
                child_ids = [row[0] for row in cur.fetchall()]
                new_ids = set(child_ids) - sampled_ids[from_table]
                
                if new_ids:
                    sampled_ids[from_table].update(new_ids)
                    total_records += len(new_ids)
                    print(f"    <- {from_table}: +{len(new_ids)} records (total: {len(sampled_ids[from_table])})")
                    
                    if from_table not in visited_tables:
                        queue.append((from_table, list(new_ids)))
                        visited_tables.add(from_table)
            except Exception as e:
                print(f"    <- {from_table}: Error - {e}")
                conn.rollback()  # Rollback transaction on error
                continue
    
    # Bước 2.3: Sample tables without primary keys based on FK relationships
    print(f"\nStep 2.3: Sampling tables without primary keys...")
    for table in tables:
        if pk_columns.get(table) != 'ctid':
            continue  # Skip tables with primary keys
        
        if table in sampled_ids and sampled_ids[table]:
            continue  # Already sampled
        
        # Sample records from this table based on foreign key relationships
        if table in fks and fks[table]:
            print(f"  Processing {table} (no PK)...")
            
            # Build WHERE clause based on all FKs
            where_conditions = []
            for fk in fks[table]:
                to_table = fk['to_table']
                from_col = fk['from_col']
                
                if to_table in sampled_ids and sampled_ids[to_table]:
                    ids_str = format_ids_for_sql(list(sampled_ids[to_table]))
                    if ids_str and ids_str != "''":
                        where_conditions.append(f"{from_col} IN ({ids_str})")
            
            if where_conditions:
                limit = min(1000, sample_size - total_records)
                where_clause = " OR ".join(where_conditions)
                
                try:
                    # Get all columns for this table
                    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' ORDER BY ordinal_position")
                    all_columns = [row[0] for row in cur.fetchall()]
                    columns_str = ', '.join(all_columns)
                    
                    cur.execute(f"""
                        SELECT {columns_str}
                        FROM {table}
                        WHERE {where_clause}
                        LIMIT {limit}
                    """)
                    
                    rows = cur.fetchall()
                    if rows:
                        # Store as tuples (can't use IDs for tables without PK)
                        sampled_ids[table] = set(rows)
                        total_records += len(rows)
                        print(f"    {table}: +{len(rows)} records (total: {len(rows)})")
                        
                        # IMPORTANT: Collect referenced entities from these rows
                        # Map column names to indices
                        col_name_to_idx = {col: idx for idx, col in enumerate(all_columns)}
                        
                        for fk in fks[table]:
                            to_table = fk['to_table']
                            from_col = fk['from_col']
                            to_col = fk['to_col']
                            
                            # Skip if target table doesn't have PK (can't collect them)
                            if pk_columns.get(to_table) == 'ctid':
                                continue
                            
                            # Extract foreign key values from sampled rows
                            fk_col_idx = col_name_to_idx.get(from_col)
                            if fk_col_idx is not None:
                                fk_values = set()
                                for row in rows:
                                    if row[fk_col_idx] is not None:
                                        fk_values.add(row[fk_col_idx])
                                
                                # Add these to sampled_ids for the referenced table
                                if fk_values:
                                    new_fk_values = fk_values - sampled_ids.get(to_table, set())
                                    if new_fk_values:
                                        if to_table not in sampled_ids:
                                            sampled_ids[to_table] = set()
                                        sampled_ids[to_table].update(new_fk_values)
                                        total_records += len(new_fk_values)
                                        print(f"      -> {to_table}: +{len(new_fk_values)} records (from {from_col})")
                except Exception as e:
                    print(f"    {table}: Error - {e}")
                    conn.rollback()  # Rollback transaction on error
    
    # Bước 2.5: Validate và ensure FK integrity by adding missing referenced records
    print(f"\nStep 2.5: Ensuring foreign key integrity...")
    added_count = 0
    max_iterations = 10
    
    for iteration in range(max_iterations):
        added_this_iteration = 0
        
        for table in list(sampled_ids.keys()):
            # Skip tables without primary keys - can't validate them the same way
            if pk_columns.get(table) == 'ctid':
                continue
                
            if table not in fks or not fks[table]:
                continue
                
            ids_list = list(sampled_ids[table])
            
            # Check each FK constraint and ADD missing referenced records
            for fk in fks[table]:
                to_table = fk['to_table']
                from_col = fk['from_col']
                to_col = fk['to_col']
                
                # Skip if target table has no primary key
                if pk_columns.get(to_table) == 'ctid':
                    continue
                
                # Get FK values for sampled records
                ids_str = format_ids_for_sql(ids_list)
                if not ids_str or ids_str == "''":
                    continue
                
                try:
                    # Find FK values that are NOT in the sampled set
                    cur.execute(f"""
                        SELECT DISTINCT t.{from_col}
                        FROM {table} t
                        WHERE t.{pk_columns[table]} IN ({ids_str})
                        AND t.{from_col} IS NOT NULL
                    """)
                    
                    fk_values = set(row[0] for row in cur.fetchall())
                    
                    # Find which ones are missing from sampled_ids
                    existing_ids = sampled_ids.get(to_table, set())
                    missing_ids = fk_values - existing_ids
                    
                    if missing_ids:
                        # Add these missing IDs to the sample
                        if to_table not in sampled_ids:
                            sampled_ids[to_table] = set()
                        sampled_ids[to_table].update(missing_ids)
                        added_this_iteration += len(missing_ids)
                        added_count += len(missing_ids)
                        print(f"  {table}.{from_col} -> {to_table}: +{len(missing_ids)} referenced records")
                        
                except Exception as e:
                    print(f"  Warning: Could not check FK {table}.{from_col} -> {to_table}: {e}")
                    conn.rollback()
                    continue
        
        if added_this_iteration == 0:
            print(f"  Validation complete after {iteration + 1} iteration(s)")
            break
    
    if added_count > 0:
        print(f"\nTotal FK-referenced records added: {added_count}")
        total_records = sum(len(ids) for ids in sampled_ids.values())
        print(f"Clean sample size: {total_records} records")
    else:
        print(f"  No orphaned records found - data integrity is perfect!")
    
    # Bước 3: Tạo SQL script
    print(f"\nStep 3: Generating SQL script...")
    sql_lines = []
    sql_lines.append(f"-- Sample data from {db_name}")
    sql_lines.append(f"-- Total records to import: {total_records}")
    sql_lines.append(f"-- Generated at: {import_timestamp}")
    sql_lines.append("")
    
    # Tạo bảng theo thứ tự dependency
    table_order = []
    remaining = set(sampled_ids.keys())
    
    while remaining:
        added = False
        for table in list(remaining):
            # Check if all dependencies are satisfied
            deps = set(fk['to_table'] for fk in fks.get(table, []))
            if deps.issubset(set(table_order)):
                table_order.append(table)
                remaining.remove(table)
                added = True
        
        if not added and remaining:
            # Break circular dependency
            table_order.append(remaining.pop())
    
    print(f"\nTable processing order: {' -> '.join(table_order)}")
    
    for table in table_order:
        ids = sampled_ids[table]
        if not ids:
            continue
        
        print(f"\nProcessing {table}: {len(ids)} records")
        
        # Lấy primary key
        cur.execute(f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{table}'::regclass AND i.indisprimary
        """)
        pk_result = cur.fetchone()
        if pk_result:
            pk_col = pk_result[0]
        else:
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' LIMIT 1")
            pk_col = cur.fetchone()[0]
        
        # Lấy danh sách các cột
        cur.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns 
            WHERE table_name = '{table}'
            AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        columns = cur.fetchall()
        col_names = [col[0] for col in columns]
        
        sql_lines.append(f"-- Table: {table} ({len(ids)} records)")
        
        pk_col = pk_columns.get(table)
        
        # For tables without primary key, data is already stored as full rows
        if pk_col == 'ctid':
            # Lấy danh sách các cột
            cur.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            col_names = [col[0] for col in columns]
            
            # ids already contains full row tuples
            rows = list(ids)
        else:
            # For tables with primary key, query by PK
            # Lấy danh sách các cột
            cur.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns 
                WHERE table_name = '{table}'
                AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            col_names = [col[0] for col in columns]
            
            # Lấy dữ liệu thực tế từ database
            ids_str = format_ids_for_sql(list(ids))
            if not ids_str or ids_str == "''":
                print(f"  Skipping {table} - no valid IDs")
                continue
                
            query = f"""
                SELECT {', '.join(col_names)}
                FROM {table}
                WHERE {pk_col} IN ({ids_str})
            """
            
            try:
                cur.execute(query)
                rows = cur.fetchall()
            except Exception as e:
                print(f"  Error querying {table}: {e}")
                continue
        
        # Tạo INSERT statements
        if rows:
            sql_lines.append(f"INSERT INTO {table} ({', '.join(col_names)}) VALUES")
            for i, row in enumerate(rows):
                # Format values
                values = []
                for val in row:
                    if val is None:
                        values.append('NULL')
                    elif isinstance(val, str):
                        # Escape single quotes
                        escaped = val.replace("'", "''")
                        values.append(f"'{escaped}'")
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, bool):
                        values.append('TRUE' if val else 'FALSE')
                    else:
                        # For other types (date, timestamp, etc.), convert to string
                        escaped = str(val).replace("'", "''")
                        values.append(f"'{escaped}'")
                
                if i < len(rows) - 1:
                    sql_lines.append(f"  ({', '.join(values)}),")
                else:
                    sql_lines.append(f"  ({', '.join(values)});")
            
        sql_lines.append("")
    
    cur.close()
    
    print(f"\n{'='*50}")
    print(f"Sample Summary:")
    print(f"{'='*50}")
    for table in table_order:
        if table in sampled_ids:
            print(f"  {table:30} {len(sampled_ids[table]):>10,} records")
    print(f"{'='*50}")
    print(f"  {'TOTAL':30} {sum(len(ids) for ids in sampled_ids.values()):>10,} records")
    print(f"{'='*50}")
    
    return '\n'.join(sql_lines), sampled_ids

if __name__ == '__main__':
    import datetime
    import_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    db_name = sys.argv[1]
    target_db = sys.argv[2]
    sample_size = int(sys.argv[3])
    seed_records = int(sys.argv[4])
    output_file = sys.argv[5]
    pg_port = int(sys.argv[6]) if len(sys.argv) > 6 else 5432
    pg_password = sys.argv[7] if len(sys.argv) > 7 else ''
    
    print(f"Connecting to PostgreSQL on localhost:{pg_port}...")
    conn = psycopg2.connect(
        host='localhost',
        database=db_name,
        user='mlflow',
        password=pg_password,
        port=pg_port
    )
    print("Connected successfully!")
    print()
    
    sql_script, sampled_ids = create_sample_sql(conn, db_name, target_db, sample_size, seed_records)
    
    with open(output_file, 'w') as f:
        f.write(sql_script)
    
    # Save summary as JSON
    summary = {
        'total_records': sum(len(ids) for ids in sampled_ids.values()),
        'tables': {table: len(ids) for table, ids in sampled_ids.items()},
        'timestamp': import_timestamp
    }
    
    summary_file = output_file.replace('.sql', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSQL script saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    
    conn.close()
PYTHON_SCRIPT

chmod +x "$DATA_DIR/create_sample.py"

# Cài đặt psycopg2 trên host nếu chưa có
print_info "Checking psycopg2 on host machine..."
if ! python3 -c "import psycopg2" 2>/dev/null; then
    print_info "Installing psycopg2-binary on host..."
    if pip install psycopg2-binary >> "$LOG_FILE" 2>&1; then
        log "psycopg2-binary installed successfully"
    else
        log_error "Failed to install psycopg2-binary"
        exit 1
    fi
else
    log "psycopg2 is already installed"
fi

# Lấy port của PostgreSQL container
print_info "Getting PostgreSQL container port..."
PG_PORT=$(docker port $CONTAINER_NAME 5432 | cut -d: -f2)
if [ -z "$PG_PORT" ]; then
    log_error "Could not determine PostgreSQL port"
    exit 1
fi
log "PostgreSQL port: $PG_PORT"

# Chạy script Python trên host
print_info "Running intelligent sampling algorithm..."
log "Command: python3 $DATA_DIR/create_sample.py $TEMP_DB $DB_NAME $SAMPLE_SIZE $MAX_SEED_RECORDS $SAMPLE_SQL_FILE $PG_PORT $DB_PASSWORD"

if python3 "$DATA_DIR/create_sample.py" \
    "$TEMP_DB" "$DB_NAME" "$SAMPLE_SIZE" "$MAX_SEED_RECORDS" "$SAMPLE_SQL_FILE" "$PG_PORT" "$DB_PASSWORD" \
    2>&1 | tee -a "$LOG_FILE"; then
    log "Sampling completed successfully"
else
    log_error "Python sampling script failed with exit code $?"
    log_error "Check if psycopg2 is installed: pip install psycopg2-binary"
    exit 1
fi

# Kiểm tra file output
if [ ! -f "$SAMPLE_SQL_FILE" ]; then
    log_error "Sample SQL file was not created by Python script"
    exit 1
fi

if [ ! -s "$SAMPLE_SQL_FILE" ]; then
    log_error "Sample SQL file is empty"
    exit 1
fi

log "Sample SQL file created: $SAMPLE_SQL_FILE ($(wc -l < "$SAMPLE_SQL_FILE") lines)"

# Kiểm tra summary file
if [ -f "$DATA_DIR/sample_summary.json" ]; then
    log "Sample summary created: $DATA_DIR/sample_summary.json"
    cat "$DATA_DIR/sample_summary.json" | tee -a "$LOG_FILE"
else
    log "Warning: Sample summary file not created"
fi

print_header "Step 6: Import Sampled Data"
log_step "Importing sampled data into target database"

# Kiểm tra file SQL đã được tạo chưa
if [ ! -f "$SAMPLE_SQL_FILE" ]; then
    log_error "Sample SQL file was not created: $SAMPLE_SQL_FILE"
    log_error "Python sampling script may have failed. Check the log above."
    exit 1
fi

log "Sample SQL file found: $SAMPLE_SQL_FILE ($(wc -l < "$SAMPLE_SQL_FILE") lines)"

# Drop và recreate database để đảm bảo clean state
print_info "Recreating target database for clean import..."
docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;" >> "$LOG_FILE" 2>&1
if docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;" >> "$LOG_FILE" 2>&1; then
    log "Target database recreated: $DB_NAME"
else
    log_error "Failed to recreate target database"
    exit 1
fi

print_info "Creating schema in target database..."
# Tạo schema từ SQL file gốc nhưng loại bỏ foreign keys và indexes
# Sử dụng sed để tách phần CREATE TABLE
cat > "$DATA_DIR/schema_only.sql" << 'SQL_HEADER'
-- Schema creation script (tables only, no foreign keys)
-- Generated by import_relbench.sh
SQL_HEADER

# Extract only CREATE TABLE statements (multi-line) 
awk '/^CREATE TABLE/,/;$/' "$SQL_FILE" >> "$DATA_DIR/schema_only.sql"

docker cp "$DATA_DIR/schema_only.sql" $CONTAINER_NAME:/tmp/

if docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/schema_only.sql >> "$LOG_FILE" 2>&1; then
    log "Schema created successfully (without foreign keys)"
else
    log_error "Failed to create schema"
    exit 1
fi

print_info "Importing sampled data..."

# Kiểm tra file SQL có tồn tại không
if [ ! -f "$SAMPLE_SQL_FILE" ]; then
    log_error "Sample SQL file not found: $SAMPLE_SQL_FILE"
    exit 1
fi

# Kiểm tra file có rỗng không
if [ ! -s "$SAMPLE_SQL_FILE" ]; then
    log_error "Sample SQL file is empty: $SAMPLE_SQL_FILE"
    exit 1
fi

log "Sample SQL file size: $(wc -l < "$SAMPLE_SQL_FILE") lines"

# Copy SQL file vào container
if docker cp "$SAMPLE_SQL_FILE" $CONTAINER_NAME:/tmp/sample_${DATASET_SHORT}.sql >> "$LOG_FILE" 2>&1; then
    log "SQL file copied to container"
else
    log_error "Failed to copy SQL file to container"
    exit 1
fi

# Import với error handling chi tiết
print_info "Executing SQL import..."
if docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/sample_${DATASET_SHORT}.sql 2>&1 | tee -a "$LOG_FILE"; then
    log "SQL import completed successfully"
else
    log_error "SQL import failed. Check log file for details: $LOG_FILE"
    log_error "Last 20 lines of SQL file:"
    tail -20 "$SAMPLE_SQL_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# Thêm foreign keys sau khi import data
print_info "Adding foreign key constraints..."
# Extract ALTER TABLE ADD CONSTRAINT statements with their FOREIGN KEY lines (2 lines each)
grep -A 1 "^ALTER TABLE.*ADD CONSTRAINT" "$SQL_FILE" | grep -v "^--$" > "$DATA_DIR/foreign_keys.sql" || true
if [ -s "$DATA_DIR/foreign_keys.sql" ]; then
    docker cp "$DATA_DIR/foreign_keys.sql" $CONTAINER_NAME:/tmp/
    if docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/foreign_keys.sql >> "$LOG_FILE" 2>&1; then
        log "Foreign key constraints added successfully"
    else
        log "Warning: Some foreign key constraints failed (expected with partial data)"
    fi
else
    log "No foreign key constraints to add"
fi

# Xóa temporary database
print_info "Cleaning up temporary database..."
docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "DROP DATABASE IF EXISTS $TEMP_DB;" >> "$LOG_FILE" 2>&1
log "Temporary database dropped"

print_header "Step 7: Verify Import"
log_step "Verifying sampled data import"

print_info "Actual record counts in database:"
echo "" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "IMPORTED RECORDS BY TABLE" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"

# Query để lấy số lượng records trong mỗi bảng
TOTAL_COUNT=0
docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c "
SELECT 
    relname || ':' || n_live_tup as info
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
" | while IFS=: read -r table count; do
    if [ ! -z "$table" ] && [ ! -z "$count" ]; then
        # Trim whitespace
        table=$(echo "$table" | xargs)
        count=$(echo "$count" | xargs)
        printf "  %-30s %10s records\n" "$table" "$count" | tee -a "$LOG_FILE"
        TOTAL_COUNT=$((TOTAL_COUNT + count))
    fi
done

echo "=========================================" | tee -a "$LOG_FILE"

# Get total count from database
TOTAL_RECORDS=$(docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c "
SELECT SUM(n_live_tup) FROM pg_stat_user_tables WHERE schemaname = 'public';
" | xargs)

printf "  %-30s %10s records\n" "TOTAL" "$TOTAL_RECORDS" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

log "Import verification completed - Total: $TOTAL_RECORDS records imported"