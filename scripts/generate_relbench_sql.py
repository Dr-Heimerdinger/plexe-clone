#!/usr/bin/env python3
"""
Generate SQL DDL and export CSV from RelBench datasets
Support any RelBench dataset: rel-f1, rel-amazon, rel-hm, rel-stack, etc.
"""

import os
import sys
import csv
import argparse
import pandas as pd
from relbench.datasets import get_dataset
from pathlib import Path


def pandas_dtype_to_sql(dtype, col_name):
    """Convert pandas dtype to PostgreSQL type"""
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"


def get_primary_key_column(table):
    """Get primary key column from table"""
    if hasattr(table, 'pkey_col') and table.pkey_col:
        return table.pkey_col
    return None


def generate_create_table_sql(table_name, table):
    """Generate SQL CREATE TABLE statement for PostgreSQL"""
    sql = f"CREATE TABLE {table_name} (\n"
    
    columns = []
    for col_name in table.df.columns:
        dtype = table.df[col_name].dtype
        sql_type = pandas_dtype_to_sql(dtype, col_name)
        col_def = f"    {col_name} {sql_type}"
        columns.append(col_def)
    
    sql += ",\n".join(columns)
    
    pkey = get_primary_key_column(table)
    if pkey:
        sql += f",\n    PRIMARY KEY ({pkey})"
    
    sql += "\n);"
    return sql


def generate_temp_table_sql(table_name, table):
    """Generate temporary table with all columns as TEXT"""
    sql = f"CREATE TEMP TABLE temp_{table_name} (\n"
    columns = [f"    {col} TEXT" for col in table.df.columns]
    sql += ",\n".join(columns)
    sql += "\n);"
    return sql


def generate_insert_sql(table_name, table):
    """Generate INSERT statement with type conversion"""
    columns = []
    conversions = []
    
    for col_name in table.df.columns:
        dtype = table.df[col_name].dtype
        columns.append(col_name)
        
        if pd.api.types.is_integer_dtype(dtype):
            conversions.append(f"    NULLIF({col_name}, '')::INTEGER")
        elif pd.api.types.is_float_dtype(dtype):
            conversions.append(f"    NULLIF({col_name}, '')::FLOAT")
        elif pd.api.types.is_bool_dtype(dtype):
            conversions.append(f"    NULLIF({col_name}, '')::BOOLEAN")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            conversions.append(f"    NULLIF({col_name}, '')::TIMESTAMP")
        else:
            conversions.append(f"    NULLIF({col_name}, '')")
    
    sql = f"INSERT INTO {table_name}\nSELECT \n"
    sql += ",\n".join(conversions)
    sql += f"\nFROM temp_{table_name};"
    return sql


def generate_foreign_keys_sql(db):
    """Generate ALTER TABLE statements for foreign keys"""
    fk_statements = []
    
    for table_name, table in db.table_dict.items():
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fkey_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table in db.table_dict:
                    ref_pkey = get_primary_key_column(db.table_dict[ref_table])
                    if ref_pkey:
                        fk_name = f"fk_{table_name}_{fkey_col}"
                        stmt = f"ALTER TABLE {table_name} ADD CONSTRAINT {fk_name}\n"
                        stmt += f"    FOREIGN KEY ({fkey_col}) REFERENCES {ref_table}({ref_pkey});"
                        fk_statements.append(stmt)
    
    return fk_statements


def generate_indexes_sql(db):
    """Generate indexes for foreign keys and important columns"""
    index_statements = []
    
    for table_name, table in db.table_dict.items():
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fkey_col in table.fkey_col_to_pkey_table.keys():
                idx_name = f"idx_{table_name}_{fkey_col}"
                stmt = f"CREATE INDEX {idx_name} ON {table_name}({fkey_col}) WHERE {fkey_col} IS NOT NULL;"
                index_statements.append(stmt)
        
        if hasattr(table, 'time_col') and table.time_col:
            idx_name = f"idx_{table_name}_{table.time_col}"
            stmt = f"CREATE INDEX {idx_name} ON {table_name}({table.time_col});"
            index_statements.append(stmt)
    
    return index_statements


def generate_complete_sql(db, dataset_name):
    """Generate complete SQL import script"""
    sql_parts = []
    
    sql_parts.append(f"-- RelBench {dataset_name.upper()} Database Schema")
    sql_parts.append("-- Auto-generated from RelBench dataset")
    sql_parts.append(f"-- Total tables: {len(db.table_dict)}")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 1: Drop existing tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    for table_name in reversed(list(db.table_dict.keys())):
        sql_parts.append(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
    sql_parts.append("")
    sql_parts.append("\\echo 'Tables dropped'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 2: Create tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    for table_name, table in db.table_dict.items():
        sql_parts.append(generate_create_table_sql(table_name, table))
        sql_parts.append("")
    
    sql_parts.append("\\echo 'Tables created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 3: Create temp tables for import'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    for table_name, table in db.table_dict.items():
        sql_parts.append(generate_temp_table_sql(table_name, table))
        sql_parts.append("")
    
    sql_parts.append("\\echo 'Temp tables created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 4: Import CSV into temp tables'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    for table_name in db.table_dict.keys():
        sql_parts.append(f"\\echo '   Importing {table_name}...'")
        sql_parts.append(f"\\copy temp_{table_name} FROM '/tmp/{table_name}.csv' WITH (FORMAT CSV, HEADER, DELIMITER ',', QUOTE '\"');")
        sql_parts.append("")
    
    sql_parts.append("\\echo 'CSV imported to temp tables'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 5: Transfer data with type conversion'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    for table_name, table in db.table_dict.items():
        sql_parts.append(f"\\echo '   Processing {table_name}...'")
        sql_parts.append(generate_insert_sql(table_name, table))
        sql_parts.append("")
    
    sql_parts.append("\\echo 'Data transferred with NULL handling'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 6: Add Foreign Keys'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    fk_statements = generate_foreign_keys_sql(db)
    for stmt in fk_statements:
        sql_parts.append(stmt)
        sql_parts.append("")
    
    sql_parts.append("\\echo 'Foreign keys added'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Step 7: Create Indexes'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    
    index_statements = generate_indexes_sql(db)
    for stmt in index_statements:
        sql_parts.append(stmt)
    sql_parts.append("")
    
    sql_parts.append("\\echo 'Indexes created'")
    sql_parts.append("\\echo ''")
    sql_parts.append("")
    
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'Summary'")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("")
    sql_parts.append("SELECT ")
    sql_parts.append("    table_name,")
    sql_parts.append("    to_char(record_count, 'FM999,999,999') as records")
    sql_parts.append("FROM (")
    
    union_parts = []
    for table_name in db.table_dict.keys():
        union_parts.append(f"    SELECT '{table_name}' as table_name, COUNT(*) as record_count FROM {table_name}")
    
    sql_parts.append("\n    UNION ALL\n".join(union_parts))
    sql_parts.append(") t")
    sql_parts.append("ORDER BY record_count DESC;")
    sql_parts.append("")
    
    sql_parts.append("\\echo ''")
    sql_parts.append("\\echo '========================================='")
    sql_parts.append("\\echo 'IMPORT COMPLETE'")
    sql_parts.append("\\echo '========================================='")
    
    return "\n".join(sql_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SQL DDL and export CSV from RelBench datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s rel-f1
  %(prog)s rel-amazon --output-dir ./amazon_data
  %(prog)s rel-hm --no-download
  
Supported datasets:
  rel-f1, rel-amazon, rel-hm, rel-stack, rel-trial, 
  rel-event, rel-avito, rel-salt, rel-arxiv, rel-ratebeer
        '''
    )
    
    parser.add_argument('dataset', type=str, 
                       help='Dataset name (e.g., rel-f1, rel-amazon)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for CSV and SQL files (default: ./{dataset}_data)')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip download if dataset already exists in cache')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_short = dataset_name.replace('rel-', '')
        output_dir = Path(f"./{dataset_short}_data")
    
    print("=" * 60)
    print(f"RelBench {dataset_name.upper()} - SQL Generation")
    print("=" * 60)
    print()
    
    print(f"Downloading {dataset_name} dataset...")
    try:
        dataset = get_dataset(dataset_name, download=not args.no_download)
        db = dataset.get_db()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nMake sure '{dataset_name}' is a valid RelBench dataset.")
        sys.exit(1)
    
    print(f"Dataset loaded: {len(db.table_dict)} tables")
    print()
    
    output_dir.mkdir(exist_ok=True)
    
    print("Exporting CSV files...")
    for table_name, table in db.table_dict.items():
        csv_path = output_dir / f"{table_name}.csv"
        # Convert any list/array columns to strings and clean problematic characters
        df_export = table.df.copy()
        for col in df_export.columns:
            if df_export[col].dtype == 'object':
                # Convert all object types to clean strings
                def clean_value(x):
                    import numpy as np
                    # Handle None
                    if x is None:
                        return ''
                    # Handle numpy arrays and lists first (before pd.isna which fails on arrays)
                    if isinstance(x, (list, dict, tuple, np.ndarray)):
                        return str(x)
                    # Now safe to check isna for scalar values
                    try:
                        if pd.isna(x):
                            return ''
                    except (ValueError, TypeError):
                        pass
                    # Convert to string and remove problematic characters
                    s = str(x)
                    # Replace newlines and carriage returns with space
                    s = s.replace('\n', ' ').replace('\r', ' ')
                    return s
                df_export[col] = df_export[col].apply(clean_value)
        # Use QUOTE_MINIMAL with escapechar to handle special characters properly
        df_export.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, doublequote=True)
        print(f"   {table_name}.csv ({len(table.df):,} rows)")
    print()
    
    print("Generating SQL DDL...")
    sql_content = generate_complete_sql(db, dataset_name)
    
    dataset_short = dataset_name.replace('rel-', '')
    sql_file = output_dir / f"import_{dataset_short}.sql"
    sql_file.write_text(sql_content)
    
    print(f"SQL script generated: {sql_file}")
    print()
    
    print("Dataset Statistics:")
    print(f"   Tables: {len(db.table_dict)}")
    total_rows = 0
    for table_name, table in db.table_dict.items():
        pkey = get_primary_key_column(table)
        fkeys = len(table.fkey_col_to_pkey_table) if hasattr(table, 'fkey_col_to_pkey_table') else 0
        rows = len(table.df)
        total_rows += rows
        print(f"   - {table_name:30s} {rows:10,} rows, {len(table.df.columns):2} cols, PK: {pkey}, FKs: {fkeys}")
    print(f"\n   Total rows: {total_rows:,}")
    print()
    
    print("=" * 60)
    print("Generation complete")
    print("=" * 60)
    print()
    print("Output directory:", output_dir)
    print("SQL script:", sql_file)
    print()
    print("Next steps:")
    print(f"  ./import_relbench.sh {dataset_name}")


if __name__ == "__main__":
    main()