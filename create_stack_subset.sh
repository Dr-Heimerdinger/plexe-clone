#!/bin/bash

# =====================================================
# Tạo subset database stack-200 (~200 records mỗi bảng)
# Giữ nguyên relationships và lấy dữ liệu liên tục 30 ngày
# ĐẢM BẢO REFERENTIAL INTEGRITY
# =====================================================

set -e

CONTAINER_NAME="plexe-clone-postgres-1"
SOURCE_DB="stack"
TARGET_DB="stack-200"
DB_USER="mlflow"

# Tham số - mỗi bảng ~200 records
RECORDS_PER_TABLE=200     # Số records mục tiêu cho mỗi bảng
TIME_WINDOW_DAYS=30       # Lấy dữ liệu trong 30 ngày liên tục

echo "========================================="
echo "Create Stack-200 Subset Database"
echo "========================================="
echo "Source DB: $SOURCE_DB"
echo "Target DB: $TARGET_DB"
echo "Records per table: ~$RECORDS_PER_TABLE"
echo "Time Window: $TIME_WINDOW_DAYS days"
echo "========================================="
echo ""

# Kiểm tra container
echo "🔍 Checking container..."
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container not running!"
    exit 1
fi
echo "✓ Container is running"
echo ""

# Step 1: Tìm time window với nhiều activity
echo "📝 Step 1: Finding optimal time window..."
cat > /tmp/find_window.sql << 'EOSQL'
-- Find 30-day window with good activity
WITH daily_counts AS (
    SELECT 
        DATE_TRUNC('day', creation_date) as day,
        COUNT(*) as daily_count
    FROM posts
    WHERE creation_date IS NOT NULL
    GROUP BY DATE_TRUNC('day', creation_date)
),
window_counts AS (
    SELECT 
        day as start_day,
        SUM(daily_count) OVER (ORDER BY day ROWS BETWEEN CURRENT ROW AND 29 FOLLOWING) as window_count
    FROM daily_counts
)
SELECT 
    to_char(start_day, 'YYYY-MM-DD') as start_date,
    to_char(start_day + INTERVAL '30 days', 'YYYY-MM-DD') as end_date,
    window_count
FROM window_counts
WHERE window_count >= 200
ORDER BY window_count ASC
LIMIT 1;
EOSQL

docker cp /tmp/find_window.sql $CONTAINER_NAME:/tmp/
TIME_WINDOW=$(docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $SOURCE_DB -t -A -F'|' -f /tmp/find_window.sql | head -1)

# Nếu không tìm được window phù hợp, lấy window có nhiều activity nhất
if [ -z "$TIME_WINDOW" ]; then
    echo "  Finding window with most activity..."
    cat > /tmp/find_window2.sql << 'EOSQL'
WITH daily_counts AS (
    SELECT 
        DATE_TRUNC('day', creation_date) as day,
        COUNT(*) as daily_count
    FROM posts
    WHERE creation_date IS NOT NULL
    GROUP BY DATE_TRUNC('day', creation_date)
),
window_counts AS (
    SELECT 
        day as start_day,
        SUM(daily_count) OVER (ORDER BY day ROWS BETWEEN CURRENT ROW AND 29 FOLLOWING) as window_count
    FROM daily_counts
)
SELECT 
    to_char(start_day, 'YYYY-MM-DD') as start_date,
    to_char(start_day + INTERVAL '30 days', 'YYYY-MM-DD') as end_date,
    window_count
FROM window_counts
WHERE window_count IS NOT NULL
ORDER BY window_count DESC
LIMIT 1;
EOSQL
    docker cp /tmp/find_window2.sql $CONTAINER_NAME:/tmp/
    TIME_WINDOW=$(docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $SOURCE_DB -t -A -F'|' -f /tmp/find_window2.sql | head -1)
fi

START_DATE=$(echo $TIME_WINDOW | cut -d'|' -f1)
END_DATE=$(echo $TIME_WINDOW | cut -d'|' -f2)
ACTIVITY=$(echo $TIME_WINDOW | cut -d'|' -f3)

echo "✓ Selected time window:"
echo "  Start: $START_DATE"
echo "  End: $END_DATE"
echo "  Posts in window: $ACTIVITY"
echo ""

# Step 2: Extract subset với đảm bảo referential integrity, mỗi bảng ~200 records
echo "📝 Step 2: Extracting subset data (~$RECORDS_PER_TABLE per table)..."
cat > /tmp/extract_subset.sql << EOSQL
-- =====================================================
-- STRATEGY: ~200 records per table with referential integrity
-- =====================================================

-- Step 2.1: Get 200 posts in time window
CREATE TEMP TABLE base_posts AS
SELECT id, owner_user_id, accepted_answer_id, parent_id
FROM posts
WHERE creation_date >= '$START_DATE'::timestamp
  AND creation_date < '$END_DATE'::timestamp
ORDER BY creation_date
LIMIT $RECORDS_PER_TABLE;

-- Step 2.2: Iteratively add parent posts and accepted answers
CREATE TEMP TABLE all_post_ids (id INTEGER PRIMARY KEY);

-- Insert base posts
INSERT INTO all_post_ids SELECT id FROM base_posts;

-- Add parent posts and accepted answers (max 3 iterations)
DO \$\$
DECLARE
    new_count INTEGER;
BEGIN
    FOR i IN 1..3 LOOP
        INSERT INTO all_post_ids (id)
        SELECT DISTINCT p.parent_id
        FROM posts p
        WHERE p.id IN (SELECT id FROM all_post_ids)
          AND p.parent_id IS NOT NULL
          AND p.parent_id NOT IN (SELECT id FROM all_post_ids)
        ON CONFLICT DO NOTHING;
        
        INSERT INTO all_post_ids (id)
        SELECT DISTINCT p.accepted_answer_id
        FROM posts p
        WHERE p.id IN (SELECT id FROM all_post_ids)
          AND p.accepted_answer_id IS NOT NULL
          AND p.accepted_answer_id NOT IN (SELECT id FROM all_post_ids)
        ON CONFLICT DO NOTHING;
        
        GET DIAGNOSTICS new_count = ROW_COUNT;
        IF new_count = 0 THEN
            EXIT;
        END IF;
    END LOOP;
END \$\$;

-- Step 2.3: Get 200 users (prioritize those with posts, then add more)
CREATE TEMP TABLE all_user_ids AS
SELECT DISTINCT owner_user_id as id
FROM posts
WHERE id IN (SELECT id FROM all_post_ids)
  AND owner_user_id IS NOT NULL
LIMIT $RECORDS_PER_TABLE;

-- If we have fewer than 200 users from posts, add more from the time window
INSERT INTO all_user_ids
SELECT DISTINCT u.id
FROM users u
WHERE u.creation_date >= '$START_DATE'::timestamp
  AND u.creation_date < '$END_DATE'::timestamp
  AND u.id NOT IN (SELECT id FROM all_user_ids)
LIMIT ($RECORDS_PER_TABLE - (SELECT COUNT(*) FROM all_user_ids));

-- Show what we collected
\echo 'Collected IDs:'
SELECT 'posts' as type, COUNT(*) as count FROM all_post_ids
UNION ALL
SELECT 'users', COUNT(*) FROM all_user_ids;

-- Step 2.4: Export users (limit 200)
\copy (SELECT * FROM users WHERE id IN (SELECT id FROM all_user_ids) LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_users.csv' CSV HEADER;

-- Step 2.5: Export posts (already limited to ~200) - nullify missing FKs
\copy (SELECT id, CASE WHEN owner_user_id IN (SELECT id FROM all_user_ids) THEN owner_user_id ELSE NULL END as owner_user_id, post_type_id, CASE WHEN accepted_answer_id IN (SELECT id FROM all_post_ids) THEN accepted_answer_id ELSE NULL END as accepted_answer_id, CASE WHEN parent_id IN (SELECT id FROM all_post_ids) THEN parent_id ELSE NULL END as parent_id, owner_display_name, title, tags, content_license, body, creation_date FROM posts WHERE id IN (SELECT id FROM all_post_ids)) TO '/tmp/subset_posts.csv' CSV HEADER;

-- Step 2.6: Export votes (limit 200) - only valid FKs
\copy (SELECT id, CASE WHEN user_id IN (SELECT id FROM all_user_ids) THEN user_id ELSE NULL END as user_id, post_id, vote_type_id, creation_date FROM votes WHERE post_id IN (SELECT id FROM all_post_ids) ORDER BY creation_date LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_votes.csv' CSV HEADER;

-- Step 2.7: Export comments (limit 200) - only valid FKs
\copy (SELECT id, post_id, CASE WHEN user_id IN (SELECT id FROM all_user_ids) THEN user_id ELSE NULL END as user_id, content_license, user_display_name, text, creation_date FROM comments WHERE post_id IN (SELECT id FROM all_post_ids) ORDER BY creation_date LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_comments.csv' CSV HEADER;

-- Step 2.8: Export badges (limit 200) - only for users in subset
\copy (SELECT * FROM badges WHERE user_id IN (SELECT id FROM all_user_ids) ORDER BY date LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_badges.csv' CSV HEADER;

-- Step 2.9: Export post_links (limit 200) - only where BOTH posts exist
\copy (SELECT * FROM post_links WHERE post_id IN (SELECT id FROM all_post_ids) AND related_post_id IN (SELECT id FROM all_post_ids) ORDER BY creation_date LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_post_links.csv' CSV HEADER;

-- Step 2.10: Export post_history (limit 200) - only valid FKs
\copy (SELECT id, post_id, CASE WHEN user_id IN (SELECT id FROM all_user_ids) THEN user_id ELSE NULL END as user_id, post_history_type_id, user_display_name, content_license, revision_guid, text, comment, creation_date FROM post_history WHERE post_id IN (SELECT id FROM all_post_ids) ORDER BY creation_date LIMIT $RECORDS_PER_TABLE) TO '/tmp/subset_post_history.csv' CSV HEADER;

-- Show counts
\echo ''
\echo 'Exported record counts (target: $RECORDS_PER_TABLE each):'
\echo '(Note: Actual CSV exports are limited to $RECORDS_PER_TABLE)'
SELECT 'users' as table_name, LEAST(COUNT(*), $RECORDS_PER_TABLE) as count FROM users WHERE id IN (SELECT id FROM all_user_ids)
UNION ALL SELECT 'posts', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM all_post_ids
UNION ALL SELECT 'votes', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM votes WHERE post_id IN (SELECT id FROM all_post_ids)
UNION ALL SELECT 'comments', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM comments WHERE post_id IN (SELECT id FROM all_post_ids)
UNION ALL SELECT 'badges', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM badges WHERE user_id IN (SELECT id FROM all_user_ids)
UNION ALL SELECT 'post_links', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM post_links WHERE post_id IN (SELECT id FROM all_post_ids) AND related_post_id IN (SELECT id FROM all_post_ids)
UNION ALL SELECT 'post_history', LEAST(COUNT(*), $RECORDS_PER_TABLE) FROM post_history WHERE post_id IN (SELECT id FROM all_post_ids)
ORDER BY count DESC;
EOSQL

docker cp /tmp/extract_subset.sql $CONTAINER_NAME:/tmp/
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $SOURCE_DB -f /tmp/extract_subset.sql

echo "✓ Data extracted to CSV files"
echo ""

# Step 3: Create new database
echo "📝 Step 3: Creating new database..."
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d postgres << EOSQL
DROP DATABASE IF EXISTS "stack-200";
CREATE DATABASE "stack-200";
EOSQL

echo "✓ Database created"
echo ""

# Step 4: Create schema in new database
echo "📝 Step 4: Creating schema..."
cat > /tmp/create_schema.sql << 'EOSQL'
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    account_id INTEGER,
    display_name VARCHAR(255),
    location VARCHAR(500),
    profile_image_url TEXT,
    website_url TEXT,
    about_me TEXT,
    creation_date TIMESTAMP
);

CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    owner_user_id INTEGER,
    post_type_id INTEGER,
    accepted_answer_id INTEGER,
    parent_id INTEGER,
    owner_display_name VARCHAR(255),
    title TEXT,
    tags VARCHAR(500),
    content_license VARCHAR(100),
    body TEXT,
    creation_date TIMESTAMP
);

CREATE TABLE votes (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    post_id INTEGER,
    vote_type_id INTEGER,
    creation_date TIMESTAMP
);

CREATE TABLE comments (
    id INTEGER PRIMARY KEY,
    post_id INTEGER,
    user_id INTEGER,
    content_license VARCHAR(100),
    user_display_name VARCHAR(255),
    text TEXT,
    creation_date TIMESTAMP
);

CREATE TABLE badges (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    class INTEGER,
    name VARCHAR(255),
    tag_based BOOLEAN,
    date TIMESTAMP
);

CREATE TABLE post_links (
    id INTEGER PRIMARY KEY,
    related_post_id INTEGER,
    post_id INTEGER,
    link_type_id INTEGER,
    creation_date TIMESTAMP
);

CREATE TABLE post_history (
    id INTEGER PRIMARY KEY,
    post_id INTEGER,
    user_id INTEGER,
    post_history_type_id INTEGER,
    user_display_name VARCHAR(255),
    content_license VARCHAR(100),
    revision_guid VARCHAR(50),
    text TEXT,
    comment TEXT,
    creation_date TIMESTAMP
);
EOSQL

docker cp /tmp/create_schema.sql $CONTAINER_NAME:/tmp/
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $TARGET_DB -f /tmp/create_schema.sql

echo "✓ Schema created"
echo ""

# Step 5: Import data
echo "📝 Step 5: Importing data..."
cat > /tmp/import_subset.sql << 'EOSQL'
\copy users FROM '/tmp/subset_users.csv' CSV HEADER;
\copy posts FROM '/tmp/subset_posts.csv' CSV HEADER;
\copy votes FROM '/tmp/subset_votes.csv' CSV HEADER;
\copy comments FROM '/tmp/subset_comments.csv' CSV HEADER;
\copy badges FROM '/tmp/subset_badges.csv' CSV HEADER;
\copy post_links FROM '/tmp/subset_post_links.csv' CSV HEADER;
\copy post_history FROM '/tmp/subset_post_history.csv' CSV HEADER;
EOSQL

docker cp /tmp/import_subset.sql $CONTAINER_NAME:/tmp/
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $TARGET_DB -f /tmp/import_subset.sql

echo "✓ Data imported"
echo ""

# Step 6: Add constraints and indexes
echo "📝 Step 6: Adding foreign keys and indexes..."
cat > /tmp/add_constraints.sql << 'EOSQL'
-- Foreign keys
ALTER TABLE posts ADD CONSTRAINT fk_posts_owner_user 
    FOREIGN KEY (owner_user_id) REFERENCES users(id);
ALTER TABLE posts ADD CONSTRAINT fk_posts_accepted_answer 
    FOREIGN KEY (accepted_answer_id) REFERENCES posts(id);
ALTER TABLE posts ADD CONSTRAINT fk_posts_parent 
    FOREIGN KEY (parent_id) REFERENCES posts(id);

ALTER TABLE votes ADD CONSTRAINT fk_votes_user 
    FOREIGN KEY (user_id) REFERENCES users(id);
ALTER TABLE votes ADD CONSTRAINT fk_votes_post 
    FOREIGN KEY (post_id) REFERENCES posts(id);

ALTER TABLE comments ADD CONSTRAINT fk_comments_post 
    FOREIGN KEY (post_id) REFERENCES posts(id);
ALTER TABLE comments ADD CONSTRAINT fk_comments_user 
    FOREIGN KEY (user_id) REFERENCES users(id);

ALTER TABLE badges ADD CONSTRAINT fk_badges_user 
    FOREIGN KEY (user_id) REFERENCES users(id);

ALTER TABLE post_links ADD CONSTRAINT fk_post_links_related_post 
    FOREIGN KEY (related_post_id) REFERENCES posts(id);
ALTER TABLE post_links ADD CONSTRAINT fk_post_links_post 
    FOREIGN KEY (post_id) REFERENCES posts(id);

ALTER TABLE post_history ADD CONSTRAINT fk_post_history_post 
    FOREIGN KEY (post_id) REFERENCES posts(id);
ALTER TABLE post_history ADD CONSTRAINT fk_post_history_user 
    FOREIGN KEY (user_id) REFERENCES users(id);

-- Indexes
CREATE INDEX idx_posts_owner_user ON posts(owner_user_id) WHERE owner_user_id IS NOT NULL;
CREATE INDEX idx_posts_parent ON posts(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_posts_type ON posts(post_type_id);
CREATE INDEX idx_votes_user ON votes(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_votes_post ON votes(post_id);
CREATE INDEX idx_comments_post ON comments(post_id);
CREATE INDEX idx_comments_user ON comments(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_badges_user ON badges(user_id);
CREATE INDEX idx_post_links_related ON post_links(related_post_id);
CREATE INDEX idx_post_links_post ON post_links(post_id);
CREATE INDEX idx_post_history_post ON post_history(post_id);
CREATE INDEX idx_post_history_user ON post_history(user_id) WHERE user_id IS NOT NULL;
EOSQL

docker cp /tmp/add_constraints.sql $CONTAINER_NAME:/tmp/
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $TARGET_DB -f /tmp/add_constraints.sql

echo "✓ Constraints and indexes added"
echo ""

# Step 7: Show summary
echo "📊 Step 7: Summary..."
cat > /tmp/show_summary.sql << 'EOSQL'
\echo '========================================='
\echo 'Record Counts'
\echo '========================================='

SELECT 
    table_name,
    to_char(record_count, 'FM999,999') as records
FROM (
    SELECT 'users' as table_name, COUNT(*) as record_count FROM users
    UNION ALL SELECT 'posts', COUNT(*) FROM posts
    UNION ALL SELECT 'votes', COUNT(*) FROM votes
    UNION ALL SELECT 'comments', COUNT(*) FROM comments
    UNION ALL SELECT 'badges', COUNT(*) FROM badges
    UNION ALL SELECT 'post_links', COUNT(*) FROM post_links
    UNION ALL SELECT 'post_history', COUNT(*) FROM post_history
) t
ORDER BY record_count DESC;

\echo ''
\echo '========================================='
\echo 'Time Range'
\echo '========================================='

SELECT 
    to_char(MIN(creation_date), 'YYYY-MM-DD HH24:MI') as earliest,
    to_char(MAX(creation_date), 'YYYY-MM-DD HH24:MI') as latest,
    EXTRACT(DAY FROM (MAX(creation_date) - MIN(creation_date))) || ' days' as range
FROM posts
WHERE creation_date IS NOT NULL;

\echo ''
\echo '========================================='
\echo 'Total Records'
\echo '========================================='

SELECT 
    to_char(
        (SELECT COUNT(*) FROM users) +
        (SELECT COUNT(*) FROM posts) +
        (SELECT COUNT(*) FROM votes) +
        (SELECT COUNT(*) FROM comments) +
        (SELECT COUNT(*) FROM badges) +
        (SELECT COUNT(*) FROM post_links) +
        (SELECT COUNT(*) FROM post_history),
        'FM999,999'
    ) as total_records;
EOSQL

docker cp /tmp/show_summary.sql $CONTAINER_NAME:/tmp/
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $TARGET_DB -f /tmp/show_summary.sql

# Cleanup temp files
rm -f /tmp/find_window.sql
rm -f /tmp/find_window2.sql
rm -f /tmp/extract_subset.sql
rm -f /tmp/create_schema.sql
rm -f /tmp/import_subset.sql
rm -f /tmp/add_constraints.sql
rm -f /tmp/show_summary.sql

echo ""
echo "========================================="
echo "✅ DONE!"
echo "========================================="
echo ""
echo "Connect to subset database:"
echo "  docker exec -it $CONTAINER_NAME psql -U $DB_USER -d stack-200"
echo ""
echo "Verify data:"
echo "  SELECT COUNT(*) FROM users;"
echo "  SELECT MIN(creation_date), MAX(creation_date) FROM posts;"
echo "========================================="