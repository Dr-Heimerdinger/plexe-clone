#!/bin/bash

# =====================================================
# Script import thuần SQL - không dùng Python
# =====================================================

set -e

CONTAINER_NAME="plexe-clone-postgres-1"
DB_NAME="stack"
DB_USER="mlflow"
CSV_SOURCE="/home/ta/kl/plexe-clone/data/database"

echo "========================================="
echo "RelBench Stack Import - Pure SQL"
echo "========================================="
echo ""

# Kiểm tra container
echo "🔍 Kiểm tra container..."
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "❌ Container không chạy!"
    exit 1
fi
echo "✓ Container đang chạy"
echo ""

# Kiểm tra và tạo database
echo "🔍 Kiểm tra database..."
if ! docker exec $CONTAINER_NAME psql -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "⚠️ Database $DB_NAME chưa tồn tại. Đang tạo..."
    docker exec $CONTAINER_NAME psql -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME;"
    echo "✓ Database $DB_NAME đã được tạo."
else
    echo "✓ Database $DB_NAME đã tồn tại."
fi
echo ""

# Copy CSV files vào container
echo "📁 Copy CSV files vào container..."
docker cp "$CSV_SOURCE/users.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/posts.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/votes.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/comments.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/badges.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/postLinks.csv" $CONTAINER_NAME:/tmp/
docker cp "$CSV_SOURCE/postHistory.csv" $CONTAINER_NAME:/tmp/
echo "✓ Files copied"
echo ""

# Tạo SQL script
echo "📝 Tạo SQL script..."
cat > /tmp/import_stack.sql << 'EOSQL'
\echo '========================================='
\echo 'Bước 1: Drop old tables'
\echo '========================================='

DROP TABLE IF EXISTS post_history CASCADE;
DROP TABLE IF EXISTS post_links CASCADE;
DROP TABLE IF EXISTS badges CASCADE;
DROP TABLE IF EXISTS comments CASCADE;
DROP TABLE IF EXISTS votes CASCADE;
DROP TABLE IF EXISTS posts CASCADE;
DROP TABLE IF EXISTS users CASCADE;

\echo '✓ Tables dropped'
\echo ''

\echo '========================================='
\echo 'Bước 2: Create tables (allow NULL)'
\echo '========================================='

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

\echo '✓ Tables created'
\echo ''

\echo '========================================='
\echo 'Bước 3: Create temp tables for import'
\echo '========================================='

-- Temp tables với tất cả columns là TEXT
CREATE TEMP TABLE temp_users (
    id TEXT,
    account_id TEXT,
    display_name TEXT,
    location TEXT,
    profile_image_url TEXT,
    website_url TEXT,
    about_me TEXT,
    creation_date TEXT
);

CREATE TEMP TABLE temp_posts (
    id TEXT,
    owner_user_id TEXT,
    post_type_id TEXT,
    accepted_answer_id TEXT,
    parent_id TEXT,
    owner_display_name TEXT,
    title TEXT,
    tags TEXT,
    content_license TEXT,
    body TEXT,
    creation_date TEXT
);

CREATE TEMP TABLE temp_votes (
    id TEXT,
    user_id TEXT,
    post_id TEXT,
    vote_type_id TEXT,
    creation_date TEXT
);

CREATE TEMP TABLE temp_comments (
    id TEXT,
    post_id TEXT,
    user_id TEXT,
    content_license TEXT,
    user_display_name TEXT,
    text TEXT,
    creation_date TEXT
);

CREATE TEMP TABLE temp_badges (
    id TEXT,
    user_id TEXT,
    class TEXT,
    name TEXT,
    tag_based TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_post_links (
    id TEXT,
    related_post_id TEXT,
    post_id TEXT,
    link_type_id TEXT,
    creation_date TEXT
);

CREATE TEMP TABLE temp_post_history (
    id TEXT,
    post_id TEXT,
    user_id TEXT,
    post_history_type_id TEXT,
    user_display_name TEXT,
    content_license TEXT,
    revision_guid TEXT,
    text TEXT,
    comment TEXT,
    creation_date TEXT
);

\echo '✓ Temp tables created'
\echo ''

\echo '========================================='
\echo 'Bước 4: Import CSV into temp tables'
\echo '========================================='

\echo '   Importing users...'
\copy temp_users FROM '/tmp/users.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing posts...'
\copy temp_posts FROM '/tmp/posts.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing votes...'
\copy temp_votes FROM '/tmp/votes.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing comments...'
\copy temp_comments FROM '/tmp/comments.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing badges...'
\copy temp_badges FROM '/tmp/badges.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing post_links...'
\copy temp_post_links FROM '/tmp/postLinks.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing post_history...'
\copy temp_post_history FROM '/tmp/postHistory.csv' WITH CSV HEADER DELIMITER ',';

\echo '✓ CSV imported to temp tables'
\echo ''

\echo '========================================='
\echo 'Bước 5: Transfer data with type conversion'
\echo '========================================='

\echo '   Processing users...'
INSERT INTO users
SELECT 
    id::INTEGER,
    NULLIF(account_id, '')::INTEGER,
    NULLIF(display_name, ''),
    NULLIF(location, ''),
    NULLIF(profile_image_url, ''),
    NULLIF(website_url, ''),
    NULLIF(about_me, ''),
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_users;

\echo '   Processing posts...'
INSERT INTO posts
SELECT 
    id::INTEGER,
    NULLIF(owner_user_id, '')::INTEGER,
    NULLIF(post_type_id, '')::INTEGER,
    NULLIF(accepted_answer_id, '')::INTEGER,
    NULLIF(parent_id, '')::INTEGER,
    NULLIF(owner_display_name, ''),
    NULLIF(title, ''),
    NULLIF(tags, ''),
    NULLIF(content_license, ''),
    NULLIF(body, ''),
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_posts;

\echo '   Processing votes...'
INSERT INTO votes
SELECT 
    id::INTEGER,
    NULLIF(user_id, '')::INTEGER,
    NULLIF(post_id, '')::INTEGER,
    NULLIF(vote_type_id, '')::INTEGER,
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_votes;

\echo '   Processing comments...'
INSERT INTO comments
SELECT 
    id::INTEGER,
    NULLIF(post_id, '')::INTEGER,
    NULLIF(user_id, '')::INTEGER,
    NULLIF(content_license, ''),
    NULLIF(user_display_name, ''),
    NULLIF(text, ''),
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_comments;

\echo '   Processing badges...'
INSERT INTO badges
SELECT 
    id::INTEGER,
    NULLIF(user_id, '')::INTEGER,
    NULLIF(class, '')::INTEGER,
    NULLIF(name, ''),
    NULLIF(tag_based, '')::BOOLEAN,
    NULLIF(date, '')::TIMESTAMP
FROM temp_badges;

\echo '   Processing post_links...'
INSERT INTO post_links
SELECT 
    id::INTEGER,
    NULLIF(related_post_id, '')::INTEGER,
    NULLIF(post_id, '')::INTEGER,
    NULLIF(link_type_id, '')::INTEGER,
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_post_links;

\echo '   Processing post_history...'
INSERT INTO post_history
SELECT 
    id::INTEGER,
    NULLIF(post_id, '')::INTEGER,
    NULLIF(user_id, '')::INTEGER,
    NULLIF(post_history_type_id, '')::INTEGER,
    NULLIF(user_display_name, ''),
    NULLIF(content_license, ''),
    NULLIF(revision_guid, ''),
    NULLIF(text, ''),
    NULLIF(comment, ''),
    NULLIF(creation_date, '')::TIMESTAMP
FROM temp_post_history;

\echo '✓ Data transferred with NULL handling'
\echo ''

\echo '========================================='
\echo 'Bước 6: Add Foreign Keys'
\echo '========================================='

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

\echo '✓ Foreign keys added'
\echo ''

\echo '========================================='
\echo 'Bước 7: Create Indexes'
\echo '========================================='

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

\echo '✓ Indexes created'
\echo ''

\echo '========================================='
\echo 'Summary'
\echo '========================================='

SELECT 
    table_name,
    to_char(record_count, 'FM999,999,999') as records
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
\echo '✅ IMPORT COMPLETE!'
\echo '========================================='
EOSQL

# Copy SQL vào container
docker cp /tmp/import_stack.sql $CONTAINER_NAME:/tmp/

# Chạy import
echo "🚀 Đang import vào database..."
echo "========================================="
docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -f /tmp/import_stack.sql

# Cleanup
rm /tmp/import_stack.sql

echo ""
echo "========================================="
echo "✅ DONE!"
echo "========================================="
echo ""
echo "Kiểm tra:"
echo "  docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME"
echo ""
echo "Queries:"
echo "  SELECT COUNT(*) FROM users;"
echo "  SELECT COUNT(*) FROM users WHERE account_id IS NULL;"
echo "========================================="