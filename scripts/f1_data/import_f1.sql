-- RelBench REL-F1 Database Schema
-- Auto-generated from RelBench dataset
-- Total tables: 9

\echo '========================================='
\echo 'Step 1: Drop existing tables'
\echo '========================================='

DROP TABLE IF EXISTS drivers CASCADE;
DROP TABLE IF EXISTS races CASCADE;
DROP TABLE IF EXISTS circuits CASCADE;
DROP TABLE IF EXISTS constructor_standings CASCADE;
DROP TABLE IF EXISTS constructors CASCADE;
DROP TABLE IF EXISTS qualifying CASCADE;
DROP TABLE IF EXISTS standings CASCADE;
DROP TABLE IF EXISTS results CASCADE;
DROP TABLE IF EXISTS constructor_results CASCADE;

\echo 'Tables dropped'
\echo ''

\echo '========================================='
\echo 'Step 2: Create tables'
\echo '========================================='

CREATE TABLE constructor_results (
    constructorResultsId INTEGER,
    raceId INTEGER,
    constructorId INTEGER,
    points FLOAT,
    date TIMESTAMP,
    PRIMARY KEY (constructorResultsId)
);

CREATE TABLE results (
    resultId INTEGER,
    raceId INTEGER,
    driverId INTEGER,
    constructorId INTEGER,
    number FLOAT,
    grid INTEGER,
    position FLOAT,
    positionOrder INTEGER,
    points FLOAT,
    laps INTEGER,
    milliseconds FLOAT,
    fastestLap FLOAT,
    rank FLOAT,
    statusId INTEGER,
    date TIMESTAMP,
    PRIMARY KEY (resultId)
);

CREATE TABLE standings (
    driverStandingsId INTEGER,
    raceId INTEGER,
    driverId INTEGER,
    points FLOAT,
    position INTEGER,
    wins INTEGER,
    date TIMESTAMP,
    PRIMARY KEY (driverStandingsId)
);

CREATE TABLE qualifying (
    qualifyId INTEGER,
    raceId INTEGER,
    driverId INTEGER,
    constructorId INTEGER,
    number INTEGER,
    position INTEGER,
    date TIMESTAMP,
    PRIMARY KEY (qualifyId)
);

CREATE TABLE constructors (
    constructorId INTEGER,
    constructorRef TEXT,
    name TEXT,
    nationality TEXT,
    PRIMARY KEY (constructorId)
);

CREATE TABLE constructor_standings (
    constructorStandingsId INTEGER,
    raceId INTEGER,
    constructorId INTEGER,
    points FLOAT,
    position INTEGER,
    wins INTEGER,
    date TIMESTAMP,
    PRIMARY KEY (constructorStandingsId)
);

CREATE TABLE circuits (
    circuitId INTEGER,
    circuitRef TEXT,
    name TEXT,
    location TEXT,
    country TEXT,
    lat FLOAT,
    lng FLOAT,
    alt FLOAT,
    PRIMARY KEY (circuitId)
);

CREATE TABLE races (
    raceId INTEGER,
    year INTEGER,
    round INTEGER,
    circuitId INTEGER,
    name TEXT,
    date TIMESTAMP,
    time TEXT,
    PRIMARY KEY (raceId)
);

CREATE TABLE drivers (
    driverId INTEGER,
    driverRef TEXT,
    code TEXT,
    forename TEXT,
    surname TEXT,
    dob TIMESTAMP,
    nationality TEXT,
    PRIMARY KEY (driverId)
);

\echo 'Tables created'
\echo ''

\echo '========================================='
\echo 'Step 3: Create temp tables for import'
\echo '========================================='

CREATE TEMP TABLE temp_constructor_results (
    constructorResultsId TEXT,
    raceId TEXT,
    constructorId TEXT,
    points TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_results (
    resultId TEXT,
    raceId TEXT,
    driverId TEXT,
    constructorId TEXT,
    number TEXT,
    grid TEXT,
    position TEXT,
    positionOrder TEXT,
    points TEXT,
    laps TEXT,
    milliseconds TEXT,
    fastestLap TEXT,
    rank TEXT,
    statusId TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_standings (
    driverStandingsId TEXT,
    raceId TEXT,
    driverId TEXT,
    points TEXT,
    position TEXT,
    wins TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_qualifying (
    qualifyId TEXT,
    raceId TEXT,
    driverId TEXT,
    constructorId TEXT,
    number TEXT,
    position TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_constructors (
    constructorId TEXT,
    constructorRef TEXT,
    name TEXT,
    nationality TEXT
);

CREATE TEMP TABLE temp_constructor_standings (
    constructorStandingsId TEXT,
    raceId TEXT,
    constructorId TEXT,
    points TEXT,
    position TEXT,
    wins TEXT,
    date TEXT
);

CREATE TEMP TABLE temp_circuits (
    circuitId TEXT,
    circuitRef TEXT,
    name TEXT,
    location TEXT,
    country TEXT,
    lat TEXT,
    lng TEXT,
    alt TEXT
);

CREATE TEMP TABLE temp_races (
    raceId TEXT,
    year TEXT,
    round TEXT,
    circuitId TEXT,
    name TEXT,
    date TEXT,
    time TEXT
);

CREATE TEMP TABLE temp_drivers (
    driverId TEXT,
    driverRef TEXT,
    code TEXT,
    forename TEXT,
    surname TEXT,
    dob TEXT,
    nationality TEXT
);

\echo 'Temp tables created'
\echo ''

\echo '========================================='
\echo 'Step 4: Import CSV into temp tables'
\echo '========================================='

\echo '   Importing constructor_results...'
\copy temp_constructor_results FROM '/tmp/constructor_results.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing results...'
\copy temp_results FROM '/tmp/results.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing standings...'
\copy temp_standings FROM '/tmp/standings.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing qualifying...'
\copy temp_qualifying FROM '/tmp/qualifying.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing constructors...'
\copy temp_constructors FROM '/tmp/constructors.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing constructor_standings...'
\copy temp_constructor_standings FROM '/tmp/constructor_standings.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing circuits...'
\copy temp_circuits FROM '/tmp/circuits.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing races...'
\copy temp_races FROM '/tmp/races.csv' WITH CSV HEADER DELIMITER ',';

\echo '   Importing drivers...'
\copy temp_drivers FROM '/tmp/drivers.csv' WITH CSV HEADER DELIMITER ',';

\echo 'CSV imported to temp tables'
\echo ''

\echo '========================================='
\echo 'Step 5: Transfer data with type conversion'
\echo '========================================='

\echo '   Processing constructor_results...'
INSERT INTO constructor_results
SELECT 
    NULLIF(constructorResultsId, '')::INTEGER,
    NULLIF(raceId, '')::INTEGER,
    NULLIF(constructorId, '')::INTEGER,
    NULLIF(points, '')::FLOAT,
    NULLIF(date, '')::TIMESTAMP
FROM temp_constructor_results;

\echo '   Processing results...'
INSERT INTO results
SELECT 
    NULLIF(resultId, '')::INTEGER,
    NULLIF(raceId, '')::INTEGER,
    NULLIF(driverId, '')::INTEGER,
    NULLIF(constructorId, '')::INTEGER,
    NULLIF(number, '')::FLOAT,
    NULLIF(grid, '')::INTEGER,
    NULLIF(position, '')::FLOAT,
    NULLIF(positionOrder, '')::INTEGER,
    NULLIF(points, '')::FLOAT,
    NULLIF(laps, '')::INTEGER,
    NULLIF(milliseconds, '')::FLOAT,
    NULLIF(fastestLap, '')::FLOAT,
    NULLIF(rank, '')::FLOAT,
    NULLIF(statusId, '')::INTEGER,
    NULLIF(date, '')::TIMESTAMP
FROM temp_results;

\echo '   Processing standings...'
INSERT INTO standings
SELECT 
    NULLIF(driverStandingsId, '')::INTEGER,
    NULLIF(raceId, '')::INTEGER,
    NULLIF(driverId, '')::INTEGER,
    NULLIF(points, '')::FLOAT,
    NULLIF(position, '')::INTEGER,
    NULLIF(wins, '')::INTEGER,
    NULLIF(date, '')::TIMESTAMP
FROM temp_standings;

\echo '   Processing qualifying...'
INSERT INTO qualifying
SELECT 
    NULLIF(qualifyId, '')::INTEGER,
    NULLIF(raceId, '')::INTEGER,
    NULLIF(driverId, '')::INTEGER,
    NULLIF(constructorId, '')::INTEGER,
    NULLIF(number, '')::INTEGER,
    NULLIF(position, '')::INTEGER,
    NULLIF(date, '')::TIMESTAMP
FROM temp_qualifying;

\echo '   Processing constructors...'
INSERT INTO constructors
SELECT 
    NULLIF(constructorId, '')::INTEGER,
    NULLIF(constructorRef, ''),
    NULLIF(name, ''),
    NULLIF(nationality, '')
FROM temp_constructors;

\echo '   Processing constructor_standings...'
INSERT INTO constructor_standings
SELECT 
    NULLIF(constructorStandingsId, '')::INTEGER,
    NULLIF(raceId, '')::INTEGER,
    NULLIF(constructorId, '')::INTEGER,
    NULLIF(points, '')::FLOAT,
    NULLIF(position, '')::INTEGER,
    NULLIF(wins, '')::INTEGER,
    NULLIF(date, '')::TIMESTAMP
FROM temp_constructor_standings;

\echo '   Processing circuits...'
INSERT INTO circuits
SELECT 
    NULLIF(circuitId, '')::INTEGER,
    NULLIF(circuitRef, ''),
    NULLIF(name, ''),
    NULLIF(location, ''),
    NULLIF(country, ''),
    NULLIF(lat, '')::FLOAT,
    NULLIF(lng, '')::FLOAT,
    NULLIF(alt, '')::FLOAT
FROM temp_circuits;

\echo '   Processing races...'
INSERT INTO races
SELECT 
    NULLIF(raceId, '')::INTEGER,
    NULLIF(year, '')::INTEGER,
    NULLIF(round, '')::INTEGER,
    NULLIF(circuitId, '')::INTEGER,
    NULLIF(name, ''),
    NULLIF(date, '')::TIMESTAMP,
    NULLIF(time, '')
FROM temp_races;

\echo '   Processing drivers...'
INSERT INTO drivers
SELECT 
    NULLIF(driverId, '')::INTEGER,
    NULLIF(driverRef, ''),
    NULLIF(code, ''),
    NULLIF(forename, ''),
    NULLIF(surname, ''),
    NULLIF(dob, '')::TIMESTAMP,
    NULLIF(nationality, '')
FROM temp_drivers;

\echo 'Data transferred with NULL handling'
\echo ''

\echo '========================================='
\echo 'Step 6: Add Foreign Keys'
\echo '========================================='

ALTER TABLE constructor_results ADD CONSTRAINT fk_constructor_results_raceId
    FOREIGN KEY (raceId) REFERENCES races(raceId);

ALTER TABLE constructor_results ADD CONSTRAINT fk_constructor_results_constructorId
    FOREIGN KEY (constructorId) REFERENCES constructors(constructorId);

ALTER TABLE results ADD CONSTRAINT fk_results_raceId
    FOREIGN KEY (raceId) REFERENCES races(raceId);

ALTER TABLE results ADD CONSTRAINT fk_results_driverId
    FOREIGN KEY (driverId) REFERENCES drivers(driverId);

ALTER TABLE results ADD CONSTRAINT fk_results_constructorId
    FOREIGN KEY (constructorId) REFERENCES constructors(constructorId);

ALTER TABLE standings ADD CONSTRAINT fk_standings_raceId
    FOREIGN KEY (raceId) REFERENCES races(raceId);

ALTER TABLE standings ADD CONSTRAINT fk_standings_driverId
    FOREIGN KEY (driverId) REFERENCES drivers(driverId);

ALTER TABLE qualifying ADD CONSTRAINT fk_qualifying_raceId
    FOREIGN KEY (raceId) REFERENCES races(raceId);

ALTER TABLE qualifying ADD CONSTRAINT fk_qualifying_driverId
    FOREIGN KEY (driverId) REFERENCES drivers(driverId);

ALTER TABLE qualifying ADD CONSTRAINT fk_qualifying_constructorId
    FOREIGN KEY (constructorId) REFERENCES constructors(constructorId);

ALTER TABLE constructor_standings ADD CONSTRAINT fk_constructor_standings_raceId
    FOREIGN KEY (raceId) REFERENCES races(raceId);

ALTER TABLE constructor_standings ADD CONSTRAINT fk_constructor_standings_constructorId
    FOREIGN KEY (constructorId) REFERENCES constructors(constructorId);

ALTER TABLE races ADD CONSTRAINT fk_races_circuitId
    FOREIGN KEY (circuitId) REFERENCES circuits(circuitId);

\echo 'Foreign keys added'
\echo ''

\echo '========================================='
\echo 'Step 7: Create Indexes'
\echo '========================================='

CREATE INDEX idx_constructor_results_raceId ON constructor_results(raceId) WHERE raceId IS NOT NULL;
CREATE INDEX idx_constructor_results_constructorId ON constructor_results(constructorId) WHERE constructorId IS NOT NULL;
CREATE INDEX idx_constructor_results_date ON constructor_results(date);
CREATE INDEX idx_results_raceId ON results(raceId) WHERE raceId IS NOT NULL;
CREATE INDEX idx_results_driverId ON results(driverId) WHERE driverId IS NOT NULL;
CREATE INDEX idx_results_constructorId ON results(constructorId) WHERE constructorId IS NOT NULL;
CREATE INDEX idx_results_date ON results(date);
CREATE INDEX idx_standings_raceId ON standings(raceId) WHERE raceId IS NOT NULL;
CREATE INDEX idx_standings_driverId ON standings(driverId) WHERE driverId IS NOT NULL;
CREATE INDEX idx_standings_date ON standings(date);
CREATE INDEX idx_qualifying_raceId ON qualifying(raceId) WHERE raceId IS NOT NULL;
CREATE INDEX idx_qualifying_driverId ON qualifying(driverId) WHERE driverId IS NOT NULL;
CREATE INDEX idx_qualifying_constructorId ON qualifying(constructorId) WHERE constructorId IS NOT NULL;
CREATE INDEX idx_qualifying_date ON qualifying(date);
CREATE INDEX idx_constructor_standings_raceId ON constructor_standings(raceId) WHERE raceId IS NOT NULL;
CREATE INDEX idx_constructor_standings_constructorId ON constructor_standings(constructorId) WHERE constructorId IS NOT NULL;
CREATE INDEX idx_constructor_standings_date ON constructor_standings(date);
CREATE INDEX idx_races_circuitId ON races(circuitId) WHERE circuitId IS NOT NULL;
CREATE INDEX idx_races_date ON races(date);

\echo 'Indexes created'
\echo ''

\echo '========================================='
\echo 'Summary'
\echo '========================================='

SELECT 
    table_name,
    to_char(record_count, 'FM999,999,999') as records
FROM (
    SELECT 'constructor_results' as table_name, COUNT(*) as record_count FROM constructor_results
    UNION ALL
    SELECT 'results' as table_name, COUNT(*) as record_count FROM results
    UNION ALL
    SELECT 'standings' as table_name, COUNT(*) as record_count FROM standings
    UNION ALL
    SELECT 'qualifying' as table_name, COUNT(*) as record_count FROM qualifying
    UNION ALL
    SELECT 'constructors' as table_name, COUNT(*) as record_count FROM constructors
    UNION ALL
    SELECT 'constructor_standings' as table_name, COUNT(*) as record_count FROM constructor_standings
    UNION ALL
    SELECT 'circuits' as table_name, COUNT(*) as record_count FROM circuits
    UNION ALL
    SELECT 'races' as table_name, COUNT(*) as record_count FROM races
    UNION ALL
    SELECT 'drivers' as table_name, COUNT(*) as record_count FROM drivers
) t
ORDER BY record_count DESC;

\echo ''
\echo '========================================='
\echo 'IMPORT COMPLETE'
\echo '========================================='