CREATE TABLE IF NOT EXISTS frame_events (
    id SERIAL PRIMARY KEY,
    frame_id INT,
    equipment_id VARCHAR(50),
    equipment_class VARCHAR(50),
    timestamp_sec FLOAT,
    current_state VARCHAR(20),
    current_activity VARCHAR(50),
    motion_source VARCHAR(50),
    total_tracked_seconds FLOAT,
    total_active_seconds FLOAT,
    total_idle_seconds FLOAT,
    total_downtime_seconds FLOAT,
    current_stop_seconds FLOAT,
    last_stop_seconds FLOAT,
    stop_count INT DEFAULT 0,
    utilization_percent FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS equipment_summary (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_class VARCHAR(50),
    total_tracked_seconds FLOAT,
    total_active_seconds FLOAT,
    total_idle_seconds FLOAT,
    total_downtime_seconds FLOAT,
    current_stop_seconds FLOAT,
    last_stop_seconds FLOAT,
    stop_count INT DEFAULT 0,
    utilization_percent FLOAT,
    last_activity VARCHAR(50),
    last_state VARCHAR(20),
    updated_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE frame_events ADD COLUMN IF NOT EXISTS total_downtime_seconds FLOAT;
ALTER TABLE frame_events ADD COLUMN IF NOT EXISTS current_stop_seconds FLOAT;
ALTER TABLE frame_events ADD COLUMN IF NOT EXISTS last_stop_seconds FLOAT;
ALTER TABLE frame_events ADD COLUMN IF NOT EXISTS stop_count INT DEFAULT 0;

ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS total_downtime_seconds FLOAT;
ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS current_stop_seconds FLOAT;
ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS last_stop_seconds FLOAT;
ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS stop_count INT DEFAULT 0;
