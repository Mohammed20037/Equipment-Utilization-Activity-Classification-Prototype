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
    utilization_percent FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS equipment_summary (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_class VARCHAR(50),
    total_tracked_seconds FLOAT,
    total_active_seconds FLOAT,
    total_idle_seconds FLOAT,
    utilization_percent FLOAT,
    last_activity VARCHAR(50),
    last_state VARCHAR(20),
    updated_at TIMESTAMP DEFAULT NOW()
);
