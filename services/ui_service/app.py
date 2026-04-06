import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

from shared.db import get_engine

st.set_page_config(page_title="Equipment Utilization", layout="wide")
st.title("Equipment Utilization Dashboard")

refresh_sec = int(os.getenv("UI_REFRESH_SECONDS", "3"))
frame_path = Path(os.getenv("OUTPUT_FRAME_PATH", "data/processed/latest.jpg"))
st.caption(f"Refresh every {refresh_sec}s (manual refresh in browser if needed).")

engine = get_engine()

target_classes = {
    c.strip().lower().replace("_", " ").replace("-", " ")
    for c in os.getenv("TARGET_EQUIPMENT_CLASSES", "excavator,dump truck,loader,roller,bulldozer,truck").split(",")
    if c.strip()
}

summary_query = text(
    """
    SELECT equipment_id, equipment_class, total_tracked_seconds,
           total_active_seconds, total_idle_seconds, total_downtime_seconds,
           current_stop_seconds, last_stop_seconds, stop_count, utilization_percent,
           last_activity, last_state, updated_at
    FROM equipment_summary
    ORDER BY updated_at DESC
    """
)

recent_events_query = text(
    """
    SELECT equipment_id, timestamp_sec, utilization_percent, current_state, current_activity,
           total_downtime_seconds, current_stop_seconds
    FROM frame_events
    ORDER BY id DESC
    LIMIT 500
    """
)

with engine.begin() as conn:
    summary_rows = conn.execute(summary_query).fetchall()
    event_rows = conn.execute(recent_events_query).fetchall()

col_video, col_status = st.columns([3, 2])
with col_video:
    st.subheader("Processed Video Feed (latest frame)")
    if frame_path.exists():
        st.image(str(frame_path), channels="BGR", use_container_width=True)
    else:
        st.info("No processed frame yet. Start cv_service.")

with col_status:
    st.subheader("Live Machine Status")
    if summary_rows:
        status_df = pd.DataFrame(summary_rows, columns=[
            "equipment_id", "equipment_class", "total_tracked_seconds",
            "total_active_seconds", "total_idle_seconds", "total_downtime_seconds",
            "current_stop_seconds", "last_stop_seconds", "stop_count", "utilization_percent",
            "last_activity", "last_state", "updated_at"
        ])
        status_df = status_df[status_df["equipment_class"].str.lower().isin(target_classes)]
        st.dataframe(
            status_df[[
                "equipment_id", "equipment_class", "last_state", "last_activity",
                "current_stop_seconds", "total_downtime_seconds", "utilization_percent"
            ]],
            use_container_width=True,
        )
    else:
        st.info("No machine status available yet.")

if not summary_rows:
    st.info("No data yet. Start cv_service + analytics_service.")
else:
    summary_df = pd.DataFrame(summary_rows, columns=[
        "equipment_id", "equipment_class", "total_tracked_seconds",
        "total_active_seconds", "total_idle_seconds", "total_downtime_seconds",
        "current_stop_seconds", "last_stop_seconds", "stop_count", "utilization_percent",
        "last_activity", "last_state", "updated_at"
    ])

    summary_df = summary_df[summary_df["equipment_class"].str.lower().isin(target_classes)]
    if summary_df.empty:
        st.info("No relevant equipment tracks yet (after class filtering).")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked machines", int(summary_df.shape[0]))
    c2.metric("Avg utilization %", round(float(summary_df["utilization_percent"].mean()), 2))
    c3.metric("Fleet downtime (s)", round(float(summary_df["total_downtime_seconds"].sum()), 2))

    st.subheader("Utilization & Downtime Dashboard")
    st.dataframe(
        summary_df[[
            "equipment_id",
            "total_active_seconds",
            "total_idle_seconds",
            "total_downtime_seconds",
            "last_stop_seconds",
            "stop_count",
            "utilization_percent",
        ]],
        use_container_width=True,
    )
    st.bar_chart(summary_df.set_index("equipment_id")[["utilization_percent", "total_downtime_seconds"]])

    if event_rows:
        events_df = pd.DataFrame(
            event_rows,
            columns=[
                "equipment_id", "timestamp_sec", "utilization_percent", "current_state",
                "current_activity", "total_downtime_seconds", "current_stop_seconds"
            ],
        )
        events_df = events_df[events_df["equipment_id"].isin(set(summary_df["equipment_id"]))]
        events_df = events_df.sort_values("timestamp_sec")
        st.subheader("Recent Utilization Trend")
        st.line_chart(events_df, x="timestamp_sec", y="utilization_percent", color="equipment_id")
