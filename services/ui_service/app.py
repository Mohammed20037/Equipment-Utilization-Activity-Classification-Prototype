import os

import pandas as pd
import streamlit as st
from sqlalchemy import text

from shared.db import get_engine

st.set_page_config(page_title="Equipment Utilization", layout="wide")
st.title("Equipment Utilization Dashboard")

refresh_sec = int(os.getenv("UI_REFRESH_SECONDS", "3"))
st.caption(f"Auto-refresh every {refresh_sec}s (manual browser refresh if needed).")

engine = get_engine()

summary_query = text(
    """
    SELECT equipment_id, equipment_class, total_tracked_seconds,
           total_active_seconds, total_idle_seconds, utilization_percent,
           last_activity, last_state, updated_at
    FROM equipment_summary
    ORDER BY updated_at DESC
    """
)

recent_events_query = text(
    """
    SELECT equipment_id, timestamp_sec, utilization_percent
    FROM frame_events
    ORDER BY id DESC
    LIMIT 500
    """
)

with engine.begin() as conn:
    summary_rows = conn.execute(summary_query).fetchall()
    event_rows = conn.execute(recent_events_query).fetchall()

if not summary_rows:
    st.info("No data yet. Start cv_service + analytics_service.")
else:
    summary_df = pd.DataFrame(summary_rows, columns=[
        "equipment_id", "equipment_class", "total_tracked_seconds",
        "total_active_seconds", "total_idle_seconds", "utilization_percent",
        "last_activity", "last_state", "updated_at"
    ])

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked machines", int(summary_df.shape[0]))
    c2.metric("Avg utilization %", round(float(summary_df["utilization_percent"].mean()), 2))
    c3.metric("Max utilization %", round(float(summary_df["utilization_percent"].max()), 2))

    st.subheader("Equipment Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Utilization by Machine")
    st.bar_chart(summary_df.set_index("equipment_id")["utilization_percent"])

    if event_rows:
        events_df = pd.DataFrame(event_rows, columns=["equipment_id", "timestamp_sec", "utilization_percent"])
        events_df = events_df.sort_values("timestamp_sec")
        st.subheader("Recent Utilization Trend")
        st.line_chart(events_df, x="timestamp_sec", y="utilization_percent", color="equipment_id")
