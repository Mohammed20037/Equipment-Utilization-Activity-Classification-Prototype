import pandas as pd
import streamlit as st
from sqlalchemy import text

from shared.db import get_engine

st.set_page_config(page_title="Equipment Utilization", layout="wide")
st.title("Equipment Utilization Dashboard")

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

with engine.begin() as conn:
    rows = conn.execute(summary_query).fetchall()

if not rows:
    st.info("No data yet. Start cv_service + analytics_service.")
else:
    df = pd.DataFrame(rows, columns=[
        "equipment_id", "equipment_class", "total_tracked_seconds",
        "total_active_seconds", "total_idle_seconds", "utilization_percent",
        "last_activity", "last_state", "updated_at"
    ])
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("equipment_id")["utilization_percent"])
