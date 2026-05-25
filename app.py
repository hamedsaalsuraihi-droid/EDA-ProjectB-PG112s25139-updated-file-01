
import os
import io
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = r"""SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>"""

st.set_page_config(page_title="Project B Forecasting Starter", layout="wide")

st.title("Mini Project B — Time-Series Forecasting")

st.sidebar.header("Student Information")
student_name = st.sidebar.text_input("Student Name", "Hamed salim")
student_id = st.sidebar.text_input("Student ID", "PG112S25139")
project_title = st.sidebar.text_input("Project Title", "Energy Consumption Forecasting")
project_goal = st.sidebar.text_area("Project Goal", "Forecast household appliance energy consumption.")
deployed_url = st.sidebar.text_input("Deployed App URL", "")

dataset_path = st.text_input("Dataset Path", "data/dataset_sample.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(dataset_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Dataset Audit")
audit_df = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.astype(str).values,
    "missing_percent": (df.isna().mean() * 100).round(2).values
})
st.dataframe(audit_df)

timestamp_col = st.selectbox(
    "Select Timestamp Column",
    options=df.columns,
    index=list(df.columns).index("date")
)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target_col = st.selectbox(
    "Select Target Column",
    options=numeric_cols,
    index=numeric_cols.index("Appliances")
)

resample_rule = st.selectbox(
    "Optional Resampling",
    options=["None", "H", "D", "W"],
    index=0
)

forecast_horizon = st.number_input(
    "Forecast Horizon",
    min_value=1,
    max_value=168,
    value=24
)

clean_df = df.copy()
clean_df[timestamp_col] = pd.to_datetime(clean_df[timestamp_col], errors="coerce", dayfirst=True)
clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")

clean_df = clean_df.dropna(subset=[timestamp_col, target_col])
clean_df = clean_df.sort_values(timestamp_col)

if resample_rule != "None":
    clean_df = (
        clean_df
        .set_index(timestamp_col)
        .resample(resample_rule)
        .mean(numeric_only=True)
        .reset_index()
    )

st.subheader("Cleaned Time-Series")
st.dataframe(clean_df.head())

feature_df = clean_df[[timestamp_col, target_col]].copy()

feature_df["lag_1"] = feature_df[target_col].shift(1)
feature_df["lag_24"] = feature_df[target_col].shift(24)
feature_df["rolling_mean_24"] = (
    feature_df[target_col]
    .shift(1)
    .rolling(24)
    .mean()
)

feature_df["hour"] = feature_df[timestamp_col].dt.hour
feature_df["weekend"] = feature_df[timestamp_col].dt.weekday >= 5
feature_df["month"] = feature_df[timestamp_col].dt.month

feature_df["y_target"] = feature_df[target_col].shift(-forecast_horizon)

feature_df = feature_df.dropna()

X = feature_df.drop(columns=[timestamp_col, "y_target"])
y = feature_df["y_target"]

st.subheader("Baseline Feature Table")
st.dataframe(feature_df.head())

st.write("Feature matrix shape:", X.shape)
st.write("Target vector shape:", y.shape)

# ============================================================
# STUDENT ADDITIONS — MODELING
# Add your forecasting models here.
# Create a results_df pandas DataFrame containing metrics.
# ============================================================

results_df = None

st.code("""
# Example placeholder:
# from sklearn.model_selection import train_test_split
# Add your own forecasting model here
""")

# ============================================================
# STUDENT ADDITIONS — DASHBOARD
# Add additional charts, KPIs, and visualizations here.
# ============================================================

fig, ax = plt.subplots()
ax.plot(feature_df[timestamp_col], feature_df[target_col])
ax.set_title("Target Over Time")
st.pyplot(fig)

submission_data = {
    "student_name": student_name,
    "student_id": student_id,
    "project_title": project_title,
    "project_goal": project_goal,
    "deployed_url": deployed_url,
    "timestamp_column": timestamp_col,
    "target_column": target_col,
    "forecast_horizon": int(forecast_horizon),
    "rows_used": int(len(feature_df)),
    "feature_columns": list(X.columns),
    "has_metrics_table": isinstance(results_df, pd.DataFrame),
    "results_table": [] if results_df is None else results_df.to_dict(orient="records")
}

project_card = f"""
# Project Card

## Student
- Name: {student_name}
- ID: {student_id}

## Project
- Title: {project_title}
- Goal: {project_goal}

## Dataset
- Timestamp Column: {timestamp_col}
- Target Column: {target_col}
- Rows Used: {len(feature_df)}

## Forecast Horizon
- {forecast_horizon}

## Notes
Students should extend this starter app with:
- additional features
- forecasting models
- metrics tables
- dashboard enhancements
"""

submission_json_str = json.dumps(submission_data, indent=2)

st.subheader("Export Files")

st.download_button(
    "Download submission.json",
    data=submission_json_str,
    file_name="submission.json",
    mime="application/json"
)

st.download_button(
    "Download project_card.md",
    data=project_card,
    file_name="project_card.md",
    mime="text/markdown"
)

st.subheader("AI Grader (/80)")

default_key = ""
if "OPENROUTER_API_KEY" in st.secrets:
    default_key = st.secrets["OPENROUTER_API_KEY"]
elif os.getenv("OPENROUTER_API_KEY"):
    default_key = os.getenv("OPENROUTER_API_KEY")

api_key = st.text_input(
    "OpenRouter API Key",
    value=default_key,
    type="password"
)

if st.button("Run AI Grader"):
    if not api_key:
        st.error("Please provide an OpenRouter API key.")
    else:
        prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
            "<insert submission.json contents here>",
            submission_json_str
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )

            result = response.json()

            content = result["choices"][0]["message"]["content"]

            parsed = None

            try:
                parsed = json.loads(content)
            except Exception:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = None

            if parsed is not None:
                st.json(parsed)
            else:
                st.text_area("Raw Model Output", content, height=300)

        except Exception as e:
            st.error(f"AI grading failed: {e}")
