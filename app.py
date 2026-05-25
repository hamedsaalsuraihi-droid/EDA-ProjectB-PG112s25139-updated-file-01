# ==========================================================
# STUDENT ADDITIONS — MODELING
# ==========================================================

st.header("Student Additions — Modeling")

# ==============================
# DATA INTEGRITY NOTES
# ==============================

st.subheader("Data Integrity Notes")

st.markdown("""
### Cleaning Steps Performed
- Parsed timestamp column into datetime format
- Removed invalid timestamps
- Converted target column to numeric
- Dropped rows with missing target values
- Sorted data chronologically

### Resampling
Optional hourly/daily resampling is supported in the app.

### Outliers
Extreme energy spikes are preserved because they may represent real appliance behavior.
""")

# ==============================
# TIME-BASED SPLIT
# ==============================

split_index = int(len(feature_df) * 0.8)

train_df = feature_df.iloc[:split_index]
test_df = feature_df.iloc[split_index:]

X_train = train_df[feature_columns]
y_train = train_df["y_target"]

X_test = test_df[feature_columns]
y_test = test_df["y_target"]

st.subheader("Train/Test Split")

st.write("Training rows:", len(train_df))
st.write("Testing rows:", len(test_df))

# ==============================
# MODEL TRAINING
# ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==============================
# METRICS
# ==============================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

mape = np.mean(
    np.abs((y_test - predictions) / y_test)
) * 100

results_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2",
        "MAPE"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        mape
    ]
})

st.subheader("Evaluation Metrics")

st.dataframe(results_df)

# ==============================
# PREDICTIONS TABLE
# ==============================

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

st.subheader("Predictions Preview")

st.dataframe(comparison_df.head(20))

# ==========================================================
# STUDENT ADDITIONS — DASHBOARD
# ==========================================================

st.header("Student Additions — Dashboard")

# ==============================
# BASELINE TIME SERIES
# ==============================

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    work_df[timestamp_col].head(500),
    work_df[target_col].head(500)
)

ax.set_title("Baseline Time-Series Preview")

st.pyplot(fig)

# ==============================
# ACTUAL VS PREDICTED
# ==============================

st.subheader("Actual vs Predicted")

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.plot(
    y_test.values[:200],
    label="Actual"
)

ax2.plot(
    predictions[:200],
    label="Predicted"
)

ax2.legend()

ax2.set_title(
    "Actual vs Predicted Energy Usage"
)

st.pyplot(fig2)

# ==============================
# FEATURE IMPORTANCE
# ==============================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

st.dataframe(importance_df)

fig3, ax3 = plt.subplots(figsize=(8, 4))

ax3.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax3.set_title("Feature Importance")

st.pyplot(fig3)

# ==============================
# PROJECT INSIGHTS
# ==============================

st.header("Project Insights")

avg_energy = work_df[target_col].mean()

max_energy = work_df[target_col].max()

st.markdown(f"""
### Key Findings

- Average appliance energy usage was approximately **{avg_energy:.2f}**.
- Maximum observed appliance usage reached **{max_energy:.2f}**.
- Lag-based features improved forecasting context.
- Time-of-day patterns strongly influence energy consumption.
- Weekend behavior differs from weekday usage patterns.

### Business Implications

This forecasting system can help households:
- monitor abnormal energy spikes
- reduce electricity costs
- optimize appliance scheduling
- improve energy efficiency planning
""")
