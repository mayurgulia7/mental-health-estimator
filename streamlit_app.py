"""Streamlit interface for the trained student score regression model."""
from pathlib import Path
import streamlit as st

try:
    import joblib
    import pandas as pd
except ModuleNotFoundError as exc:
    st.set_page_config(page_title="Student Wellbeing Score", page_icon="🌿")
    st.title("🌿 Student Wellbeing Score")
    st.error(f"A required Python package is missing: `{exc.name}`.")
    st.markdown("Install the project dependencies in the **same Python environment** used to run Streamlit, then restart the app:")
    st.code("python -m pip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")
    st.stop()

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "mental_health_model.joblib"
DATA_PATH = ROOT / "Student Social Media And Mental Health Impact.csv"

st.set_page_config(page_title="Student Wellbeing Score", page_icon="🌿", layout="wide")
st.title("🌿 Student Wellbeing Score")
st.markdown("Estimate the dataset's mental health score from social media habits and lifestyle factors.")
st.info("This tool provides a model estimate for an educational project. It is not a diagnosis or a substitute for professional support.")

if not ARTIFACT_PATH.exists():
    st.warning("The model has not been trained yet. Train it first to create the required model artifact.")
    st.code("python train_model.py", language="bash")
    st.stop()

@st.cache_resource
def load_model(path: str, modified: float):
    return joblib.load(path)["model"]

@st.cache_data
def load_categories(path: str, modified: float):
    data = pd.read_csv(path)
    categories = {}
    for column in ["Gender", "Country", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use"]:
        categories[column] = sorted(data[column].dropna().astype(str).unique().tolist())
    return categories

try:
    model = load_model(str(ARTIFACT_PATH), ARTIFACT_PATH.stat().st_mtime)
    categories = load_categories(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()

with st.form("student_features"):
    st.subheader("Student profile")
    profile = st.columns(3)
    with profile[0]:
        age = st.number_input("Age", min_value=10, max_value=120, value=21, step=1)
        gender = st.selectbox("Gender", categories["Gender"])
    with profile[1]:
        academic_level = st.selectbox("Academic level", categories["Academic_Level"])
        country = st.selectbox("Country", categories["Country"])
    with profile[2]:
        platform = st.selectbox("Most used platform", categories["Most_Used_Platform"])
        purpose = st.selectbox("Main purpose of use", categories["Purpose_Of_Use"])

    st.subheader("Daily habits")
    habits = st.columns(3)
    with habits[0]:
        usage = st.slider("Social media use (hours/day)", 0.0, 24.0, 5.0, 0.1)
        unlocks = st.number_input("Phone unlocks per day", min_value=0, max_value=2000, value=170, step=1)
    with habits[1]:
        study = st.slider("Study (hours/day)", 0.0, 24.0, 3.0, 0.1)
        activity = st.slider("Physical activity (hours/day)", 0.0, 24.0, 1.5, 0.1)
    with habits[2]:
        sleep = st.slider("Sleep (hours/night)", 0.0, 24.0, 7.0, 0.1)
        stress = st.select_slider("Stress level", options=["Low", "Medium", "High", "Very High"], value="Medium")

    submitted = st.form_submit_button("Estimate score", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic_level,
        "Most_Used_Platform": platform,
        "Purpose_Of_Use": purpose,
        "Avg_Daily_Usage_Hours": usage,
        "Daily_Unlocks": unlocks,
        "Study_Hours": study,
        "Physical_Activity_Hours": activity,
        "Sleep_Hours_Per_Night": sleep,
        "Stress_Level": stress,
    }])
    score = float(model.predict(row)[0])
    st.divider()
    st.subheader("Estimated score")
    st.metric("Mental health score", f"{score:.2f} / 10")
    st.caption("The estimate reflects patterns in the training dataset. It should not be used to assess an individual's mental health.")
