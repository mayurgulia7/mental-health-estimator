# Student Social Media and Mental Health Score

A regression project using student social-media, lifestyle, and demographic data to estimate the dataset's `Mental_Health_Score`. This score is not a clinical diagnosis or substitute for professional care.

## Project files

- `ML_Project.ipynb` — data review, EDA, cleaning, preprocessing, model comparison, tuning, and evaluation.
- `train_model.py` — reproducible training and held-out evaluation; writes `artifacts/mental_health_model.joblib`.
- `app.py` — FastAPI inference service.
- `streamlit_app.py` — interactive browser UI for entering student features and viewing a prediction.
- `Student Social Media And Mental Health Impact.csv` — source dataset.

## Setup

Use Python 3.10 or newer. From this directory, create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Open and run `ML_Project.ipynb` from this directory so its relative CSV path resolves. To train the serving artifact and print held-out R², MAE, and RMSE:

```bash
python train_model.py
```

## Run the API

After training, start the service:

```bash
uvicorn app:app --reload
```

- Health: `GET http://127.0.0.1:8000/health`
- Interactive schema and request form: `http://127.0.0.1:8000/docs`
- Prediction: `POST http://127.0.0.1:8000/predict`

Example request body (all 12 feature fields are required):

```json
{
  "Age": 21, "Gender": "Female", "Country": "India",
  "Academic_Level": "Undergraduate", "Most_Used_Platform": "Instagram",
  "Purpose_Of_Use": "Entertainment", "Avg_Daily_Usage_Hours": 5.2,
  "Daily_Unlocks": 170, "Study_Hours": 3.0,
  "Physical_Activity_Hours": 1.5, "Sleep_Hours_Per_Night": 7.0,
  "Stress_Level": "Medium"
}
```

Country categories are handled by the fitted one-hot encoder; categories not seen during training are safely ignored. The API validates numeric ranges and allowed stress levels.


## Run the Streamlit UI

Install dependencies in the same Python environment that runs Streamlit, then train the model and launch the interactive form:

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run streamlit_app.py
```

Streamlit opens the app in your browser. The form collects the same 12 feature fields as the API and displays the estimated score with an educational-use disclaimer.

If Streamlit reports `No module named 'joblib'`, it is running from an environment that does not have the project dependencies. Run the install command above using that environment's Python, then restart Streamlit.
