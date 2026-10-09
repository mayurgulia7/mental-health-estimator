"""Train and save the student mental-health-score regression pipeline."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Student Social Media And Mental Health Impact.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "mental_health_model.joblib"
TARGET = "Mental_Health_Score"
NUMERIC = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Study_Hours",
           "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
CATEGORICAL = ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use", "Stress_Level", "Country"]
STRESS_ORDER = ["Low", "Medium", "High", "Very High"]


def build_preprocessor():
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[STRESS_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    nominal_columns = [c for c in CATEGORICAL if c != "Stress_Level"]
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("study", skewed, ["Study_Hours"]),
        ("numeric", numeric, [c for c in NUMERIC if c != "Study_Hours"]),
        ("stress", ordinal, ["Stress_Level"]),
        ("categorical", nominal, nominal_columns),
    ])


def main():
    df = pd.read_csv(DATA_PATH).drop_duplicates().copy()
    df["Physical_Activity_Hours"] = df["Physical_Activity_Hours"].clip(lower=0)
    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("random forest", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipeline,
        {
            "random forest__n_estimators": [100, 200, 300],
            "random forest__max_depth": [5, 10, 15, None],
            "random forest__min_samples_split": [2, 5, 10],
            "random forest__min_samples_leaf": [1, 2, 4],
        },
        n_iter=12, cv=3, scoring="r2", random_state=42, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    predictions = search.best_estimator_.predict(X_test)
    print(f"Held-out R2:   {r2_score(y_test, predictions):.3f}")
    print(f"Held-out MAE:  {mean_absolute_error(y_test, predictions):.3f} score points")
    print(f"Held-out RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f} score points")
    print("Best parameters:", search.best_params_)

    # Refit the selected configuration on all cleaned records for serving.
    final_model = search.best_estimator_
    final_model.fit(X, y)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": NUMERIC + CATEGORICAL}, ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
