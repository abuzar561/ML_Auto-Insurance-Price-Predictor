from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "AutoInsurance.csv"
TARGET_COLUMN = "Monthly Premium Auto"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RAW_COLUMNS = [
    "Customer",
    "State",
    "Customer Lifetime Value",
    "Response",
    "Coverage",
    "Education",
    "Effective To Date",
    "EmploymentStatus",
    "Gender",
    "Income",
    "Location Code",
    "Marital Status",
    TARGET_COLUMN,
    "Months Since Last Claim",
    "Months Since Policy Inception",
    "Number of Open Complaints",
    "Number of Policies",
    "Policy Type",
    "Policy",
    "Renew Offer Type",
    "Sales Channel",
    "Total Claim Amount",
    "Vehicle Class",
    "Vehicle Size",
]

FEATURE_COLUMNS = [
    "State",
    "Customer Lifetime Value",
    "Response",
    "Coverage",
    "Education",
    "EmploymentStatus",
    "Gender",
    "Income",
    "Location Code",
    "Marital Status",
    "Months Since Last Claim",
    "Months Since Policy Inception",
    "Number of Open Complaints",
    "Number of Policies",
    "Policy Type",
    "Policy",
    "Renew Offer Type",
    "Sales Channel",
    "Total Claim Amount",
    "Vehicle Class",
    "Vehicle Size",
]

NUMERIC_FEATURES = [
    "Customer Lifetime Value",
    "Income",
    "Months Since Last Claim",
    "Months Since Policy Inception",
    "Number of Open Complaints",
    "Number of Policies",
    "Total Claim Amount",
]

CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES]


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    metrics: dict[str, dict[str, float]]
    best_model_name: str
    row_count: int
    target_mean: float
    target_min: float
    target_max: float
    categorical_options: dict[str, list[str]]


def normalize_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in RAW_COLUMNS if column not in raw_data.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = raw_data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    data = data.replace(["", "?", "NA", "N/A"], pd.NA)

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(subset=[TARGET_COLUMN])

    return data


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return normalize_dataset(pd.read_csv(path))


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "Ridge Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def get_categorical_options(dataset: pd.DataFrame) -> dict[str, list[str]]:
    return {
        column: sorted(value for value in dataset[column].dropna().unique().tolist())
        for column in CATEGORICAL_FEATURES
    }


def train_candidate_models(path: Path = DATA_PATH) -> TrainingResult:
    dataset = load_dataset(path)
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}
    best_model_name = ""
    best_mae = float("inf")

    for model_name, model in build_candidate_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics[model_name] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, predictions)), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }
        models[model_name] = model

        if metrics[model_name]["mae"] < best_mae:
            best_mae = metrics[model_name]["mae"]
            best_model_name = model_name

    return TrainingResult(
        models=models,
        metrics=metrics,
        best_model_name=best_model_name,
        row_count=len(dataset),
        target_mean=float(y.mean()),
        target_min=float(y.min()),
        target_max=float(y.max()),
        categorical_options=get_categorical_options(dataset),
    )
