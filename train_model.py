import json
from pathlib import Path

import joblib

from src.insurance_pipeline import DATA_PATH, FEATURE_COLUMNS, train_candidate_models


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = REPORTS_DIR / "training_metrics.json"
MODEL_PATH = MODELS_DIR / "auto_insurance_model.joblib"


def main() -> None:
    result = train_candidate_models(DATA_PATH)
    selected_model = result.models[result.best_model_name]

    report = {
        "dataset": {
            "source": DATA_PATH.name,
            "rows": result.row_count,
            "target": "Monthly Premium Auto",
            "target_mean": round(result.target_mean, 4),
            "target_min": result.target_min,
            "target_max": result.target_max,
        },
        "selected_model": result.best_model_name,
        "selection_metric": "mae",
        "models": result.metrics,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": selected_model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": result.metrics[result.best_model_name],
        },
        MODEL_PATH,
    )

    print(f"Rows: {result.row_count:,}")
    print(f"Selected model: {result.best_model_name}")
    print(f"MAE: ${result.metrics[result.best_model_name]['mae']:,.2f}")
    print(f"Metrics written to: {METRICS_PATH}")
    print(f"Model written to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
