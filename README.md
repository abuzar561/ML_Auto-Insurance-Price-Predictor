# Auto Insurance Premium Predictor

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-web%20app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-F7931E?logo=scikitlearn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A Streamlit machine learning app that estimates monthly auto insurance premiums from customer, policy, vehicle, and claims attributes. The project trains multiple scikit-learn regression models from the included dataset, compares validation metrics, and presents a practical quote estimate with a model-error range.

## Highlights

- Interactive Streamlit quote estimator for auto insurance premiums.
- Reusable preprocessing and model-training pipeline in `src/insurance_pipeline.py`.
- Compares Ridge Regression, Random Forest, and Gradient Boosting models.
- Handles numeric imputation, categorical imputation, one-hot encoding, scaling, and model selection.
- Generates training metrics and optional local model artifacts with `train_model.py`.
- Includes documentation, a sample policy payload, repository validation, and GitHub Actions CI.
- Keeps generated model binaries out of source control so the repo stays lightweight and reproducible.

## Dataset

`AutoInsurance.csv` contains 9,134 customer-policy records with 24 columns. The target column is `Monthly Premium Auto`, with values ranging from 61 to 298 in the included dataset.

See [docs/DATASET.md](docs/DATASET.md) for schema notes, feature descriptions, and limitations.

## Project Structure

```text
.
+-- app.py
+-- train_model.py
+-- AutoInsurance.csv
+-- src/
|   +-- insurance_pipeline.py
+-- docs/
|   +-- DATASET.md
|   +-- MODEL_CARD.md
|   +-- USAGE.md
+-- examples/
|   +-- sample-policy.json
+-- scripts/
|   +-- validate-project.js
+-- requirements.txt
```

## Quick Start

```bash
git clone https://github.com/abuzar561/ML_Auto-Insurance-Price-Predictor.git
cd ML_Auto-Insurance-Price-Predictor
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Train and Export Metrics

```bash
python train_model.py
```

The script trains candidate models, selects the best model by mean absolute error, writes `reports/training_metrics.json`, and saves a local model artifact to `models/auto_insurance_model.joblib`.

## Validation

```bash
node scripts/validate-project.js
python -m py_compile app.py train_model.py src/insurance_pipeline.py
python train_model.py
```

GitHub Actions runs these checks on every push and pull request.

## Important Disclaimer

This project is for education and portfolio demonstration only. It is not a production pricing system and must not be used to issue real insurance quotes. Real insurance pricing requires regulated rating plans, actuarial review, compliance approval, monitoring, auditability, and human oversight.

## License

This project is licensed under the [MIT License](LICENSE).
