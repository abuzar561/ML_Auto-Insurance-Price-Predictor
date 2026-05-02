# Usage Guide

## Run the App

```bash
streamlit run app.py
```

The app trains the candidate models from `AutoInsurance.csv` on startup and caches the result for the Streamlit session.

## Train from the Command Line

```bash
python train_model.py
```

This writes:

```text
reports/training_metrics.json
models/auto_insurance_model.joblib
```

Both files are generated artifacts and are ignored by git.

## Example Policy

```json
{
  "State": "California",
  "Customer Lifetime Value": 8000,
  "Response": "No",
  "Coverage": "Basic",
  "Education": "Bachelor",
  "EmploymentStatus": "Employed",
  "Gender": "F",
  "Income": 50000,
  "Location Code": "Suburban",
  "Marital Status": "Married",
  "Months Since Last Claim": 12,
  "Months Since Policy Inception": 36,
  "Number of Open Complaints": 0,
  "Number of Policies": 1,
  "Policy Type": "Personal Auto",
  "Policy": "Personal L3",
  "Renew Offer Type": "Offer1",
  "Sales Channel": "Agent",
  "Total Claim Amount": 450,
  "Vehicle Class": "Four-Door Car",
  "Vehicle Size": "Medsize"
}
```

## Troubleshooting

If the app cannot load data, confirm `AutoInsurance.csv` is in the repository root.

If dependencies are missing, run:

```bash
pip install -r requirements.txt
```

If validation fails after changing the dataset, confirm the CSV headers still match `RAW_COLUMNS` in `src/insurance_pipeline.py`.
