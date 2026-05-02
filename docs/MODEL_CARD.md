# Model Card

## Model Details

- Task: supervised regression
- Target: `Monthly Premium Auto`
- Inputs: customer, policy, vehicle, location, and claims attributes
- Framework: scikit-learn
- Interface: Streamlit

## Candidate Models

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The training pipeline selects the model with the lowest mean absolute error on a held-out test split.

## Feature Handling

The pipeline:

- drops `Customer` and `Effective To Date`
- imputes missing numeric and categorical values
- scales numeric features
- one-hot encodes categorical features
- compares model metrics on the same validation split

## Intended Use

This project is intended for:

- machine learning portfolio demonstration
- Streamlit app demonstration
- regression-model practice
- discussion of responsible ML in insurance pricing contexts

## Not Intended For

- real insurance quotes
- regulated pricing decisions
- automated policy issuance
- use with private customer data

## Limitations

- The dataset is historical and limited.
- Metrics are based on a single train/test split.
- Model estimates do not represent approved insurance rates.
- No fairness, drift, calibration, or actuarial validation is included.
- Some source features may be inappropriate for regulated production pricing without review.
