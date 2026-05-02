# Dataset Notes

`AutoInsurance.csv` contains 9,134 auto-insurance customer-policy records and 24 columns.

## Columns

| Column | Description |
| --- | --- |
| `Customer` | Customer identifier. Not used as a model feature. |
| `State` | Customer or policy state. |
| `Customer Lifetime Value` | Customer value metric from the source dataset. |
| `Response` | Marketing response flag. |
| `Coverage` | Coverage tier. |
| `Education` | Customer education level. |
| `Effective To Date` | Policy effective date. Not used as a model feature. |
| `EmploymentStatus` | Employment status. |
| `Gender` | Customer gender field from the dataset. |
| `Income` | Customer income. |
| `Location Code` | Rural, suburban, or urban location code. |
| `Marital Status` | Customer marital status. |
| `Monthly Premium Auto` | Target variable for prediction. |
| `Months Since Last Claim` | Time since last claim. |
| `Months Since Policy Inception` | Policy tenure in months. |
| `Number of Open Complaints` | Open complaint count. |
| `Number of Policies` | Policies associated with the customer. |
| `Policy Type` | Policy category. |
| `Policy` | Policy plan. |
| `Renew Offer Type` | Renewal offer category. |
| `Sales Channel` | Acquisition or sales channel. |
| `Total Claim Amount` | Total claim amount field in the source dataset. |
| `Vehicle Class` | Vehicle category. |
| `Vehicle Size` | Vehicle size band. |

## Target Summary

| Metric | Value |
| --- | ---: |
| Rows | 9,134 |
| Minimum monthly premium | 61 |
| Mean monthly premium | 93.22 |
| Maximum monthly premium | 298 |

## Preprocessing Notes

- `Customer` and `Effective To Date` are excluded from the model feature set.
- Numeric columns use median imputation and standard scaling.
- Categorical columns use most-frequent imputation and one-hot encoding.
- Model selection uses a held-out train/test split and mean absolute error.

## Responsible Use

This dataset is useful for ML education and portfolio demos. Real insurance pricing needs approved rating data, actuarial review, regulatory compliance, bias monitoring, explainability, and controlled deployment.
