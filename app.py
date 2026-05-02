from pathlib import Path

import pandas as pd
import streamlit as st

from src.insurance_pipeline import DATA_PATH, FEATURE_COLUMNS, train_candidate_models


st.set_page_config(page_title="Auto Insurance Premium Predictor", layout="wide")


@st.cache_resource(show_spinner=False)
def load_training_result(data_path: str):
    return train_candidate_models(Path(data_path))


def build_policy_input(options: dict[str, list[str]]) -> pd.DataFrame:
    st.sidebar.header("Policy Profile")

    state = st.sidebar.selectbox("State", options["State"])
    coverage = st.sidebar.selectbox("Coverage", options["Coverage"], index=options["Coverage"].index("Basic"))
    policy_type = st.sidebar.selectbox("Policy Type", options["Policy Type"])
    policy = st.sidebar.selectbox("Policy", options["Policy"])
    renew_offer = st.sidebar.selectbox("Renew Offer Type", options["Renew Offer Type"])
    sales_channel = st.sidebar.selectbox("Sales Channel", options["Sales Channel"])

    st.sidebar.header("Customer Profile")
    education = st.sidebar.selectbox("Education", options["Education"])
    employment = st.sidebar.selectbox("Employment Status", options["EmploymentStatus"])
    gender = st.sidebar.selectbox("Gender", options["Gender"])
    location = st.sidebar.selectbox("Location Code", options["Location Code"])
    marital = st.sidebar.selectbox("Marital Status", options["Marital Status"])
    response = st.sidebar.selectbox("Marketing Response", options["Response"])

    income = st.sidebar.number_input("Annual Income", min_value=0, max_value=200000, value=50000, step=1000)
    customer_lifetime_value = st.sidebar.number_input(
        "Customer Lifetime Value",
        min_value=0,
        max_value=100000,
        value=8000,
        step=500,
    )

    st.sidebar.header("Claims and Vehicle")
    vehicle_class = st.sidebar.selectbox("Vehicle Class", options["Vehicle Class"])
    vehicle_size = st.sidebar.selectbox("Vehicle Size", options["Vehicle Size"])
    total_claim_amount = st.sidebar.number_input(
        "Total Claim Amount",
        min_value=0,
        max_value=3000,
        value=450,
        step=25,
    )
    months_since_last_claim = st.sidebar.number_input(
        "Months Since Last Claim",
        min_value=0,
        max_value=60,
        value=12,
        step=1,
    )
    months_since_policy_inception = st.sidebar.number_input(
        "Months Since Policy Inception",
        min_value=0,
        max_value=120,
        value=36,
        step=1,
    )
    open_complaints = st.sidebar.number_input("Number of Open Complaints", min_value=0, max_value=10, value=0, step=1)
    number_of_policies = st.sidebar.number_input("Number of Policies", min_value=1, max_value=10, value=1, step=1)

    policy_input = {
        "State": state,
        "Customer Lifetime Value": float(customer_lifetime_value),
        "Response": response,
        "Coverage": coverage,
        "Education": education,
        "EmploymentStatus": employment,
        "Gender": gender,
        "Income": float(income),
        "Location Code": location,
        "Marital Status": marital,
        "Months Since Last Claim": float(months_since_last_claim),
        "Months Since Policy Inception": float(months_since_policy_inception),
        "Number of Open Complaints": float(open_complaints),
        "Number of Policies": float(number_of_policies),
        "Policy Type": policy_type,
        "Policy": policy,
        "Renew Offer Type": renew_offer,
        "Sales Channel": sales_channel,
        "Total Claim Amount": float(total_claim_amount),
        "Vehicle Class": vehicle_class,
        "Vehicle Size": vehicle_size,
    }

    return pd.DataFrame([policy_input], columns=FEATURE_COLUMNS)


def quote_signals(policy: pd.DataFrame) -> list[str]:
    row = policy.iloc[0]
    signals: list[str] = []

    if row["Coverage"] == "Premium":
        signals.append("Premium coverage is associated with higher quoted monthly premiums.")

    if row["Vehicle Class"] in {"Luxury Car", "Luxury SUV", "Sports Car"}:
        signals.append("Higher-value vehicle classes tend to increase premium estimates.")

    if row["Total Claim Amount"] > 900:
        signals.append("A high recent claim amount can raise the model estimate.")

    if row["Number of Open Complaints"] > 0:
        signals.append("Open complaints are included as a customer-service risk signal.")

    if row["Location Code"] == "Urban":
        signals.append("Urban location codes may carry different risk patterns than rural or suburban policies.")

    return signals


def main() -> None:
    st.title("Auto Insurance Premium Predictor")
    st.caption("Educational regression demo built with Streamlit and scikit-learn.")

    try:
        training_result = load_training_result(str(DATA_PATH))
    except Exception as error:
        st.error(f"Unable to train models from {DATA_PATH.name}: {error}")
        st.stop()

    policy = build_policy_input(training_result.categorical_options)
    selected_model = training_result.models[training_result.best_model_name]
    selected_metrics = training_result.metrics[training_result.best_model_name]

    st.sidebar.header("Model")
    st.sidebar.write(f"Selected model: `{training_result.best_model_name}`")
    st.sidebar.write(f"Training rows: `{training_result.row_count:,}`")
    st.sidebar.info("This is an educational pricing demo, not a real insurance quote engine.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected Model", training_result.best_model_name)
    metric_columns[1].metric("MAE", f"${selected_metrics['mae']:,.2f}")
    metric_columns[2].metric("RMSE", f"${selected_metrics['rmse']:,.2f}")
    metric_columns[3].metric("R2 Score", f"{selected_metrics['r2']:.3f}")

    st.subheader("Model Comparison")
    comparison = pd.DataFrame(training_result.metrics).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.subheader("Policy Input")
    st.dataframe(policy, use_container_width=True, hide_index=True)

    st.divider()

    if st.button("Estimate Monthly Premium", type="primary", use_container_width=True):
        prediction = float(selected_model.predict(policy)[0])
        lower_bound = max(training_result.target_min, prediction - selected_metrics["mae"])
        upper_bound = prediction + selected_metrics["mae"]

        result_columns = st.columns(3)
        result_columns[0].metric("Estimated Monthly Premium", f"${prediction:,.2f}")
        result_columns[1].metric("Typical Error Range", f"${lower_bound:,.2f} - ${upper_bound:,.2f}")
        result_columns[2].metric("Dataset Average", f"${training_result.target_mean:,.2f}")

        premium_position = min(1.0, max(0.0, prediction / training_result.target_max))
        st.progress(premium_position, text=f"Relative to dataset maximum: {premium_position:.1%}")

        signals = quote_signals(policy)
        if signals:
            st.subheader("Quote Signals")
            for signal in signals:
                st.write(f"- {signal}")

    with st.expander("Responsible-use note"):
        st.write(
            "This model is a portfolio demonstration. Production insurance pricing requires approved rating logic, "
            "actuarial validation, compliance review, monitoring, audit logs, and human oversight."
        )


if __name__ == "__main__":
    main()
