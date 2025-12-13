import streamlit as st
import pandas as pd
import joblib

# --- 1. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_data():
    return joblib.load('insurance_model.pkl')

data = load_data()
model = data["model"]
encoders = data["encoders"]
feature_names = data["columns"]

# --- 2. THE CHAT INTERFACE ---
st.set_page_config(page_title="Auto Insurance Bot", page_icon="🚗")

st.title("🚗 Insurance Price Predictor")
st.write("Chat with the AI to get your estimated monthly premium.")

# We use a Form to simulate a structured "conversation"
with st.form("prediction_form"):
    st.header("Tell me about yourself:")
    
    # SPLIT LAYOUT (2 Columns)
    col1, col2 = st.columns(2)
    
    with col1:
        # NUMERIC INPUTS
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        months_policy = st.number_input("Months Since Policy Inception", min_value=0, value=12)
        months_claim = st.number_input("Months Since Last Claim", min_value=0, value=6)
        policies = st.number_input("Number of Policies", min_value=1, value=1)
        complaints = st.number_input("Number of Open Complaints", min_value=0, value=0)

    with col2:
        # DROPDOWNS (Using the saved encoders to get options)
        state = st.selectbox("State", encoders['State'].classes_)
        coverage = st.selectbox("Coverage", encoders['Coverage'].classes_)
        education = st.selectbox("Education", encoders['Education'].classes_)
        employment = st.selectbox("Employment Status", encoders['EmploymentStatus'].classes_)
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        location = st.selectbox("Location Code", encoders['Location Code'].classes_)
        marital = st.selectbox("Marital Status", encoders['Marital Status'].classes_)
        vehicle_class = st.selectbox("Vehicle Class", encoders['Vehicle Class'].classes_)
        vehicle_size = st.selectbox("Vehicle Size", encoders['Vehicle Size'].classes_)
        
        # Hardcoding less important ones to default 
        policy_type = encoders['Policy Type'].classes_[0]
        policy = encoders['Policy'].classes_[0]
        renew_offer = encoders['Renew Offer Type'].classes_[0]
        sales_channel = encoders['Sales Channel'].classes_[0]
        response = encoders['Response'].classes_[0]

    # SUBMIT BUTTON
    submitted = st.form_submit_button("💰 Get Quote")

    if submitted:
        # --- 3. PROCESS DATA ---
        # We must convert the Text inputs back to Numbers using the loaded Encoders
        input_data = {
            'State': encoders['State'].transform([state])[0],
            'Response': encoders['Response'].transform([response])[0],
            'Coverage': encoders['Coverage'].transform([coverage])[0],
            'Education': encoders['Education'].transform([education])[0],
            'EmploymentStatus': encoders['EmploymentStatus'].transform([employment])[0],
            'Gender': encoders['Gender'].transform([gender])[0],
            'Income': income,
            'Location Code': encoders['Location Code'].transform([location])[0],
            'Marital Status': encoders['Marital Status'].transform([marital])[0],
            'Months Since Last Claim': months_claim,
            'Months Since Policy Inception': months_policy,
            'Number of Open Complaints': complaints,
            'Number of Policies': policies,
            'Policy Type': encoders['Policy Type'].transform([policy_type])[0],
            'Policy': encoders['Policy'].transform([policy])[0],
            'Renew Offer Type': encoders['Renew Offer Type'].transform([renew_offer])[0],
            'Sales Channel': encoders['Sales Channel'].transform([sales_channel])[0],
            'Vehicle Class': encoders['Vehicle Class'].transform([vehicle_class])[0],
            'Vehicle Size': encoders['Vehicle Size'].transform([vehicle_size])[0]
        }

        # Create a DataFrame with the exact same column order as training
        df_input = pd.DataFrame([input_data])
        df_input = df_input[feature_names] # Reorder columns to match model

        # --- 4. PREDICT ---
        prediction = model.predict(df_input)[0]

        st.success(f"### 🤖 Estimated Premium: ${prediction:.2f} / month")