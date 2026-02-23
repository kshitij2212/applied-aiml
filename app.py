import streamlit as st
import pandas as pd
from ml_pipeline import NoShowPredictor

st.set_page_config(page_title="No-Show Predictor", layout="wide")

if 'predictor' not in st.session_state:
    st.session_state.predictor = NoShowPredictor()

if 'df' not in st.session_state:
    st.session_state.df = None

if 'is_trained' not in st.session_state:
    st.session_state.trained = False

st.title("🏥 No-Show Predictor")

page = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Train Model", "Predict"]
)

if page == "Upload Data":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')
        st.success("File uploaded successfully!")
        st.dataframe(st.session_state.df.head())

elif page == "Train Model":
    st.header("Train Model")

    if st.session_state.df is None:
        st.warning("Pehle data upload karo!")
    else:
        model_name = st.selectbox(
            "Model choose karo",
            ["Logistic Regression", "Decision Tree", "Random Forest"]
        )

        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        threshold = st.slider("Threshold", 0.1, 0.9, 0.35)

        if st.button("Train"):
            with st.spinner("Training..."):
                results = st.session_state.predictor.train(
                    st.session_state.df,
                    model_name=model_name,
                    test_size=test_size,
                    threshold=threshold
                )
                st.session_state.trained = True
                st.session_state.results = results

elif page == "Predict":
    st.header("Patient Risk Prediction")

    if not st.session_state.trained:
        st.warning("Train the model first!")
    elif page == "Predict":
    st.header("Patient Risk Prediction")

    if not st.session_state.trained:
        st.warning("Train the model first!")
    else:
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 0, 115, 30)
            gender = st.selectbox("Gender", ["M", "F"])
            scholarship = st.selectbox("Scholarship", [0, 1])
            hipertension = st.selectbox("Hipertension", [0, 1]) 
            diabetes = st.selectbox("Diabetes", [0, 1])
            alcoholism = st.selectbox("Alcoholism", [0, 1])

        with col2:
            handcap = st.selectbox("Handcap", [0, 1, 2, 3, 4])
            sms = st.selectbox("SMS Received", [0, 1])
            lead_time = st.number_input("Lead Time (days)", 0, 365, 7)
            day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6])

        if st.button("Predict"):
            patient_data = {
                'Age': age,
                'Gender': gender,
                'Scholarship': scholarship,
                'Hypertension': hipertension,
                'Diabetes': diabetes,
                'Alcoholism': alcoholism,
                'Handcap': handcap,
                'SMS_received': sms,
                'LeadTime': lead_time,
                'DayOfWeek': day_of_week,
                'ScheduledDay': '2024-01-01T00:00:00Z',
                'AppointmentDay': '2024-01-01T00:00:00Z',
            }