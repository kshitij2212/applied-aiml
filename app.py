import streamlit as st
import pandas as pd
import plotly.express as px
from ml_pipeline import NoShowPredictor

st.set_page_config(page_title="No-Show Predictor", layout="wide")

if 'predictor' not in st.session_state:
    st.session_state.predictor = NoShowPredictor()

if 'df' not in st.session_state:
    st.session_state.df = None

if 'trained' not in st.session_state:
    st.session_state.trained = False

st.title(":hospital: No-Show Predictor")

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
            st.session_state.df = pd.read_csv(
                uploaded_file,
                encoding='latin-1',
                on_bad_lines='skip'
            )
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

        if st.session_state.trained:
            results = st.session_state.results

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{results['accuracy']:.2%}")
            col2.metric("Precision", f"{results['precision']:.2%}")
            col3.metric("Recall", f"{results['recall']:.2%}")
            col4.metric("F1 Score", f"{results['f1']:.2%}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Feature Importance")
                fi = pd.Series(results['feature_importance']).sort_values(ascending=False)
                st.bar_chart(fi)

            with col2:
                st.subheader("Confusion Matrix")
                cm = results['confusion_matrix']
                labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
                values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
                fig = px.pie(
                    names=labels,
                    values=values,
                    hole=0.4,
                    color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444', '#3B82F6']
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, width='stretch')

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
                'Hipertension': hipertension,
                'Diabetes': diabetes,
                'Alcoholism': alcoholism,
                'Handcap': handcap,
                'SMS_received': sms,
                'LeadTime': lead_time,
                'DayOfWeek': day_of_week,
                'IsWeekend': 1 if day_of_week >= 5 else 0,
                'ScheduledDay': '2024-01-01T00:00:00Z',
                'AppointmentDay': '2024-01-01T00:00:00Z',
            }

            result = st.session_state.predictor.predict(patient_data)
            prob = result['probability']
            prediction = result['prediction']
            threshold_used = result['threshold_used']

            st.subheader(f"No-Show Probability: {prob:.2%}")
            st.caption(f"Threshold used: {threshold_used}")

            if prediction == 1:
                if prob >= 0.5:
                    st.error("High Risk — Call Patient!")
                else:
                    st.warning("Medium Risk — Send SMS!")
            else:
                st.success("Low Risk — Patient will come!")