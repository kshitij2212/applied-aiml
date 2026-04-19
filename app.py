import streamlit as st
st.write("App is starting...")

import warnings

warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
import logging
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.express as px
from ml_pipeline import NoShowPredictor
try:
    from agent_logic import run_agent
except Exception as e:
    run_agent = None
    print("Agent import failed:", e)
from pdf_generator import generate_pdf

st.set_page_config(
    page_title="ClinIQ — No-Show Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

import os

if os.path.exists("style.css"):
    local_css("style.css")
else:
    st.warning("style.css not found")

@st.cache_resource
def load_predictor():
    return NoShowPredictor()

if 'predictor' not in st.session_state:
    st.session_state.predictor = None

if st.session_state.predictor is None:
    st.session_state.predictor = load_predictor()
if 'df'           not in st.session_state: st.session_state.df           = None
if 'trained'      not in st.session_state: st.session_state.trained      = bool(st.session_state.predictor.trained)
if 'agent_result' not in st.session_state: st.session_state.agent_result = None
if 'last_prob'    not in st.session_state: st.session_state.last_prob    = 0.0
if 'last_patient' not in st.session_state: st.session_state.last_patient = {}
if 'results'      not in st.session_state: st.session_state.results      = None
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div>
            <div class="sidebar-brand-text">ClinIQ</div>
            <div class="sidebar-brand-sub">Care Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Upload Data", "Train Model", "Predict & Assist"],
        index=2,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    model_status = "Ready with Pretrained Model" if st.session_state.trained else "Not Trained"
    data_status  = f"{len(st.session_state.df)} rows" if st.session_state.df is not None else "No Manual Dataset"

    st.markdown(f"""
    <div style="background:#0C1118;border:1px solid #1E2A38;border-radius:10px;padding:14px 16px;">
        <div style="color:#4A6580;font-size:10px;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">System Status</div>
        <div style="color:#8FA3B8;font-size:12px;margin-bottom:6px;">{model_status}</div>
        <div style="color:#8FA3B8;font-size:12px;">{data_status}</div>
    </div>
    """, unsafe_allow_html=True)

if page == "Upload Data":
    st.markdown("""
    <div class="page-header">
        <div class="badge">STEP 1</div>
        <h1>Upload Dataset</h1>
        <p>Import your appointment CSV to train or evaluate the no-show prediction model.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            st.session_state.df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')

        st.success(f"Dataset loaded — {len(st.session_state.df):,} rows × {len(st.session_state.df.columns)} columns")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Records",  f"{len(st.session_state.df):,}")
        col_b.metric("Features",       f"{len(st.session_state.df.columns)}")
        col_c.metric("Missing Values", f"{st.session_state.df.isnull().sum().sum():,}")

        st.markdown("<div class='section-title' style='margin-top:20px'>Preview</div>", unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head(8), use_container_width=True)

        if st.button("Quick Retrain on This Dataset"):
            with st.spinner("Retraining model…"):
                try:
                    results = st.session_state.predictor.train(
                        st.session_state.df,
                        model_name='Logistic Regression',
                        test_size=0.2,
                        threshold=st.session_state.predictor.threshold
                    )
                    st.session_state.trained  = True
                    st.session_state.results  = results
                    st.success("Model retrained successfully.")
                except Exception as e:
                    st.error(f"Retrain failed: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:48px 24px;color:#2A3D55;">
            <div style="font-size:40px;margin-bottom:12px;">📁</div>
            <div style="color:#4A6580;font-size:14px;">No file selected yet</div>
            <div style="color:#2A3D55;font-size:12px;margin-top:6px;">Supports UTF-8 and Latin-1 encoded CSVs</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "Train Model":
    st.markdown("""
    <div class="page-header">
        <div class="badge">STEP 2</div>
        <h1>Train Model</h1>
        <p>Configure and train the no-show prediction classifier on your dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("No dataset uploaded — using the bundled pretrained model. Upload data to retrain.")
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            model_name = st.selectbox("Algorithm", ["Logistic Regression", "Decision Tree", "Random Forest"])
        with col_cfg2:
            test_size_pct = st.slider("Test Split", 10, 40, 20, 5, format="%d%%",
                                      help="Percentage of data held out for evaluation")
            test_size = test_size_pct / 100.0
        with col_cfg3:
            threshold  = st.slider("Decision Threshold", 0.1, 0.9, 0.35, 0.05, format="%.2f",
                                   help="Probability cutoff for positive prediction")

        if st.button("Train Model"):
            with st.spinner("Training…"):
                results = st.session_state.predictor.train(
                    st.session_state.df,
                    model_name=model_name,
                    test_size=test_size,
                    threshold=threshold
                )
                st.session_state.trained = True
                st.session_state.results = results

    if st.session_state.trained and st.session_state.results:
        r = st.session_state.results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card success">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{r['accuracy']:.1%}</div>
            </div>
            <div class="metric-card info">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{r['precision']:.1%}</div>
            </div>
            <div class="metric-card warn">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{r['recall']:.1%}</div>
            </div>
            <div class="metric-card danger">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{r['f1']:.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-card'><div class='section-title'>Feature Importance</div>", unsafe_allow_html=True)
            fi  = pd.Series(r['feature_importance']).sort_values(ascending=True)
            fig = px.bar(
                fi, orientation='h',
                color_discrete_sequence=['#1FD6A0'],
                template='plotly_dark',
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(gridcolor='#1E2A38', tickfont=dict(size=11, color='#4A6580')),
                yaxis=dict(gridcolor='#1E2A38', tickfont=dict(size=11, color='#8FA3B8')),
                showlegend=False,
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-card'><div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
            cm     = r['confusion_matrix']
            labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
            values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
            fig2   = px.pie(
                names=labels, values=values, hole=0.5,
                color_discrete_sequence=['#1FD6A0', '#FBB724', '#F87171', '#38BDF8'],
                template='plotly_dark',
            )
            fig2.update_traces(textfont_size=12, textfont_color='#E8F0F7')
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(font=dict(size=11, color='#8FA3B8'), bgcolor='rgba(0,0,0,0)'),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Predict & Assist":
    st.markdown("""
    <div class="page-header">
        <div class="badge">STEP 3</div>
        <h1>Patient Risk Assessment</h1>
        <p>Enter patient details to predict no-show probability and generate an AI care strategy.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("Model not trained yet. Upload a dataset or ensure the default dataset is available.")

    if st.session_state.trained:
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

        with col1:
            st.markdown("<div class='section-title'>Demographics</div>", unsafe_allow_html=True)
            age        = st.number_input("Age", 0, 115, 30)
            gender     = st.selectbox("Gender", ["M", "F"])
            scholarship = st.selectbox("Scholarship", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            st.markdown("<div class='section-title'>Clinical Profile</div>", unsafe_allow_html=True)
            hipertension = st.selectbox("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x else "No")
            diabetes     = st.selectbox("Diabetes",      [0, 1], format_func=lambda x: "Yes" if x else "No")
            alcoholism   = st.selectbox("Alcoholism",    [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col3:
            st.markdown("<div class='section-title'>Accessibility & Timing</div>", unsafe_allow_html=True)
            handcap = st.selectbox("Disability Level", [0, 1, 2, 3, 4])
            sms     = st.selectbox("SMS Reminder", [0, 1], format_func=lambda x: "Sent" if x else "Not Sent")
            lead_time    = st.number_input("Lead Time (days)", 0, 365, 7)

        col_day = st.columns([1])[0]
        with col_day:
            day_of_week  = st.select_slider("Appointment Day", options=[0,1,2,3,4,5,6], value=2,
                                            format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", use_container_width=True)

        if predict_btn:
            patient_data = {
                'Age': age, 'Gender': gender, 'Scholarship': scholarship,
                'Hipertension': hipertension, 'Diabetes': diabetes, 'Alcoholism': alcoholism,
                'Handcap': handcap, 'SMS_received': sms, 'LeadTime': lead_time,
                'DayOfWeek': day_of_week, 'IsWeekend': 1 if day_of_week >= 5 else 0,
                'ScheduledDay': '2024-01-01T00:00:00Z',
                'AppointmentDay': '2024-01-01T00:00:00Z',
            }

            result       = st.session_state.predictor.predict(patient_data)
            prob         = result['probability']
            prediction   = result['prediction']
            threshold_used = result['threshold_used']

            st.session_state.last_prob    = prob
            st.session_state.last_patient = patient_data

            if prediction == 1:
                if prob >= 0.5:
                    risk_cls    = "high"
                    risk_label  = "HIGH RISK"
                    risk_action = "Recommend immediate phone call to patient"
                else:
                    risk_cls    = "medium"
                    risk_label  = "MEDIUM RISK"
                    risk_action = "Send SMS reminder before appointment"
            else:
                risk_cls    = "low"
                risk_label  = "LOW RISK"
                risk_action = "Patient likely to attend — standard follow-up"

            st.markdown(f"""
            <div class="risk-panel {risk_cls}">
                <div class="risk-label">{risk_label}</div>
                <div class="risk-prob">{prob:.1%}</div>
                <div class="risk-action">{risk_action}</div>
                <div style="color:#4A6580;font-size:11px;margin-top:8px;">Threshold: {threshold_used}</div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()
        
            st.markdown("""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
                <div style="font-size:18px;font-weight:700;letter-spacing:-0.01em;">Agentic Care Strategy</div>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Agent is reasoning over clinical guidelines and patient history..."):
                try:
                    agent_response = run_agent(patient_data, result)
                    st.session_state.agent_result = agent_response['final_report']
                    st.success("Care strategy generated successfully")
                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.info("Ensure GROQ_API_KEY is configured in your .env file.")
    if st.session_state.agent_result:
        report = st.session_state.agent_result

        def fmt(obj):
            if isinstance(obj, dict):
                return "\n\n".join([f"**{str(k).replace('_',' ').title()}**: {fmt(v)}" for k, v in obj.items()])
            elif isinstance(obj, list):
                return "\n".join([f"- {fmt(v)}" for v in obj])
            return str(obj)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Risk Summary & Narrative", expanded=True):
            st.markdown(fmt(report.get('summary', 'No summary available.')))

        with st.expander("Key Contributing Factors"):
            st.markdown(fmt(report.get('factors', [])))

        with st.expander("Recommended Clinical Interventions", expanded=True):
            st.markdown(fmt(report.get('strategies', [])))

        st.markdown("<br>", unsafe_allow_html=True)
        
        col_act, col_eth = st.columns([1, 1], gap="large")
        
        with col_act:
            st.subheader("Download Report")
            try:
                pdf_data = generate_pdf(report, st.session_state.last_patient, st.session_state.last_prob)
                st.download_button(
                    label="Download Full Clinical PDF",
                    data=pdf_data,
                    file_name=f"care_report_{age}_{gender}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Generation Error: {e}")

        with col_eth:
            with st.expander("Medical Sources & Ethical Disclaimers", expanded=True):
                st.markdown(f"**Sources:**\n{fmt(report.get('sources', []))}")
                st.divider()
                st.warning(fmt(report.get('disclaimers', 'Use with clinical caution. All AI recommendations must be verified by a medical professional.')))