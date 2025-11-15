import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import shap
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

from groq import Groq


# ================================================================
# 0. LOAD DATA
# ================================================================
@st.cache_data
def load_credit_data():
    df = pd.read_csv("german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# ================================================================
# 1. TRAIN MODEL + SHAP EXPLAINER
# ================================================================
@st.cache_resource
def train_credit_model():
    df = load_credit_data().copy()

    df["default"] = df["Risk"].map({"good": 0, "bad": 1})
    df = df.drop(columns=["Risk"])

    if "Saving accounts" in df.columns:
        df["Saving accounts"] = df["Saving accounts"].fillna("none")
    if "Checking account" in df.columns:
        df["Checking account"] = df["Checking account"].fillna("none")

    X = df.drop(columns=["default"])
    y = df["default"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    feature_columns = X_encoded.columns.tolist()

    shap_explainer = shap.LinearExplainer(
        model,
        X_train,
        feature_perturbation="interventional"
    )

    return model, feature_columns, accuracy, shap_explainer


model, feature_columns, accuracy, shap_explainer = train_credit_model()


# ================================================================
# 2. GLOBAL UI SETTINGS – LIGHT PASTEL THEME ONLY
# ================================================================
st.set_page_config(
    page_title="Credit Risk Prediction Platform",
    layout="centered",
    page_icon="🏦",
)

# Custom CSS
st.markdown(
    """
    <style>
    body { background-color: #F9FAFB; }

    .block-container {
        padding-top: 1.2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-bottom: 3rem !important;
        background: linear-gradient(180deg, #F9FAFB 0%, #F5F3FF 60%, #F9FAFB 100%);
    }

    .card {
        background-color: #FFFFFF;
        padding: 1.2rem 1.4rem;
        border-radius: 1rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 8px 24px rgba(148, 163, 184, 0.18);
        margin-bottom: 1rem;
    }

    .pastel-header {
        height: 10px;
        border-radius: 999px;
        background: linear-gradient(90deg, #BFDBFE, #E0E7FF, #FBCFE8);
        margin-bottom: 1.1rem;
    }

    .small-label {
        font-size: 0.86rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #38BDF8, #A5B4FC);
        color: #FFFFFF;
        border-radius: 999px;
        height: 3rem;
        font-size: 1.05rem;
        font-weight: 500;
        border: none;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #0EA5E9, #818CF8);
        color: #FFFFFF;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }

    .stTabs [data-baseweb="tab"] {
        background-color: #E0F2FE;
        border-radius: 999px;
        padding-top: 6px;
        padding-bottom: 6px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #A5B4FC !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# 3. SIDEBAR
# ================================================================
st.sidebar.header("Model Overview")
st.sidebar.write(f"**Test Accuracy:** {accuracy:.2%}")
st.sidebar.write("**Target:** `default` (1 = bad risk, 0 = good risk)")
st.sidebar.info("Prototype built for educational purposes only.")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Credit scoring app", "Model dashboard", "About the project"],
)

groq_key = st.sidebar.text_input(
    "Groq API key (for AI assistant)",
    type="password",
)


# ================================================================
# 4. PAGE 1 – CREDIT SCORING APP
# ================================================================
if page == "Credit scoring app":

    st.title("Credit Risk Prediction Platform")
    st.caption(
        "Logistic Regression scoring model with SHAP explainability, PDF reporting, "
        "and a Generative AI risk analyst (Llama 3.1 via Groq)."
    )
    st.markdown("---")

    st.markdown("#### 1. Applicant Information")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="pastel-header"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job (0 = unskilled, 3 = highly skilled)", [0, 1, 2, 3], index=2)
            credit_amount = st.number_input("Credit Amount", 100, 100000, 5000, step=500)
            duration = st.number_input("Duration (months)", 4, 72, 24)

        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            housing = st.selectbox("Housing", ["own", "rent", "free"])
            saving_accounts = st.selectbox("Saving accounts", ["none", "little", "moderate", "quite rich", "rich"], 1)
            checking_account = st.selectbox("Checking account", ["none", "little", "moderate", "rich"], 1)
            purpose = st.selectbox(
                "Purpose",
                ["radio/TV", "education", "furniture/equipment", "car", "business",
                 "domestic appliances", "repairs", "vacation/others"],
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 2. Prediction & Decision")

    # Predict
    if st.button("Run Credit Risk Prediction", use_container_width=True):

        input_data = {
            "Age": [age],
            "Sex": [sex],
            "Job": [job],
            "Housing": [housing],
            "Saving accounts": [saving_accounts],
            "Checking account": [checking_account],
            "Credit amount": [credit_amount],
            "Duration": [duration],
            "Purpose": [purpose],
        }

        X_new = pd.DataFrame(input_data)
        X_new = pd.get_dummies(X_new, drop_first=True)
        X_new = X_new.reindex(columns=feature_columns, fill_value=0)

        prob_default = model.predict_proba(X_new)[0, 1]
        prob_default_percent = prob_default * 100

        if prob_default < 0.2:
            risk_label, color, decision = "Low Risk", "🟢", "ACCEPT"
            decision_comment = "Low risk profile. Application can be accepted."
        elif prob_default < 0.5:
            risk_label, color, decision = "Medium Risk", "🟡", "REVIEW"
            decision_comment = "Medium risk. Recommend manual review or guarantees."
        else:
            risk_label, color, decision = "High Risk", "🔴", "REJECT"
            decision_comment = "High predicted risk. Loan should be rejected."

        st.session_state["results"] = {
            "input_data": input_data,
            "X_new_encoded": X_new,
            "prob_default_percent": prob_default_percent,
            "risk_label": risk_label,
            "decision": decision,
            "decision_comment": decision_comment,
        }

    # Display results
    if "results" in st.session_state:
        r = st.session_state["results"]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="pastel-header"></div>', unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Probability of Default", f"{r['prob_default_percent']:.2f} %")
        colB.metric("Risk Level", r["risk_label"])
        colC.metric("Decision", r["decision"])

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='small-label'>{r['decision_comment']}</p>", unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["SHAP Explanations", "PDF Report", "AI Risk Analyst"])

        # TAB 1 - SHAP
        with tab1:
            st.subheader("Feature-level Explanation")
            shap_values = shap_explainer.shap_values(r["X_new_encoded"])
            sv = shap_values[0] if isinstance(shap_values, list) else shap_values

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(sv, r["X_new_encoded"], plot_type="bar", show=False)
            st.pyplot(fig)

        # TAB 2 - PDF
        with tab2:
            st.subheader("Generate PDF Report")

            buffer = BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)

            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, 750, "Credit Risk Scoring Report")

            pdf.setFont("Helvetica", 10)
            y = 720
            for k, v in r["input_data"].items():
                pdf.drawString(50, y, f"{k}: {v[0]}")
                y -= 15

            pdf.save()
            buffer.seek(0)

            st.download_button(
                "Download PDF",
                buffer,
                file_name="credit_report.pdf",
                mime="application/pdf",
            )

        # TAB 3 - AI Risk Analyst
        with tab3:
            st.subheader("AI-Generated Risk Analysis")

            instruction = st.text_area(
                "Optional: extra instructions for the AI",
                "Write a concise professional report for a credit committee.",
            )

            if not groq_key:
                st.warning("Add your Groq API key in the sidebar.")
            else:
                if st.button("Generate AI Report"):
                    try:
                        client = Groq(api_key=groq_key)

                        prompt = f"""
You are a senior credit risk analyst.

Predicted PD: {r['prob_default_percent']:.2f}%
Risk category: {r['risk_label']}
Decision: {r['decision']}
Comment: {r['decision_comment']}

Customer profile:
{r['input_data']}

User instructions:
{instruction}
                        """

                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{"role": "user", "content": prompt}],
                        )

                        st.write(response.choices[0].message.content)

                    except Exception as e:
                        st.error(f"Groq API Error: {e}")


# ================================================================
# 5. MODEL DASHBOARD
# ================================================================
elif page == "Model dashboard":

    st.title("Model Dashboard – German Credit Dataset")
    st.caption("Global view of the dataset and risk distribution.")
    st.markdown("---")

    df = load_credit_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Clients", len(df))
    col2.metric("Bad risk rate", f"{(df['Risk']=='bad').mean()*100:.2f}%")
    col3.metric("Avg Credit", f"{df['Credit amount'].mean():,.0f}")

    st.markdown("---")

    st.subheader("Risk Distribution")
    risk_counts = df["Risk"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(risk_counts.index, risk_counts.values,
           color=["#4ade80", "#f97373"])
    st.pyplot(fig)

    st.subheader("Distribution of Credit Amount")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(df["Credit amount"], bins=30, color="#60a5fa", edgecolor="white")
    st.pyplot(fig2)


# ================================================================
# 6. ABOUT PAGE
# ================================================================
elif page == "About the project":

    st.title("About this project")
    st.caption("Credit Risk Prediction Platform with Explainable AI and Generative AI.")
    st.markdown("---")

    st.markdown("""
    ## 1. Objective

    This project is part of the **Generative AI module**.  
    It demonstrates how AI can support decision-making in **credit risk assessment**,
    combining:
    - Machine learning  
    - Explainability (SHAP)  
    - Generative AI (Groq Llama 3.1)  
    """)

    st.markdown("""
    ## 2. Dataset & Model

    - Dataset: **German Credit Dataset**  
    - Target: `default` (0 = good, 1 = bad)  
    - Model: **Logistic Regression**  
    - Explainability: **SHAP values**  
    """)

    st.markdown("""
    ## 3. Application Features

    - Credit scoring  
    - SHAP explanations  
    - PDF report generation  
    - AI-generated risk report  
    - Interactive dashboard  
    """)

    st.markdown("""
    ## 4. Limitations & Future Improvements

    - Only one simple model  
    - Small dataset  
    - No fairness analysis  

    Possible extensions:
    - Add more ML models  
    - Add drift detection  
    - Deploy on cloud  
    """)

    st.success("This section provides a structured overview of the project.")
