import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BidTrust AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        color: #dbeafe;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        text-align: center;
    }

    .pass-box {
        padding: 25px;
        border-radius: 15px;
        background: #dcfce7;
        border: 2px solid #22c55e;
        text-align: center;
    }

    .fail-box {
        padding: 25px;
        border-radius: 15px;
        background: #fee2e2;
        border: 2px solid #ef4444;
        text-align: center;
    }

    .warning-box {
        padding: 20px;
        border-radius: 12px;
        background: #fef3c7;
        border: 1px solid #f59e0b;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "best_tender_classifier.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # Optional because Gradient Boosting doesn't require scaling
    scaler = None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <h1>🛡️ BidTrust AI</h1>

    <p>
        AI-Powered GeM Tender Bid Compliance & Risk Assessment Platform
    </p>

    <p>
        Analyze financial, technical, legal and documentation compliance
        before submitting your tender bid.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ BidTrust AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Bid Compliance",
            "📊 Dataset Explorer",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.caption("AI-powered tender compliance analysis")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">📊 Compliance Dashboard</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Model",
            "Gradient Boosting"
        )

    with col2:
        st.metric(
            "🎯 Accuracy",
            "95.9%"
        )

    with col3:
        st.metric(
            "📈 F1 Score",
            "94.8%"
        )

    with col4:
        st.metric(
            "📋 Dataset",
            "5,000 Bids"
        )

    st.markdown("---")

    st.subheader("How BidTrust AI Works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        ### 1️⃣ Enter Bid Data

        Provide:

        - Company turnover
        - Required turnover
        - Experience
        - GST/PAN/Udyam
        - OEM authorization
        - Technical score
        - Financial compliance
        """)

    with c2:
        st.markdown("""
        ### 2️⃣ AI Analysis

        BidTrust AI evaluates:

        - Financial eligibility
        - Technical compliance
        - Documentation
        - Local content
        - Certificates
        - Blacklist status
        """)

    with c3:
        st.markdown("""
        ### 3️⃣ Get Decision

        The AI generates:

        - PASS / FAIL
        - PASS probability
        - Compliance score
        - Risk indicators
        - Compliance recommendations
        """)


# ============================================================
# BID COMPLIANCE
# ============================================================

elif page == "🔍 Bid Compliance":

    st.markdown(
        '<div class="section-title">🔍 Tender Compliance Assessment</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter the tender and bidder information below. "
        "Derived compliance margins will be calculated automatically."
    )

    # --------------------------------------------------------
    # TENDER ID
    # --------------------------------------------------------

    tender_id = st.text_input(
        "Tender ID",
        value="TENDER-001"
    )

    # --------------------------------------------------------
    # FINANCIAL INFORMATION
    # --------------------------------------------------------

    st.subheader("💰 Financial Compliance")

    col1, col2 = st.columns(2)

    with col1:
        turnover = st.number_input(
            "Company Turnover (₹ Crore)",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )

    with col2:
        required_turnover = st.number_input(
            "Required Turnover (₹ Crore)",
            min_value=0.0,
            max_value=1000.0,
            value=5.0,
            step=0.1
        )

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    st.subheader("🏢 Experience & Eligibility")

    col1, col2 = st.columns(2)

    with col1:
        experience = st.number_input(
            "Company Experience (Years)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1
        )

    with col2:
        required_experience = st.number_input(
            "Required Experience (Years)",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.1
        )

    # --------------------------------------------------------
    # DOCUMENT VALIDATION
    # --------------------------------------------------------

    st.subheader("📄 Legal & Document Compliance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gst_valid = st.selectbox(
            "GST Valid",
            ["Yes", "No"]
        )

    with col2:
        pan_valid = st.selectbox(
            "PAN Valid",
            ["Yes", "No"]
        )

    with col3:
        udyam_valid = st.selectbox(
            "Udyam Valid",
            ["Yes", "No"]
        )

    with col4:
        oem_authorization = st.selectbox(
            "OEM Authorization",
            ["Yes", "No"]
        )

    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    st.subheader("⚙️ Technical Compliance")

    col1, col2 = st.columns(2)

    with col1:
        technical_score = st.slider(
            "Technical Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.1
        )

    with col2:
        minimum_technical_score = st.number_input(
            "Minimum Required Technical Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

    # --------------------------------------------------------
    # COMPLIANCE
    # --------------------------------------------------------

    st.subheader("📋 Compliance Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        technical_compliance = st.selectbox(
            "Technical Compliance",
            ["Compliant", "Not Compliant"]
        )

    with col2:
        financial_compliance = st.selectbox(
            "Financial Compliance",
            ["Compliant", "Not Compliant"]
        )

    with col3:
        eligibility_compliance = st.selectbox(
            "Eligibility Compliance",
            ["Compliant", "Not Compliant"]
        )

    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    st.subheader("📁 Mandatory Documents")

    col1, col2 = st.columns(2)

    with col1:
        mandatory_documents_complete = st.selectbox(
            "Mandatory Documents Complete",
            ["Yes", "No"]
        )

    with col2:
        missing_documents = st.number_input(
            "Number of Missing Documents",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

    # --------------------------------------------------------
    # LOCAL CONTENT
    # --------------------------------------------------------

    st.subheader("🇮🇳 Local Content")

    col1, col2 = st.columns(2)

    with col1:
        local_content = st.number_input(
            "Local Content (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1
        )

    with col2:
        required_local_content = st.number_input(
            "Required Local Content (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    # --------------------------------------------------------
    # CERTIFICATE / BLACKLIST
    # --------------------------------------------------------

    st.subheader("🛡️ Certificate & Legal Status")

    col1, col2 = st.columns(2)

    with col1:
        certificate_validity = st.selectbox(
            "Certificate Validity",
            ["Valid", "Expired"]
        )

    with col2:
        blacklist_clear = st.selectbox(
            "Blacklist Status",
            ["Clear", "Blacklisted"]
        )

    # --------------------------------------------------------
    # COMPLIANCE SCORE
    # --------------------------------------------------------

    st.subheader("📊 Existing Compliance Score")

    compliance_score = st.slider(
        "Compliance Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.1
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    st.markdown("---")

    predict_button = st.button(
        "🚀 Analyze Bid Compliance",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        # ----------------------------------------------------
        # Convert categorical values to 0/1
        # ----------------------------------------------------

        record = {
            "turnover_crore": turnover,
            "required_turnover_crore": required_turnover,
            "experience_years": experience,
            "required_experience_years": required_experience,

            "gst_valid": 1 if gst_valid == "Yes" else 0,
            "pan_valid": 1 if pan_valid == "Yes" else 0,
            "udyam_valid": 1 if udyam_valid == "Yes" else 0,
            "oem_authorization": 1 if oem_authorization == "Yes" else 0,

            "technical_score": technical_score,
            "minimum_technical_score": minimum_technical_score,

            "technical_compliance":
                1 if technical_compliance == "Compliant" else 0,

            "financial_compliance":
                1 if financial_compliance == "Compliant" else 0,

            "eligibility_compliance":
                1 if eligibility_compliance == "Compliant" else 0,

            "mandatory_documents_complete":
                1 if mandatory_documents_complete == "Yes" else 0,

            "missing_documents": missing_documents,

            "local_content_percent": local_content,
            "required_local_content_percent": required_local_content,

            "certificate_validity":
                1 if certificate_validity == "Valid" else 0,

            "blacklist_clear":
                1 if blacklist_clear == "Clear" else 0,

            "compliance_score": compliance_score
        }

        # ----------------------------------------------------
        # Derived features
        # Same feature engineering as notebook
        # ----------------------------------------------------

        row = pd.DataFrame([record])

        row["turnover_margin"] = (
            row["turnover_crore"]
            - row["required_turnover_crore"]
        )

        row["experience_margin"] = (
            row["experience_years"]
            - row["required_experience_years"]
        )

        row["technical_margin"] = (
            row["technical_score"]
            - row["minimum_technical_score"]
        )

        row["local_content_margin"] = (
            row["local_content_percent"]
            - row["required_local_content_percent"]
        )

        # ----------------------------------------------------
        # Exact feature order from training
        # ----------------------------------------------------

        feature_names = [
            "turnover_crore",
            "required_turnover_crore",
            "experience_years",
            "required_experience_years",
            "gst_valid",
            "pan_valid",
            "udyam_valid",
            "oem_authorization",
            "technical_score",
            "minimum_technical_score",
            "technical_compliance",
            "financial_compliance",
            "eligibility_compliance",
            "mandatory_documents_complete",
            "missing_documents",
            "local_content_percent",
            "required_local_content_percent",
            "certificate_validity",
            "blacklist_clear",
            "compliance_score",
            "turnover_margin",
            "experience_margin",
            "technical_margin",
            "local_content_margin"
        ]

        row = row[feature_names]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(row)[0]

        probability = model.predict_proba(row)[0]

        predicted_label = label_encoder.inverse_transform(
            [prediction]
        )[0]

        # Probability of PASS
        pass_index = list(label_encoder.classes_).index("PASS")
        pass_probability = probability[pass_index]

        fail_probability = 1 - pass_probability

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("🎯 AI Assessment Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Result",
                predicted_label
            )

        with col2:
            st.metric(
                "PASS Probability",
                f"{pass_probability * 100:.1f}%"
            )

        with col3:
            st.metric(
                "FAIL Probability",
                f"{fail_probability * 100:.1f}%"
            )

        if predicted_label == "PASS":

            st.markdown(
                f"""
                <div class="pass-box">
                    <h2>✅ BID LIKELY TO PASS</h2>
                    <h3>Confidence: {pass_probability * 100:.1f}%</h3>
                    <p>
                    The submitted bid satisfies the learned compliance
                    pattern according to the trained classifier.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="fail-box">
                    <h2>❌ BID LIKELY TO FAIL</h2>
                    <h3>FAIL Probability: {fail_probability * 100:.1f}%</h3>
                    <p>
                    Potential compliance issues were detected.
                    Review the failed requirements before submission.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # MARGINS
        # ====================================================

        st.markdown("---")

        st.subheader("🔎 Compliance Gap Analysis")

        gap1, gap2, gap3, gap4 = st.columns(4)

        turnover_margin = (
            turnover - required_turnover
        )

        experience_margin = (
            experience - required_experience
        )

        technical_margin = (
            technical_score - minimum_technical_score
        )

        local_content_margin = (
            local_content - required_local_content
        )

        with gap1:
            st.metric(
                "Turnover Margin",
                f"{turnover_margin:.2f} Cr"
            )

        with gap2:
            st.metric(
                "Experience Margin",
                f"{experience_margin:.2f} Years"
            )

        with gap3:
            st.metric(
                "Technical Margin",
                f"{technical_margin:.2f}"
            )

        with gap4:
            st.metric(
                "Local Content Margin",
                f"{local_content_margin:.2f}%"
            )

        # ====================================================
        # RISK FLAGS
        # ====================================================

        st.markdown("---")

        st.subheader("⚠️ Risk Indicators")

        risks = []

        if turnover_margin < 0:
            risks.append(
                "Company turnover is below the required turnover."
            )

        if experience_margin < 0:
            risks.append(
                "Company experience is below the required experience."
            )

        if technical_margin < 0:
            risks.append(
                "Technical score is below the minimum requirement."
            )

        if local_content_margin < 0:
            risks.append(
                "Local content is below the required percentage."
            )

        if missing_documents > 0:
            risks.append(
                f"{missing_documents} mandatory document(s) are missing."
            )

        if gst_valid == "No":
            risks.append("GST validation failed.")

        if pan_valid == "No":
            risks.append("PAN validation failed.")

        if udyam_valid == "No":
            risks.append("Udyam validation failed.")

        if oem_authorization == "No":
            risks.append("OEM authorization is unavailable.")

        if certificate_validity == "Expired":
            risks.append("Certificate is expired.")

        if blacklist_clear == "Blacklisted":
            risks.append("Blacklist check failed.")

        if not risks:

            st.success(
                "✅ No major rule-based risk indicators were detected."
            )

        else:

            for risk in risks:
                st.warning("⚠️ " + risk)


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.markdown(
        '<div class="section-title">📊 GeM Bid Dataset Explorer</div>',
        unsafe_allow_html=True
    )

    dataset_path = "gem_bid_compliance_dataset_5000.csv"

    if os.path.exists(dataset_path):

        df = pd.read_csv(dataset_path)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Bids",
                len(df)
            )

        with col2:
            pass_count = (
                df["result"] == "PASS"
            ).sum()

            st.metric(
                "PASS Bids",
                pass_count
            )

        with col3:
            fail_count = (
                df["result"] == "FAIL"
            ).sum()

            st.metric(
                "FAIL Bids",
                fail_count
            )

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        st.subheader("Result Distribution")

        result_counts = df["result"].value_counts()

        st.bar_chart(result_counts)

    else:

        st.warning(
            "Dataset file not found. "
            "Place gem_bid_compliance_dataset_5000.csv "
            "in the same directory as app.py."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">ℹ️ About BidTrust AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ## 🛡️ BidTrust AI

    BidTrust AI is an AI-powered tender compliance assessment
    platform designed to help organizations evaluate whether a
    GeM tender bid is likely to pass or fail compliance checks.

    ### Core Modules

    **💰 Financial Compliance**
    - Company turnover
    - Required turnover

    **🏢 Experience & Eligibility**
    - Company experience
    - Required experience

    **📄 Document Compliance**
    - GST
    - PAN
    - Udyam
    - OEM authorization
    - Mandatory documents

    **⚙️ Technical Compliance**
    - Technical score
    - Minimum technical score

    **🇮🇳 Local Content**
    - Local content percentage
    - Required local content

    **🛡️ Legal Checks**
    - Certificate validity
    - Blacklist status

    ### Machine Learning

    The classification system compares:

    - Logistic Regression
    - Random Forest
    - Gradient Boosting

    The trained Gradient Boosting model is used for the final
    prediction based on F1-score.
    """)

    st.success(
        "BidTrust AI — Making tender compliance faster, smarter and more transparent."
    )