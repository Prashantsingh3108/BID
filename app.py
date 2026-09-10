
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GeM-AI Integrated Bid Compliance",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS
# ============================================================
st.markdown(
    """
<style>
    /* ---------- Global ---------- */
    .stApp {
        background: #060b17;
        color: #e8eefc;
    }

    .main .block-container {
        max-width: 1700px;
        padding: 0.8rem 1.5rem 2rem 1.5rem;
    }

    [data-testid="stSidebar"] {
        background: #0b132b;
        border-right: 1px solid #263457;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] * {
        color: #cbd6f6;
    }

    /* ---------- Top SIH banner ---------- */
    .top-banner {
        background: #3a1707;
        border: 1px solid #6d3510;
        color: #ffb12b;
        padding: 8px 15px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .4px;
        margin-bottom: 10px;
    }

    .top-banner-right {
        float: right;
        color: #4fe09b;
    }

    /* ---------- Header ---------- */
    .header-card {
        background: #0b132b;
        border: 1px solid #1d2a49;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 10px;
    }

    .brand {
        font-size: 23px;
        font-weight: 800;
        color: #f4f7ff;
    }

    .brand-sub {
        color: #8191b6;
        font-size: 12px;
        margin-top: 2px;
    }

    .header-pill {
        background: #16213c;
        border: 1px solid #2b3a61;
        border-radius: 9px;
        padding: 9px 13px;
        color: #dce6ff;
        font-size: 12px;
    }

    /* ---------- Quick selector ---------- */
    .selector {
        background: #0a1122;
        border-top: 1px solid #1a2745;
        border-bottom: 1px solid #1a2745;
        padding: 10px 0;
        margin: 0 -1.5rem 14px -1.5rem;
    }

    .selector-title {
        color: #b7c5e5;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .bid-chip {
        padding: 8px 13px;
        border-radius: 7px;
        background: #17233e;
        border: 1px solid #2a3a60;
        font-size: 12px;
        font-weight: 700;
    }

    .bid-green { color: #46dda0; }
    .bid-amber { color: #ffb22e; border-color: #d87c08; }
    .bid-red { color: #ff6666; }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #111c38 0%, #0d1730 100%);
        border: 1px solid #1d2b4c;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 20px;
        font-weight: 800;
        color: #f4f7ff;
    }

    .hero-sub {
        color: #7f8fb4;
        font-size: 12px;
        margin-top: 4px;
    }

    .live {
        display: inline-block;
        margin-left: 7px;
        padding: 3px 8px;
        border-radius: 5px;
        background: #083c3b;
        border: 1px solid #0c6b62;
        color: #41d7a0;
        font-size: 10px;
        font-weight: 800;
        vertical-align: middle;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: #0f192f;
        border: 1px solid #202e4e;
        border-radius: 12px;
        padding: 15px;
        min-height: 112px;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #7f8fb3;
        font-size: 10px;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: .5px;
    }

    .metric-value {
        color: #f0f5ff;
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-foot {
        color: #7182a8;
        font-size: 10px;
        margin-top: 5px;
    }

    .green { color: #35d391 !important; }
    .amber { color: #ffb21d !important; }
    .red { color: #ff6666 !important; }
    .blue { color: #5790ff !important; }
    .purple { color: #8c7cff !important; }

    /* ---------- Panels ---------- */
    .panel {
        background: #0e172c;
        border: 1px solid #202e4e;
        border-radius: 12px;
        padding: 16px;
        min-height: 315px;
    }

    .panel-title {
        color: #eef3ff;
        font-weight: 800;
        font-size: 15px;
    }

    .panel-sub {
        float: right;
        color: #7485aa;
        font-size: 10px;
        font-weight: 500;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 800;
        border: 1px solid #2a59a7;
    }

    /* ---------- Inputs ---------- */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: #111c38;
        border-color: #2a3a60;
    }

    label, .stSlider label {
        color: #b8c5e4 !important;
    }

    /* ---------- Status boxes ---------- */
    .status-pass {
        background: #08352a;
        border: 1px solid #1aa873;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .status-fail {
        background: #3a1116;
        border: 1px solid #d54855;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .status-title {
        font-size: 21px;
        font-weight: 900;
    }

    .small-muted {
        color: #7182a8;
        font-size: 11px;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_artifact(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


model = load_artifact("best_tender_classifier.pkl")
label_encoder = load_artifact("label_encoder.pkl")
scaler = load_artifact("scaler.pkl")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-size:22px;font-weight:900;color:#f4f7ff;">
            🏛️ GeM-AI
        </div>
        <div class="brand-sub">Integrated Bid Compliance Hub</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "CORE DASHBOARD",
        [
            "🏠 Main Dashboard",
            "📑 Tender Management",
            "📋 Tender Requirements",
            "👥 Bidder Management",
            "📤 Upload Documents",
            "🤖 AI Document Analysis",
            "⚙️ Compliance Engine",
            "⚠️ Risk Assessment",
            "🔎 Evidence Explorer",
            "🚨 Discrepancy Center",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="small-muted">
        <b>AI PROCESSING & PIPELINE</b><br><br>
        Document OCR → Requirement Extraction → Rule Engine →
        ML Risk Classification → Human Review → Decision
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("SIH Prototype • Synthetic Demo Data")


# ============================================================
# TOP HEADER
# ============================================================
st.markdown(
    """
    <div class="top-banner">
        ⚠️ SIH PROTOTYPE DEMO MODE
        &nbsp; | &nbsp; SYNTHETIC DATA — FOR SIH PROTOTYPE ONLY
        &nbsp; | &nbsp; DEMO DOCUMENT — NOT A GOVERNMENT DOCUMENT
        <span class="top-banner-right">🛡 Human-in-the-Loop Procurement Officer Mandatory Authorization Active</span>
    </div>
    """,
    unsafe_allow_html=True,
)

h1, h2, h3, h4 = st.columns([2.8, 1.1, 1.5, 1.0])

with h1:
    st.markdown(
        """
        <div class="header-card">
            <div class="brand">🏛️ GeM-AI Compliance Hub
                <span class="live">SIH PROBLEM 26100</span>
            </div>
            <div class="brand-sub">
                Government e-Marketplace Integrated Bid Compliance Verification Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        """
        <div class="header-card">
            <div class="metric-label">ACTIVE TENDER</div>
            <div style="font-weight:800;color:#eaf0ff;margin-top:6px;">CPCL/DEMO/2026/001</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        """
        <div class="header-card">
            <div class="metric-label">EVALUATION DATE</div>
            <div style="font-weight:800;color:#eaf0ff;margin-top:6px;">09 Sep 2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h4:
    st.markdown(
        """
        <div class="header-card">
            <div style="font-weight:800;color:#eaf0ff;">🔵 SR</div>
            <div class="brand-sub">Senior Procurement Officer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# QUICK DEMO SELECTOR
# ============================================================
st.markdown(
    """
    <div class="selector">
        <span class="selector-title">✨ SIH Judge Quick Demo Selector</span>
    </div>
    """,
    unsafe_allow_html=True,
)

q1, q2, q3, q4 = st.columns([1.4, 1.4, 1.4, .9])

with q1:
    if st.button("🟢 Bidder A — Apex • Compliant 94%", use_container_width=True):
        st.session_state.demo_bidder = "A"

with q2:
    if st.button("🟠 Bidder B — Nova • Medium Risk 72%", use_container_width=True):
        st.session_state.demo_bidder = "B"

with q3:
    if st.button("🔴 Bidder C — Vertex • High Risk 44%", use_container_width=True):
        st.session_state.demo_bidder = "C"

with q4:
    if st.button("▶ Auto 3-Min SIH Demo", use_container_width=True):
        st.session_state.demo_bidder = "B"


# ============================================================
# DEMO DATA
# ============================================================
demo_scores = {"A": 94, "B": 72, "C": 44}
demo_risk = {"A": "Low", "B": "Medium", "C": "High"}
demo_name = {"A": "Apex", "B": "Nova", "C": "Vertex"}

selected = st.session_state.get("demo_bidder", "B")


# ============================================================
# MAIN DASHBOARD
# ============================================================
if page == "🏠 Main Dashboard":

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">
                Procurement Executive Command Center
                <span class="live">Live AI Verification System</span>
            </div>
            <div class="hero-sub">
                Real-time multi-source bid compliance analytics for Tender
                <b>CPCL/DEMO/2026/001</b>
                (Procurement of Industrial High-Pressure Centrifugal Pumps)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    cols = st.columns(8)

    cards = [
        ("ACTIVE TENDERS", "1", "CPCL Demo", "blue"),
        ("TOTAL BIDDERS", "3", "Bids Submitted", "purple"),
        ("VERIFIED COMPLIANT", "1", "Low Risk", "green"),
        ("MANUAL REVIEW", "1", "Action Needed", "amber"),
        ("HIGH RISK BIDDERS", "1", "Flagged", "red"),
        ("PENDING DOCS", "3", "In Pipeline", "blue"),
        ("COMPLIANCE RATE", "70%", "Tender Average", "blue"),
        ("VERIFICATION SPEED", "4.2s", "Per Document", "purple"),
    ]

    for col, (label, value, foot, cls) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {cls}">{value}</div>
                    <div class="metric-foot">◉ {foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Charts
    c1, c2, c3 = st.columns([1.05, 1.0, 1.1])

    with c1:
        st.markdown(
            """
            <div class="panel">
                <span class="panel-title">Compliance Score Comparison</span>
                <span class="panel-sub">100-Point Weighted</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        names = ["Apex", "Nova", "Vertex"]
        values = [94, 72, 44]
        ax.bar(names, values)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=.15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown(
            """
            <div class="panel">
                <span class="panel-title">Bidder Risk Classification</span>
                <span class="panel-sub">Multi-Factor Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        ax.pie(
            [1, 1, 1],
            labels=["Low (1)", "Medium (1)", "High (1)"],
            startangle=90,
            wedgeprops={"width": 0.38},
        )
        ax.set_aspect("equal")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c3:
        st.markdown(
            """
            <div class="panel">
                <span class="panel-title">Compliance Coverage by Domain</span>
                <span class="panel-sub">6 Domain Audit</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Radar-like coverage using a polar chart
        categories = [
            "Statutory",
            "Financial",
            "Technical",
            "MSME",
            "Labour",
            "Make in India",
        ]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values = [78, 82, 72, 75, 70, 80]
        values += values[:1]
        angles += angles[:1]

        fig = plt.figure(figsize=(5.5, 3.5))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=.15)
        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=7)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Bidder table
    st.markdown("### 📌 Bidder Verification Snapshot")

    bidder_df = pd.DataFrame(
        {
            "Bidder": ["Apex", "Nova", "Vertex"],
            "Compliance Score": [94, 72, 44],
            "Risk": ["Low", "Medium", "High"],
            "Financial": ["PASS", "PASS", "FAIL"],
            "Technical": ["PASS", "PASS", "FAIL"],
            "Documents": ["Complete", "2 Missing", "5 Missing"],
            "Decision": ["Verified", "Manual Review", "High Risk"],
        }
    )

    st.dataframe(
        bidder_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🎯 Selected Demo Bidder")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Bidder", demo_name[selected])
    with sc2:
        st.metric("Compliance Score", f"{demo_scores[selected]}%")
    with sc3:
        st.metric("Risk Level", demo_risk[selected])

    if demo_risk[selected] == "Low":
        st.success("✅ Bidder is currently in the low-risk compliant category.")
    elif demo_risk[selected] == "Medium":
        st.warning("⚠️ Bidder requires procurement-officer manual review.")
    else:
        st.error("🚨 Bidder is high risk. Review failed requirements before decision.")


# ============================================================
# TENDER MANAGEMENT
# ============================================================
elif page == "📑 Tender Management":
    st.title("📑 Tender Management")
    st.info("Manage active GeM tenders and evaluation status.")

    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Tender ID", "CPCL/DEMO/2026/001")
        st.text_input("Tender Title", "Industrial High-Pressure Centrifugal Pumps")
        st.selectbox("Tender Status", ["Active", "Under Evaluation", "Closed"])
    with c2:
        st.number_input("Total Clauses", min_value=1, value=12)
        st.date_input("Evaluation Date")
        st.selectbox("Procurement Type", ["Goods", "Services", "Works"])


# ============================================================
# TENDER REQUIREMENTS
# ============================================================
elif page == "📋 Tender Requirements":
    st.title("📋 Tender Requirements")
    st.caption("12 compliance clauses configured for the active demo tender.")

    requirements = pd.DataFrame(
        {
            "Clause": [
                "Financial Turnover",
                "Experience",
                "GST / PAN",
                "Udyam / MSME",
                "OEM Authorization",
                "Technical Specification",
                "Mandatory Documents",
                "Local Content",
                "Certificate Validity",
                "Blacklist Status",
                "Labour Compliance",
                "Make in India",
            ],
            "Domain": [
                "Financial",
                "Eligibility",
                "Statutory",
                "MSME",
                "Technical",
                "Technical",
                "Documentation",
                "Statutory",
                "Legal",
                "Legal",
                "Labour",
                "Policy",
            ],
            "Weight": [12, 10, 8, 7, 12, 15, 10, 8, 5, 5, 4, 4],
            "Mandatory": ["Yes"] * 12,
        }
    )
    st.dataframe(requirements, use_container_width=True, hide_index=True)


# ============================================================
# BIDDER MANAGEMENT
# ============================================================
elif page == "👥 Bidder Management":
    st.title("👥 Bidder Management")

    bidders = pd.DataFrame(
        {
            "Bidder": ["Apex", "Nova", "Vertex"],
            "Bid ID": ["BID-001", "BID-002", "BID-003"],
            "Score": [94, 72, 44],
            "Risk": ["Low", "Medium", "High"],
            "Review": ["Verified", "Manual Review", "Flagged"],
        }
    )
    st.dataframe(bidders, use_container_width=True, hide_index=True)


# ============================================================
# UPLOAD DOCUMENTS
# ============================================================
elif page == "📤 Upload Documents":
    st.title("📤 Upload Tender / Bid Documents")
    st.write("Upload PDF, DOCX or image evidence for the compliance pipeline.")

    uploaded = st.file_uploader(
        "Choose document",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded:
        st.success(f"{len(uploaded)} document(s) received.")
        for file in uploaded:
            st.write(f"📄 {file.name} — {file.size / 1024:.1f} KB")


# ============================================================
# AI DOCUMENT ANALYSIS
# ============================================================
elif page == "🤖 AI Document Analysis":
    st.title("🤖 AI Document Analysis")

    st.info(
        "Prototype analysis view. Connect your OCR / LLM pipeline here to "
        "extract clauses, entities, dates and compliance evidence."
    )

    text = st.text_area(
        "Document text",
        "The bidder confirms GST registration, OEM authorization, "
        "minimum turnover and technical compliance.",
        height=180,
    )

    if st.button("🔍 Analyze Document", type="primary"):
        keywords = [
            "GST",
            "OEM",
            "turnover",
            "technical",
            "authorization",
        ]
        found = [k for k in keywords if k.lower() in text.lower()]

        st.success("Document analysis completed.")
        st.write("Detected evidence:", ", ".join(found) if found else "None")


# ============================================================
# COMPLIANCE ENGINE
# ============================================================
elif page == "⚙️ Compliance Engine":
    st.title("⚙️ Compliance Engine")

    st.write("Configure the weighted compliance rule engine.")

    rules = {
        "Financial": 20,
        "Technical": 25,
        "Documentation": 20,
        "Statutory": 15,
        "Eligibility": 10,
        "Policy / Local Content": 10,
    }

    for name, default in rules.items():
        st.slider(name, 0, 50, default)

    st.success("Rule configuration ready for evaluation.")


# ============================================================
# RISK ASSESSMENT
# ============================================================
elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")

    score = st.slider("Overall Risk Score", 0, 100, 38)

    if score < 30:
        st.success("🟢 LOW RISK")
    elif score < 65:
        st.warning("🟠 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")

    st.progress(score / 100)
    st.caption(
        "Risk classification should support — not replace — procurement-officer review."
    )


# ============================================================
# EVIDENCE EXPLORER
# ============================================================
elif page == "🔎 Evidence Explorer":
    st.title("🔎 Evidence Explorer")

    evidence = pd.DataFrame(
        {
            "Evidence ID": ["EV-001", "EV-002", "EV-003", "EV-004"],
            "Requirement": [
                "GST Validity",
                "OEM Authorization",
                "Turnover",
                "Technical Score",
            ],
            "Source": [
                "GST Certificate.pdf",
                "OEM Letter.pdf",
                "Audited Financials.pdf",
                "Technical Bid.pdf",
            ],
            "Confidence": ["98%", "96%", "94%", "91%"],
            "Status": ["Verified", "Verified", "Verified", "Review"],
        }
    )

    st.dataframe(evidence, use_container_width=True, hide_index=True)


# ============================================================
# DISCREPANCY CENTER
# ============================================================
elif page == "🚨 Discrepancy Center":
    st.title("🚨 Discrepancy Center")

    discrepancies = [
        ("Nova", "2 mandatory documents missing", "Medium"),
        ("Vertex", "Technical score below threshold", "High"),
        ("Vertex", "OEM authorization unavailable", "High"),
        ("Nova", "Local content evidence incomplete", "Medium"),
    ]

    for bidder, issue, severity in discrepancies:
        if severity == "High":
            st.error(f"🔴 {bidder} — {issue}")
        else:
            st.warning(f"🟠 {bidder} — {issue}")


# ============================================================
# OPTIONAL REAL ML PREDICTION
# ============================================================
with st.sidebar.expander("🧠 ML Bid Predictor", expanded=False):
    st.caption(
        "Uses best_tender_classifier.pkl when the model and label encoder "
        "are available in the same folder."
    )

    turnover = st.number_input("Company Turnover (₹ Cr)", 0.0, 1000.0, 10.0)
    required_turnover = st.number_input("Required Turnover (₹ Cr)", 0.0, 1000.0, 5.0)
    experience = st.number_input("Experience (Years)", 0.0, 100.0, 5.0)
    required_experience = st.number_input("Required Experience", 0.0, 100.0, 3.0)
    technical_score = st.slider("Technical Score", 0.0, 100.0, 75.0)
    minimum_technical_score = st.number_input("Minimum Technical Score", 0.0, 100.0, 70.0)

    gst = st.selectbox("GST Valid", ["Yes", "No"])
    pan = st.selectbox("PAN Valid", ["Yes", "No"])
    udyam = st.selectbox("Udyam Valid", ["Yes", "No"])
    oem = st.selectbox("OEM Authorization", ["Yes", "No"])
    documents = st.selectbox("Mandatory Documents Complete", ["Yes", "No"])
    missing_documents = st.number_input("Missing Documents", 0, 100, 0)
    local_content = st.number_input("Local Content %", 0.0, 100.0, 60.0)
    required_local_content = st.number_input("Required Local Content %", 0.0, 100.0, 50.0)
    certificate = st.selectbox("Certificate", ["Valid", "Expired"])
    blacklist = st.selectbox("Blacklist", ["Clear", "Blacklisted"])
    compliance_score = st.slider("Existing Compliance Score", 0.0, 100.0, 80.0)

    if st.button("🚀 Run ML Prediction", use_container_width=True):

        if model is None or label_encoder is None:
            st.warning(
                "ML files not found. Put best_tender_classifier.pkl and "
                "label_encoder.pkl beside app.py to enable real predictions."
            )
        else:
            record = {
                "turnover_crore": turnover,
                "required_turnover_crore": required_turnover,
                "experience_years": experience,
                "required_experience_years": required_experience,
                "gst_valid": 1 if gst == "Yes" else 0,
                "pan_valid": 1 if pan == "Yes" else 0,
                "udyam_valid": 1 if udyam == "Yes" else 0,
                "oem_authorization": 1 if oem == "Yes" else 0,
                "technical_score": technical_score,
                "minimum_technical_score": minimum_technical_score,
                "technical_compliance": 1 if technical_score >= minimum_technical_score else 0,
                "financial_compliance": 1 if turnover >= required_turnover else 0,
                "eligibility_compliance": 1 if experience >= required_experience else 0,
                "mandatory_documents_complete": 1 if documents == "Yes" else 0,
                "missing_documents": missing_documents,
                "local_content_percent": local_content,
                "required_local_content_percent": required_local_content,
                "certificate_validity": 1 if certificate == "Valid" else 0,
                "blacklist_clear": 1 if blacklist == "Clear" else 0,
                "compliance_score": compliance_score,
            }

            row = pd.DataFrame([record])
            row["turnover_margin"] = turnover - required_turnover
            row["experience_margin"] = experience - required_experience
            row["technical_margin"] = technical_score - minimum_technical_score
            row["local_content_margin"] = local_content - required_local_content

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
                "local_content_margin",
            ]

            try:
                row = row[feature_names]

                prediction = model.predict(row)[0]
                predicted_label = label_encoder.inverse_transform([prediction])[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(row)[0]
                    classes = list(label_encoder.classes_)

                    if "PASS" in classes:
                        pass_probability = probabilities[classes.index("PASS")]
                    else:
                        pass_probability = float(np.max(probabilities))
                else:
                    pass_probability = 1.0 if predicted_label == "PASS" else 0.0

                if predicted_label == "PASS":
                    st.success(
                        f"✅ BID LIKELY TO PASS — Confidence: "
                        f"{pass_probability * 100:.1f}%"
                    )
                else:
                    st.error(
                        f"❌ BID LIKELY TO FAIL — PASS probability: "
                        f"{pass_probability * 100:.1f}%"
                    )

            except Exception as exc:
                st.error("Prediction failed because the model feature schema differs.")
                st.code(str(exc))


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align:center;color:#536486;font-size:10px;margin-top:25px;">
        GeM-AI Integrated Bid Compliance Hub • SIH Prototype •
        Human-in-the-Loop Decision Support • Synthetic Demonstration Data
    </div>
    """,
    unsafe_allow_html=True,
)
