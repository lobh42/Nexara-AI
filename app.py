"""
Nexara AI - Streamlit Application
Predictive Maintenance Scheduling with Multi-LLM Support
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

# Add current directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pattern_detector import (
    detect_degradation_patterns,
    detect_near_misses,
    generate_maintenance_schedule,
    get_equipment_health_scores,
    get_trend_data,
)
from ai_engine import analyze_logs_with_llm, chat_query
from ml_features import (
    detect_anomalies_isolation_forest,
    predict_failure_probability,
    generate_report_data,
)
from notifications import check_and_notify
from digital_twin import (
    analyze_digital_twin,
    get_equipment_twin_params,
    get_twin_visualization_data,
)
from youtube_util import (
    get_maintenance_videos,
    get_video_thumbnail,
    EQUIPMENT_VIDEO_FALLBACKS,
)

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexara AI - Maintenance Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Theme ───────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono&display=swap');

    /* Variables & Tokens */
    :root {
        --bg-main: #FFFFFF;
        --bg-alt: #F8FAFC;
        --text-primary: #0F172A;
        --text-secondary: #64748B;
        --brand-primary: #4F46E5;
        --brand-secondary: #3B82F6;
        --brand-accent: #8B5CF6;
        --border-color: #E2E8F0;
        --glass-bg: rgba(255, 255, 255, 0.8);
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    /* Global Overrides */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Noise Texture Overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0.015;
        z-index: 0;
        pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3 Forts%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
    }

    /* Animations */
    @keyframes entranceFade {
        from { opacity: 0; transform: translateY(12px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    @keyframes subtlePulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.05); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }

    /* Layout Spacing Rhythm */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* Header Styling - Animated Gradient */
    .main-header {
        position: relative;
        background: linear-gradient(-45deg, #4F46E5, #3B82F6, #8B5CF6, #2563EB);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite, entranceFade 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        padding: 3rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        color: white;
        box-shadow: var(--shadow-xl);
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at top right, rgba(255,255,255,0.1), transparent);
    }
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        color: white !important;
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.25rem;
        font-weight: 500;
    }

    /* SaaS Cards - Glassmorphism & Depth */
    .premium-card {
        background: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.75rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: entranceFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
        box-shadow: var(--shadow-sm);
        position: relative;
    }
    .premium-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px -12px rgba(50, 50, 93, 0.1), 0 18px 36px -18px rgba(0, 0, 0, 0.15);
        border-color: var(--brand-secondary);
    }
    .hero-metric {
        grid-column: span 2;
        background: linear-gradient(to bottom right, #FFFFFF, #F8FAFC);
        border-right: 4px solid var(--brand-primary);
    }

    /* Staggered Entry for cards */
    .card-row > div:nth-child(1) .premium-card { animation-delay: 0.1s; }
    .card-row > div:nth-child(2) .premium-card { animation-delay: 0.2s; }
    .card-row > div:nth-child(3) .premium-card { animation-delay: 0.3s; }

    /* Typography Hierarchy */
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Status Indicators */
    .indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .indicator-critical { background: #EF4444; box-shadow: 0 0 12px #EF4444; animation: subtlePulse 2s infinite; }
    .indicator-warning { background: #F59E0B; box-shadow: 0 0 8px #F59E0B; }
    .indicator-healthy { background: #10B981; }

    /* Buttons - Tactile Feedback */
    .stButton > button {
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        border: 1px solid var(--border-color) !important;
        background: white !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stButton > button:hover {
        border-color: var(--brand-primary) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
    }
    .stButton > button:active {
        transform: scale(0.96) !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.06) !important;
    }
    .stButton [data-testid="baseButton-primary"] {
        background: var(--brand-primary) !important;
        color: white !important;
        border: none !important;
    }

    /* Sidebar - High End SaaS Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-alt);
        border-right: 1px solid var(--border-color);
        padding: 2rem 1rem;
    }
    .sidebar-brand {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .sidebar-brand span {
        background: linear-gradient(135deg, #4F46E5 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 8px 4px;
        color: var(--text-secondary);
        border: none;
        border-bottom: 2px solid transparent;
        transition: all 0.3s;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: var(--brand-primary) !important;
        border-bottom-color: var(--brand-primary) !important;
    }

    /* Chat UI - Linear Inspired */
    .chat-user {
        background: var(--brand-primary);
        color: white;
        padding: 1.25rem 1.75rem;
        border-radius: 20px 20px 4px 20px;
        margin: 1.2rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: var(--shadow-lg);
        animation: slideInRight 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .chat-assistant {
        background: var(--bg-alt);
        color: var(--text-primary);
        padding: 1.25rem 1.75rem;
        border-radius: 20px 20px 20px 4px;
        margin: 1.2rem 0;
        max-width: 80%;
        border: 1px solid var(--border-color);
        animation: entranceFade 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Hide Streamlit default decorations */
    div[data-testid="stDecoration"] {
        display: none;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(12px);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State Initialization ─────────────────────────────────
def init_session_state():
    defaults = {
        "data_loaded": False,
        "raw_data": None,
        "cleaned_data": None,
        "patterns": [],
        "near_misses": [],
        "schedule": [],
        "health_scores": [],
        "analysis": None,
        "anomaly_data": None,
        "failure_predictions": [],
        "chat_history": [],
        "notifications": [],
        "technician_config": {
            "name": "",
            "email": "",
            "phone": "",
            "alert_enabled": True,
        },
        "analysis_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Notification System ─────────────────────────────────────────────
def send_notification(title: str, message: str, severity: str = "info"):
    """Add a notification to the session state."""
    notification = {
        "id": len(st.session_state.notifications) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": title,
        "message": message,
        "severity": severity,
        "read": False,
    }
    st.session_state.notifications.append(notification)


def check_and_send_alerts():
    """Check data and send automatic alerts for critical conditions."""
    if not st.session_state.data_loaded:
        return


def get_equipment_emoji(eq_type: str) -> str:
    """Get emoji for equipment type."""
    emojis = {
        "pump": "⚙️",
        "motor": "🔌",
        "compressor": "💨",
        "fan": "🌀",
        "gearbox": "⚙️",
        "conveyor": "🏭",
        "valve": "🔀",
    }
    return emojis.get(eq_type, "🔧")


def render_notifications_panel():
    """Render the notifications UI in sidebar."""
    st.markdown("---")
    st.markdown("### 🔔 Notifications")
    
    # Auto-notification status
    if st.session_state.get("auto_notification_sent"):
        st.success(f"🔔 {st.session_state.auto_notification_sent} alert(s) sent to technicians!")
    
    unread = [n for n in st.session_state.notifications if not n.get("read", False)]
    if unread:
        st.markdown(f"**{len(unread)} unread alert(s)**")
    else:
        st.markdown("No new notifications")
    
    if st.session_state.notifications:
        with st.expander("View All Notifications"):
            for n in reversed(st.session_state.notifications[-10:]):
                icon = "🔴" if n["severity"] == "critical" else "🟡" if n["severity"] == "warning" else "ℹ️"
                st.markdown(f"**{icon} {n['title']}**")
                st.caption(f"{n['timestamp']} - {n['message']}")
                if not n.get("read"):
                    if st.button("Mark Read", key=f"read_{n['id']}"):
                        n["read"] = True
                        st.rerun()
    
    if st.button("Clear All", key="clear_notifications_btn"):
        st.session_state.notifications = []
        st.rerun()


def render_header():
    """Render the premium animated header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>Nexara AI</h1>
            <p>Intelligence-Driven Predictive Maintenance for the Modern Factory</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing():
    """Render the premium asymmetrical landing page."""
    render_header()
    
    # Asymmetrical Hero Section
    col1, col2 = st.columns([1.8, 1])
    
    with col1:
        st.markdown(
            """
            <div class="premium-card hero-metric">
                <div class="metric-label">System Architecture</div>
                <h2 style="font-family: 'Outfit', sans-serif; letter-spacing: -0.02em; margin-top: 0.5rem;">
                    Unified Maintenance Intelligence
                </h2>
                <p style="color: var(--text-secondary); font-size: 1.1rem; line-height: 1.6; margin-top: 1rem;">
                    Nexara AI utilizes multi-model reasoning to synthesize sensor data, 
                    historical logs, and digital twin simulations into actionable maintenance 
                    schedules—drastically reducing unplanned downtime.
                </p>
                <div style="margin-top: 2rem; display: flex; gap: 1rem;">
                    <span class="badge-info" style="background: var(--bg-alt); border: 1px solid var(--border-color);">Predictive ML</span>
                    <span class="badge-info" style="background: var(--bg-alt); border: 1px solid var(--border-color);">Digital Twin</span>
                    <span class="badge-info" style="background: var(--bg-alt); border: 1px solid var(--border-color);">LLM Insights</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            """
            <div class="premium-card">
                <div class="metric-label">Quick Start</div>
                <div style="margin-top: 1.5rem;">
                    <p style="font-size: 0.95rem; color: var(--text-secondary);">
                        Upload your equipment telemetry to begin real-time analysis.
                    </p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("📋 Explore Sample Analytics", type="primary", key="explore_sample_btn", width="stretch"):
            load_sample_data()
            st.rerun()

    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    # Secondary Features Grid
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="premium-card"><h3>⚡ Speed</h3><p>Instant pattern detection across millions of logs.</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="premium-card"><h3>🔒 Secure</h3><p>Enterprise-grade data isolation and processing.</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="premium-card"><h3>🛠️ Precision</h3><p>98% accuracy in early-stage degradation detection.</p></div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">⚡ <span>Nexara</span> AI</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: -1.5rem; margin-bottom: 2rem;">V3.0 | Enterprise Intelligence</p>', unsafe_allow_html=True)
        
        # Data upload
        st.markdown('<div class="metric-label" style="margin-bottom: 0.5rem;">Data Source</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Upload equipment log CSV file", label_visibility="collapsed")
        if uploaded_file is not None:
            process_upload(uploaded_file)

        if st.button("📋 Load Sample Data", key="load_sample_sidebar", width="stretch"):
            load_sample_data()

        if st.session_state.data_loaded:
            st.markdown(f'<div class="info-box">Loaded: {len(st.session_state.cleaned_data)} records</div>', unsafe_allow_html=True)

        st.markdown("---")
        render_notifications_panel()
        
        st.markdown('<div style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
        st.caption("Nexara AI | Designed for Precision")


def process_upload(uploaded_file):
    """Process an uploaded CSV file."""
    file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
    if st.session_state.get("current_file_id") == file_id:
        return
        
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()

        encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
        df = None
        last_error = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(io.StringIO(content.decode(encoding)))
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
        
        if df is None:
            st.sidebar.error(f"Could not decode file. Tried: {', '.join(encodings)}. Last error: {last_error}")
            return

        if df.empty:
            st.sidebar.error("The uploaded file is empty.")
            return

        original_cols = list(df.columns)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        
        required_cols = ["timestamp", "equipment_id"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.sidebar.error(f"Missing required columns: {missing}. Your data needs: timestamp, equipment_id. Found: {list(df.columns)}")
            return

        # Enforce numeric types
        numeric_cols = ["temperature", "vibration", "pressure", "rpm"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse timestamps and remove invalid rows
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "equipment_id"])

        if df.empty:
            st.sidebar.error("The uploaded file resulted in empty data after cleaning timestamps.")
            return

        # Ensure severity column exists and has bounded values
        if "severity" not in df.columns:
            df["severity"] = "info"
        else:
            df["severity"] = df["severity"].fillna("info").astype(str).str.lower()
            df["severity"] = df["severity"].apply(lambda x: x if x in ["info", "warning", "critical"] else "info")

        st.sidebar.info(f"Detected columns: {list(df.columns)}")
        
        st.session_state.raw_data = df.copy()
        st.session_state.cleaned_data = df.copy()
        st.session_state.data_loaded = True
        st.session_state.analysis_done = False

        st.session_state.current_file_id = file_id
        run_full_analysis()
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")


def load_sample_data():
    """Load the sample dataset."""
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "sample_logs.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.session_state.raw_data = df.copy()
        st.session_state.cleaned_data = df.copy()
        st.session_state.data_loaded = True
        st.session_state.analysis_done = False
        run_full_analysis()
    else:
        st.sidebar.error("Sample data file not found.")


def run_full_analysis(force: bool = False):
    """Run all analysis modules on the cleaned data."""
    df = st.session_state.cleaned_data
    if df is None:
        return
    
    if st.session_state.get("analysis_done") and not force:
        return

    with st.spinner("Running pattern detection..."):
        st.session_state.patterns = detect_degradation_patterns(df)
    with st.spinner("Detecting near-miss events..."):
        st.session_state.near_misses = detect_near_misses(df)
    with st.spinner("Computing health scores..."):
        st.session_state.health_scores = get_equipment_health_scores(df)
    with st.spinner("Generating maintenance schedule..."):
        st.session_state.schedule = generate_maintenance_schedule(
            df,
            st.session_state.patterns,
            st.session_state.near_misses,
            {},
        )

    # ML Features
    with st.spinner("Running anomaly detection..."):
        st.session_state.anomaly_data = detect_anomalies_isolation_forest(df)
    with st.spinner("Predicting failure probabilities..."):
        st.session_state.failure_predictions = predict_failure_probability(df)

    # Auto-notify for critical equipment
    if st.session_state.failure_predictions:
        notification_result = check_and_notify(
            st.session_state.failure_predictions, 
            threshold=70.0,
            technician_config=st.session_state.technician_config
        )
        if notification_result.get("notifications_sent", 0) > 0:
            st.session_state.auto_notification_sent = notification_result.get("notifications_sent", 0)

    # AI Analysis
    logs_summary = {
        "total_records": len(df),
        "date_range": {
            "start": str(df["timestamp"].min()) if "timestamp" in df.columns else None,
            "end": str(df["timestamp"].max()) if "timestamp" in df.columns else None,
        },
        "equipment_count": df["equipment_id"].nunique() if "equipment_id" in df.columns else 0,
        "severity_distribution": df["severity"].value_counts().to_dict() if "severity" in df.columns else {},
    }

    # AI Analysis (with timeout handling)
    try:
        with st.spinner("Running AI analysis (this may take a moment)..."):
            st.session_state.analysis = analyze_logs_with_llm(
                logs_summary,
                st.session_state.patterns,
                st.session_state.near_misses,
            )
    except Exception as e:
        st.session_state.analysis = {
            "error": str(e),
            "executive_summary": "AI analysis temporarily unavailable. Rule-based analysis is active.",
            "critical_findings": [],
            "root_causes": [],
            "immediate_actions": [],
            "long_term_strategy": []
        }
    
    st.session_state.analysis_done = True


# ── Main Content ─────────────────────────────────────────────────
def render_header():
    st.markdown(
        """
    <div class="main-header">
        <h1>⚡ Nexara AI - Maintenance Intelligence</h1>
        <p>AI-powered predictive maintenance scheduling for manufacturing equipment</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_landing():
    """Render landing page when no data is loaded."""
    st.markdown('<div class="section-header">Welcome to Nexara AI</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload your equipment log CSV or load the sample dataset from the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">5</div>
            <div class="metric-label">LLM Providers Supported</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">9+</div>
            <div class="metric-label">Analysis Features</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value">ML</div>
            <div class="metric-label">Anomaly & Failure Prediction</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Features")
    features = {
        "📈 Degradation Pattern Detection": "Identify temperature, vibration, and pressure anomalies",
        "⚠️ Near-Miss Detection": "Spot parameters approaching critical thresholds",
        "📅 Confidence-Based Scheduling": "Maintenance schedules with confidence scores",
        "🤖 Multi-LLM Chat Assistant": "Chat with 5 different AI providers",
        "🔬 ML Anomaly Detection": "Isolation Forest-based anomaly detection",
        "📊 Failure Prediction": "ML-based failure probability estimation",
        "📄 Report Export": "Download comprehensive maintenance reports",
    }
    cols = st.columns(3)
    for i, (title, desc) in enumerate(features.items()):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            st.caption(desc)


@st.fragment
def render_dashboard():
    """Render the premium SaaS dashboard with asymmetrical layout."""
    df = st.session_state.cleaned_data
    health_scores = st.session_state.health_scores

    # Summary Metrics - Asymmetrical Row
    m1, m2, m3 = st.columns([1.5, 1, 1])
    
    critical_count = len([h for h in health_scores if h["status"] == "critical"])
    
    with m1:
        st.markdown(
            f"""
            <div class="premium-card hero-metric">
                <div class="metric-label">System Status</div>
                <div class="metric-value">{100 - (critical_count * 10)}%</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Overall Operational Efficiency</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with m2:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="metric-label">Alerts</div>
                <div class="metric-value" style="color: #EF4444;">{critical_count}</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Critical Actions Required</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="premium-card">
                <div class="metric-label">Fleet Size</div>
                <div class="metric-value">{df["equipment_id"].nunique() if "equipment_id" in df.columns else 0}</div>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Active Connected Assets</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

    # Main tabs
    tabs = st.tabs([
        "📊 Overview",
        "🔍 Patterns",
        "⚠️ Near-Misses",
        "📅 Schedule",
        "🔬 Anomaly Detection",
        "📈 Failure Prediction",
        "👥 Digital Twin",
        "🎬 Maintenance Videos",
        "🤖 AI Chat",
        "📋 Machine Schedule",
        "📄 Export Report",
    ])

    with tabs[0]: render_overview_tab()
    with tabs[1]: render_patterns_tab()
    with tabs[2]: render_near_misses_tab()
    with tabs[3]: render_schedule_tab()
    with tabs[4]: render_anomaly_tab()
    with tabs[5]: render_failure_prediction_tab()
    with tabs[6]: render_digital_twin_tab()
    with tabs[7]: render_videos_tab()
    with tabs[8]: render_chat_tab()
    with tabs[9]: render_machine_schedule_tab()
    with tabs[10]: render_export_tab()


def render_overview_tab():
    """Render the premium overview tab with AI insights and health grid."""
    df = st.session_state.cleaned_data
    health_scores = st.session_state.health_scores
    analysis = st.session_state.analysis

    if df is None:
        return

    # Asymmetrical Layout: AI Insights (Top) + Health Details (Bottom)
    if analysis and analysis.get("success"):
        a = analysis.get("analysis", {})
        source = analysis.get("source", "unknown")
        
        st.markdown(f'<div class="metric-label" style="margin-top: 1rem;">AI Intelligence Engine — {source.upper()}</div>', unsafe_allow_html=True)
        
        col_summary, col_findings = st.columns([1.2, 1])
        
        with col_summary:
            st.markdown(
                f"""
                <div class="premium-card" style="height: 100%;">
                    <h3>Intelligence Summary</h3>
                    <p style="color: var(--text-secondary); line-height: 1.6;">{a.get("executive_summary", "Analyzing fleet data...")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_findings:
            if "critical_findings" in a and a["critical_findings"]:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0;">Critical Findings</h4>', unsafe_allow_html=True)
                for f in a["critical_findings"][:3]:
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-color);">
                            <span class="indicator indicator-critical"></span>
                            <strong>{f.get("equipment", "Asset")}</strong>: {f.get("finding", "")}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)

    # Equipment Health Overview - Grid
    st.markdown('<div class="metric-label">Real-time Asset Health</div>', unsafe_allow_html=True)
    if health_scores:
        # Health score bar chart
        health_df = pd.DataFrame(health_scores)
        fig = px.bar(
            health_df,
            x="equipment_name",
            y="health_score",
            color="status",
            color_discrete_map={
                "healthy": "#22C55E",
                "attention": "#F59E0B",
                "warning": "#EF4444",
                "critical": "#DC2626",
            },
            title="Equipment Health Scores",
            labels={"equipment_name": "Equipment", "health_score": "Health Score"},
        )
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_color="#1E293B",
            xaxis_tickangle=-45,
        )
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning threshold")
        fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Critical threshold")
        st.plotly_chart(fig, width='stretch', key="overview_health")

    # Sensor trend charts
    st.markdown('<div class="section-header">Sensor Trends</div>', unsafe_allow_html=True)
    trend_data = get_trend_data(df)

    col1, col2 = st.columns(2)
    with col1:
        if "temperature" in trend_data.get("series", {}):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data["timestamps"],
                y=trend_data["series"]["temperature"],
                mode="lines+markers",
                name="Temperature (F)",
                line=dict(color="#EF4444"),
            ))
            fig.update_layout(title="Temperature Trends", plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
            st.plotly_chart(fig, width='stretch', key="temp_trend")

    with col2:
        if "vibration" in trend_data.get("series", {}):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data["timestamps"],
                y=trend_data["series"]["vibration"],
                mode="lines+markers",
                name="Vibration (mm/s)",
                line=dict(color="#F59E0B"),
            ))
            fig.update_layout(title="Vibration Trends", plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
            st.plotly_chart(fig, width='stretch', key="vibration_trend")

    col3, col4 = st.columns(2)
    with col3:
        if "pressure" in trend_data.get("series", {}):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data["timestamps"],
                y=trend_data["series"]["pressure"],
                mode="lines+markers",
                name="Pressure (PSI)",
                line=dict(color="#2563EB"),
            ))
            fig.update_layout(title="Pressure Trends", plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
            st.plotly_chart(fig, width='stretch', key="pressure_trend")

    with col4:
        # Severity distribution pie chart
        if "severity" in df.columns:
            sev_counts = df["severity"].value_counts()
            fig = px.pie(
                names=sev_counts.index,
                values=sev_counts.values,
                title="Severity Distribution",
                color=sev_counts.index,
                color_discrete_map={"info": "#3B82F6", "warning": "#F59E0B", "critical": "#EF4444"},
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
            st.plotly_chart(fig, width='stretch', key="severity_dist")


def render_patterns_tab():
    patterns = st.session_state.patterns
    st.markdown(f'<div class="section-header">Degradation Patterns ({len(patterns)} detected)</div>', unsafe_allow_html=True)

    if not patterns:
        st.info("No degradation patterns detected. Upload data to analyze.")
        return

    for i, p in enumerate(patterns):
        severity = p["severity"]
        badge_class = "badge-critical" if severity == "critical" else "badge-warning"

        with st.expander(f"{'🔴' if severity == 'critical' else '🟡'} {p['equipment_name']} - {p['signal']} (Confidence: {p['confidence']}%)"):
            st.markdown(f'<span class="{badge_class}">{severity.upper()}</span>', unsafe_allow_html=True)
            st.markdown(f"**Description:** {p['description']}")
            st.markdown(f"**Recommendation:** {p['recommendation']}")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p["confidence"],
                title={"text": "Confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#EF4444" if p["confidence"] > 70 else "#F59E0B"},
                    "steps": [
                        {"range": [0, 40], "color": "#DCFCE7"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#FEE2E2"},
                    ],
                },
            ))
            fig.update_layout(height=250, plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
            st.plotly_chart(fig, width='stretch', key=f"pattern_gauge_{i}")

            st.markdown("**Evidence:**")
            for e in p.get("evidence", []):
                st.markdown(f"- {e}")
            
            # Show maintenance videos for critical patterns
            if severity == "critical" or p.get("confidence", 0) > 70:
                with st.expander("🛠️ Maintenance & Repair Videos"):
                    render_maintenance_videos(p["equipment_name"])


def render_near_misses_tab():
    near_misses = st.session_state.near_misses
    st.markdown(f'<div class="section-header">Near-Miss Events ({len(near_misses)} detected)</div>', unsafe_allow_html=True)

    if not near_misses:
        st.info("No near-miss events detected.")
        return

    for i, nm in enumerate(near_misses):
        status_icon = "🔴" if nm["status"] == "approaching_critical" else "🟡"
        with st.expander(f"{status_icon} {nm['equipment_name']} - {nm['parameter']} (Failure Prob: {nm['failure_probability']}%)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"{nm['current_value']:.1f} {nm['unit']}")
            with col2:
                st.metric("Warning Threshold", f"{nm['warning_threshold']} {nm['unit']}")
            with col3:
                st.metric("Critical Threshold", f"{nm['critical_threshold']} {nm['unit']}")

            # Threshold visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[nm["current_value"]],
                y=[nm["equipment_name"]],
                orientation="h",
                name="Current",
                marker_color="#F59E0B" if nm["status"] != "approaching_critical" else "#EF4444",
            ))
            fig.add_vline(x=nm["warning_threshold"], line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.add_vline(x=nm["critical_threshold"], line_dash="dash", line_color="red", annotation_text="Critical")
            fig.update_layout(
                height=150,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#1E293B",
                title=f"{nm['parameter']} Level",
            )
            st.plotly_chart(fig, width='stretch', key=f"near_miss_chart_{i}")

            st.markdown(f"**Description:** {nm['description']}")
            st.markdown(f"**Action Required:** {nm['action_required']}")
            st.markdown(f"**Occurrences:** {nm['occurrences']} | **Last:** {nm['last_occurrence']}")


def render_schedule_tab():
    schedule = st.session_state.schedule
    st.markdown(f'<div class="section-header">Maintenance Schedule ({len(schedule)} tasks)</div>', unsafe_allow_html=True)

    if not schedule:
        st.info("No maintenance tasks scheduled. Upload data to generate a schedule.")
        return

    # Summary by urgency
    urgency_counts = {}
    for s in schedule:
        u = s["urgency"]
        urgency_counts[u] = urgency_counts.get(u, 0) + 1

    cols = st.columns(len(urgency_counts))
    color_map = {"IMMEDIATE": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#F59E0B", "LOW": "#22C55E"}
    for i, (urgency, count) in enumerate(urgency_counts.items()):
        with cols[i]:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="color:{color_map.get(urgency, "#64748B")}">{count}</div>'
                f'<div class="metric-label">{urgency}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Timeline chart
    schedule_df = pd.DataFrame(schedule)
    if "recommended_date" in schedule_df.columns:
        fig = px.scatter(
            schedule_df,
            x="recommended_date",
            y="equipment_name",
            color="urgency",
            size="confidence",
            hover_data=["task", "estimated_duration"],
            title="Maintenance Timeline",
            color_discrete_map=color_map,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
        st.plotly_chart(fig, width='stretch', key="maintenance_timeline")

    # Detailed list
    for s in schedule:
        urgency = s["urgency"]
        icon = "🔴" if urgency == "IMMEDIATE" else "🟠" if urgency == "HIGH" else "🟡" if urgency == "MEDIUM" else "🟢"
        with st.expander(f"{icon} [{urgency}] {s['equipment_name']}: {s['task']}"):
            st.markdown(f"**Description:** {s['description']}")
            st.markdown(f"**Confidence:** {s['confidence']}%")
            st.markdown(f"**Recommended Date:** {s['recommended_date']}")
            st.markdown(f"**Duration:** {s['estimated_duration']}")
            st.markdown(f"**Reasoning:** {s['reasoning']}")
            st.markdown(f"**Production Impact:** {s['production_impact']}")
            if s.get("evidence"):
                st.markdown("**Evidence:**")
                for e in s["evidence"]:
                    st.markdown(f"- {e}")




def render_anomaly_tab():
    st.markdown('<div class="section-header">ML Anomaly Detection (Isolation Forest)</div>', unsafe_allow_html=True)

    anomaly_data = st.session_state.anomaly_data
    if anomaly_data is None or anomaly_data.empty:
        st.info("No data available for anomaly detection. Upload data first.")
        return

    anomaly_count = len(anomaly_data[anomaly_data["anomaly"] == -1])
    normal_count = len(anomaly_data[anomaly_data["anomaly"] == 1])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(anomaly_data))
    with col2:
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        st.metric("Normal Records", normal_count)

    # Anomaly scatter plot
    if "temperature" in anomaly_data.columns and "vibration" in anomaly_data.columns:
        fig = px.scatter(
            anomaly_data,
            x="temperature",
            y="vibration",
            color=anomaly_data["anomaly"].map({1: "Normal", -1: "Anomaly"}),
            color_discrete_map={"Normal": "#22C55E", "Anomaly": "#EF4444"},
            title="Anomaly Detection: Temperature vs Vibration",
            labels={"color": "Classification"},
            hover_data=["equipment_name", "severity"] if "equipment_name" in anomaly_data.columns else None,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
        st.plotly_chart(fig, width='stretch', key="anomaly_temp_vib")

    if "pressure" in anomaly_data.columns and "temperature" in anomaly_data.columns:
        fig = px.scatter(
            anomaly_data,
            x="temperature",
            y="pressure",
            color=anomaly_data["anomaly"].map({1: "Normal", -1: "Anomaly"}),
            color_discrete_map={"Normal": "#22C55E", "Anomaly": "#EF4444"},
            title="Anomaly Detection: Temperature vs Pressure",
            labels={"color": "Classification"},
            hover_data=["equipment_name", "severity"] if "equipment_name" in anomaly_data.columns else None,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
        st.plotly_chart(fig, width='stretch', key="anomaly_temp_press")

    # Anomaly score distribution
    if "anomaly_score" in anomaly_data.columns:
        fig = px.histogram(
            anomaly_data,
            x="anomaly_score",
            nbins=30,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#1E293B")
        st.plotly_chart(fig, width='stretch', key="anomaly_score_dist")

    # Anomalous records table
    anomalies = anomaly_data[anomaly_data["anomaly"] == -1]
    if not anomalies.empty:
        st.markdown("**Anomalous Records:**")
        display_cols = [c for c in ["timestamp", "equipment_name", "severity", "temperature", "vibration", "pressure", "message", "anomaly_score"] if c in anomalies.columns]
        st.dataframe(anomalies[display_cols].sort_values("anomaly_score"), width='stretch')


def render_failure_prediction_tab():
    """Render the premium ML-driven failure probability matrix."""
    st.markdown('<div class="metric-label">Predictive Horizon Analysis</div>', unsafe_allow_html=True)
    
    predictions = st.session_state.get("failure_predictions", [])
    if not predictions:
        st.info("Predictive analysis required. Synthesizing horizon metrics...")
        return

    # Matrix Visualization
    df_p = pd.DataFrame(predictions)
    fig = px.bar(
        df_p, x="equipment_name", y="failure_probability",
        color="risk_level",
        color_discrete_map={
            "CRITICAL": "#EF4444", "HIGH": "#F59E0B",
            "MEDIUM": "#3B82F6", "LOW": "#10B981"
        },
        title="Failure Probability by Asset Class",
        labels={"equipment_name": "Asset", "failure_probability": "Prob (%)"}
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, width="stretch", key="failure_prob_matrix")

    # Granular Intelligence Cards
    for p in predictions:
        risk = p["risk_level"]
        icon = "🔴" if risk == "CRITICAL" else "🟠" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        
        with st.expander(f"{icon} {p['equipment_name']} — {p['failure_probability']}% Probability"):
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="metric-label">Risk Threshold</div><div class="metric-value" style="font-size:1.5rem; color:{"#EF4444" if risk=="CRITICAL" else "var(--text-main)"};">{risk}</div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-label">Synthetic Probability</div><div class="metric-value" style="font-size:1.5rem;">{p["failure_probability"]}%</div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-label">Est. Horizon (Days)</div><div class="metric-value" style="font-size:1.5rem;">{p.get("estimated_time_to_failure", "N/A")}</div>', unsafe_allow_html=True)
            
            st.markdown(f"**Strategic Recommendation:** {p['recommended_action']}")
            st.markdown("**Core Risk Vectors:**")
            for f in p["contributing_factors"]:
                st.markdown(f"- {f}")
            
            if risk in ["CRITICAL", "HIGH"]:
                st.info(f"Targeted maintenance library available for {p['equipment_name']}. Syncing tutorials...")
            st.markdown('</div>', unsafe_allow_html=True)


def render_digital_twin_tab():
    """Render the premium Digital Twin reality-sync simulation."""
    st.markdown('<div class="metric-label">Logic Synthesis Engine</div>', unsafe_allow_html=True)

    equipment_list = st.session_state.get("health_scores", [])
    if not equipment_list:
        st.info("Logic synthesis required. Upload fleet data to begin.")
        return

    names = [e["equipment_name"] for e in equipment_list]
    col_input, col_main = st.columns([1, 2.5])
    
    with col_input:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("<h4>Reality Parameters</h4>", unsafe_allow_html=True)
        selected_equipment = st.selectbox("Select Asset Class", names, key="twin_eq_select")
        current_state = next((e for e in equipment_list if e["equipment_name"] == selected_equipment), {})
        
        st.markdown('<div style="margin-top:1.5rem;">', unsafe_allow_html=True)
        t_val = st.slider("Synthetic Temperature (°F)", 50, 250, int(current_state.get("health_score", 100) + 20), key="t_slider")
        v_val = st.slider("Vibration Intensity (ips)", 0.0, 1.0, 0.1, key="v_slider")
        p_val = st.slider("Pressure Load (psi)", 0, 1000, 450, key="p_slider")
        r_val = st.slider("Rotational RPM", 0, 5000, 1800, key="r_slider")
        st.markdown('</div>', unsafe_allow_html=True)
        
        run_sim = st.button("Initialize Reality Sync", type="primary", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    if run_sim:
        with col_main:
            params = {"temperature": t_val, "vibration": v_val, "pressure": p_val, "rpm": r_val}
            result = analyze_digital_twin(selected_equipment, params, current_state)
            vis_data = get_twin_visualization_data(selected_equipment, params)
            eq_type = vis_data.get("equipment_type", "generic")
            
            st.markdown('<div class="metric-label">Logic Synthesis Results</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="premium-card"><h4>Synthetic Health</h4><div class="metric-value">{vis_data["health_score"]}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="premium-card"><h4>Fleet Status</h4><div class="metric-value" style="color:{"#10B981" if vis_data["status"]=="healthy" else "#EF4444"};">{vis_data["status"].upper()}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="premium-card"><h4>Asset Class</h4><div class="metric-value">{eq_type.title()}</div></div>', unsafe_allow_html=True)
            
            col_vis, col_meta = st.columns([1.5, 1])
            with col_vis:
                st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("<h4>Reality Visualization</h4>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:320px; background:var(--bg-alt); border-radius:16px;">
                        <div style="font-size:5rem; margin-bottom:1rem;">{get_equipment_emoji(eq_type)}</div>
                        <div style="font-weight:600; color:var(--text-main); font-size:1.2rem;">{selected_equipment}</div>
                        <div style="margin-top:1rem; padding:0.5rem 1rem; background:white; border-radius:100px; border:1px solid var(--border-color); font-size:0.8rem;">
                            <span class="indicator indicator-{"healthy" if vis_data["status"]=="healthy" else "critical"}"></span> SYSTEM ACTIVE
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_meta:
                if result.get("success"):
                    analysis = result.get("analysis", {})
                    risk_intel = analysis.get("risk_assessment", {})
                    risk_level = risk_intel.get('risk_level', 'LOW')
                    risk_color = "#EF4444" if risk_level == "CRITICAL" else "#F59E0B" if risk_level == "HIGH" else "#10B981"
                    
                    st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
                    st.markdown("<h4>Risk Intelligence</h4>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="margin:1.5rem 0;">
                            <div class="metric-label">Risk Profile</div>
                            <div style="font-size:1.5rem; font-weight:700; color:{risk_color};">{risk_level}</div>
                        </div>
                        <div style="padding-top:1rem; border-top:1px solid var(--border-color);">
                            <p style="font-size:0.85rem; color:var(--text-secondary); line-height:1.6;">{analysis.get("behavior_prediction", "Stable operational parameters maintained.")}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        with col_main:
            st.markdown(
                """
                <div class="premium-card" style="height:550px; display:flex; align-items:center; justify-content:center; text-align:center; opacity:0.6;">
                    <div>
                        <div style="font-size:4rem; margin-bottom:1.5rem;">🔌</div>
                        <h4 style="margin-bottom:0.5rem;">Reality Sync Idle</h4>
                        <p>Adjust parameters and click <strong>Initialize Reality Sync</strong><br>to synthesize asset behavior in simulation-sync mode.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )




def render_chat_tab():
    """Render the premium Linear-inspired AI Chat."""
    st.markdown('<div class="metric-label" style="margin-bottom: 2rem;">Autonomous Maintenance Assistant</div>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about maintenance, equipment status, or schedules...", label_visibility="collapsed", placeholder="Type your message here...", key="chat_input")
        submit = st.form_submit_button("Send Intelligence Query", width="stretch")
            
    if submit and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        context = {
            "equipment_health": st.session_state.get("health_scores", []),
            "patterns": st.session_state.get("patterns", []),
            "schedule": st.session_state.get("schedule", []),
            "near_misses": st.session_state.get("near_misses", []),
        }
        result = chat_query(user_input, context, st.session_state.chat_history)
        response = result.get("response", "I couldn't generate a response.")
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.button("Clear History", key="clear_chat", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()


def render_machine_schedule_tab():
    """Render machine-wise scheduling with premium card depth."""
    st.markdown('<div class="metric-label">Asymmetric Asset Analytics</div>', unsafe_allow_html=True)
    
    df = st.session_state.get("cleaned_data", None)
    schedule = st.session_state.get("schedule", [])
    
    if df is None or df.empty:
        st.info("Upload data to begin asset-specific reporting.")
        return
    
    machine_list = sorted(df["equipment_id"].unique().tolist()) if "equipment_id" in df.columns else []
    
    if not machine_list:
        st.info("No assets detected.")
        return
    
    selected_machine = st.selectbox("Select Asset for Deep Analysis", machine_list, key="machine_schedule_select")
    
    machine_data = df[df["equipment_id"] == selected_machine] if "equipment_id" in df.columns else pd.DataFrame()
    
    # Asymmetrical Statistics Row
    col_meta, col_timeline = st.columns([1, 2])
    
    with col_meta:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown(f"<h4>{selected_machine}</h4>", unsafe_allow_html=True)
        total_records = len(machine_data)
        severity_counts = machine_data["severity"].value_counts() if "severity" in machine_data.columns else {}
        
        st.markdown(f"""
            <div style="margin-top:1.5rem;">
                <div class="metric-label">Total Logs</div>
                <div class="metric-value" style="font-size: 2rem;">{total_records}</div>
            </div>
            <div style="margin-top:1.5rem;">
                <div class="metric-label">Critical Alerts</div>
                <div class="metric-value" style="font-size: 2rem; color: #EF4444;">{severity_counts.get("critical", 0)}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_timeline:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("<h4>Activity Timeline</h4>", unsafe_allow_html=True)
        if "timestamp" in machine_data.columns and not machine_data.empty:
            try:
                machine_data_copy = machine_data.copy()
                machine_data_copy["date"] = pd.to_datetime(machine_data_copy["timestamp"]).dt.date
                daily_counts = machine_data_copy.groupby("date").size().reset_index(name="records")
                fig = px.area(daily_counts, x="date", y="records", color_discrete_sequence=["#4F46E5"])
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width="stretch")
            except: st.caption("Timeline unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Secondary Grid: Parameter Trends + Distribution
    col_trend, col_dist = st.columns([2, 1])
    
    with col_trend:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        numeric_cols = [c for c in ["temperature", "vibration", "pressure", "rpm"] if c in machine_data.columns]
        if numeric_cols:
            param = st.selectbox("Trend Parameter", numeric_cols, key="p_select")
            param_data = machine_data.sort_values("timestamp").tail(100)
            fig_p = px.line(param_data, x="timestamp", y=param, color_discrete_sequence=["#8B5CF6"])
            fig_p.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_p, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_dist:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        if "severity" in machine_data.columns:
            sev_df = machine_data["severity"].value_counts().reset_index()
            fig_s = px.pie(sev_df, values="count", names="severity", color_discrete_sequence=["#DC2626", "#F59E0B", "#10B981"])
            fig_s.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
            st.plotly_chart(fig_s, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Schedule Table
    st.markdown('<div class="premium-card" style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown("<h4>Maintenance Forecast</h4>", unsafe_allow_html=True)
    machine_schedule = [s for s in schedule if s.get("equipment_name") == selected_machine or s.get("equipment_id") == selected_machine]
    if machine_schedule:
        sched_df = pd.DataFrame(machine_schedule)[["recommended_date", "task", "urgency", "confidence"]]
        st.dataframe(sched_df, width="stretch", hide_index=True)
    else:
        st.info("No scheduled maintenance for this asset.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Download Actions
    st.markdown('<div style="margin-top: 2rem; display: flex; gap: 1rem;">', unsafe_allow_html=True)
    st.download_button("📥 Export CSV", machine_data.to_csv(index=False), f"{selected_machine}_report.csv")
    st.markdown('</div>', unsafe_allow_html=True)




def render_maintenance_videos(equipment_name: str):
    """Display maintenance guidance for specific equipment."""
    
    st.markdown(f"### 📺 Maintenance Guidance for: **{equipment_name}**")
    
    # Show general guidance based on equipment type
    st.markdown("#### 📚 Maintenance Resources")
    
    resources = get_maintenance_resources(equipment_name)
    for r in resources:
        st.markdown(f"- **{r['title']}**: {r['description']}")
    
    st.markdown("---")
    st.markdown("#### 🔧 Quick Maintenance Tips")
    tips = get_quick_tips(equipment_name)
    for tip in tips:
        st.markdown(f"✅ {tip}")
    
    st.markdown("---")
    st.markdown("#### 🎬 Recommended Video Topics")
    st.info("""
    1. **Equipment Safety Procedures** - Essential safety protocols
    2. **Routine Maintenance Checklist** - Daily/weekly tasks
    3. **Troubleshooting Common Issues** - Problem diagnosis
    4. **Parts Replacement Guide** - How to replace worn parts
    5. **Preventive Maintenance Best Practices** - Keeping equipment running
    """)


def get_maintenance_resources(equipment_name: str) -> list:
    """Get maintenance resources for equipment."""
    name_lower = equipment_name.lower()
    
    resources = {
        "cnc": [
            {"title": "CNC Machine Daily Maintenance Checklist", "description": "Daily, weekly, monthly maintenance tasks"},
            {"title": "CNC Spindle Maintenance", "description": "Spindle care and alignment"},
            {"title": "CNC Lubrication Guide", "description": "Proper lubrication procedures"},
        ],
        "hydraulic press": [
            {"title": "Hydraulic System Maintenance", "description": "Hydraulic fluid and filter changes"},
            {"title": "Press Safety Procedures", "description": "Safety protocols for press operation"},
        ],
        "conveyor": [
            {"title": "Conveyor Belt Tracking", "description": "Belt alignment and tracking"},
            {"title": "Roller and Bearing Maintenance", "description": "Roller replacement procedures"},
        ],
        "motor": [
            {"title": "Electric Motor Testing", "description": "Motor diagnostics and testing"},
            {"title": "Motor Bearing Replacement", "description": "Bearing change procedures"},
        ],
        "pump": [
            {"title": "Pump Seals and Gaskets", "description": "Seal replacement guide"},
            {"title": "Centrifugal Pump Maintenance", "description": "Pump upkeep procedures"},
        ],
        "compressor": [
            {"title": "Compressor Oil Changes", "description": "Oil type and change intervals"},
            {"title": "Air Filter Maintenance", "description": "Filter cleaning and replacement"},
        ],
    }
    
    for key in resources:
        if key in name_lower:
            return resources[key]
    
    return [
        {"title": "General Equipment Maintenance", "description": "Basic maintenance procedures"},
        {"title": "Predictive Maintenance Overview", "description": "Understanding PM strategies"},
    ]


def get_quick_tips(equipment_name: str) -> list:
    """Get quick maintenance tips."""
    name_lower = equipment_name.lower()
    
    tips_map = {
        "cnc": [
            "Check coolant levels daily",
            "Inspect tool holders for wear",
            "Clean chip conveyor regularly",
            "Check oil levels in spindle",
            "Inspect electrical connections",
        ],
        "motor": [
            "Listen for unusual noises",
            "Check for excessive vibration",
            "Monitor temperature during operation",
            "Inspect motor mounts",
            "Check electrical connections",
        ],
        "pump": [
            "Check for leaks regularly",
            "Monitor pressure readings",
            "Inspect seals for wear",
            "Check foundation bolts",
            "Listen for cavitation",
        ],
        "compressor": [
            "Drain condensate daily",
            "Check oil level daily",
            "Inspect belts for tension",
            "Clean air filters monthly",
            "Check safety valves",
        ],
        "conveyor": [
            "Check belt tension",
            "Inspect rollers for wear",
            "Lubricate chains",
            "Check sensors",
            "Clean belt surface",
        ],
    }
    
    for key, tips in tips_map.items():
        if key in name_lower:
            return tips
    
    return [
        "Follow manufacturer maintenance schedule",
        "Keep equipment clean and dry",
        "Monitor for unusual sounds or vibrations",
        "Check all safety devices regularly",
        "Document all maintenance activities",
    ]


    videos = get_maintenance_videos(equipment_name)
    
    if not videos:
        st.info(f"No videos available for {equipment_name}")
        return
    
    st.markdown(f"### 📺 Repair & Maintenance Videos for: **{equipment_name}**")
    
    for i, video in enumerate(videos):
        st.markdown(f"#### 🎬 {video['title']}")
        st.markdown(f"_{video['description']}_")
        
        video_url = f"https://www.youtube.com/watch?v={video['video_id']}"
        st.video(video_url)
        
        st.markdown(f"[📎 Open in YouTube]({video_url})")
        
        st.markdown("---")




def render_videos_tab():
    """Render the premium maintenance video gallery."""
    st.markdown('<div class="metric-label">Operational Guidance Library</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4>Reality-Sync Maintenance Tutorials</h4>", unsafe_allow_html=True)
    st.markdown("Select an asset from the Sidebar or Digital Twin to synthesize specific guidance, or browse the active library below:")
    
    v_cols = st.columns(2)
    with v_cols[0]:
        st.markdown("""
            <div style="margin-bottom: 2rem; padding: 1.5rem; background: var(--bg-alt); border-radius: 12px; height: 100%;">
                <h5 style="margin-top:0; font-size: 1.1rem; color: var(--indigo-main);">Precision Machining (CNC)</h5>
                <ul style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.8; padding-left: 1.2rem;">
                    <li>Daily Spindle Synchronization</li>
                    <li>Coolant Integrity Analysis</li>
                    <li>Axis Alignment Protocols</li>
                    <li>Lubrication Lifecycle Management</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with v_cols[1]:
        st.markdown("""
            <div style="margin-bottom: 2rem; padding: 1.5rem; background: var(--bg-alt); border-radius: 12px; height: 100%;">
                <h5 style="margin-top:0; font-size: 1.1rem; color: var(--indigo-main);">Power Transmission (Motors)</h5>
                <ul style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.8; padding-left: 1.2rem;">
                    <li>Bearing Acoustic Diagnostics</li>
                    <li>Thermal Signature Calibration</li>
                    <li>Emergency Stop Validation</li>
                    <li>Winding Resistance Testing</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)




def render_export_tab():
    """Render the premium report export interface."""
    st.markdown('<div class="metric-label">Intelligence Export Center</div>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.info("Logic synthesis is required before report generation.")
        return

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h4>Comprehensive Fleet Narrative</h4>", unsafe_allow_html=True)
    st.markdown("Synthesize a strategic maintenance report containing all detected patterns, predictive schedules, and AI-driven risk assessments.")

    report = generate_report_data(
        st.session_state.cleaned_data,
        st.session_state.patterns,
        st.session_state.near_misses,
        st.session_state.schedule,
        st.session_state.health_scores,
        st.session_state.analysis,
    )

    st.text_area("Narrative Preview", report, height=300)
    
    st.markdown('<div style="margin-top: 2rem; display: flex; gap: 1rem;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("📄 Export Narrative (TXT)", report, f"Nexara_Report_{datetime.now().strftime('%Y%m%d')}.txt", type="primary", width="stretch")
    with col2:
        st.button("📄 Export Strategic (DOCX)", disabled=True, type="secondary", width="stretch") # docx logic kept separate for stability
    with col3:
        # JSON Logic
        export_data = {
            "patterns": st.session_state.get("patterns", []),
            "near_misses": st.session_state.get("near_misses", []),
            "schedule": st.session_state.get("schedule", []),
            "health_scores": st.session_state.get("health_scores", []),
        }
        import json
        json_export = json.dumps(export_data, indent=2, default=str)
        st.download_button("📊 Export Logic (JSON)", json_export, f"Nexara_Logic_{datetime.now().strftime('%Y%m%d')}.json", type="primary", width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_header()

    if st.session_state.data_loaded:
        render_dashboard()
    else:
        render_landing()


if __name__ == "__main__":
    main()
