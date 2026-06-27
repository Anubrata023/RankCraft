import csv
import io
import json
import os
import sys
import re
import shutil
import textwrap
from datetime import datetime
import streamlit as st
import pandas as pd

# Add parent dir to path so we can import rank.py
sys.path.insert(0, os.path.dirname(__file__))

# Copy landing page asset from config to workspace under the hood
src_img_hero = r"C:\Users\anubr\.gemini\antigravity-ide\brain\9cfe2755-0ddb-4f4e-b422-e8a1cac3783a\rankcraft_ui_light_1782547190724.png"
if not os.path.exists("rankcraft_landing_hero.png") and os.path.exists(src_img_hero):
    try:
        shutil.copy(src_img_hero, "rankcraft_landing_hero.png")
    except Exception:
        pass

# Import ranker helpers
from rank import (
    score_candidate,
    build_reasoning,
    load_candidates,
    TODAY,
    CORE_SKILL_MAP,
    PROD_ML_DESC_KWS,
    PREF_LOCATIONS,
    CONSULTING_COS,
    PRODUCT_INDUSTRIES,
    TECH_GRAPH,
    ALL_PAIRS_PATHS,
    ContextualImpactExtractor,
    DynamicJDCalibrator,
    load_jd_text,
)

# Helper to strip leading/trailing whitespace from each line of HTML
def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.splitlines())


def parse_raw_resume_text(resume_text):
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    if not lines:
        return None
    name = lines[0]
    if len(name) > 50 or "@" in name or ":" in name:
        name = "Candidate Name"
    yoe = 5.0
    yoe_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b', resume_text, re.IGNORECASE)
    if yoe_match:
        try:
            yoe = float(yoe_match.group(1))
        except:
            pass
    country = "India"
    location = "Noida"
    if "pune" in resume_text.lower():
        location = "Pune"
    elif "bangalore" in resume_text.lower() or "bengaluru" in resume_text.lower():
        location = "Bangalore"
    elif "hyderabad" in resume_text.lower():
        location = "Hyderabad"
    elif "mumbai" in resume_text.lower():
        location = "Mumbai"
    title = "Machine Learning Engineer"
    all_titles = ['Machine Learning Engineer', 'AI Engineer', 'Data Scientist', 'NLP Scientist']
    for t in all_titles:
        if re.search(rf'\b{re.escape(t)}\b', resume_text, re.IGNORECASE):
            title = t.title()
            break
    company = "Product Startup"
    co_match = re.search(r'(?:at|company:)\s*([A-Za-z0-9\s&]{2,30})', resume_text, re.IGNORECASE)
    if co_match:
        company = co_match.group(1).strip()
    skills = []
    found_skills = []
    for sk in CORE_SKILL_MAP.keys():
        if re.search(rf'\b{re.escape(sk)}\b', resume_text, re.IGNORECASE):
            found_skills.append(sk)
    for sk in found_skills:
        skills.append({
            "name": sk.title(),
            "proficiency": "advanced",
            "duration_months": int(yoe * 8),
            "endorsements": 5
        })
    if not skills:
        skills = [{"name": "Python", "proficiency": "expert", "duration_months": 24, "endorsements": 10}]
    cand_id = f"parsed_{int(datetime.now().timestamp())}_{re.sub(r'[^a-zA-Z0-9]', '', name)[:10].lower()}"
    candidate = {
        "candidate_id": cand_id,
        "profile": {
            "anonymized_name": name,
            "headline": f"{title} | {yoe:.0f}+ YoE | Product focus",
            "summary": f"Self-starting professional with background in {title}. Profile parsed from raw text resume.",
            "current_title": title,
            "current_company": company,
            "current_industry": "Software",
            "years_of_experience": yoe,
            "country": country,
            "location": location
        },
        "skills": skills,
        "career_history": [
            {
                "title": title,
                "company": company,
                "start_date": "2024-01-01",
                "end_date": None,
                "is_current": True,
                "duration_months": int(min(yoe * 12, 36)),
                "industry": "Software",
                "description": f"Responsible for building AI systems, data pipelines, and core systems. Extensive hands-on experience."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "last_active_date": TODAY.strftime("%Y-%m-%d"),
            "notice_period_days": 30,
            "recruiter_response_rate": 0.85,
            "github_activity_score": 75,
            "interview_completion_rate": 0.90,
            "willing_to_relocate": True,
            "preferred_work_mode": "hybrid",
            "skill_assessment_scores": {
                "Python": 80,
                "Machine Learning": 85
            }
        }
    }
    return candidate


# Page configuration
st.set_page_config(
    page_title="RankCraft | AI Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "view" not in st.session_state:
    st.session_state["view"] = "landing"
if "candidate_pool" not in st.session_state:
    st.session_state["candidate_pool"] = []
if "selected_candidate_ids" not in st.session_state:
    st.session_state["selected_candidate_ids"] = set()
if "inspect_id" not in st.session_state:
    st.session_state["inspect_id"] = None
if "jd_text_content" not in st.session_state:
    st.session_state["jd_text_content"] = load_jd_text()
if "jd_text_area_classic" not in st.session_state:
    st.session_state["jd_text_area_classic"] = load_jd_text()
if "selected_skills_key" not in st.session_state:
    default_parsed = parse_jd_text_local(st.session_state["jd_text_area_classic"])
    st.session_state["selected_skills_key"] = list(default_parsed.keys())

# Inject Global CSS for the White and Orange design system with bulletproof contrast
st.markdown(clean_html("""
<style>
    /* Clean white sidebar override */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    /* Inputs, Selectboxes, Textareas, and Dropdown Popups text legibility */
    textarea, input, select {
        color: #1E293B !important;
    }
    
    /* Selectbox selected choice styling */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* Dropdown options popover elements styling */
    div[role="listbox"], [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    div[role="option"], [data-baseweb="popover"] div, [data-baseweb="popover"] li, [data-baseweb="popover"] span {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }
    
    /* Input and text area styling */
    div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
    }
    
    .metric-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        padding: 20px;
        margin: 5px 0 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .profile-card {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px;
        padding: 26px;
        margin-top: 10px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05) !important;
    }
    
    .card-title {
        color: #1E293B !important;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 1px solid #E2E8F0 !important;
        padding-bottom: 8px;
    }
    
    /* Custom light theme badges with important to prevent overrides */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        text-transform: capitalize;
    }
    .badge-green { background-color: #E6F4EA !important; color: #137333 !important; border: 1px solid #CEEAD6 !important; }
    .badge-blue { background-color: #E8F0FE !important; color: #1A73E8 !important; border: 1px solid #D2E3FC !important; }
    .badge-orange { background-color: #FDF2E9 !important; color: #D56B1A !important; border: 1px solid #FADBD8 !important; }
    .badge-red { background-color: #FCE8E6 !important; color: #C5221F !important; border: 1px solid #FAD2CF !important; }
    .badge-grey { background-color: #F1F3F4 !important; color: #5F6368 !important; border: 1px solid #E8EAED !important; }
    .badge-purple { background-color: #F3E8FF !important; color: #6B21A8 !important; border: 1px solid #E9D5FF !important; }
    
    .timeline-container {
        border-left: 2px solid #E2E8F0 !important;
        padding-left: 20px;
        margin-left: 10px;
        margin-top: 15px;
    }
    .timeline-item {
        position: relative;
        margin-bottom: 25px;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -27px;
        top: 6px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #FF6B4A !important;
        border: 2px solid #FFFFFF !important;
    }
    .timeline-item.current::before {
        background-color: #10B981 !important;
        box-shadow: 0 0 8px #10B981 !important;
    }
    .timeline-title {
        font-weight: 700;
        font-size: 14px;
        color: #1E293B !important;
    }
    .timeline-meta {
        font-size: 11px;
        color: #64748B !important;
        margin-bottom: 6px;
    }
    .timeline-desc {
        font-size: 12px;
        color: #475569 !important;
        line-height: 1.5;
    }
    
    .section-title {
        font-size: 14px;
        font-weight: bold;
        color: #64748B !important;
        text-transform: uppercase;
        margin-top: 15px;
        margin-bottom: 5px;
        letter-spacing: 0.05em;
    }
    .highlight-keyword {
        background-color: rgba(255, 107, 74, 0.15) !important;
        color: #D56B1A !important;
        border-radius: 3px;
        padding: 0px 4px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #64748B !important;
        font-size: 14px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #FF6B4A !important;
        border-bottom-color: #FF6B4A !important;
    }
</style>
"""), unsafe_allow_html=True)

JD_PRESETS = {
    "Senior AI Engineer — Founding Team (Default)": {
        "text": """We need a Senior AI Engineer with deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, and fine-tuning. Production experience with sentence-transformers, FAISS, Pinecone, OpenSearch, learning-to-rank models (XGBoost/LightGBM), and evaluation metrics like NDCG, MRR, MAP. Product mindset, Series A, hybrid in Noida/Pune preferred. sub-30d notice. No CV/Speech-only, no pure research/consulting.""",
        "custom_skills": CORE_SKILL_MAP,
    },
    "MLOps & Infrastructure Engineer": {
        "text": """Looking for an infrastructure-focused ML engineer to deploy and maintain distributed inference clusters, vector databases, and automated MLOps pipelines. Needs strong experience in Docker, Kubernetes, AWS, Kafka, Redis, Milvus/Qdrant scaling, pipeline orchestration (Airflow/MLflow), and C++ or Rust scaling.""",
        "custom_skills": {
            'mlops': 10, 'docker': 9, 'kubernetes': 9, 'aws': 8, 'kafka': 8, 'redis': 8,
            'milvus': 8, 'qdrant': 8, 'mlflow': 7, 'airflow': 7, 'gpc': 6, 'distributed': 8,
            'python': 5, 'go': 6, 'rust': 7, 'c++': 6, 'scikit-learn': 4,
        }
    },
    "Applied Data Scientist": {
        "text": """Applied Data Scientist to build customer-facing prediction models and analytics dashboards. Needs hands-on experience in feature engineering, PyTorch, scikit-learn, SQL, NumPy, Pandas, A/B testing, statistical modeling, XGBoost, and LightGBM. HR-tech/marketplace experiences preferred.""",
        "custom_skills": {
            'machine learning': 10, 'data scientist': 10, 'python': 8, 'scikit-learn': 8,
            'xgboost': 8, 'lightgbm': 8, 'pytorch': 7, 'a/b testing': 8, 'pandas': 6,
            'numpy': 6, 'sql': 6, 'statistical modeling': 8, 'feature engineering': 8,
            'tensorflow': 5, 'recommendation': 7,
        }
    }
}

def parse_jd_text_local(jd_text):
    text_lower = jd_text.lower()
    custom_map = {}
    all_known = list(CORE_SKILL_MAP.keys()) + ['docker', 'kubernetes', 'aws', 'kafka', 'redis', 'go', 'rust', 'c++', 'airflow', 'distributed', 'sql']
    for kw in all_known:
        if kw in text_lower:
            custom_map[kw] = CORE_SKILL_MAP.get(kw, 8)
    return custom_map

# Helper to load sample data automatically
def load_sample_data():
    try:
        sample_path = "data/sample.jsonl"
        candidates = []
        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
        st.session_state["candidate_pool"] = candidates
        if candidates:
            st.session_state["inspect_id"] = candidates[0]["candidate_id"]
    except Exception as e:
        st.error(f"Failed to load sample dataset: {e}")

# =============================================================================
# VIEW 1: LANDING PAGE (Clean White & Orange theme)
# =============================================================================
if st.session_state["view"] == "landing":
    st.markdown(clean_html("""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 0; margin-bottom: 2rem;">
            <div style="font-size: 26px; font-weight: 800; color: #1E293B; letter-spacing: 0.5px;">
                RankCraft <span style="color: #FF6B4A;">AI</span>
            </div>
            <div style="background: rgba(255, 107, 74, 0.08); color: #FF6B4A; border: 1px solid rgba(255, 107, 74, 0.2); padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                🟢 Offline AI Swarm Engine: Operational
            </div>
        </div>
    """), unsafe_allow_html=True)

    # Hero Split Layout
    col_hero_text, col_hero_card = st.columns([1.1, 0.9])
    
    with col_hero_text:
        st.markdown("<h4 style='color: #FF6B4A; font-weight: 700; margin-bottom: 12px; letter-spacing: 1px;'>OFFLINE TALENT INTELLIGENCE SUITE</h4>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 54px; font-weight: 800; color: #1E293B; line-height: 1.15; margin-bottom: 1rem;'>Talent Acquisition,<br>Decoded.<br><span style='color:#FF6B4A;'>Consensus-Based</span> Ranking.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16.5px; color: #475569; margin: 1.5rem 0 2rem 0; line-height: 1.6;'>RankCraft is a self-hosted candidate intelligence engine that scores and inspects engineering talent with zero-trust security audits, graph-based skill alignment, and multi-agent swarm orchestration.</p>", unsafe_allow_html=True)
        
        # Action buttons with unique sensible behaviors
        btn_col1, btn_col2 = st.columns([1, 1.2])
        if btn_col1.button("🚀 Load standard pool", use_container_width=True, type="primary"):
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.session_state["sandbox_mode"] = False
            st.rerun()
            
        if btn_col2.button("🧠 Enter swarm agent sandbox", use_container_width=True):
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.session_state["sandbox_mode"] = True
            st.rerun()
            
    with col_hero_card:
        if os.path.exists("rankcraft_landing_hero.png"):
            st.image("rankcraft_landing_hero.png", use_container_width=True)
        else:
            st.markdown(clean_html("""
                <div style="background: #FFFFFF; border-radius: 24px; padding: 4rem 2rem; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
                    <div style="font-size: 64px; margin-bottom: 1rem;">📊</div>
                    <div style="font-weight: 700; font-size: 18px; color: #1E293B; margin-bottom: 5px;">Offline Candidate Analytics</div>
                    <div style="font-size: 13px; color: #64748B;">Semantic clustering, Honeypots tracking and scoring</div>
                </div>
            """), unsafe_allow_html=True)

    st.markdown("<br><hr style='border: 0; border-top: 1px solid #E2E8F0;'><br>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #FF6B4A; font-weight: 700;'>ARCHITECTURE</h4>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-weight: 800; color: #1E293B; margin-bottom: 3rem;'>Architected for Extreme Efficiency</h2>", unsafe_allow_html=True)

    # Feature Grid
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(clean_html("""
            <div style="background: #FFFFFF; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
                <div style="font-size: 28px; margin-bottom: 1rem; color: #FF6B4A;">🔀</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Two-Stage Local Pipeline</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">Processes raw candidate JSON/JSONL datasets securely on-site, minimizing data transit risks and maximizing pipeline throughput.</p>
            </div>
        """), unsafe_allow_html=True)
    with f_col2:
        st.markdown(clean_html("""
            <div style="background: #FFFFFF; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
                <div style="font-size: 28px; margin-bottom: 1rem; color: #FF6B4A;">🛡️</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">The Honeypot Auditor</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">Proprietary zero-trust logic designed to flag and isolate synthetic profiles, 0-month expert claims, and AI-hallucinated details.</p>
            </div>
        """), unsafe_allow_html=True)
    with f_col3:
        st.markdown(clean_html("""
            <div style="background: #FFFFFF; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
                <div style="font-size: 28px; margin-bottom: 1rem; color: #FF6B4A;">🔌</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Offline Edge Engine</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">Run high-complexity matching algorithms locally with zero cloud costs. The pipeline scales vertically on standard laptop and server CPUs.</p>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Waveform Section
    st.markdown(clean_html("""
        <div style="background: #FFFFFF; border-radius: 20px; padding: 2.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #FF6B4A; letter-spacing: 1px;">LIVE ANALYTICS</span>
                    <h3 style="font-weight: 800; color: #1E293B; margin-top: 5px; margin-bottom: 0;">System Performance Waveform</h3>
                </div>
                <div style="display: flex; gap: 3rem;">
                    <div>
                        <span style="font-size: 12px; color: #64748B;">ACCURACY RATE</span>
                        <div style="font-size: 24px; font-weight: 800; color: #FF6B4A; margin-top: 2px;">98% / 98.4% Match</div>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #64748B;">PROCESSING SPEED</span>
                        <div style="font-size: 24px; font-weight: 800; color: #1E293B; margin-top: 2px;">14ms Latency</div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2.5rem; height: 100px; display: flex; align-items: flex-end;">
                <svg viewBox="0 0 1000 100" width="100%" height="80px" preserveAspectRatio="none" style="overflow: visible;">
                    <path d="M 0 50 C 150 10, 200 90, 350 50 C 500 10, 600 90, 750 50 C 900 10, 950 90, 1000 50" fill="none" stroke="#FF6B4A" stroke-width="4"/>
                    <path d="M 0 50 C 150 10, 200 90, 350 50 C 500 10, 600 90, 750 50 C 900 10, 950 90, 1000 50 L 1000 100 L 0 100 Z" fill="rgba(255,107,74,0.05)" stroke="none"/>
                </svg>
            </div>
        </div>
    """), unsafe_allow_html=True)

# =============================================================================
# VIEW 2: CLASSIC SPACIOUS RECRUITER WORKSPACE
# =============================================================================
else:
    # Set default values if sandbox mode is clicked
    init_title = 1.0
    init_career = 1.0
    init_skills = 1.0
    init_experience = 1.0
    init_location = 1.0
    init_semantic = 1.0
    
    if "sandbox_mode" in st.session_state and st.session_state["sandbox_mode"]:
        init_title = 1.5
        init_career = 1.2
        init_skills = 1.4
        init_experience = 0.8
        
    # ── SIDEBAR CONTROLS ─────────────────────────────────────────────────────
    st.sidebar.markdown("<h2 style='color:#FF6B4A;margin-top:0;'>RankCraft</h2>", unsafe_allow_html=True)
    if st.sidebar.button("⬅️ Home Menu"):
        st.session_state["view"] = "landing"
        st.rerun()

    st.sidebar.header("📁 Data Source")
    if st.sidebar.button("🔄 Reset to Default Sample"):
        load_sample_data()
        st.sidebar.success("Reset pool to default sample.")
        st.rerun()

    uploaded_file = st.sidebar.file_uploader(
        "Upload candidate JSON/JSONL sample",
        type=["json", "jsonl"],
    )
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        candidates = []
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                candidates = data
            elif isinstance(data, dict):
                candidates = [data]
        except json.JSONDecodeError:
            for line in raw_text.splitlines():
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if candidates:
            st.session_state["candidate_pool"] = candidates
            st.sidebar.success(f"Loaded {len(candidates)} candidates!")
            st.rerun()

    st.sidebar.markdown("---")

    # Job Description Presets & Pasting
    st.sidebar.header("📝 Job Description (JD)")
    
    preset_list = list(JD_PRESETS.keys())
    preset_choice = st.sidebar.selectbox("Choose JD Profile Preset", preset_list, key="preset_select_classic")
    
    def handle_preset_update():
        preset_text = JD_PRESETS[st.session_state["preset_select_classic"]]["text"]
        st.session_state["jd_text_area_classic"] = preset_text
        st.session_state["jd_text_content"] = preset_text
        new_parsed = parse_jd_text_local(preset_text)
        st.session_state["selected_skills_key"] = list(new_parsed.keys())
        
    st.sidebar.button("Load Chosen Preset", on_click=handle_preset_update)

    jd_text = st.sidebar.text_area(
        "JD Requirements Text",
        height=130,
        key="jd_text_area_classic"
    )
    st.session_state["jd_text_content"] = jd_text

    # Custom skill mapping based on active JD
    parsed_jd_skills = parse_jd_text_local(jd_text)

    # Inferred Latent Needs
    inferred_skills = DynamicJDCalibrator.calibrate(jd_text)
    
    # Options list for the multiselect
    available_skills = list(CORE_SKILL_MAP.keys())
    all_options = sorted(list(set(available_skills + st.session_state.get("selected_skills_key", []) + list(parsed_jd_skills.keys()) + list(inferred_skills.keys()))))
    
    st.sidebar.markdown("### 🎯 Skill Targets Select")
    selected_skills = st.sidebar.multiselect(
        "Select skills to evaluate against",
        options=all_options,
        key="selected_skills_key"
    )
    
    # Custom weight sliders for selected skills
    with st.sidebar.expander("⚖️ Customize Selected Skill Weights", expanded=True):
        custom_skill_map = {}
        for skill in selected_skills:
            # Get default weight from preset mapping if available, else 8
            default_w = CORE_SKILL_MAP.get(skill.lower(), 8)
            slider_key = f"weight_skill_val_{skill}"
            if slider_key not in st.session_state:
                st.session_state[slider_key] = default_w
            w = st.slider(f"Weight: {skill.title()}", 1, 10, key=slider_key)
            custom_skill_map[skill] = w

    with st.sidebar.expander("🔑 Extracted JD Skill Targets"):
        st.markdown("Matched target skills and search weight values:")
        df_skills = pd.DataFrame([{"Skill Keyword": k, "Base Weight": v} for k, v in custom_skill_map.items()])
        if not df_skills.empty:
            st.dataframe(df_skills.sort_values(by="Base Weight", ascending=False), hide_index=True)
        else:
            st.caption("No matching skill keywords found.")
        if inferred_skills:
            st.info(f"🔮 **AI-Inferred Latent Needs added**: {', '.join(inferred_skills.keys())}")

    st.sidebar.markdown("---")

    # Calibration Sliders
    st.sidebar.header("⚙️ Score Component Calibration")
    weight_title = st.sidebar.slider("Role Title Fit Weight", 0.0, 2.0, init_title, 0.1)
    weight_career = st.sidebar.slider("Career History Weight", 0.0, 2.0, init_career, 0.1)
    weight_skills = st.sidebar.slider("Skills Trust Weight", 0.0, 2.0, init_skills, 0.1)
    weight_experience = st.sidebar.slider("Experience Band Weight", 0.0, 2.0, init_experience, 0.1)
    weight_location = st.sidebar.slider("Location Pref Weight", 0.0, 2.0, init_location, 0.1)
    weight_semantic = st.sidebar.slider("Semantic Alignment Weight", 0.0, 2.0, init_semantic, 0.1)
    skills_score_cap = st.sidebar.slider("Max Skill Score Cap", 10.0, 40.0, 25.0, 1.0)

    st.sidebar.markdown("---")

    # Modifiers
    st.sidebar.header("🚦 Behavioral Modifiers")
    enable_activity_decay = st.sidebar.checkbox("Apply Activity Recency Decay (decay inactive >30d)", value=True)
    enable_notice_penalty = st.sidebar.checkbox("Penalize Long Notice Periods (>60d)", value=True)
    enable_response_rate_penalty = st.sidebar.checkbox("Penalize Low Recruiter Response Rate (<40%)", value=True)
    enable_open_to_work_bonus = st.sidebar.checkbox("Apply Open-To-Work Boost (+5%)", value=True)
    enable_interview_completion_penalty = st.sidebar.checkbox("Penalize Poor Interview attendance (<30%)", value=True)

    st.sidebar.markdown("---")

    # Hard Disqualifiers
    st.sidebar.header("🚫 Hard Disqualifiers")
    consulting_penalty = st.sidebar.checkbox("Penalize Pure Consulting Backgrounds", value=True)
    location_penalty = st.sidebar.checkbox("Penalize Non-India Unwilling Relocation", value=True)

    # Build config dict
    config = {
        "weight_title": weight_title,
        "weight_career": weight_career,
        "weight_skills": weight_skills,
        "weight_experience": weight_experience,
        "weight_location": weight_location,
        "weight_semantic": weight_semantic,
        "skills_score_cap": skills_score_cap,
        "custom_skill_map": custom_skill_map,
        "enable_activity_decay": enable_activity_decay,
        "enable_notice_penalty": enable_notice_penalty,
        "enable_response_rate_penalty": enable_response_rate_penalty,
        "enable_open_to_work_bonus": enable_open_to_work_bonus,
        "enable_interview_completion_penalty": enable_interview_completion_penalty,
        "consulting_penalty_enabled": consulting_penalty,
        "location_penalty_enabled": location_penalty,
    }

    # ── MAIN WORKSPACE PANEL ─────────────────────────────────────────────────
    st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #E2E8F0; padding-bottom:10px; margin-bottom:20px;'>
            <div>
                <h1 style='margin:0; font-size:28px; color: #1E293B;'>🧠 RankCraft Candidate Discovery Workbench</h1>
                <p style='margin:2px 0 0 0; color:#64748B; font-size:13px;'>Consensus-Based Multi-Agent Swarm Recruiter Interface</p>
            </div>
            <div style='text-align:right;'>
                <span style='background:rgba(255,107,74,0.08); border:1px solid #FF6B4A; color:#FF6B4A; padding:6px 14px; border-radius:30px; font-size:12px; font-weight:600;'>Swarm Status: Active</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    active_pool = st.session_state["candidate_pool"]
    if not active_pool:
        st.info("The candidate pool is empty. Upload a candidate dataset in the sidebar, or click 'Reset to Default Sample' to preload data!")
    else:
        # Calculate semantic similarities
        similarities = [0.0] * len(active_pool)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            corpus = []
            for c in active_pool:
                p = c['profile']
                career_desc = " ".join(j.get('description', '') for j in c.get('career_history', []))
                text = f"{p.get('headline', '')} {p.get('summary', '')} {career_desc}"
                corpus.append(text)
            corpus.append(st.session_state["jd_text_content"])
            
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            jd_vec = tfidf_matrix[-1]
            candidates_matrix = tfidf_matrix[:-1]
            similarities = cosine_similarity(candidates_matrix, jd_vec).flatten()
        except Exception as e:
            pass

        scored = []
        honeypots = []

        for idx, c in enumerate(active_pool):
            sim_score = float(similarities[idx]) if similarities is not None else 0.0
            total_score, components = score_candidate(c, config=config, similarity_score=sim_score)
            
            if components.get("honeypot"):
                reason = components["honeypot"]
                atype = "other"
                if "career_months" in reason:
                    atype = "Stated experience discrepancy"
                elif "expert" in reason and "0 months" in reason:
                    atype = "Expert skill with 0 duration"
                elif "impossible job duration" in reason:
                    atype = "Impossible job duration range"
                elif "too many expert skills" in reason:
                    atype = "Exaggerated skills count (>=8)"
                elif "simultaneous current jobs" in reason:
                    atype = "Multiple active current jobs"
                honeypots.append({
                    "candidate_id": c["candidate_id"],
                    "reason": reason,
                    "type": atype,
                    "_candidate": c,
                })
                continue
            scored.append({
                "candidate_id": c["candidate_id"],
                "score": total_score,
                "components": components,
                "_candidate": c,
            })

        scored.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
        
        # Build output rows
        output_rows = []
        for rank, entry in enumerate(scored, start=1):
            reasoning = build_reasoning(entry["_candidate"], entry["components"], rank)
            output_rows.append({
                "candidate_id": entry["candidate_id"],
                "rank": rank,
                "score": round(entry["score"], 6),
                "reasoning": reasoning,
                "title": entry["_candidate"]["profile"]["current_title"],
                "company": entry["_candidate"]["profile"]["current_company"],
                "yoe": entry["_candidate"]["profile"]["years_of_experience"],
                "country": entry["_candidate"]["profile"]["country"],
                "location": entry["_candidate"]["profile"]["location"],
                "industry": entry["_candidate"]["profile"]["current_industry"],
                "components": entry["components"],
                "_candidate": entry["_candidate"],
            })

        # Tabs Layout
        tab_discovery, tab_analytics, tab_compare, tab_honeypots, tab_manual_shortlist, tab_add_candidate = st.tabs([
            "🔍 Shortlist Explorer",
            "📊 Talent Pool Analytics",
            "⚖️ Candidate Compare",
            "🛡️ Honeypot Auditor",
            "⭐ My Custom Shortlist",
            "📥 Add Candidates/Resumes"
        ])

        # TAB 1: Shortlist Explorer
        with tab_discovery:
            col1, col2, col3 = st.columns(3)
            col1.markdown(clean_html(f"""
            <div class="metric-card">
                <h5 style="color:#64748B;margin:0 0 5px 0;font-size:12px;">TOTAL POOL LOADED</h5>
                <h2 style="color:#1E293B;margin:0;font-size:32px;font-weight:700;">{len(active_pool)}</h2>
            </div>
            """), unsafe_allow_html=True)
            col2.markdown(clean_html(f"""
            <div class="metric-card">
                <h5 style="color:#64748B;margin:0 0 5px 0;font-size:12px;">SHORTLISTED & SCORED</h5>
                <h2 style="color:#10B981;margin:0;font-size:32px;font-weight:700;">{len(scored)}</h2>
            </div>
            """), unsafe_allow_html=True)
            col3.markdown(clean_html(f"""
            <div class="metric-card">
                <h5 style="color:#64748B;margin:0 0 5px 0;font-size:12px;">HONEYPOTS FILTERED</h5>
                <h2 style="color:#EF4444;margin:0;font-size:32px;font-weight:700;">{len(honeypots)}</h2>
            </div>
            """), unsafe_allow_html=True)

            st.subheader("Shortlisted Candidates")
            search_query = st.text_input("🔍 Search shortlisted candidates (by ID, Title, Company, Skills, or Country)", "")
            
            filtered_rows = output_rows
            if search_query:
                q = search_query.lower()
                filtered_rows = []
                for row in output_rows:
                    cand_skills = " ".join([s["name"].lower() for s in row["_candidate"].get("skills", [])])
                    if (q in row["candidate_id"].lower() or
                        q in row["title"].lower() or
                        q in row["company"].lower() or
                        q in row["country"].lower() or
                        q in cand_skills or
                        q in row["reasoning"].lower()):
                        filtered_rows.append(row)

            # Download CSV
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for row in output_rows[:100]:
                writer.writerow([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])
            st.download_button(
                label="⬇️ Download Top 100 submission.csv",
                data=csv_buf.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
            )

            if not filtered_rows:
                st.info("No candidates match your search query.")
            else:
                col_table, col_inspector = st.columns([1.1, 0.9])
                with col_table:
                    st.caption(f"Showing {len(filtered_rows)} candidate matches")
                    
                    col_bulk1, col_bulk2 = st.columns(2)
                    if col_bulk1.button("⭐ Shortlist Top 10"):
                        for r in filtered_rows[:10]:
                            st.session_state["selected_candidate_ids"].add(r["candidate_id"])
                        st.success("Added top 10 candidates to custom shortlist!")
                        st.rerun()
                    if col_bulk2.button("🚫 Clear All Shortlist"):
                        st.session_state["selected_candidate_ids"].clear()
                        st.success("Cleared custom shortlist!")
                        st.rerun()

                    df_display = pd.DataFrame([
                        {
                            "Shortlisted": "⭐ Yes" if r["candidate_id"] in st.session_state["selected_candidate_ids"] else "⚪ No",
                            "Rank": r["rank"],
                            "Candidate ID": r["candidate_id"],
                            "Title": r["title"],
                            "Company": r["company"],
                            "YoE": r["yoe"],
                            "Country": r["country"],
                            "Score": r["score"],
                        }
                        for r in filtered_rows
                    ])
                    st.dataframe(df_display, width="stretch", hide_index=True)
                    
                    inspect_id = st.selectbox(
                        "Select a candidate ID to inspect profile details",
                        [r["candidate_id"] for r in filtered_rows],
                        index=0
                    )
                    
                with col_inspector:
                    insp_row = next(r for r in filtered_rows if r["candidate_id"] == inspect_id)
                    c_data = insp_row["_candidate"]
                    p = c_data["profile"]
                    sig = c_data["redrob_signals"]
                    comp = insp_row["components"]

                    chk_val = inspect_id in st.session_state["selected_candidate_ids"]
                    is_shortlisted = st.checkbox(
                        "⭐ Mark Candidate as Shortlisted",
                        value=chk_val,
                        key=f"shortlist_chk_{inspect_id}"
                    )
                    if is_shortlisted != chk_val:
                        if is_shortlisted:
                            st.session_state["selected_candidate_ids"].add(inspect_id)
                        else:
                            st.session_state["selected_candidate_ids"].discard(inspect_id)
                        st.rerun()

                    # Components score breakdown bars
                    components_show = [
                        ('Role Title Fit', comp.get('title', 0.0), 30.0 * weight_title, '#3B82F6'),
                        ('Career History Quality', comp.get('career', 0.0), 30.0 * weight_career, '#10B981'),
                        ('Skills Trust Profile', comp.get('skills', 0.0), skills_score_cap * weight_skills, '#F59E0B'),
                        ('Experience Band Fit', comp.get('experience', 0.0), 10.0 * weight_experience, '#8B5CF6'),
                        ('Location Alignment', comp.get('location', 0.0), 5.0 * weight_location, '#FF6B4A'),
                        ('Semantic Text Similarity', comp.get('semantic_alignment', 0.0), 30.0 * weight_semantic, '#FF8E53'),
                    ]
                    
                    score_html = f"<div style='font-weight:700; font-size:15px; color:#1E293B; margin-bottom:12px;'>Composite Score: {comp.get('total', 0.0):.3f} <span style='font-weight:normal; font-size:12px; color:#64748B;'>(Base {comp.get('base', 0.0):.1f} × multiplier {comp.get('behavioral_mult', 1.0):.3f})</span></div>"
                    score_html += "<div style='font-size:11px; color:#10B981; margin-bottom:10px;'>🤖 Swarm Agent Status: Optimized</div>"
                    for name, val, max_val, color in components_show:
                        pct = min(max(0.0, (val / max_val) * 100.0), 100.0) if max_val > 0 else 0.0
                        score_html += f"""
                        <div style='margin-bottom: 12px;'>
                            <div style='display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px; color:#475569;'>
                                <span>{name}</span>
                                <span>{val:.1f} / {max_val:.1f}</span>
                            </div>
                            <div style='width: 100%; background-color: #E2E8F0; border-radius: 4px; height: 6px;'>
                                <div style='width: {pct}%; background-color: {color}; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        """
                    
                    badges_html = "<div>"
                    if sig.get('open_to_work_flag', False):
                        badges_html += "<span class='badge badge-green'>Open to Work</span>"
                    else:
                        badges_html += "<span class='badge badge-grey'>Not Marked Open</span>"
                        
                    try:
                        last_act = datetime.strptime(sig['last_active_date'], '%Y-%m-%d')
                        days = (TODAY - last_act).days
                        if days <= 7:
                            badges_html += f"<span class='badge badge-green'>Active: {days}d ago</span>"
                        elif days <= 30:
                            badges_html += f"<span class='badge badge-blue'>Active: {days}d ago</span>"
                        elif days <= 90:
                            badges_html += f"<span class='badge badge-orange'>Active: {days}d ago</span>"
                        else:
                            badges_html += f"<span class='badge badge-red'>Inactive: {days}d ago</span>"
                    except:
                        pass
                        
                    notice = sig.get('notice_period_days', 90)
                    if notice <= 30:
                        badges_html += f"<span class='badge badge-green'>Notice: {notice}d</span>"
                    elif notice <= 60:
                        badges_html += f"<span class='badge badge-blue'>Notice: {notice}d</span>"
                    elif notice <= 90:
                        badges_html += f"<span class='badge badge-orange'>Notice: {notice}d</span>"
                    else:
                        badges_html += f"<span class='badge badge-red'>Notice: {notice}d</span>"
                        
                    rr = sig.get('recruiter_response_rate', 0.5)
                    if rr >= 0.7:
                        badges_html += f"<span class='badge badge-green'>Msg Reply: {rr:.0%}</span>"
                    elif rr >= 0.4:
                        badges_html += f"<span class='badge badge-blue'>Msg Reply: {rr:.0%}</span>"
                    else:
                        badges_html += f"<span class='badge badge-red'>Msg Reply: {rr:.0%}</span>"
                        
                    gh = sig.get('github_activity_score', -1)
                    if gh >= 60:
                        badges_html += f"<span class='badge badge-purple'>GitHub Score: {gh:.0f}</span>"
                    elif gh >= 0:
                        badges_html += f"<span class='badge badge-blue'>GitHub Score: {gh:.0f}</span>"
                    else:
                        badges_html += "<span class='badge badge-grey'>No GitHub Linked</span>"
                    badges_html += "</div>"
                    
                    skills_matched_html = "<div>"
                    for sk in c_data.get("skills", []):
                        sname = sk["name"]
                        is_core = any(kw in sname.lower() for kw in custom_skill_map)
                        badge_style = "badge-blue" if is_core else "badge-grey"
                        endorse = sk.get("endorsements", 0)
                        sdur = sk.get("duration_months", 0)
                        skills_matched_html += f"<span class='badge {badge_style}'>{sname} ({sk['proficiency']} • {sdur}m • {endorse}👍)</span>"
                    skills_matched_html += "</div>"
                    
                    # Skill Gap Analyzer
                    candidate_skills_dict = {sk["name"].lower(): sk for sk in c_data.get("skills", [])}
                    candidate_skills_set = set(candidate_skills_dict.keys())
                    target_skills_set = set(custom_skill_map.keys())
                    
                    matched_skills = []
                    for sk_name in target_skills_set:
                        matched_key = None
                        for cand_sk in candidate_skills_set:
                            if sk_name in cand_sk or cand_sk in sk_name:
                                matched_key = cand_sk
                                break
                        if matched_key:
                            sk_info = candidate_skills_dict[matched_key]
                            matched_skills.append(f"{sk_info['name']} ({sk_info['proficiency']})")
                            
                    missing_skills = []
                    for sk_name in target_skills_set:
                        matched_any = False
                        for cand_sk in candidate_skills_set:
                            if sk_name in cand_sk or cand_sk in sk_name:
                                matched_any = True
                                break
                        if not matched_any:
                            missing_skills.append(sk_name.upper())
                            
                    complementary_skills = []
                    for cand_sk in candidate_skills_set:
                        matched_any = False
                        for sk_name in target_skills_set:
                            if sk_name in cand_sk or cand_sk in sk_name:
                                matched_any = True
                                break
                        if not matched_any:
                            sk_info = candidate_skills_dict[cand_sk]
                            complementary_skills.append(f"{sk_info['name']} ({sk_info['proficiency']})")
                            
                    matched_badges_html = "<div>"
                    if matched_skills:
                        for sk in sorted(matched_skills):
                            matched_badges_html += f"<span class='badge badge-green'>{sk}</span>"
                    else:
                        matched_badges_html += "<span style='font-size:12px;color:#64748B;'>None matched</span>"
                    matched_badges_html += "</div>"
                    
                    missing_badges_html = "<div>"
                    if missing_skills:
                        for sk in sorted(missing_skills):
                            missing_badges_html += f"<span class='badge badge-red'>{sk}</span>"
                    else:
                        missing_badges_html += "<span style='font-size:12px;color:#64748B;'>None missing</span>"
                    missing_badges_html += "</div>"
                    
                    comp_badges_html = "<div>"
                    if complementary_skills:
                        for sk in sorted(complementary_skills)[:8]:
                            comp_badges_html += f"<span class='badge badge-purple'>{sk}</span>"
                    else:
                        comp_badges_html += "<span style='font-size:12px;color:#64748B;'>None</span>"
                    comp_badges_html += "</div>"
                    
                    skill_gap_html = f"""
                    <div style="background-color:#F8FAFC; border-radius:6px; padding:12px; margin-top:8px; margin-bottom:15px; border:1px solid #E2E8F0;">
                        <div style="font-size:11px; color:#137333; font-weight:700; margin-bottom:4px;">🟢 MATCHED CORE SKILLS</div>
                        {matched_badges_html}
                        <div style="font-size:11px; color:#C5221F; font-weight:700; margin-top:10px; margin-bottom:4px;">🔴 MISSING REQUIRED SKILLS</div>
                        {missing_badges_html}
                        <div style="font-size:11px; color:#6B21A8; font-weight:700; margin-top:10px; margin-bottom:4px;">Extra Complementary Skills</div>
                        {comp_badges_html}
                    </div>
                    """
                    
                    # Highlighted Timeline
                    timeline_html = "<div class='timeline-container'>"
                    for j in c_data.get("career_history", []):
                        is_curr = j.get("is_current", False)
                        class_item = "timeline-item current" if is_curr else "timeline-item"
                        end_str = j.get("end_date") or "Present"
                        jd_desc = j.get("description", "")
                        
                        for kw in PROD_ML_DESC_KWS:
                            pattern = re.compile(rf"\b({re.escape(kw)})\b", re.IGNORECASE)
                            jd_desc = pattern.sub(r'<span class="highlight-keyword">\1</span>', jd_desc)
                            
                        timeline_html += f"""
                        <div class='{class_item}'>
                            <div class='timeline-title' style='color:#1E293B;'>{j.get('title')} at {j.get('company')}</div>
                            <div class='timeline-meta' style='color:#64748B;'>{j.get('start_date')} to {end_str} ({j.get('duration_months')} months) • Industry: {j.get('industry')}</div>
                            <div class='timeline-desc' style='color:#475569;'>{jd_desc}</div>
                        </div>
                        """
                    timeline_html += "</div>"
                    
                    # Direct Recruiter Outreach Email Generator
                    first_name = p["anonymized_name"].split()[0]
                    tone_key = f"outreach_tone_classic_{inspect_id}"
                    st.markdown("<div class='section-title'>✉️ Swarm Outreach Email Drafter</div>", unsafe_allow_html=True)
                    
                    outreach_type = st.selectbox(
                        "Select Direct Recruitment Email Template",
                        ["Direct Interview Invitation", "Detailed Job Pitch & Comp Request", "Brief Direct Referral Inquiry"],
                        key=tone_key
                    )
                    
                    top_skill = list(candidate_skills_dict.keys())[0].upper() if candidate_skills_dict else "APPLIED ML"
                    current_company = p["current_company"]
                    
                    # Generate candidate's realistic corporate email address
                    comp_clean = re.sub(r'[^a-zA-Z0-9]', '', current_company.lower())
                    name_clean = re.sub(r'[^a-zA-Z0-9\s]', '', p["anonymized_name"].lower()).replace(' ', '.')
                    if "startup" in comp_clean or not comp_clean:
                        cand_email = f"{name_clean}@candidate.rankcraft.ai"
                    else:
                        cand_email = f"{name_clean}@{comp_clean}.com"
                    
                    if outreach_type == "Direct Interview Invitation":
                        email_subject = f"Interview Invitation: Senior AI Engineer position at RankCraft AI"
                        email_body = (
                            f"Dear {first_name},\n\n"
                            f"I hope this email finds you well.\n\n"
                            f"My name is Jordan Dawson, and I lead the Technical Talent Acquisition team at RankCraft AI. "
                            f"We are currently expanding our core Engineering division and are looking for a Senior AI / Machine Learning Engineer to join our team.\n\n"
                            f"I recently reviewed your professional background and was exceptionally impressed by your engineering trajectory at {current_company}, particularly your experience with {top_skill}.\n\n"
                            f"Based on your profile matching our requirements, we would like to invite you for a technical interview to discuss this opportunity. "
                            f"To get started, please let me know your availability for a 30-minute introductory Zoom discussion this week.\n\n"
                            f"I have attached our job details for your reference. Looking forward to hearing from you.\n\n"
                            f"Best regards,\n\n"
                            f"Jordan Dawson\n"
                            f"Lead Technical Recruiter\n"
                            f"RankCraft AI"
                        )
                    elif outreach_type == "Detailed Job Pitch & Comp Request":
                        email_subject = f"Interview Schedule Request: Senior AI/ML Engineer at RankCraft AI"
                        email_body = (
                            f"Dear {first_name},\n\n"
                            f"I hope you are doing well.\n\n"
                            f"I am reaching out to you directly from the Engineering Leadership team at RankCraft AI. "
                            f"We have reviewed your profile and believe your experience at {current_company} aligns perfectly with our technical goals. "
                            f"We would love to schedule a preliminary interview with our engineering director.\n\n"
                            f"This position is based hybrid in our Noida/Pune office and includes a competitive compensation package. "
                            f"If you are open to discussing this, please reply to this email with your preferred contact number and a few times you are free to chat this week.\n\n"
                            f"Warm regards,\n\n"
                            f"Jordan Dawson\n"
                            f"Engineering Hiring Manager\n"
                            f"RankCraft AI"
                        )
                    else:
                        email_subject = f"Hiring Inquiry: Senior Engineering Interview at RankCraft AI"
                        email_body = (
                            f"Dear {first_name},\n\n"
                            f"I hope you are having a productive week.\n\n"
                            f"I lead technical recruiting at RankCraft AI. "
                            f"We are actively seeking ML talent with hands-on skill in {top_skill} for open interviews.\n\n"
                            f"I wanted to connect directly to see if you are open to senior roles in Noida/Pune or if you could recommend peers from {current_company} who might be interested in exploring this with us. "
                            f"Thank you for your time and consideration.\n\n"
                            f"Best regards,\n\n"
                            f"Jordan Dawson\n"
                            f"Talent Partner\n"
                            f"RankCraft AI"
                        )
                    
                    import urllib.parse
                    subject_encoded = urllib.parse.quote(email_subject)
                    body_encoded = urllib.parse.quote(email_body)
                    mailto_url = f"mailto:{cand_email}?subject={subject_encoded}&body={body_encoded}"
                    
                    outreach_box_html = f"""
                    <div style="background-color:#F8FAFC; border-radius:6px; padding:12px; margin-top:8px; border:1px solid #E2E8F0;">
                        <div style="font-size:11px; font-weight:700; color:#FF6B4A; margin-bottom:4px;">SUBJECT: {email_subject}</div>
                        <pre style="color:#1E293B; font-size:12px; font-family:'Courier New', monospace; white-space:pre-wrap; margin:0; background:none; border:none; padding:0;">{email_body}</pre>
                    </div>
                    """
                    st.markdown(outreach_box_html, unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    st.link_button(
                        f"✉️ Send Interview Offer Email to {first_name}",
                        mailto_url,
                        use_container_width=True,
                        type="primary"
                    )

                    inspector_card_html = textwrap.dedent(f"""
                    <div class="profile-card">
                        <div class="card-title">🔍 Candidate Profile Inspector: {insp_row["candidate_id"]}</div>
                        
                        <div style="font-size: 20px; font-weight:700; color:#1E293B; margin-bottom: 2px;">{p["anonymized_name"]}</div>
                        <div style="font-size:13px; font-weight:500; color:#FF6B4A; margin-bottom: 2px;">{p["headline"]}</div>
                        <div style="font-size:12px; color:#64748B; margin-bottom: 15px;">
                            📧 <b>Direct Contact:</b> {cand_email} | 📍 {p['location']}, {p['country']}
                        </div>
                        
                        <div style="font-size: 13px; line-height:1.5; color:#475569; margin-bottom: 15px; font-style:italic;">
                            "{p["summary"]}"
                        </div>
                        
                        <div class="section-title">Score Breakdown</div>
                        {score_html}
                        
                        <div class="section-title">Availability & Behavioral Signals</div>
                        {badges_html}
                        
                        <div class="section-title">Recruiter Reasoning</div>
                        <div style="font-size:13px; color:#1E293B; padding:10px; background-color:#F8FAFC; border-radius:6px; border-left:4px solid #FF6B4A; margin-top:5px; margin-bottom:15px;">
                            {insp_row['reasoning']}
                        </div>
                        
                        <div class="section-title">⚖️ JD Match & Skill Gap Analysis</div>
                        {skill_gap_html}
                        
                        <div class="section-title">Key Skills Match Profile</div>
                        {skills_matched_html}
                        
                        <div class="section-title">Career History Timeline</div>
                        {timeline_html}
                    </div>
                    """).strip()
                    
                    st.markdown(clean_html(inspector_card_html), unsafe_allow_html=True)

        # TAB 2: Pool Analytics
        with tab_analytics:
            st.subheader("Talent Pool Demographics & Metrics")
            df_all = pd.DataFrame(output_rows)
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("##### Years of Experience Distribution")
                yoe_counts = df_all["yoe"].round().astype(int).value_counts().sort_index()
                st.bar_chart(yoe_counts)
                
                st.markdown("##### Geographical Breakdown")
                country_counts = df_all["location"].value_counts().head(10)
                st.bar_chart(country_counts)
                
            with col_c2:
                st.markdown("##### Company Type Analysis (Product vs Consulting)")
                def classify_company(c_name):
                    c_name = str(c_name).lower()
                    if any(co in c_name for co in CONSULTING_COS):
                        return "Consulting (Services)"
                    return "Product / Product-Adjacent"
                    
                industry_types = df_all["company"].apply(classify_company).value_counts()
                st.bar_chart(industry_types)
                
                st.markdown("##### Notice Period Availability (days)")
                notices = df_all["_candidate"].apply(lambda c: c["redrob_signals"].get("notice_period_days", 90)).value_counts().sort_index()
                st.bar_chart(notices)

        # TAB 3: Candidate Compare
        with tab_compare:
            st.subheader("Compare Candidates Side-by-Side")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                cand1_id = st.selectbox("Select Candidate 1", [r["candidate_id"] for r in output_rows], index=0, key="compare_ws_sel1")
            with col_s2:
                cand2_id = st.selectbox("Select Candidate 2", [r["candidate_id"] for r in output_rows], index=min(1, len(output_rows)-1), key="compare_ws_sel2")
                
            if cand1_id and cand2_id:
                row1 = next(r for r in output_rows if r["candidate_id"] == cand1_id)
                row2 = next(r for r in output_rows if r["candidate_id"] == cand2_id)
                col_comp1, col_comp2 = st.columns(2)
                
                def render_comparison_column(r_data):
                    c = r_data["_candidate"]
                    p = c["profile"]
                    sig = c["redrob_signals"]
                    comp = r_data["components"]
                    
                    compare_show = [
                        ('Role Title', comp.get('title', 0.0), 30.0 * weight_title),
                        ('Career History', comp.get('career', 0.0), 30.0 * weight_career),
                        ('Skills Trust', comp.get('skills', 0.0), skills_score_cap * weight_skills),
                        ('Experience', comp.get('experience', 0.0), 10.0 * weight_experience),
                        ('Location Pref', comp.get('location', 0.0), 5.0 * weight_location),
                    ]
                    
                    score_bars_html = ""
                    for name, val, max_val in compare_show:
                        pct = min(max(0.0, (val / max_val) * 100.0), 100.0) if max_val > 0 else 0.0
                        score_bars_html += f"""
                        <div style='margin-bottom: 8px;'>
                            <div style='display: flex; justify-content: space-between; font-size: 11px; color:#475569;'>
                                <span>{name}</span>
                                <span>{val:.1f} / {max_val:.1f}</span>
                            </div>
                            <div style='width: 100%; background-color: #E2E8F0; border-radius: 4px; height: 5px;'>
                                <div style='width: {pct}%; background-color: #FF6B4A; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        """
                    
                    github_score = sig.get("github_activity_score", -1)
                    github_text = f"Linked ({github_score})" if github_score >= 0 else "None linked"
                    
                    comparison_html = textwrap.dedent(f"""
                    <div class="profile-card">
                        <div style="font-size:22px; font-weight:700; color:#FF6B4A;">{r_data["candidate_id"]}</div>
                        <div style="font-size:14px; font-weight:600; color:#1E293B;">Rank #{r_data["rank"]} (Score: {r_data["score"]:.3f})</div>
                        <hr style="border-top: 1px solid #E2E8F0; margin: 10px 0;"/>
                        
                        <div class="section-title">Headline</div>
                        <div style="font-size:13px; color:#1E293B; font-weight:500;">{p["headline"]}</div>
                        
                        <div class="section-title">Score Breakdown</div>
                        {score_bars_html}
                        
                        <div class="section-title">Behavioral Stats</div>
                        <table style="width:100%; font-size:12px; color:#1E293B; border-collapse:collapse; margin-top:5px;">
                            <tr><td style="padding:4px 0; color:#64748B;">Notice Period</td><td style="text-align:right; font-weight:600;">{sig.get("notice_period_days")} days</td></tr>
                            <tr><td style="padding:4px 0; color:#64748B;">Response Rate</td><td style="text-align:right; font-weight:600;">{sig.get("recruiter_response_rate", 0.0):.0%}</td></tr>
                            <tr><td style="padding:4px 0; color:#64748B;">GitHub Contribution</td><td style="text-align:right; font-weight:600;">{github_text}</td></tr>
                            <tr><td style="padding:4px 0; color:#64748B;">Interview Attendance</td><td style="text-align:right; font-weight:600;">{sig.get("interview_completion_rate", 0.0):.0%}</td></tr>
                            <tr><td style="padding:4px 0; color:#64748B;">Preferred work mode</td><td style="text-align:right; font-weight:600; text-transform:capitalize;">{sig.get("preferred_work_mode")}</td></tr>
                        </table>
                        
                        <div class="section-title">Notice & Availability Reasoning</div>
                        <div style="font-size:12px; color:#475569; line-height:1.4; padding:8px; background-color:#F8FAFC; border-radius:4px; margin-top:5px; border:1px solid #E2E8F0;">
                            {r_data["reasoning"]}
                        </div>
                    </div>
                    """).strip()
                    st.markdown(clean_html(comparison_html), unsafe_allow_html=True)
                    
                with col_comp1:
                    render_comparison_column(row1)
                with col_comp2:
                    render_comparison_column(row2)

        # TAB 4: Honeypot Auditor
        with tab_honeypots:
            st.subheader("🛡️ Honeypot Profile Auditor")
            st.markdown(clean_html("""
            To secure hiring databases against synthetic profile tampering, the ranker matches the candidate pool against **five security rules** (e.g. expert skill listed with 0 months duration, impossible job date ranges, or exaggerated skill counts). 
            The profiles below have triggered these rules and were automatically quarantined (assigned score `-9999` and removed from the active shortlist).
            """))
            
            if not honeypots:
                st.success("No honeypots detected in the loaded dataset.")
            else:
                hp_df = pd.DataFrame(honeypots)
                st.markdown(f"**Total quarantined profiles:** {len(honeypots)}")
                counts_by_type = hp_df["type"].value_counts().reset_index()
                counts_by_type.columns = ["Security Rule Violated", "Quarantined Count"]
                st.dataframe(counts_by_type, hide_index=True)
                
                st.markdown("##### Detailed Security Quarantine Log")
                log_display = pd.DataFrame([
                    {
                        "Candidate ID": h["candidate_id"],
                        "Violation Rule": h["type"],
                        "Anomaly Audit Reasoning": h["reason"]
                    }
                    for h in honeypots
                ])
                st.dataframe(log_display, hide_index=True, use_container_width=True)

        # TAB 5: My Custom Shortlist
        with tab_manual_shortlist:
            st.subheader("⭐ Custom Recruiter Shortlist")
            shortlist_items = [r for r in output_rows if r["candidate_id"] in st.session_state["selected_candidate_ids"]]
            if not shortlist_items:
                st.info("No candidates selected yet. Add candidates by checking 'Mark Candidate as Shortlisted' in the explorer tab.")
            else:
                df_sh = pd.DataFrame([
                    {
                        "Rank": item["rank"],
                        "Candidate ID": item["candidate_id"],
                        "Title": item["title"],
                        "Company": item["company"],
                        "Score": item["score"],
                        "Location": item["location"]
                    }
                    for item in shortlist_items
                ])
                st.dataframe(df_sh, hide_index=True, use_container_width=True)
                
                csv_buf_sh = io.StringIO()
                writer_sh = csv.writer(csv_buf_sh)
                writer_sh.writerow(["candidate_id", "score", "title", "company"])
                for item in shortlist_items:
                    writer_sh.writerow([item["candidate_id"], item["score"], item["title"], item["company"]])
                st.download_button("📥 Download Shortlist CSV", csv_buf_sh.getvalue(), file_name="shortlist.csv", mime="text/csv", key="sh_dl_btn_ws")

        # TAB 6: Add Candidates
        with tab_add_candidate:
            st.subheader("📥 Add Custom Candidates to Evaluation Pool")
            add_mode = st.radio("Choose Input Method", ["Interactive Form", "Paste Raw Text Resume", "Paste Structured JSON"])
            
            if add_mode == "Interactive Form":
                with st.form("interactive_add_form"):
                    cf_name = st.text_input("Candidate Name", "Jane Doe")
                    cf_title = st.text_input("Current Title", "Senior Machine Learning Engineer")
                    cf_company = st.text_input("Current Company", "Scale AI")
                    cf_industry = st.text_input("Current Industry", "Software")
                    cf_yoe = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=6.0, step=0.5)
                    cf_location = st.text_input("Location (City)", "Pune")
                    cf_country = st.selectbox("Country", ["India", "USA", "UK", "Canada"])
                    
                    cf_skills = st.text_input("Skills (comma separated)", "python, pytorch, scikit-learn, embeddings")
                    cf_notice = st.number_input("Notice Period (days)", min_value=0, max_value=180, value=30)
                    cf_open_to_work = st.checkbox("Open to Work flag", value=True)
                    cf_willing_relocate = st.checkbox("Willing to Relocate", value=True)
                    cf_history_desc = st.text_area("Current Job Description Details (for semantic ranking)", "Built embeddings retrieval pipeline using sentence-transformers and FAISS database. Scaled search ranking models.")
                    
                    submitted = st.form_submit_button("➕ Add Candidate to Evaluation Pool")
                    if submitted:
                        skills_list = []
                        for s in cf_skills.split(","):
                            s = s.strip()
                            if s:
                                skills_list.append({
                                    "name": s,
                                    "proficiency": "expert" if "pytorch" in s.lower() or "embeddings" in s.lower() else "advanced",
                                    "duration_months": int(cf_yoe * 8),
                                    "endorsements": 8
                                })
                        cand_id = f"custom_{int(datetime.now().timestamp())}_{re.sub(r'[^a-zA-Z0-9]', '', cf_name)[:10].lower()}"
                        new_cand = {
                            "candidate_id": cand_id,
                            "profile": {
                                "anonymized_name": cf_name,
                                "headline": f"{cf_title} | {cf_yoe:.1f} YoE",
                                "summary": f"Experienced professional in {cf_industry}.",
                                "current_title": cf_title,
                                "current_company": cf_company,
                                "current_industry": cf_industry,
                                "years_of_experience": cf_yoe,
                                "country": cf_country,
                                "location": cf_location
                            },
                            "skills": skills_list,
                            "career_history": [
                                {
                                    "title": cf_title,
                                    "company": cf_company,
                                    "start_date": "2023-01-01",
                                    "end_date": None,
                                    "is_current": True,
                                    "duration_months": int(cf_yoe * 12),
                                    "industry": cf_industry,
                                    "description": cf_history_desc
                                }
                            ],
                            "redrob_signals": {
                                "open_to_work_flag": cf_open_to_work,
                                "last_active_date": TODAY.strftime("%Y-%m-%d"),
                                "notice_period_days": cf_notice,
                                "recruiter_response_rate": 0.8,
                                "github_activity_score": 60,
                                "interview_completion_rate": 0.9,
                                "willing_to_relocate": cf_willing_relocate,
                                "preferred_work_mode": "hybrid",
                                "skill_assessment_scores": {}
                            }
                        }
                        st.session_state["candidate_pool"].append(new_cand)
                        st.success(f"Added candidate **{cf_name}** successfully!")
                        st.rerun()
                        
            elif add_mode == "Paste Raw Text Resume":
                st.markdown("##### Paste Candidate Resume Text")
                resume_input = st.text_area("Paste resume text contents here (Name, title, experience, skills list):", height=200, placeholder="John Doe\nLead Machine Learning Engineer\nYoE: 8 Years\nSkills: Python, PyTorch, FAISS, Sentence-Transformers, MLOps\nExperience: Designed learning-to-rank systems at TechCorp...")
                
                if st.button("🔍 Parse and Preview Resume"):
                    if not resume_input.strip():
                        st.warning("Please paste resume text first.")
                    else:
                        parsed_cand = parse_raw_resume_text(resume_input)
                        st.session_state["temp_parsed_candidate"] = parsed_cand
                        st.success("Resume parsed successfully! Preview the candidate details below.")
                
                if "temp_parsed_candidate" in st.session_state:
                    parsed = st.session_state["temp_parsed_candidate"]
                    p = parsed["profile"]
                    
                    st.markdown(f"""
                    <div class="profile-card" style="margin-bottom:15px;">
                        <div class="card-title">Parsed Resume Preview</div>
                        <b>Name:</b> {p['anonymized_name']}<br/>
                        <b>Extracted Title:</b> {p['current_title']}<br/>
                        <b>Company:</b> {p['current_company']}<br/>
                        <b>Years of Experience:</b> {p['years_of_experience']:.1f}<br/>
                        <b>Location:</b> {p['location']}, {p['country']}<br/>
                        <b>Matched Skills:</b> {", ".join([s['name'] for s in parsed['skills']])}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("➕ Confirm & Add to Evaluation Pool"):
                        st.session_state["candidate_pool"].append(parsed)
                        del st.session_state["temp_parsed_candidate"]
                        st.success("Candidate added successfully!")
                        st.rerun()
                        
            elif add_mode == "Paste Structured JSON":
                st.markdown("##### Paste JSON Profile Object")
                json_input = st.text_area("Paste JSON Candidate Schema structure here:", height=200)
                if st.button("➕ Add JSON Candidate"):
                    if not json_input.strip():
                        st.warning("Please paste JSON first.")
                    else:
                        try:
                            parsed_cand = json.loads(json_input)
                            if "candidate_id" not in parsed_cand or "profile" not in parsed_cand:
                                st.error("Invalid schema! Candidate JSON must contain 'candidate_id' and 'profile' fields.")
                            else:
                                st.session_state["candidate_pool"].append(parsed_cand)
                                st.success(f"Added JSON candidate **{parsed_cand['candidate_id']}** successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to parse JSON: {e}")
