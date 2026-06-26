"""
Redrob Hackathon — Sandbox Demo (Upgraded Recruiter Workbench)
==============================================================
Run locally:  streamlit run app.py
Deploy to:    HuggingFace Spaces (Streamlit SDK) or Streamlit Cloud

Features:
- Dynamic Weight tuning sliders (real-time re-ranking)
- Profile Card viewer & highlighted Career Timeline
- Talent pool Analytics dashboard
- Side-by-Side Candidate Comparison
- Honeypot Security Auditor
- Job Description keyword mapping presets
"""

import csv
import io
import json
import os
import sys
import re
import textwrap
from datetime import datetime
import streamlit as st
import pandas as pd

# Add parent dir to path so we can import rank.py
sys.path.insert(0, os.path.dirname(__file__))

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
)

# Helper to strip all leading/trailing whitespace from each line of HTML to prevent Markdown code block parsing
def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.splitlines())


def parse_raw_resume_text(resume_text):
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    if not lines:
        return None
    
    # 1. Name: Assume first line is the name
    name = lines[0]
    if len(name) > 50 or "@" in name or ":" in name:
        name = "Candidate Name"
        
    # 2. Extract Years of Experience
    yoe = 5.0 # Default fallback
    yoe_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b', resume_text, re.IGNORECASE)
    if yoe_match:
        try:
            yoe = float(yoe_match.group(1))
        except:
            pass
            
    # 3. Extract Country and Location
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
        
    # 4. Extract Current Title
    title = "Machine Learning Engineer" # Default
    all_titles = list(CORE_ML_TITLES) + list(PARTIAL_ML_TITLES)
    for t in all_titles:
        if re.search(rf'\b{re.escape(t)}\b', resume_text, re.IGNORECASE):
            title = t.title()
            break
            
    # 5. Extract Current Company
    company = "Product Startup"
    co_match = re.search(r'(?:at|company:)\s*([A-Za-z0-9\s&]{2,30})', resume_text, re.IGNORECASE)
    if co_match:
        company = co_match.group(1).strip()
        
    # 6. Extract Skills
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
    page_title="Redrob Talent Discovery Sandbox",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(clean_html("""
<style>
    /* Styling general visual containers */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 18px;
        margin: 5px 0 15px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .profile-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        margin-top: 10px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        color: #F8FAFC;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }
    
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
    .badge-green { background-color: #064E3B; color: #6EE7B7; border: 1px solid #047857; }
    .badge-blue { background-color: #1E3A8A; color: #93C5FD; border: 1px solid #1D4ED8; }
    .badge-orange { background-color: #78350F; color: #FCD34D; border: 1px solid #D97706; }
    .badge-red { background-color: #7F1D1D; color: #FCA5A5; border: 1px solid #B91C1C; }
    .badge-grey { background-color: #374151; color: #D1D5DB; border: 1px solid #4B5563; }
    .badge-purple { background-color: #4C1D95; color: #DDD6FE; border: 1px solid #6D28D9; }
    
    /* Career history timeline styles */
    .timeline-container {
        border-left: 2px solid #334155;
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
        background-color: #3B82F6;
        border: 2px solid #0F172A;
    }
    .timeline-item.current::before {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    .timeline-title {
        font-weight: 700;
        font-size: 14px;
        color: #F8FAFC;
    }
    .timeline-meta {
        font-size: 11px;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .timeline-desc {
        font-size: 12px;
        color: #CBD5E1;
        line-height: 1.5;
    }
    
    /* Custom spacing / text utility classes */
    .section-title {
        font-size: 14px;
        font-weight: bold;
        color: #94A3B8;
        text-transform: uppercase;
        margin-top: 15px;
        margin-bottom: 5px;
        letter-spacing: 0.05em;
    }
    .highlight-keyword {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border-radius: 3px;
        padding: 0px 4px;
        font-weight: 500;
    }
</style>
"""), unsafe_allow_html=True)

# ── JD PRESET UTILITIES ───────────────────────────────────────────────────────

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

def parse_jd_text(jd_text):
    text_lower = jd_text.lower()
    custom_map = {}
    # Scan standard CORE_SKILL_MAP + MLOps skills
    all_known = list(CORE_SKILL_MAP.keys()) + ['docker', 'kubernetes', 'aws', 'kafka', 'redis', 'go', 'rust', 'c++', 'airflow', 'distributed', 'sql']
    for kw in all_known:
        if kw in text_lower:
            # Map default or preset score
            custom_map[kw] = CORE_SKILL_MAP.get(kw, 8)
    return custom_map

# ── APP INITIALIZATION & SIDEBAR ─────────────────────────────────────────────

st.title("🔍 Redrob Candidate Discovery & Ranking Workbench")
st.caption("Advanced AI Recruiter Sandbox — calibrate scoring weights, verify filters, and explore the talent pool")

# Initialize session states
if "candidate_pool" not in st.session_state:
    st.session_state["candidate_pool"] = []

if "selected_candidate_ids" not in st.session_state:
    st.session_state["selected_candidate_ids"] = set()

# Auto-load default sample on first startup if empty
if not st.session_state["candidate_pool"]:
    default_path = "data/sample.jsonl"
    if os.path.exists(default_path):
        try:
            sample_cands = []
            with open(default_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample_cands.append(json.loads(line))
            st.session_state["candidate_pool"] = sample_cands
        except Exception as e:
            pass

# Sidebar controls
st.sidebar.header("📁 Data Source")

# Reset button
if st.sidebar.button("🔄 Reset to Default Sample"):
    default_path = "data/sample.jsonl"
    if os.path.exists(default_path):
        try:
            sample_cands = []
            with open(default_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample_cands.append(json.loads(line))
            st.session_state["candidate_pool"] = sample_cands
            st.sidebar.success(f"Reset pool to default sample ({len(sample_cands)} candidates).")
        except Exception as e:
            st.sidebar.error(f"Error resetting: {e}")

uploaded_file = st.sidebar.file_uploader(
    "Upload candidate JSON/JSONL sample",
    type=["json", "jsonl"],
)

if uploaded_file:
    import_mode = st.sidebar.radio("Import Mode", ["Replace pool", "Append to pool"], key="import_mode")
    if st.sidebar.button("📥 Import Uploaded File"):
        try:
            raw = uploaded_file.read().decode("utf-8")
            imported_candidates = []
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    imported_candidates = data
                elif isinstance(data, dict):
                    imported_candidates = [data]
            except json.JSONDecodeError:
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            imported_candidates.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if imported_candidates:
                if import_mode == "Replace pool":
                    st.session_state["candidate_pool"] = imported_candidates
                else:
                    existing_ids = {c["candidate_id"] for c in st.session_state["candidate_pool"]}
                    for c in imported_candidates:
                        if c["candidate_id"] not in existing_ids:
                            st.session_state["candidate_pool"].append(c)
                st.sidebar.success(f"Successfully loaded {len(imported_candidates)} candidates!")
            else:
                st.sidebar.error("No valid candidate entries found in uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Error importing file: {e}")

st.sidebar.markdown("---")

# Job Description Presets
st.sidebar.header("📝 Job Description (JD)")

if "jd_text_content" not in st.session_state:
    st.session_state["jd_text_content"] = JD_PRESETS[list(JD_PRESETS.keys())[0]]["text"]

def on_preset_change():
    preset_name = st.session_state["preset_select"]
    st.session_state["jd_text_content"] = JD_PRESETS[preset_name]["text"]

# Upload JD file
uploaded_jd_file = st.sidebar.file_uploader(
    "Upload custom Job Description (.txt)",
    type=["txt"],
    key="jd_file_uploader"
)

if uploaded_jd_file:
    try:
        jd_file_text = uploaded_jd_file.getvalue().decode("utf-8")
        if jd_file_text != st.session_state["jd_text_content"]:
            st.session_state["jd_text_content"] = jd_file_text
            st.sidebar.success("Loaded JD from file!")
    except Exception as e:
        st.sidebar.error(f"Error reading JD: {e}")

preset_list = list(JD_PRESETS.keys())
st.sidebar.selectbox(
    "Choose JD Profile Preset",
    preset_list,
    key="preset_select",
    on_change=on_preset_change
)

jd_text = st.sidebar.text_area(
    "JD Requirements Text",
    value=st.session_state["jd_text_content"],
    height=120,
    key="jd_text_area"
)
st.session_state["jd_text_content"] = jd_text

# Build custom skill mapping based on active JD
custom_skill_map = parse_jd_text(jd_text)

with st.sidebar.expander("🔑 Extracted JD Skill Targets"):
    st.markdown("Matched target skills and search weight values:")
    df_skills = pd.DataFrame([{"Skill Keyword": k, "Base Weight": v} for k, v in custom_skill_map.items()])
    if not df_skills.empty:
        st.dataframe(df_skills.sort_values(by="Base Weight", ascending=False), hide_index=True)
    else:
        st.caption("No matching skill keywords found.")

st.sidebar.markdown("---")

# Dynamic weights config
st.sidebar.header("⚙️ Score Component Calibration")
weight_title = st.sidebar.slider("Role Title Fit Weight", 0.0, 2.0, 1.0, 0.1)
weight_career = st.sidebar.slider("Career History Weight", 0.0, 2.0, 1.0, 0.1)
weight_skills = st.sidebar.slider("Skills Trust Weight", 0.0, 2.0, 1.0, 0.1)
weight_experience = st.sidebar.slider("Experience Band Weight", 0.0, 2.0, 1.0, 0.1)
weight_location = st.sidebar.slider("Location Pref Weight", 0.0, 2.0, 1.0, 0.1)

skills_score_cap = st.sidebar.slider("Max Skill Score Cap", 10.0, 40.0, 25.0, 1.0)

st.sidebar.markdown("---")

# Behavioral modifiers config
st.sidebar.header("🚦 Behavioral Modifiers")
enable_activity_decay = st.sidebar.checkbox("Apply Activity Recency Decay (decay inactive >30d)", value=True)
enable_notice_penalty = st.sidebar.checkbox("Penalize Long Notice Periods (>60d)", value=True)
enable_response_rate_penalty = st.sidebar.checkbox("Penalize Low Recruiter Response Rate (<40%)", value=True)
enable_open_to_work_bonus = st.sidebar.checkbox("Apply Open-To-Work Boost (+5%)", value=True)
enable_interview_completion_penalty = st.sidebar.checkbox("Penalize Poor Interview attendance (<30%)", value=True)

st.sidebar.markdown("---")

# Disqualifications / Penalty Toggles
st.sidebar.header("🚫 Hard Disqualifiers")
consulting_penalty = st.sidebar.checkbox("Penalize Pure Consulting Backgrounds", value=True)
location_penalty = st.sidebar.checkbox("Penalize Non-India Unwilling Relocation", value=True)

# Build configuration dict
config = {
    "weight_title": weight_title,
    "weight_career": weight_career,
    "weight_skills": weight_skills,
    "weight_experience": weight_experience,
    "weight_location": weight_location,
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

# ── DATA PARSING ─────────────────────────────────────────────────────────────

active_pool = st.session_state["candidate_pool"]

# Sandbox cap warning
if len(active_pool) > 1000:
    st.sidebar.warning(f"Loaded {len(active_pool)} candidates — capping at 1,000 for web sandbox performance.")
    active_pool = active_pool[:1000]

# ── ENGINE PROCESSING ─────────────────────────────────────────────────────────

if active_pool:
    scored = []
    honeypots = []

    for c in active_pool:
        total_score, components = score_candidate(c, config=config)
        
        # Check if flagged as honeypot
        if components.get("honeypot"):
            # Determine type of honeypot rule violated
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

    # Sort deterministically
    scored.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
    
    # Ranks rows
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

    # ── INTERACTIVE WORKBENCH TABS ────────────────────────────────────────────
    
    tab_discovery, tab_analytics, tab_compare, tab_honeypots, tab_manual_shortlist, tab_add_candidate = st.tabs([
        "🔍 Shortlist Explorer",
        "📊 Talent Pool Analytics",
        "⚖️ Candidate Compare",
        "🛡️ Honeypot Auditor",
        "⭐ My Custom Shortlist",
        "📥 Add Candidates/Resumes"
    ])

    # ── TAB 1: SHORTLIST & DETAILED EXPLORER ──────────────────────────────────
    with tab_discovery:
        # Metrics summary row
        col1, col2, col3 = st.columns(3)
        col1.markdown(clean_html(f"""
        <div class="metric-card">
            <h5 style="color:#94A3B8;margin:0 0 5px 0;font-size:12px;">TOTAL POOL LOADED</h5>
            <h2 style="color:#F8FAFC;margin:0;font-size:32px;font-weight:700;">{len(active_pool)}</h2>
        </div>
        """), unsafe_allow_html=True)
        col2.markdown(clean_html(f"""
        <div class="metric-card">
            <h5 style="color:#94A3B8;margin:0 0 5px 0;font-size:12px;">SHORTLISTED & SCORED</h5>
            <h2 style="color:#10B981;margin:0;font-size:32px;font-weight:700;">{len(scored)}</h2>
        </div>
        """), unsafe_allow_html=True)
        col3.markdown(clean_html(f"""
        <div class="metric-card">
            <h5 style="color:#94A3B8;margin:0 0 5px 0;font-size:12px;">HONEYPOTS FILTERED</h5>
            <h2 style="color:#EF4444;margin:0;font-size:32px;font-weight:700;">{len(honeypots)}</h2>
        </div>
        """), unsafe_allow_html=True)

        st.subheader("Shortlisted Candidates")
        
        # Search & Filter box
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

        # Download CSV option
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
            # Table layout & inspector card split
            col_table, col_inspector = st.columns([1.1, 0.9])
            
            with col_table:
                st.caption(f"Showing {len(filtered_rows)} candidate matches")
                
                # Bulk shortlisting actions
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

                # Make a selectable dataframe with Shortlist Status
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
                
                # Dropdown selector to inspect candidate profile
                inspect_id = st.selectbox(
                    "Select a candidate ID to inspect profile details",
                    [r["candidate_id"] for r in filtered_rows],
                    index=0
                )
                
            with col_inspector:
                # Find matching row for details
                insp_row = next(r for r in filtered_rows if r["candidate_id"] == inspect_id)
                c_data = insp_row["_candidate"]
                p = c_data["profile"]
                sig = c_data["redrob_signals"]
                comp = insp_row["components"]

                # Individual Shortlist checkbox
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

                # Scores Stacked/Progress display
                score_html = ""
                components_show = [
                    ('Role Title Fit', comp.get('title', 0.0), 30.0 * weight_title, '#3B82F6'),
                    ('Career History Quality', comp.get('career', 0.0), 30.0 * weight_career, '#10B981'),
                    ('Skills Trust Profile', comp.get('skills', 0.0), skills_score_cap * weight_skills, '#F59E0B'),
                    ('Experience Band Fit', comp.get('experience', 0.0), 10.0 * weight_experience, '#8B5CF6'),
                    ('Location Alignment', comp.get('location', 0.0), 5.0 * weight_location, '#EC4899'),
                ]
                
                score_html += f"<div style='font-weight:700; font-size:15px; color:#F8FAFC; margin-bottom:12px;'>Composite Score: {comp.get('total', 0.0):.3f} <span style='font-weight:normal; font-size:12px; color:#94A3B8;'>(Base {comp.get('base', 0.0):.1f} × multiplier {comp.get('behavioral_mult', 1.0):.3f})</span></div>"
                for name, val, max_val, color in components_show:
                    pct = min(max(0.0, (val / max_val) * 100.0), 100.0) if max_val > 0 else 0.0
                    score_html += f"""
                    <div style='margin-bottom: 12px;'>
                        <div style='display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px; color:#E2E8F0;'>
                            <span>{name}</span>
                            <span>{val:.1f} / {max_val:.1f}</span>
                        </div>
                        <div style='width: 100%; background-color: #334155; border-radius: 4px; height: 6px;'>
                            <div style='width: {pct}%; background-color: {color}; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                    """
                
                # Behavioral Pill Badges
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
                
                # Skills matching inventory
                skills_matched_html = "<div>"
                for sk in c_data.get("skills", []):
                    sname = sk["name"]
                    is_core = any(kw in sname.lower() for kw in custom_skill_map)
                    badge_style = "badge-blue" if is_core else "badge-grey"
                    endorse = sk.get("endorsements", 0)
                    sdur = sk.get("duration_months", 0)
                    skills_matched_html += f"<span class='badge {badge_style}'>{sname} ({sk['proficiency']} • {sdur}m • {endorse}👍)</span>"
                skills_matched_html += "</div>"
                
                # Skill Gap Analysis computation
                candidate_skills_dict = {sk["name"].lower(): sk for sk in c_data.get("skills", [])}
                candidate_skills_set = set(candidate_skills_dict.keys())
                target_skills_set = set(custom_skill_map.keys())

                # Matched core skills
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

                # Missing core skills
                missing_skills = []
                for sk_name in target_skills_set:
                    matched_any = False
                    for cand_sk in candidate_skills_set:
                        if sk_name in cand_sk or cand_sk in sk_name:
                            matched_any = True
                            break
                    if not matched_any:
                        missing_skills.append(sk_name.upper())

                # Complementary skills (in profile but not in target list)
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
                    matched_badges_html += "<span style='font-size:12px;color:#94A3B8;'>None matched</span>"
                matched_badges_html += "</div>"

                missing_badges_html = "<div>"
                if missing_skills:
                    for sk in sorted(missing_skills):
                        missing_badges_html += f"<span class='badge badge-red'>{sk}</span>"
                else:
                    missing_badges_html += "<span style='font-size:12px;color:#94A3B8;'>None missing</span>"
                missing_badges_html += "</div>"

                comp_badges_html = "<div>"
                if complementary_skills:
                    for sk in sorted(complementary_skills)[:8]:
                        comp_badges_html += f"<span class='badge badge-purple'>{sk}</span>"
                else:
                    comp_badges_html += "<span style='font-size:12px;color:#94A3B8;'>None</span>"
                comp_badges_html += "</div>"

                skill_gap_html = f"""
                <div style="background-color:#1E293B; border-radius:6px; padding:12px; margin-top:8px; margin-bottom:15px; border:1px solid #334155;">
                    <div style="font-size:11px; color:#10B981; font-weight:700; margin-bottom:4px;">🟢 MATCHED CORE SKILLS</div>
                    {matched_badges_html}
                    <div style="font-size:11px; color:#FCA5A5; font-weight:700; margin-top:10px; margin-bottom:4px;">🔴 MISSING REQUIRED SKILLS</div>
                    {missing_badges_html}
                    <div style="font-size:11px; color:#DDD6FE; font-weight:700; margin-top:10px; margin-bottom:4px;">🟣 EXTRA COMPLEMENTARY SKILLS</div>
                    {comp_badges_html}
                </div>
                """

                # Highlighted Career History Timeline
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
                        <div class='timeline-title'>{j.get('title')} at {j.get('company')}</div>
                        <div class='timeline-meta'>{j.get('start_date')} to {end_str} ({j.get('duration_months')} months) • Industry: {j.get('industry')}</div>
                        <div class='timeline-desc'>{jd_desc}</div>
                    </div>
                    """
                timeline_html += "</div>"

                # Render entire Candidate Inspector Card in a single unified HTML block (solving indented Markdown block issue)
                inspector_card_html = textwrap.dedent(f"""
                <div class="profile-card">
                    <div class="card-title">🔍 Candidate Profile Inspector: {insp_row["candidate_id"]}</div>
                    
                    <div style="font-size: 20px; font-weight:700; color:#F1F5F9; margin-bottom: 2px;">{p["anonymized_name"]}</div>
                    <div style="font-size:13px; font-weight:500; color:#3B82F6; margin-bottom: 12px;">{p["headline"]}</div>
                    
                    <div style="font-size: 13px; line-height:1.5; color:#CBD5E1; margin-bottom: 15px; font-style:italic;">
                        "{p["summary"]}"
                    </div>
                    
                    <div class="section-title">Score Breakdown</div>
                    {score_html}
                    
                    <div class="section-title">Availability & Behavioral Signals</div>
                    {badges_html}
                    
                    <div class="section-title">Recruiter Reasoning</div>
                    <div style="font-size:13px; color:#E2E8F0; padding:10px; background-color:#1E293B; border-radius:6px; border-left:4px solid #10B981; margin-top:5px; margin-bottom:15px;">
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

    # ── TAB 2: TALENT POOL ANALYTICS ──────────────────────────────────────────
    with tab_analytics:
        st.subheader("Talent Pool Demographics & Metrics")
        
        # Load df for charts
        df_all = pd.DataFrame(output_rows)
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # 1. Experience bands distribution
            st.markdown("##### Years of Experience Distribution")
            yoe_counts = df_all["yoe"].round().astype(int).value_counts().sort_index()
            st.bar_chart(yoe_counts)
            
            # 2. Preferred Location distributions
            st.markdown("##### Geographical Breakdown")
            country_counts = df_all["location"].value_counts().head(10)
            st.bar_chart(country_counts)
            
        with col_c2:
            # 3. Industry type distribution (Product vs Consulting)
            st.markdown("##### Company Type Analysis (Product vs Consulting)")
            def classify_company(c_name):
                c_name = str(c_name).lower()
                if any(co in c_name for co in CONSULTING_COS):
                    return "Consulting (Services)"
                return "Product / Product-Adjacent"
                
            industry_types = df_all["company"].apply(classify_company).value_counts()
            st.bar_chart(industry_types)
            
            # 4. Notice Period distribution
            st.markdown("##### Notice Period Availability (days)")
            notices = df_all["_candidate"].apply(lambda c: c["redrob_signals"].get("notice_period_days", 90)).value_counts().sort_index()
            st.bar_chart(notices)

    # ── TAB 3: CANDIDATE COMPARE WORKBENCH ────────────────────────────────────
    with tab_compare:
        st.subheader("Compare Candidates Side-by-Side")
        st.caption("Select two candidates to compare their fit parameters side-by-side.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            cand1_id = st.selectbox("Select Candidate 1", [r["candidate_id"] for r in output_rows], index=0)
        with col_s2:
            cand2_id = st.selectbox("Select Candidate 2", [r["candidate_id"] for r in output_rows], index=min(1, len(output_rows)-1))
            
        if cand1_id and cand2_id:
            row1 = next(r for r in output_rows if r["candidate_id"] == cand1_id)
            row2 = next(r for r in output_rows if r["candidate_id"] == cand2_id)
            
            col_c1, col_c2 = st.columns(2)
            
            # Helper function to render a single candidate column in comparison
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
                        <div style='display: flex; justify-content: space-between; font-size: 11px; color:#CBD5E1;'>
                            <span>{name}</span>
                            <span>{val:.1f} / {max_val:.1f}</span>
                        </div>
                        <div style='width: 100%; background-color: #334155; border-radius: 4px; height: 5px;'>
                            <div style='width: {pct}%; background-color: #3B82F6; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                    """
                
                github_score = sig.get("github_activity_score", -1)
                github_text = f"Linked ({github_score})" if github_score >= 0 else "None linked"
                
                comparison_html = textwrap.dedent(f"""
                <div class="profile-card">
                    <div style="font-size:22px; font-weight:700; color:#3B82F6;">{r_data["candidate_id"]}</div>
                    <div style="font-size:14px; font-weight:600; color:#F1F5F9;">Rank #{r_data["rank"]} (Score: {r_data["score"]:.3f})</div>
                    <hr style="border-top: 1px solid #334155; margin: 10px 0;"/>
                    
                    <div class="section-title">Headline</div>
                    <div style="font-size:13px; color:#E2E8F0; font-weight:500;">{p["headline"]}</div>
                    
                    <div class="section-title">Score Breakdown</div>
                    {score_bars_html}
                    
                    <div class="section-title">Behavioral Stats</div>
                    <table style="width:100%; font-size:12px; color:#E2E8F0; border-collapse:collapse; margin-top:5px;">
                        <tr><td style="padding:4px 0; color:#94A3B8;">Notice Period</td><td style="text-align:right; font-weight:600;">{sig.get("notice_period_days")} days</td></tr>
                        <tr><td style="padding:4px 0; color:#94A3B8;">Response Rate</td><td style="text-align:right; font-weight:600;">{sig.get("recruiter_response_rate", 0.0):.0%}</td></tr>
                        <tr><td style="padding:4px 0; color:#94A3B8;">GitHub Contribution</td><td style="text-align:right; font-weight:600;">{github_text}</td></tr>
                        <tr><td style="padding:4px 0; color:#94A3B8;">Interview Attendance</td><td style="text-align:right; font-weight:600;">{sig.get("interview_completion_rate", 0.0):.0%}</td></tr>
                        <tr><td style="padding:4px 0; color:#94A3B8;">Preferred work mode</td><td style="text-align:right; font-weight:600; text-transform:capitalize;">{sig.get("preferred_work_mode")}</td></tr>
                    </table>
                    
                    <div class="section-title">Notice & Availability Reasoning</div>
                    <div style="font-size:12px; color:#CBD5E1; line-height:1.4; padding:8px; background-color:#1E293B; border-radius:4px; margin-top:5px;">
                        {r_data["reasoning"]}
                    </div>
                </div>
                """).strip()
                
                st.markdown(clean_html(comparison_html), unsafe_allow_html=True)
                
            with col_c1:
                render_comparison_column(row1)
            with col_c2:
                render_comparison_column(row2)

    # ── TAB 4: HONEYPOT AUDITOR PANEL ─────────────────────────────────────────
    with tab_honeypots:
        st.subheader("🛡️ Honeypot Profile Auditor")
        st.markdown(clean_html("""
        To secure hiring databases against synthetic profile tampering, the ranker matches the candidate pool against **five security rules** (e.g. expert skill listed with 0 months duration, impossible job date ranges, or exaggerated skill counts). 
        The profiles below have triggered these rules and were automatically quarantined (assigned score `-9999` and removed from the active shortlist).
        """))
        
        if not honeypots:
            st.success("No honeypots detected in the loaded dataset.")
        else:
            # Display summary metric details
            hp_df = pd.DataFrame(honeypots)
            st.markdown(f"**Total quarantined profiles:** {len(honeypots)}")
            
            # Anomaly summary breakdown
            counts_by_type = hp_df["type"].value_counts().reset_index()
            counts_by_type.columns = ["Security Rule Violated", "Quarantined Count"]
            st.dataframe(counts_by_type, hide_index=True)
            
            # Detailed tables
            st.markdown("##### Detailed Security Quarantine Log")
            log_display = pd.DataFrame([
                {
                    "Candidate ID": h["candidate_id"],
                    "Violation Category": h["type"],
                    "Vulnerability Explanation": h["reason"],
                    "Current Title": h["_candidate"]["profile"]["current_title"],
                    "Country": h["_candidate"]["profile"]["country"],
                }
                for h in honeypots
            ])
            st.dataframe(log_display, width="stretch", hide_index=True)

    # ── TAB 5: MY CUSTOM SHORTLIST ────────────────────────────────────────────
    with tab_manual_shortlist:
        st.subheader("⭐ My Custom Shortlist Workbench")
        st.markdown("Here you can manage and download candidates you have manually selected from the Shortlist Explorer.")
        
        selected_ids = st.session_state["selected_candidate_ids"]
        if not selected_ids:
            st.info("No candidates selected yet. Go to the 'Shortlist Explorer' tab and check the 'Shortlist Candidate' box on any profile!")
        else:
            # Get selected candidates' row data
            selected_rows = [r for r in output_rows if r["candidate_id"] in selected_ids]
            # Re-sort them by score descending
            selected_rows.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))
            
            st.markdown(f"**Total candidates selected:** {len(selected_rows)}")
            
            # Metrics for selected
            col_m1, col_m2, col_m3 = st.columns(3)
            avg_score = sum(r["score"] for r in selected_rows) / len(selected_rows)
            avg_yoe = sum(r["yoe"] for r in selected_rows) / len(selected_rows)
            col_m1.metric("Average Score", f"{avg_score:.2f}")
            col_m2.metric("Average Experience", f"{avg_yoe:.1f} Yrs")
            col_m3.metric("Locations Represented", f"{len(set(r['location'] for r in selected_rows))}")
            
            # Action to download
            custom_csv_buf = io.StringIO()
            custom_writer = csv.writer(custom_csv_buf)
            custom_writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for idx, r in enumerate(selected_rows, start=1):
                custom_writer.writerow([r["candidate_id"], idx, r["score"], r["reasoning"]])
            
            # Executive Report Generation
            report_text = f"""=============================================================================
             EXECUTIVE SHORTLIST SUMMARY REPORT — REDROB WORKBENCH
=============================================================================
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Hiring Target Role Description: {jd_text[:150]}...
Total Candidates Shortlisted: {len(selected_rows)}
Average Evaluation Score: {avg_score:.2f}
Average Candidate Experience: {avg_yoe:.1f} Years
Total Unique Cities: {len(set(r['location'] for r in selected_rows))}

-----------------------------------------------------------------------------
SHORTLISTED CANDIDATES EVALUATION SUMMARY:
-----------------------------------------------------------------------------
"""
            for idx, r in enumerate(selected_rows, start=1):
                c_data = r["_candidate"]
                sig = c_data["redrob_signals"]
                report_text += f"""
{idx}. CANDIDATE ID: {r["candidate_id"]} (Score: {r["score"]:.3f})
   - Rank: #{idx} (Relative Shortlist Rank)
   - Current Title: {r["title"]}
   - Current Company: {r["company"]}
   - Experience: {r["yoe"]:.1f} Years
   - Location: {r["location"]}, {r["country"]}
   - Notice Period: {sig.get("notice_period_days")} Days
   - Evaluation Reasoning Summary:
     "{r["reasoning"]}"
   --------------------------------------------------------------------------"""

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label=f"⬇️ Download Custom Shortlist CSV ({len(selected_rows)} candidates)",
                    data=custom_csv_buf.getvalue(),
                    file_name="custom_shortlist.csv",
                    mime="text/csv",
                    key="download_custom_csv"
                )
            with col_dl2:
                st.download_button(
                    label=f"📝 Download Executive Report (.txt)",
                    data=report_text,
                    file_name="executive_summary_report.txt",
                    mime="text/plain",
                    key="download_custom_report"
                )
            
            # Show list/table of selected candidates with deselect option
            st.markdown("##### Selected Candidates Checklist")
            for r in selected_rows:
                col_cname, col_cdel = st.columns([8, 2])
                col_cname.markdown(f"**{r['candidate_id']}** — {r['title']} at {r['company']} (Score: **{r['score']:.3f}**)")
                if col_cdel.button("Remove candidate from list", key=f"remove_sel_{r['candidate_id']}"):
                    st.session_state["selected_candidate_ids"].remove(r["candidate_id"])
                    st.success("Removed candidate!")
                    st.rerun()

    # ── TAB 6: ADD CANDIDATES / RESUMES ─────────────────────────────────────────
    with tab_add_candidate:
        st.subheader("📥 Add Custom Candidates & Resumes")
        st.markdown("Add individual candidates to the evaluation pool by filling out a form, pasting raw resume text, or inputting structured JSON.")
        
        add_mode = st.radio("Choose Input Method", ["Interactive Profile Form", "Paste Raw Text Resume", "Paste Structured JSON"], horizontal=True)
        
        if add_mode == "Interactive Profile Form":
            st.markdown("##### Fill Candidate Profile Details")
            with st.form("custom_candidate_form", clear_on_submit=True):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    cf_name = st.text_input("Full Name (Anonymized)", "Alexander Watson")
                    cf_title = st.text_input("Current Job Title", "Senior AI Engineer")
                    cf_company = st.text_input("Current Company", "CoreML Solutions")
                    cf_industry = st.selectbox("Current Industry", list(PRODUCT_INDUSTRIES) + ["Consulting", "Services", "Other"])
                    cf_yoe = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=6.5, step=0.5)
                with col_f2:
                    cf_country = st.text_input("Country", "India")
                    cf_location = st.text_input("City / Location", "Noida")
                    cf_willing_relocate = st.checkbox("Willing to Relocate", value=True)
                    cf_notice = st.number_input("Notice Period (Days)", min_value=0, max_value=180, value=30, step=5)
                    cf_open_to_work = st.checkbox("Open to Work flag", value=True)
                    
                cf_skills = st.text_area("Skills (Comma-separated, e.g. Python, PyTorch, Embeddings, FAISS)", "Python, PyTorch, FAISS, Embeddings, NLP")
                cf_history_desc = st.text_area("Current Job Description Details (for semantic ranking)", "Built embeddings retrieval pipeline using sentence-transformers and FAISS database. Scaled search ranking models.")
                
                submitted = st.form_submit_button("➕ Add Candidate to Evaluation Pool")
                if submitted:
                    # Construct Candidate Object
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
                    st.success(f"Added candidate **{cf_name}** ({cand_id}) successfully!")
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
                
                # Show details in a card
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
            json_input = st.text_area("Paste JSON Candidate Schema structure here:", height=200, placeholder='{\n  "candidate_id": "cust_999",\n  "profile": {\n    "anonymized_name": "Sarah Connor",\n    "current_title": "Senior NLP Scientist",\n    ...\n  },\n  ...\n}')
            
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

else:
    st.info("The candidate pool is empty. Add candidates in the 'Add Candidates/Resumes' tab or upload/import a file in the sidebar to begin!")
    
    # Static visual about the ranker details
    st.subheader("📖 Candidate Scoring Design System")
    st.markdown(clean_html("""
    The Redrob Intelligent Ranker utilizes a five-component weighted scoring system coupled with a behavioral availability multiplier to score candidates based on relevance rather than keyword density.
    
    ### How candidates are scored:
    
    1. **Role/Title Fit (0–30 pts)**: Penalizes title stuffing and rewards core ML/AI profiles (e.g. AI Engineers, NLP Scientists) while penalizing disqualified titles.
    2. **Career History (0–30 pts)**: Calculates career duration at product companies. Pure consulting service firm histories are penalized to zero.
    3. **Skills Trust Match (0–25 pts)**: Scales core skill weights by endorsement confidence and duration metrics.
    4. **Experience Level (0–10 pts)**: Evaluates optimal target experience bands (JD specifies 5–9 years).
    5. **Location Alignment (0–5 pts)**: Prioritizes local candidates (Noida/Pune) and relocalizing applicants from Tier-1 cities.
    
    **Behavioral multiplier (×0.3 to ×1.2)**: Adjusted based on availability markers (notice period, response rates, recent login activity, and interview completion history).
    """))
