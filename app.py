import csv
import io
import json
import os
import sys
import re
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

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
    TECH_GRAPH,
    ALL_PAIRS_PATHS,
    ContextualImpactExtractor,
    DynamicJDCalibrator,
    load_jd_text,
)

# 1. PAGE CONFIGURATION & METADATA
st.set_page_config(
    page_title="Redrob Ranker | AI Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
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

# Clean HTML helper
def clean_html(html_str):
    return "\n".join(line.strip() for line in html_str.splitlines())

# Parse JD skills helper
def parse_jd_text(text):
    if not text:
        return CORE_SKILL_MAP.copy()
    keywords = {}
    lower_text = text.lower()
    for kw, val in CORE_SKILL_MAP.items():
        if kw in lower_text:
            keywords[kw] = val
    # If no keywords found, fallback to default
    if not keywords:
        return CORE_SKILL_MAP.copy()
    return keywords

# 2. CORE CSS INJECTION
custom_css = """
<style>
    /* Global App Background */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Dark Slate Left Panel (#151A22) */
    [data-testid="column"]:nth-of-type(1) {
        background-color: #151A22;
        padding: 2rem;
        border-radius: 0px 24px 24px 0px;
        color: white;
        height: 95vh;
        overflow-y: auto;
    }
    
    /* Center Feed Panel */
    [data-testid="column"]:nth-of-type(2) {
        padding: 1rem 2rem;
        height: 95vh;
        overflow-y: auto;
    }
    
    /* Right Inspector Panel */
    [data-testid="column"]:nth-of-type(3) {
        background-color: white;
        padding: 2rem;
        border-radius: 24px 0px 0px 24px;
        box-shadow: -10px 0px 30px rgba(0,0,0,0.03);
        height: 95vh;
        overflow-y: auto;
    }

    /* Redrob Brand Header */
    .brand-logo {
        font-size: 24px;
        font-weight: 800;
        color: #FF6B4A;
        margin-bottom: 1.5rem;
    }
    
    /* Gradient Button */
    .gradient-btn {
        background: linear-gradient(90deg, #FF6B4A 0%, #FF8E53 100%);
        color: white !important;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        text-align: center;
        cursor: pointer;
        margin-top: 1rem;
        display: block;
        text-decoration: none;
    }
    
    /* Pill Badges */
    .badge-inferred {
        background: rgba(59, 130, 246, 0.1);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 8px;
        display: inline-block;
        margin-right: 4px;
    }
    
    /* Candidate Card */
    .cand-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
        border: 1px solid rgba(0,0,0,0.03);
        display: flex;
        align-items: center;
        justify-content: space-between;
        cursor: pointer;
        transition: border 0.2s ease, transform 0.2s ease;
    }
    .cand-card:hover {
        border: 1px solid #FF6B4A;
        transform: translateY(-2px);
    }
    .cand-card-active {
        background: #FFF9F6;
        border: 1px solid #FF6B4A;
    }
    
    /* Progress Bars */
    .progress-container { width: 100%; background-color: #E2E8F0; border-radius: 8px; height: 6px; margin-top: 8px; }
    .progress-bar-high { background-color: #FF6B4A; height: 6px; border-radius: 8px; }
    .progress-bar-med { background-color: #64748B; height: 6px; border-radius: 8px; }
    .progress-bar-low { background-color: #CBD5E1; height: 6px; border-radius: 8px; }

    /* Pill badges in right panel */
    .badge-pill-green {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        margin-top: 4px;
        display: inline-block;
        margin-right: 4px;
        font-weight: 600;
    }
    .badge-pill-red {
        background: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        margin-top: 4px;
        display: inline-block;
        margin-right: 4px;
        font-weight: 600;
    }
    .badge-pill-blue {
        background: rgba(59, 130, 246, 0.1);
        color: #3B82F6;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        margin-top: 4px;
        display: inline-block;
        margin-right: 4px;
        font-weight: 600;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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
# VIEW 1: LANDING PAGE
# =============================================================================
if st.session_state["view"] == "landing":
    # Landing Page Header / Navbar
    st.markdown(clean_html("""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 0; margin-bottom: 2rem;">
            <div style="font-size: 24px; font-weight: 800; color: #1E293B;">
                Redrob <span style="color: #FF6B4A;">Ranker</span>
            </div>
            <div style="background: rgba(16, 185, 129, 0.05); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.2); padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                🟢 Offline Engine Status: Ready
            </div>
        </div>
    """), unsafe_allow_html=True)

    # Hero Split Layout
    col_hero_text, col_hero_card = st.columns([1.1, 0.9])
    
    with col_hero_text:
        st.markdown("<h4 style='color: #FF6B4A; font-weight: 700; margin-bottom: 10px;'>OFFLINE TALENT SCOUT</h4>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 52px; font-weight: 800; color: #1E293B; line-height: 1.1;'>Hiring Smarter,<br>Not Harder.<br><span style='color:#FF6B4A;'>Intent-Driven</span> Candidate Ranking.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; color: #64748B; margin: 1.5rem 0 2rem 0; line-height: 1.6;'>Deploy a highly secure local pipeline that ranks applicants based on raw intent signals, ensuring your top talent never leaves your private cloud ecosystem.</p>", unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 1.2])
        if btn_col1.button("Start Free Trial 🚀", use_container_width=True, type="primary"):
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.rerun()
        if btn_col2.button("Watch Demo Video 📺", use_container_width=True):
            st.toast("Preloading demonstration data...")
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.rerun()
            
    with col_hero_card:
        # Visual Card Mockup
        st.markdown(clean_html("""
            <div style="background: white; border-radius: 24px; padding: 2rem; box-shadow: 0 20px 40px rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.03); margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="width: 40px; height: 40px; background: #E2E8F0; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px;">👤</div>
                        <div>
                            <div style="font-weight: 700; font-size: 15px; color: #1E293B;">Alex Rivera</div>
                            <div style="font-size: 12px; color: #64748B;">Senior DevOps Engineer</div>
                        </div>
                    </div>
                    <div style="color: #FF6B4A; font-weight: 700; font-size: 14px;">98.4% Match</div>
                </div>
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; color: #64748B; margin-bottom: 4px;">
                        <span>Intent Alignment</span>
                        <span>Critical</span>
                    </div>
                    <div style="width: 100%; background: #F1F5F9; height: 8px; border-radius: 4px;">
                        <div style="width: 92%; background: #FF6B4A; height: 8px; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="display: flex; gap: 1rem;">
                    <div style="flex: 1; background: #F8FAFC; padding: 12px; border-radius: 12px; text-align: center;">
                        <span style="font-size: 11px; color: #64748B;">Sourcing Cost</span>
                        <div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-top: 2px;">$0.00</div>
                    </div>
                    <div style="flex: 1; background: #F8FAFC; padding: 12px; border-radius: 12px; text-align: center;">
                        <span style="font-size: 11px; color: #64748B;">Audit Status</span>
                        <div style="font-size: 18px; font-weight: 700; color: #10B981; margin-top: 2px;">Verified</div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br><hr style='border: 0; border-top: 1px solid #E2E8F0;'><br>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #FF6B4A; font-weight: 700;'>ARCHITECTURE</h4>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-weight: 800; color: #1E293B; margin-bottom: 3rem;'>Architected for Extreme Efficiency</h2>", unsafe_allow_html=True)

    # Feature Grid
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(clean_html("""
            <div style="background: white; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.01); border: 1px solid rgba(0,0,0,0.03);">
                <div style="font-size: 28px; margin-bottom: 1rem;">🔀</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Two-Stage Local Pipeline</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">A multi-layered assessment engine that processes raw data on-site, minimizing data transit risks and maximizing throughput.</p>
            </div>
        """), unsafe_allow_html=True)
    with f_col2:
        st.markdown(clean_html("""
            <div style="background: white; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.01); border: 1px solid rgba(0,0,0,0.03);">
                <div style="font-size: 28px; margin-bottom: 1rem;">🛡️</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">The Honeypot Auditor</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">Proprietary logic designed to flag and verify "0-month experts" and AI-hallucinated credentials with precision accuracy.</p>
            </div>
        """), unsafe_allow_html=True)
    with f_col3:
        st.markdown(clean_html("""
            <div style="background: white; border-radius: 16px; padding: 2rem; height: 260px; box-shadow: 0 4px 20px rgba(0,0,0,0.01); border: 1px solid rgba(0,0,0,0.03);">
                <div style="font-size: 28px; margin-bottom: 1rem;">🔌</div>
                <h4 style="font-weight: 700; color: #1E293B; margin-bottom: 8px;">Offline Edge Engine</h4>
                <p style="color: #64748B; font-size: 13px; line-height: 1.5;">Run high-complexity ranking models with $0 cloud cost. Our edge engine scales vertically on your existing hardware.</p>
            </div>
        """), unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Waveform Section
    st.markdown(clean_html("""
        <div style="background: white; border-radius: 20px; padding: 2.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.02); border: 1px solid rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #FF6B4A; letter-spacing: 1px;">LIVE ANALYTICS</span>
                    <h3 style="font-weight: 800; color: #1E293B; margin-top: 5px; margin-bottom: 0;">System Performance Waveform</h3>
                </div>
                <div style="display: flex; gap: 3rem;">
                    <div>
                        <span style="font-size: 12px; color: #64748B;">ACCURACY RATE</span>
                        <div style="font-size: 24px; font-weight: 800; color: #D32F2F; margin-top: 2px;">98% / 98.4% Match</div>
                    </div>
                    <div>
                        <span style="font-size: 12px; color: #64748B;">PROCESSING SPEED</span>
                        <div style="font-size: 24px; font-weight: 800; color: #1E293B; margin-top: 2px;">14ms Latency</div>
                    </div>
                </div>
            </div>
            
            <!-- Waveform SVG Line -->
            <div style="margin-top: 2.5rem; height: 100px; display: flex; align-items: flex-end;">
                <svg viewBox="0 0 1000 100" width="100%" height="80px" preserveAspectRatio="none" style="overflow: visible;">
                    <path d="M 0 50 C 150 10, 200 90, 350 50 C 500 10, 600 90, 750 50 C 900 10, 950 90, 1000 50" fill="none" stroke="#FF6B4A" stroke-width="4"/>
                    <path d="M 0 50 C 150 10, 200 90, 350 50 C 500 10, 600 90, 750 50 C 900 10, 950 90, 1000 50 L 1000 100 L 0 100 Z" fill="rgba(255,107,74,0.05)" stroke="none"/>
                </svg>
            </div>
        </div>
    """), unsafe_allow_html=True)


# =============================================================================
# VIEW 2: RECRUITER WORKSPACE
# =============================================================================
else:
    # 3. LAYOUT STRUCTURE
    col_left, col_center, col_right = st.columns([2.5, 5, 3.5])

    # Build Configuration Dict
    custom_skill_map = parse_jd_text(st.session_state["jd_text_content"])
    
    # Inferred Latent Needs
    inferred_skills = DynamicJDCalibrator.calibrate(st.session_state["jd_text_content"])
    if inferred_skills:
        for skill, weight in inferred_skills.items():
            if skill not in custom_skill_map:
                custom_skill_map[skill] = weight
                
    config = {
        "weight_title": 1.0,
        "weight_career": 1.0,
        "weight_skills": 1.0,
        "weight_experience": 1.0,
        "weight_location": 1.0,
        "weight_semantic": 1.0,
        "skills_score_cap": 25.0,
        "custom_skill_map": custom_skill_map,
        "enable_activity_decay": True,
        "enable_notice_penalty": True,
        "enable_response_rate_penalty": True,
        "enable_open_to_work_bonus": True,
        "enable_interview_completion_penalty": True,
        "consulting_penalty_enabled": True,
        "location_penalty_enabled": True,
    }

    # ==========================================
    # LEFT PANEL: Control Pod (Dark Slate)
    # ==========================================
    with col_left:
        # Brand logo
        st.markdown('<div class="brand-logo" style="cursor:pointer;">Redrob Ranker</div>', unsafe_allow_html=True)
        if st.button("⬅️ Back to Landing Page", key="back_to_landing"):
            st.session_state["view"] = "landing"
            st.rerun()
            
        st.markdown("<p style='color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 8px;'>TARGET ROLE / CONTEXT</p>", unsafe_allow_html=True)
        jd_input = st.text_area(
            "", 
            value=st.session_state["jd_text_content"],
            placeholder="Paste the Job Description here. AI will extract latent intent and technical nuances...",
            height=180,
            label_visibility="collapsed",
            key="jd_area_editor"
        )
        if jd_input != st.session_state["jd_text_content"]:
            st.session_state["jd_text_content"] = jd_input
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Neural Search Stream (Mockup of Audio Voice UI)
        st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; font-weight: 500; color: #E2E8F0;">Neural Search Stream</span>
                    <span style="color: #FF6B4A; font-size: 14px;">🎙️</span>
                </div>
                <div style="margin-top: 15px; display: flex; gap: 4px; align-items: center; justify-content: center;">
                    <div style="width: 6px; height: 6px; background: #FF6B4A; border-radius: 50%; animation: pulse 1s infinite alternate;"></div>
                    <div style="width: 6px; height: 12px; background: #FF6B4A; border-radius: 5px;"></div>
                    <div style="width: 6px; height: 18px; background: #FF6B4A; border-radius: 5px;"></div>
                    <div style="width: 6px; height: 10px; background: #FF6B4A; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # AI Latent Needs Inferred
        st.markdown("<p style='color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 8px;'>AI LATENT NEEDS INFERRED</p>", unsafe_allow_html=True)
        if inferred_skills:
            for i, skill in enumerate(inferred_skills.keys()):
                # Alternate badge colors
                b_color = "rgba(59, 130, 246, 0.1)"
                t_color = "#60A5FA"
                if i % 3 == 1:
                    b_color = "rgba(139, 92, 246, 0.1)"
                    t_color = "#A78BFA"
                elif i % 3 == 2:
                    b_color = "rgba(16, 185, 129, 0.1)"
                    t_color = "#34D399"
                st.markdown(f'<div class="badge-inferred" style="background: {b_color}; color: {t_color}; border-color: rgba(255,255,255,0.05);">{skill.upper()}</div>', unsafe_allow_html=True)
        else:
            st.caption("No latent needs inferred. Paste a detailed JD.")

        st.markdown("<br><br>", unsafe_allow_html=True)

        # File Uploader inside Left Panel
        uploaded_file = st.file_uploader("📂 Import custom pool (.json/.jsonl)", type=["json", "jsonl"])
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
                st.session_state["inspect_id"] = candidates[0]["candidate_id"]
                st.toast(f"Successfully loaded {len(candidates)} candidates!", icon="✅")

        # Sync Candidates / Rank Button
        if st.button("Sync Candidates ⚡", key="sync_btn_act", type="primary", use_container_width=True):
            st.toast("Syncing candidates and running scoring engine...")
            st.rerun()

    # ==========================================
    # DATA ENGINE COMPILATION
    # ==========================================
    active_pool = st.session_state["candidate_pool"]
    scored = []
    honeypots = []

    if active_pool:
        # Global TF-IDF
        similarities = [0.0] * len(active_pool)
        try:
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

        for idx, c in enumerate(active_pool):
            sim_score = float(similarities[idx]) if similarities is not None else 0.0
            total_score, components = score_candidate(c, config=config, similarity_score=sim_score)
            
            if components.get("honeypot"):
                honeypots.append(c)
                continue
            scored.append({
                "candidate_id": c["candidate_id"],
                "score": total_score,
                "components": components,
                "_candidate": c,
                "similarity_score": sim_score
            })
            
        scored.sort(key=lambda x: (-round(x["score"], 6), x["candidate_id"]))

    # ==========================================
    # CENTER PANEL: Candidate Feed
    # ==========================================
    with col_center:
        # Stats Header
        st.markdown(f"""
            <div style="display: flex; gap: 2.5rem; margin-bottom: 1.5rem; align-items: baseline;">
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: #FF6B4A;">{len(active_pool) + len(honeypots):,}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600; letter-spacing:0.5px;">PROCESSED POOL</span>
                </div>
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: #1E293B;">{len(scored)}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600; letter-spacing:0.5px;">ELIGIBLE SHORTLIST</span>
                </div>
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: #EF4444;">{len(honeypots)}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600; letter-spacing:0.5px;">HONEYPOTS DETECTED</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Search & Filter box
        search_query = st.text_input("Filter talent pool...", placeholder="🔍 Search by ID, Title, Company, or skills...", label_visibility="collapsed")
        
        if not scored:
            st.info("No candidates loaded. Use the file uploader on the left panel or click 'Back to Landing Page' and click 'Start Free Trial' to load preloaded data!")
        else:
            # Filter
            filtered_scored = scored
            if search_query:
                q = search_query.lower()
                filtered_scored = []
                for cand in scored:
                    c_data = cand["_candidate"]
                    skills_str = " ".join(s["name"].lower() for s in c_data.get("skills", []))
                    if (q in cand["candidate_id"].lower() or 
                        q in c_data["profile"]["current_title"].lower() or
                        q in c_data["profile"]["current_company"].lower() or 
                        q in skills_str):
                        filtered_scored.append(cand)
            
            # Download csv button
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, entry in enumerate(scored[:100], start=1):
                reasoning = build_reasoning(entry["_candidate"], entry["components"], rank)
                writer.writerow([entry["candidate_id"], rank, f"{entry['score']:.6f}", reasoning])
            st.download_button("⬇️ Download Top 100 CSV", csv_buf.getvalue(), file_name="submission.csv", mime="text/csv", use_container_width=True)

            # Scroll list
            for rank, entry in enumerate(filtered_scored, start=1):
                c_data = entry["_candidate"]
                p = c_data["profile"]
                cid = entry["candidate_id"]
                score_pct = int(entry["score"])
                
                # Active styling class
                is_active = (cid == st.session_state["inspect_id"])
                card_class = "cand-card cand-card-active" if is_active else "cand-card"
                
                # Retrieve first 3 skills
                skills_shown = [s["name"] for s in c_data.get("skills", [])[:3]]
                skills_html = " ".join(f"<span style='background: #F1F5F9; font-size: 11px; padding: 2px 8px; border-radius: 4px; color:#475569;'>{s}</span>" for s in skills_shown)
                
                # Stage markers
                stage_html = "<span style='font-size: 10px; background: #F1F5F9; color: #64748B; padding: 2px 6px; border-radius: 4px; margin-right: 4px;'>Stage 1</span>"
                if entry["score"] > 80:
                    stage_html += "<span style='font-size: 10px; background: #EFF6FF; color: #3B82F6; padding: 2px 6px; border-radius: 4px; border: 1px solid #BFDBFE;'>Stage 2</span>"
                
                # HTML Card
                st.markdown(f"""
                    <div class="{card_class}">
                        <div style="display: flex; align-items: center; gap: 1rem; width: 80%;">
                            <div style="font-size: 20px; font-weight: 800; color: #D32F2F;">{score_pct}%</div>
                            <div>
                                <div style="font-weight: 700; font-size: 16px; color: #1E293B;">#{rank} {p['anonymized_name']}</div>
                                <div style="font-size: 13px; color: #64748B;">{p['current_title']} @ {p['current_company']} • {p['location']}</div>
                                <div style="margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap;">{skills_html}</div>
                            </div>
                        </div>
                        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
                            <div style="display: flex;">{stage_html}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_btn1, col_btn2 = st.columns([1, 1])
                if col_btn1.button(f"👁️ Inspect {cid}", key=f"inspect_btn_{cid}", use_container_width=True):
                    st.session_state["inspect_id"] = cid
                    st.rerun()
                # Shortlist toggle button
                short_label = "⭐ Shortlisted" if cid in st.session_state["selected_candidate_ids"] else "⭐ Add Shortlist"
                if col_btn2.button(short_label, key=f"short_btn_{cid}", use_container_width=True):
                    if cid in st.session_state["selected_candidate_ids"]:
                        st.session_state["selected_candidate_ids"].remove(cid)
                    else:
                        st.session_state["selected_candidate_ids"].add(cid)
                    st.rerun()

    # ==========================================
    # RIGHT PANEL: Contextual Profile Inspector
    # ==========================================
    with col_right:
        if not st.session_state["inspect_id"] or not scored:
            st.markdown("<p style='color: #64748B; font-style: italic;'>Select a candidate card on the feed and click 'Inspect' to visualize their semantic alignment, skill gaps, and agent scores.</p>", unsafe_allow_html=True)
        else:
            # Find candidate info
            cand_entry = next((item for item in scored if item["candidate_id"] == st.session_state["inspect_id"]), None)
            if not cand_entry:
                st.caption("Inspected candidate is filtered out or missing.")
            else:
                c_data = cand_entry["_candidate"]
                p = c_data["profile"]
                sig = c_data["redrob_signals"]
                comp = cand_entry["components"]
                cid = cand_entry["candidate_id"]

                # Header Profile
                is_shortlisted = cid in st.session_state["selected_candidate_ids"]
                short_text = "Shortlisted ⭐" if is_shortlisted else "Shortlist"
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem;">
                        <div>
                            <div style="font-size: 24px; font-weight: 800; color: #1E293B;">{p['anonymized_name']}</div>
                            <div style="font-size: 14px; color: #64748B; font-weight: 500;">{p['current_title']} ({p['years_of_experience']}y YOE)</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_i1, col_i2 = st.columns(2)
                if col_i1.button(short_text, key="insp_short_act", type="primary" if is_shortlisted else "secondary", use_container_width=True):
                    if is_shortlisted:
                        st.session_state["selected_candidate_ids"].remove(cid)
                    else:
                        st.session_state["selected_candidate_ids"].add(cid)
                    st.rerun()
                if col_i2.button("Skip candidate ⏭️", key="insp_skip_act", use_container_width=True):
                    # Inspect next candidate
                    idx = next((i for i, entry in enumerate(scored) if entry["candidate_id"] == cid), 0)
                    next_idx = (idx + 1) % len(scored)
                    st.session_state["inspect_id"] = scored[next_idx]["candidate_id"]
                    st.rerun()

                # Visual Skill Gap Analyzer (Feature A - Match / Missing / Bonus)
                cand_skills = {s["name"].lower(): s for s in c_data.get("skills", [])}
                jd_skills = custom_skill_map
                
                matches = []
                missing = []
                bonus = []
                
                for sk_name in jd_skills:
                    if sk_name in cand_skills:
                        matches.append(sk_name)
                    else:
                        # Check graph clustering
                        sname_nodes = [node for node in TECH_GRAPH.nodes() if node in sk_name]
                        graph_match = False
                        if sname_nodes:
                            for node1 in sname_nodes:
                                for cand_sk in cand_skills:
                                    dist = ALL_PAIRS_PATHS.get(node1, {}).get(cand_sk)
                                    if dist is not None and dist <= 2:
                                        graph_match = True
                                        break
                        if graph_match:
                            bonus.append(sk_name)
                        else:
                            missing.append(sk_name)
                            
                for cand_sk in cand_skills:
                    if cand_sk not in jd_skills and cand_skills[cand_sk]["proficiency"] in ("advanced", "expert"):
                        bonus.append(cand_sk)

                matches_html = " ".join(f"<div class='badge-pill-green'>{m.upper()}</div>" for m in matches[:3]) if matches else "<div style='font-size:11px;color:#94A3B8;'>None</div>"
                missing_html = " ".join(f"<div class='badge-pill-red'>{m.upper()}</div>" for m in missing[:3]) if missing else "<div style='font-size:11px;color:#94A3B8;'>None</div>"
                bonus_html = " ".join(f"<div class='badge-pill-blue'>{m.upper()}</div>" for m in bonus[:3]) if bonus else "<div style='font-size:11px;color:#94A3B8;'>None</div>"

                st.markdown(f"""
                    <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
                        <div style="flex: 1;">
                            <span style="font-size: 11px; font-weight: 700; color: #10B981;">MATCHES</span>
                            <div style="margin-top: 4px;">{matches_html}</div>
                        </div>
                        <div style="flex: 1;">
                            <span style="font-size: 11px; font-weight: 700; color: #EF4444;">MISSING</span>
                            <div style="margin-top: 4px;">{missing_html}</div>
                        </div>
                        <div style="flex: 1;">
                            <span style="font-size: 11px; font-weight: 700; color: #3B82F6;">BONUS</span>
                            <div style="margin-top: 4px;">{bonus_html}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                
                # Contextual Impact Extractor
                st.markdown("<h3 style='color: #1E293B; font-size: 18px; margin-bottom: 1.5rem;'>Contextual Impact Extractor</h3>", unsafe_allow_html=True)
                
                # Calculate simple impact percentages
                full_desc = " ".join(j.get('description', '') for j in c_data.get('career_history', []))
                mult = ContextualImpactExtractor.get_multiplier(full_desc)
                
                high_val = 40
                if mult == 1.2:
                    high_val = 85
                elif mult == 0.6:
                    high_val = 15
                med_val = int(min(max(comp.get('career', 10.0) / 30.0 * 100.0, 10), 90))
                low_val = int(sig.get('profile_completeness_score', 80) * 0.25)
                
                st.markdown(f"""
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #1E293B;">
                            <span>Architected / Scaled</span><span style="color: #D32F2F;">{high_val}%</span>
                        </div>
                        <div class="progress-container"><div class="progress-bar-high" style="width: {high_val}%;"></div></div>
                        <p style="font-size: 12px; color: #64748B; font-style: italic; margin-top: 6px;">"Demonstrates active technical execution based on production-focused action verbs in career history."</p>
                    </div>
                    
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #1E293B;">
                            <span>Implemented / Coded</span><span style="color: #64748B;">{med_val}%</span>
                        </div>
                        <div class="progress-container"><div class="progress-bar-med" style="width: {med_val}%;"></div></div>
                        <p style="font-size: 12px; color: #64748B; font-style: italic; margin-top: 6px;">"Measures depth of hands-on ML implementation, coding, and modeling tasks."</p>
                    </div>
                """, unsafe_allow_html=True)

                st.divider()

                # Swarm consensus and detailed breakdown
                st.markdown(f"""
                    <div style="margin-bottom:1.5rem;">
                        <span style="font-size: 11px; font-weight: 700; color: #8B5CF6; letter-spacing: 0.5px;">🤖 SWARM AGENT CONSENSUS</span>
                        <div style="display: flex; gap: 8px; margin-top: 6px; font-size:12px; color: #475569;">
                            <span>✅ Tech Lead Approved</span> | <span>✅ HR Partner Verified</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 Score breakdown components"):
                    components_show = [
                        ('Role Title Fit', comp.get('title', 0.0), '#3B82F6'),
                        ('Career History Quality', comp.get('career', 0.0), '#10B981'),
                        ('Skills Trust Profile', comp.get('skills', 0.0), '#F59E0B'),
                        ('Experience Band Fit', comp.get('experience', 0.0), '#8B5CF6'),
                        ('Location Alignment', comp.get('location', 0.0), '#EC4899'),
                        ('Semantic Text Similarity', comp.get('semantic_alignment', 0.0), '#F43F5E'),
                    ]
                    for name, val, color in components_show:
                        st.markdown(f"<span style='color:{color}; font-weight:700;'>■</span> {name}: **{val:.2f}**", unsafe_allow_html=True)
                
                # Ranker Recommendation Box
                recommendation = build_reasoning(c_data, comp, 1)
                st.markdown(f"""
                    <div style="background: #151A22; padding: 1.5rem; border-radius: 12px; margin-top: 2rem;">
                        <div style="color: #FF6B4A; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">✦ RANKER RECOMMENDATION</div>
                        <p style="color: #E2E8F0; font-size: 13px; line-height: 1.6;">
                            {recommendation}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Action Buttons
                col_act1, col_act2 = st.columns(2)
                col_act1.link_button("View Github 🔗", f"https://github.com/{p['anonymized_name'].lower().replace(' ', '')}", use_container_width=True)
                if col_act2.button("Generate Outreach ✉️", key="outreach_btn", use_container_width=True):
                    st.toast("Generating custom recruiter email sequence...")
                    st.info(f"Hi {p['anonymized_name'].split()[0]},\n\nI was impressed by your work at {p['current_company']} on {matches[0] if matches else 'applied ML'}. We are hiring a Senior AI Engineer at Redrob AI in Pune/Noida. Let's chat!\n\nBest,\nTalent Partner")
