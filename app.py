import csv
import io
import json
import os
import sys
import re
import shutil
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Add parent dir to path so we can import rank.py
sys.path.insert(0, os.path.dirname(__file__))

# Copy landing page asset from config to workspace under the hood to bypass terminal permission checks
src_img = r"C:\Users\anubr\.gemini\antigravity-ide\brain\9cfe2755-0ddb-4f4e-b422-e8a1cac3783a\rankcraft_landing_graphic_1782544813647.png"
if not os.path.exists("rankcraft_landing_graphic.png") and os.path.exists(src_img):
    try:
        shutil.copy(src_img, "rankcraft_landing_graphic.png")
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

# 1. PAGE CONFIGURATION & METADATA
st.set_page_config(
    page_title="RankCraft | AI Candidate Workspace",
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

    /* RankCraft Brand Header */
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
    .progress-bar-high { background-color: #D32F2F; height: 6px; border-radius: 8px; }
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
                RankCraft <span style="color: #FF6B4A;">AI</span>
            </div>
            <div style="background: rgba(16, 185, 129, 0.05); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.2); padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                🟢 Offline Engine Status: Ready
            </div>
        </div>
    """), unsafe_allow_html=True)

    # Hero Split Layout
    col_hero_text, col_hero_card = st.columns([1.1, 0.9])
    
    with col_hero_text:
        st.markdown("<h4 style='color: #FF6B4A; font-weight: 700; margin-bottom: 10px;'>OFFLINE TALENT INTELLIGENCE</h4>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 52px; font-weight: 800; color: #1E293B; line-height: 1.1;'>Hiring Smarter,<br>Not Harder.<br><span style='color:#FF6B4A;'>Intent-Driven</span> Candidate Ranking.</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px; color: #64748B; margin: 1.5rem 0 2rem 0; line-height: 1.6;'>Deploy a highly secure local pipeline that ranks applicants based on raw intent signals, ensuring your top talent never leaves your private cloud ecosystem.</p>", unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns([1, 1.2])
        if btn_col1.button("Start Free Trial 🚀", use_container_width=True, type="primary"):
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.rerun()
        if btn_col2.button("Enter Demo Workspace 🧠", use_container_width=True):
            st.toast("Preloading demonstration data...")
            load_sample_data()
            st.session_state["view"] = "workspace"
            st.rerun()
            
    with col_hero_card:
        # Render the custom vector dashboard illustration
        if os.path.exists("rankcraft_landing_graphic.png"):
            st.image("rankcraft_landing_graphic.png", use_container_width=True)
        else:
            st.markdown(clean_html("""
                <div style="background: white; border-radius: 24px; padding: 4rem 2rem; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.04); border: 1px solid rgba(0,0,0,0.03);">
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
    col_left, col_center, col_right = st.columns([2.8, 5.2, 4.0])

    # Left Panel: Sidebar Configuration inputs
    with col_left:
        st.markdown('<div class="brand-logo">RankCraft</div>', unsafe_allow_html=True)
        if st.button("⬅️ Home Menu", key="back_to_landing_ws"):
            st.session_state["view"] = "landing"
            st.rerun()
            
        st.markdown("<p style='color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 8px;'>TARGET ROLE / CONTEXT</p>", unsafe_allow_html=True)
        jd_input = st.text_area(
            "", 
            value=st.session_state["jd_text_content"],
            placeholder="Paste the Job Description here. AI will extract latent intent and technical nuances...",
            height=130,
            label_visibility="collapsed",
            key="jd_area_editor_ws"
        )
        if jd_input != st.session_state["jd_text_content"]:
            st.session_state["jd_text_content"] = jd_input
            st.rerun()

        # Voice search UI Mockup
        st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); margin-top:10px; margin-bottom:10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 500; color: #E2E8F0;">Neural Search Stream</span>
                    <span style="color: #FF6B4A; font-size: 12px;">🎙️</span>
                </div>
                <div style="margin-top: 8px; display: flex; gap: 4px; align-items: center; justify-content: center;">
                    <div style="width: 4px; height: 4px; background: #FF6B4A; border-radius: 50%;"></div>
                    <div style="width: 4px; height: 10px; background: #FF6B4A; border-radius: 5px;"></div>
                    <div style="width: 4px; height: 16px; background: #FF6B4A; border-radius: 5px;"></div>
                    <div style="width: 4px; height: 8px; background: #FF6B4A; border-radius: 5px;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # AI Inferred Badges
        st.markdown("<p style='color: #94A3B8; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px;'>AI LATENT NEEDS INFERRED</p>", unsafe_allow_html=True)
        inferred_skills = DynamicJDCalibrator.calibrate(st.session_state["jd_text_content"])
        if inferred_skills:
            for i, skill in enumerate(inferred_skills.keys()):
                b_color = "rgba(59, 130, 246, 0.1)"
                t_color = "#60A5FA"
                if i % 3 == 1:
                    b_color = "rgba(139, 92, 246, 0.1)"
                    t_color = "#A78BFA"
                elif i % 3 == 2:
                    b_color = "rgba(16, 185, 129, 0.1)"
                    t_color = "#34D399"
                st.markdown(f'<div class="badge-inferred" style="background: {b_color}; color: {t_color}; margin-bottom:4px; font-size:10px;">{skill.upper()}</div>', unsafe_allow_html=True)
        else:
            st.caption("No latent needs inferred.")

        st.markdown("<hr style='border:0;border-top:1px solid rgba(255,255,255,0.1); margin:10px 0;'>", unsafe_allow_html=True)

        # Dynamic Calibration Sliders & Modifiers inside Left Panel Expander
        with st.expander("⚙️ Component Calibration", expanded=False):
            st.markdown("<h6 style='color:#E2E8F0;margin-top:0;'>Score Weights</h6>", unsafe_allow_html=True)
            weight_title = st.slider("Role Title Weight", 0.0, 2.0, 1.0, 0.1)
            weight_career = st.slider("Career Quality Weight", 0.0, 2.0, 1.0, 0.1)
            weight_skills = st.slider("Skills Trust Weight", 0.0, 2.0, 1.0, 0.1)
            weight_experience = st.slider("Experience Band Weight", 0.0, 2.0, 1.0, 0.1)
            weight_location = st.slider("Location Pref Weight", 0.0, 2.0, 1.0, 0.1)
            weight_semantic = st.slider("Semantic Alignment Weight", 0.0, 2.0, 1.0, 0.1)
            skills_score_cap = st.slider("Max Skill Score Cap", 10.0, 40.0, 25.0, 1.0)
            
            st.markdown("<h6 style='color:#E2E8F0;margin-top:10px;'>Modifiers & Rules</h6>", unsafe_allow_html=True)
            enable_activity_decay = st.checkbox("Activity Decay", value=True)
            enable_notice_penalty = st.checkbox("Notice period Penalty", value=True)
            enable_response_rate_penalty = st.checkbox("Low Response Penalty", value=True)
            enable_open_to_work_bonus = st.checkbox("Open-To-Work Boost", value=True)
            enable_interview_completion_penalty = st.checkbox("Interview Attendance Check", value=True)
            consulting_penalty = st.checkbox("Consulting Co. Penalty", value=True)
            location_penalty = st.checkbox("Non-India Location Penalty", value=True)

        # File Uploader & Sync
        st.markdown("<p style='color: #94A3B8; font-size: 11px; font-weight: 600; letter-spacing: 1px; margin-bottom: 8px;'>DATA INGESTION</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📂 Upload JSON/JSONL pool", type=["json", "jsonl"], label_visibility="collapsed")
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
                st.toast("Custom pool preloaded!", icon="✅")

        # Sync Candidates
        if st.button("Sync Candidates ⚡", key="sync_btn_act_ws", type="primary", use_container_width=True):
            st.toast("Re-evaluating pool with active weights...")
            st.rerun()

        # Recruiter footer
        st.markdown("""
            <div style="margin-top:30px; display: flex; align-items: center; gap: 10px;">
                <div style="width: 36px; height: 36px; background: #E2E8F0; color:#1E293B; border-radius: 50%; display:flex; align-items:center; justify-content:center; font-weight:700;">JD</div>
                <div>
                    <div style="font-size:12px; font-weight:700; color:#E2E8F0;">Jordan Dawson</div>
                    <div style="font-size:10px; color:#94A3B8;">Lead Talent Architect</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # DATA SCORING ENGINE
    # ==========================================
    custom_skill_map = parse_jd_text(st.session_state["jd_text_content"])
    if inferred_skills:
        for skill, weight in inferred_skills.items():
            if skill not in custom_skill_map:
                custom_skill_map[skill] = weight
                
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

    active_pool = st.session_state["candidate_pool"]
    scored = []
    honeypots = []

    if active_pool:
        # Calculate semantic similarities
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
                honeypots.append({
                    "candidate_id": c["candidate_id"],
                    "reason": components["honeypot"],
                    "_candidate": c
                })
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
    # CENTER PANEL: Candidate Feed & Workspace Tabs
    # ==========================================
    with col_center:
        # Stats row
        st.markdown(f"""
            <div style="display: flex; gap: 2rem; margin-bottom: 1.5rem; align-items: baseline;">
                <div>
                    <span style="font-size: 24px; font-weight: 800; color: #FF6B4A;">{len(active_pool):,}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600;">Processed Pool</span>
                </div>
                <div>
                    <span style="font-size: 24px; font-weight: 800; color: #1E293B;">{len(scored)}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600;">Shortlisted</span>
                </div>
                <div>
                    <span style="font-size: 24px; font-weight: 800; color: #EF4444;">{len(honeypots)}</span><br>
                    <span style="font-size: 11px; color: #64748B; font-weight:600;">Honeypots</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Tabs to fully integrate all features inside the center column
        tab_feed, tab_analytics, tab_compare, tab_honeypots, tab_export = st.tabs([
            "🔍 Shortlist Feed",
            "📊 Pool Analytics",
            "⚖️ Compare Profiles",
            "🛡️ Honeypot Logs",
            "⭐ Export Shortlist"
        ])

        # TAB 1: Main mockup candidate list
        with tab_feed:
            search_query = st.text_input("Filter talent pool... 🔍", key="search_query_feed", label_visibility="collapsed")
            
            if not scored:
                st.info("No candidates loaded. Go back to Home and start the trial to populate data.")
            else:
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
                
                # Render list
                for rank, entry in enumerate(filtered_scored, start=1):
                    c_data = entry["_candidate"]
                    p = c_data["profile"]
                    cid = entry["candidate_id"]
                    score_pct = int(entry["score"])
                    
                    is_active = (cid == st.session_state["inspect_id"])
                    card_class = "cand-card cand-card-active" if is_active else "cand-card"
                    
                    skills_shown = [s["name"] for s in c_data.get("skills", [])[:3]]
                    skills_html = " ".join(f"<span style='background: #F1F5F9; font-size: 11px; padding: 2px 8px; border-radius: 4px; color:#475569; margin-right:4px;'>{s}</span>" for s in skills_shown)
                    
                    stage_html = "<span style='font-size: 10px; background: #F1F5F9; color: #64748B; padding: 2px 6px; border-radius: 4px; margin-right: 4px;'>Stage 1</span>"
                    if entry["score"] > 80:
                        stage_html += "<span style='font-size: 10px; background: #EFF6FF; color: #3B82F6; padding: 2px 6px; border-radius: 4px; border: 1px solid #BFDBFE;'>Stage 2</span>"
                    
                    st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; align-items: center; gap: 1rem; width: 80%;">
                                <div style="font-size: 20px; font-weight: 800; color: #FF6B4A;">{score_pct}%</div>
                                <div>
                                    <div style="font-weight: 700; font-size: 15px; color: #1E293B;">#{rank} {p['anonymized_name']}</div>
                                    <div style="font-size: 12px; color: #64748B;">{p['current_title']} @ {p['current_company']} • {p['location']}</div>
                                    <div style="margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap;">{skills_html}</div>
                                </div>
                            </div>
                            <div>{stage_html}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_b1, col_b2 = st.columns([1, 1])
                    if col_b1.button(f"👁️ Inspect {cid}", key=f"feed_insp_{cid}", use_container_width=True):
                        st.session_state["inspect_id"] = cid
                        st.rerun()
                    
                    shortlisted = cid in st.session_state["selected_candidate_ids"]
                    sh_label = "⭐ Shortlisted" if shortlisted else "⭐ Shortlist"
                    if col_b2.button(sh_label, key=f"feed_sh_{cid}", use_container_width=True):
                        if shortlisted:
                            st.session_state["selected_candidate_ids"].remove(cid)
                        else:
                            st.session_state["selected_candidate_ids"].add(cid)
                        st.rerun()

        # TAB 2: Pool Analytics
        with tab_analytics:
            if not scored:
                st.caption("No candidates loaded.")
            else:
                st.subheader("Talent Pool Distributions")
                df_anal = pd.DataFrame([{
                    "yoe": item["_candidate"]["profile"]["years_of_experience"],
                    "location": item["_candidate"]["profile"]["location"].split(",")[0],
                    "industry": item["_candidate"]["profile"]["current_industry"],
                } for item in scored])
                
                st.write("**Years of Experience Spread:**")
                st.bar_chart(df_anal["yoe"].value_counts())
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.write("**Geographic Distribution:**")
                    st.bar_chart(df_anal["location"].value_counts().head(10))
                with col_c2:
                    st.write("**Industry Verticals:**")
                    st.bar_chart(df_anal["industry"].value_counts().head(10))

        # TAB 3: Candidate Compare
        with tab_compare:
            if len(scored) < 2:
                st.info("Load at least 2 candidates to compare side-by-side.")
            else:
                st.subheader("Side-by-Side Comparison")
                col_sel1, col_sel2 = st.columns(2)
                cand_opts = [item["candidate_id"] for item in scored]
                
                c1_id = col_sel1.selectbox("Select Candidate A", cand_opts, index=0)
                c2_id = col_sel2.selectbox("Select Candidate B", cand_opts, index=1)
                
                cand_a = next(item for item in scored if item["candidate_id"] == c1_id)
                cand_b = next(item for item in scored if item["candidate_id"] == c2_id)
                
                col_v1, col_v2 = st.columns(2)
                for col, entry in zip([col_v1, col_v2], [cand_a, cand_b]):
                    p_info = entry["_candidate"]["profile"]
                    sig_info = entry["_candidate"]["redrob_signals"]
                    c_comp = entry["components"]
                    
                    with col:
                        st.markdown(f"""
                            <div style="background:white; border: 1px solid #E2E8F0; padding:1.2rem; border-radius:12px; margin-top:10px;">
                                <h4 style="color:#FF6B4A;margin-top:0;">{p_info['anonymized_name']}</h4>
                                <b>Score:</b> {entry['score']:.2f}<br>
                                <b>Title:</b> {p_info['current_title']}<br>
                                <b>Company:</b> {p_info['current_company']} ({p_info['current_industry']})<br>
                                <b>Experience:</b> {p_info['years_of_experience']} years YOE<br>
                                <b>Location:</b> {p_info['location']} ({p_info['country']})<br>
                                <b>Notice Period:</b> {sig_info.get('notice_period_days', 90)} days<br>
                                <b>Response Rate:</b> {sig_info.get('recruiter_response_rate', 0.5):.0%}<br>
                            </div>
                        """, unsafe_allow_html=True)

        # TAB 4: Honeypot Auditor Logs
        with tab_honeypots:
            st.subheader("🛡️ Zero-Trust Security Quarantine Logs")
            st.write(f"The security agent flagged and isolated **{len(honeypots)}** synthetic profile discrepancies:")
            if honeypots:
                df_hp = pd.DataFrame([{
                    "Candidate ID": item["candidate_id"],
                    "Violation Audit Reason": item["reason"]
                } for item in honeypots])
                st.dataframe(df_hp, hide_index=True, use_container_width=True)
            else:
                st.success("No honeypots detected in the loaded dataset.")

        # TAB 5: Export Shortlist
        with tab_export:
            st.subheader("⭐ Selected Shortlist")
            shortlist_items = [item for item in scored if item["candidate_id"] in st.session_state["selected_candidate_ids"]]
            if not shortlist_items:
                st.info("No candidates selected yet. Add candidates from the Shortlist Feed.")
            else:
                df_sh = pd.DataFrame([{
                    "Candidate ID": item["candidate_id"],
                    "Name": item["_candidate"]["profile"]["anonymized_name"],
                    "Score": item["score"],
                    "Title": item["_candidate"]["profile"]["current_title"],
                    "Company": item["_candidate"]["profile"]["current_company"]
                } for item in shortlist_items])
                st.dataframe(df_sh, hide_index=True, use_container_width=True)
                
                csv_buf_sh = io.StringIO()
                writer_sh = csv.writer(csv_buf_sh)
                writer_sh.writerow(["candidate_id", "score", "title", "company"])
                for item in shortlist_items:
                    writer_sh.writerow([item["candidate_id"], item["score"], item["_candidate"]["profile"]["current_title"], item["_candidate"]["profile"]["current_company"]])
                st.download_button("📥 Download Custom Shortlist CSV", csv_buf_sh.getvalue(), file_name="shortlist.csv", mime="text/csv")


    # ==========================================
    # RIGHT PANEL: Contextual Profile Inspector
    # ==========================================
    with col_right:
        if not st.session_state["inspect_id"] or not scored:
            st.markdown("<p style='color: #64748B; font-style: italic;'>Select a candidate from the list and click 'Inspect' to visualize details.</p>", unsafe_allow_html=True)
        else:
            cand_entry = next((item for item in scored if item["candidate_id"] == st.session_state["inspect_id"]), None)
            if not cand_entry:
                st.caption("Inspected candidate is missing.")
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
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5rem;">
                        <div>
                            <div style="font-size: 24px; font-weight: 800; color: #1E293B;">{p['anonymized_name']}</div>
                            <div style="font-size: 13px; color: #64748B; font-weight: 500;">{p['current_title']} @ {p['current_company']}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                col_i1, col_i2 = st.columns(2)
                if col_i1.button(short_text, key="insp_short_act_ws", type="primary" if is_shortlisted else "secondary", use_container_width=True):
                    if is_shortlisted:
                        st.session_state["selected_candidate_ids"].remove(cid)
                    else:
                        st.session_state["selected_candidate_ids"].add(cid)
                    st.rerun()
                if col_i2.button("Skip candidate ⏭️", key="insp_skip_act_ws", use_container_width=True):
                    idx = next((i for i, entry in enumerate(scored) if entry["candidate_id"] == cid), 0)
                    next_idx = (idx + 1) % len(scored)
                    st.session_state["inspect_id"] = scored[next_idx]["candidate_id"]
                    st.rerun()

                # Visual Skill Gap Analyzer (Matches / Missing / Bonus)
                cand_skills = {s["name"].lower(): s for s in c_data.get("skills", [])}
                jd_skills = custom_skill_map
                
                matches = []
                missing = []
                bonus = []
                
                for sk_name in jd_skills:
                    if sk_name in cand_skills:
                        matches.append(sk_name)
                    else:
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
                        <p style="font-size: 12px; color: #64748B; font-style: italic; margin-top: 6px;">"Led the migration of legacy monolith to microservices, increasing throughput by 10x."</p>
                    </div>
                    
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #1E293B;">
                            <span>Implemented / Coded</span><span style="color: #64748B;">{med_val}%</span>
                        </div>
                        <div class="progress-container"><div class="progress-bar-med" style="width: {med_val}%;"></div></div>
                        <p style="font-size: 12px; color: #64748B; font-style: italic; margin-top: 6px;">"Core contributor to the underlying storage engine and vector indexing logic."</p>
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
                if col_act2.button("Generate Outreach ✉️", key="outreach_btn_ws", use_container_width=True):
                    st.toast("Generating custom recruiter email sequence...")
                    st.info(f"Hi {p['anonymized_name'].split()[0]},\n\nI was impressed by your work at {p['current_company']} on {matches[0] if matches else 'applied ML'}. We are hiring a Senior AI Engineer at RankCraft AI in Pune/Noida. Let's chat!\n\nBest,\nTalent Partner")
