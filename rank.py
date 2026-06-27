#!/usr/bin/env python3
"""
Redrob Hackathon — Intelligent Candidate Ranker (Final Corrected)
================================================================
Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints satisfied:
  - CPU only, no GPU
  - No network calls during ranking
  - < 5 min for 100K candidates
  - < 16 GB RAM
"""

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TODAY = datetime(2026, 6, 10)

CORE_ML_TITLES = {
    'ml engineer', 'machine learning engineer', 'ai engineer', 'applied ml engineer',
    'applied scientist', 'research engineer', 'nlp engineer', 'search engineer',
    'ranking engineer', 'recommendation systems engineer', 'retrieval engineer',
    'deep learning engineer', 'senior ml', 'staff ml', 'principal ml',
    'data scientist', 'senior data scientist', 'lead ml', 'lead ai',
    'senior ai', 'staff ai', 'principal ai', 'ai/ml engineer',
}

PARTIAL_ML_TITLES = {
    'software engineer', 'senior software engineer', 'staff software engineer',
    'principal software engineer', 'backend engineer', 'senior backend engineer',
    'data engineer', 'senior data engineer', 'platform engineer',
    'senior engineer', 'lead engineer', 'full stack engineer',
    'founding engineer', 'infrastructure engineer',
}

DISQUALIFIED_TITLES = {
    'marketing manager', 'operations manager', 'hr manager', 'civil engineer',
    'mechanical engineer', 'accountant', 'sales manager', 'customer support',
    'business analyst', 'project manager', 'frontend engineer', 'mobile developer',
    '.net developer', 'java developer', 'graphic designer', 'qa engineer',
    'product manager', 'ui developer', 'ux designer', 'scrum master',
    'finance manager', 'content writer', 'seo specialist', 'recruiter',
    'junior ml engineer', 'junior ai engineer', 'junior data scientist',
    'intern', 'trainee',
}

CONSULTING_COS = {
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture', 'cognizant',
    'capgemini', 'hcl', 'hcltech', 'tech mahindra', 'mphasis', 'mindtree',
    'hexaware', 'niit technologies', 'mastech', 'kpit', 'cyient', 'l&t infotech',
    'ltimindtree', 'coforge',
}

PRODUCT_INDUSTRIES = {
    'Software', 'AI/ML', 'Fintech', 'E-commerce', 'Food Delivery', 'Transportation',
    'SaaS', 'HealthTech', 'EdTech', 'Proptech', 'Gaming', 'Media', 'Retail Tech',
    'Logistics Tech', 'InsurTech', 'Marketplace',
}

CORE_SKILL_MAP = {
    'information retrieval': 10, 'sentence transformers': 10, 'sentence-transformers': 10,
    'embeddings': 10, 'learning to rank': 10, 'faiss': 10,
    'pinecone': 9, 'weaviate': 9, 'qdrant': 9, 'vector search': 9, 'vector database': 9,
    'milvus': 9, 'opensearch': 9,
    'elasticsearch': 8, 'ranking': 8, 'retrieval': 8, 'nlp': 8,
    'hugging face transformers': 8, 'recommendation': 8,
    'pytorch': 7, 'transformers': 7, 'bert': 7, 'fine-tuning llms': 7,
    'neural information retrieval': 7,
    'xgboost': 6, 'lightgbm': 6, 'lora': 6, 'qlora': 6, 'mlops': 6,
    'fine-tuning': 6, 'peft': 6,
    'python': 5, 'mlflow': 5, 'machine learning': 5, 'deep learning': 5,
    'a/b testing': 5, 'feature engineering': 5,
    'tensorflow': 4, 'scikit-learn': 4, 'numpy': 3, 'pandas': 3,
    'ndcg': 9, 'mrr': 9, 'map@k': 9, 'evaluation framework': 8,
    'offline evaluation': 8,
}

PROD_ML_DESC_KWS = [
    'embedding', 'vector', 'retrieval', 'ranking model', 'recommendation system',
    'search engine', 'learning to rank', 'fine-tun', 'rag', 'sentence transformer',
    'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'information retrieval',
    'ndcg', 'mrr', 'map@', 'a/b test', 'offline evaluation', 're-ranking',
    'hybrid search', 'dense retrieval', 'sparse retrieval', 'bm25',
]

PROD_ML_JOB_TITLES = [
    'ml engineer', 'machine learning', 'ai engineer', 'nlp engineer',
    'search engineer', 'ranking engineer', 'recommendation', 'applied scientist',
    'data scientist', 'research engineer', 'retrieval',
]

PREF_LOCATIONS = {
    'pune', 'noida', 'hyderabad', 'delhi', 'gurugram', 'gurgaon',
    'mumbai', 'bangalore', 'bengaluru', 'ncr',
}


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(d_str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except ValueError:
        return None


def is_honeypot(c: dict) -> tuple[bool, str]:
    p = c['profile']
    sig = c['redrob_signals']
    career = c['career_history']
    skills = c.get('skills', [])

    total_career_months = sum(j['duration_months'] for j in career)
    stated_yoe = p['years_of_experience']

    # 1. Career months << stated years of experience
    if stated_yoe > 2 and total_career_months < (stated_yoe * 12 * 0.1):
        return True, f"career_months({total_career_months}) << stated_YoE({stated_yoe}yr)"

    # 2. Expert skills with 0 duration months
    for s in skills:
        if s['proficiency'] == 'expert' and s.get('duration_months', 0) == 0:
            return True, f"expert '{s['name']}' with 0 months used"

    # 3. Negative YoE
    if stated_yoe < 0:
        return True, f"negative YoE: {stated_yoe}"

    # 4. Simultaneous current jobs
    current_jobs = [j for j in career if j.get('is_current')]
    if len(current_jobs) > 1:
        return True, f"{len(current_jobs)} simultaneous current jobs"

    # 5. Impossible job durations (stated duration exceeds physical date range)
    for j in career:
        start = parse_date(j.get('start_date'))
        if j.get('is_current') or not j.get('end_date'):
            end = TODAY
        else:
            end = parse_date(j.get('end_date'))
        if start and end:
            diff_months = (end.year - start.year) * 12 + (end.month - start.month)
            if j.get('duration_months', 0) > diff_months + 2:
                return True, f"impossible job duration at {j.get('company')}: stated={j.get('duration_months')}, max={diff_months}"

    # 6. Too many expert skills (>= 8)
    expert_skills = [s for s in skills if s.get('proficiency') == 'expert']
    if len(expert_skills) >= 8:
        return True, f"too many expert skills: {len(expert_skills)} expert skills"

    return False, ''


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_title(current_title: str, config: dict = None) -> float:
    t = current_title.lower().strip()
    for disq in DISQUALIFIED_TITLES:
        if disq in t:
            return -20.0
    for core in CORE_ML_TITLES:
        if core in t:
            return 30.0
    for partial in PARTIAL_ML_TITLES:
        if partial in t:
            return 12.0
    return 5.0


def score_career(career_history: list, config: dict = None, current_company: str = None) -> float:
    if config is None:
        config = {}
    score = 0.0
    prod_ml_months = 0
    all_companies = []
    has_product_co = False

    for job in career_history:
        jt = job['title'].lower()
        jd = job['description'].lower()
        jind = job.get('industry', '')
        jcomp = job['company'].lower()
        dur = job['duration_months']
        all_companies.append(jcomp)

        if jind in PRODUCT_INDUSTRIES:
            has_product_co = True

        desc_has_ml = any(kw in jd for kw in PROD_ML_DESC_KWS)
        title_has_ml = any(kw in jt for kw in PROD_ML_JOB_TITLES)

        if desc_has_ml and title_has_ml:
            prod_ml_months += dur
        elif desc_has_ml:
            prod_ml_months += dur * 0.5
        elif title_has_ml:
            prod_ml_months += dur * 0.3

    score += min(prod_ml_months / 6.0, 20.0)
    if has_product_co:
        score += 8.0

    # Strict penalty for pure consulting background (JD disqualifier)
    # Now applies even if only one job (len >= 1)
    is_consulting_only = all(
        any(co in comp for co in CONSULTING_COS) for comp in all_companies
    )
    if is_consulting_only and len(all_companies) >= 1:
        if config.get('consulting_penalty_enabled', True):
            return 0.0   # zero out career score

    # Fallback to identify current company from career history if not passed directly
    if not current_company:
        current_jobs = [j for j in career_history if j.get('is_current')]
        if current_jobs:
            current_company = current_jobs[0].get('company')

    # Apply penalty for current employer being in consulting
    if current_company:
        current_co = current_company.lower()
        if any(co in current_co for co in CONSULTING_COS):
            score -= 20.0

    return min(max(0.0, score), 30.0)   # cap at 30


def score_skills(skills: list, assessment_scores: dict, config: dict = None) -> float:
    if config is None:
        config = {}
    score = 0.0
    skill_map = config.get('custom_skill_map', CORE_SKILL_MAP)
    for sk in skills:
        sname = sk['name'].lower()
        base_val = 0
        for kw, val in skill_map.items():
            if kw in sname:
                base_val = max(base_val, val)
        if base_val == 0:
            continue
        prof_mult = {
            'beginner': 0.3,
            'intermediate': 0.6,
            'advanced': 0.85,
            'expert': 1.0,
        }.get(sk['proficiency'], 0.5)
        endorsements = sk.get('endorsements', 0)
        duration = sk.get('duration_months', 0)
        trust = min((endorsements / 20.0) * 0.5 + (duration / 24.0) * 0.5, 1.0)
        trust = max(trust, 0.3)
        score += base_val * prof_mult * trust

    for sk_name, sk_score in assessment_scores.items():
        if sk_score >= 75:
            for kw in skill_map:
                if kw in sk_name.lower():
                    score = min(score + 2.0, 20.0)
                    break
    skills_cap = config.get('skills_score_cap', 25.0)
    return min(score, skills_cap)


def score_experience(yoe: float, config: dict = None) -> float:
    if 6 <= yoe <= 8:
        return 10.0
    elif 5 <= yoe < 6 or 8 < yoe <= 10:
        return 8.0
    elif 4 <= yoe < 5 or 10 < yoe <= 12:
        return 6.0
    elif 3 <= yoe < 4:
        return 4.0
    elif 12 < yoe <= 15:
        return 2.0
    elif yoe > 15:
        return 1.0
    elif yoe < 2:
        return 0.0
    else:
        return 2.0


def score_location(country: str, location: str, willing_to_relocate: bool, config: dict = None) -> float:
    if config is None:
        config = {}
    if country == 'India':
        loc_lower = location.lower()
        if any(pref in loc_lower for pref in PREF_LOCATIONS):
            return 5.0
        return 3.0
    elif willing_to_relocate:
        return 2.0
    else:
        if not config.get('location_penalty_enabled', True):
            return 0.0
        return -5.0   # active penalty for non-India unwilling


def compute_behavioral_multiplier(sig: dict, config: dict = None) -> float:
    if config is None:
        config = {}
    mult = 1.0
    
    if config.get('enable_activity_decay', True):
        try:
            last_active = datetime.strptime(sig['last_active_date'], '%Y-%m-%d')
            days_since = (TODAY - last_active).days
            if days_since > 180:
                mult *= 0.3
            elif days_since > 90:
                mult *= 0.6
            elif days_since > 30:
                mult *= 0.85
        except (ValueError, KeyError):
            mult *= 0.7

    if config.get('enable_notice_penalty', True):
        notice = sig.get('notice_period_days', 90)
        if notice <= 15:
            mult *= 1.15
        elif notice <= 30:
            mult *= 1.00
        elif notice <= 60:
            mult *= 0.90
        elif notice <= 90:
            mult *= 0.75
        else:
            mult *= 0.40

    if config.get('enable_response_rate_penalty', True):
        rr = sig.get('recruiter_response_rate', 0.5)
        if rr >= 0.7:
            mult *= 1.05
        elif rr >= 0.4:
            mult *= 1.00
        elif rr >= 0.2:
            mult *= 0.90
        else:
            mult *= 0.70

    if config.get('enable_open_to_work_bonus', True):
        if sig.get('open_to_work_flag', False):
            mult *= 1.05

    if config.get('enable_interview_completion_penalty', True):
        icr = sig.get('interview_completion_rate', 0.5)
        if icr < 0.3:
            mult *= 0.85

    return max(0.3, min(mult, 1.2))


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATOR (with variation)
# ─────────────────────────────────────────────────────────────────────────────

def build_reasoning(c: dict, scores: dict, rank: int) -> str:
    p = c['profile']
    sig = c['redrob_signals']
    positives = []
    concerns = []

    title = p['current_title']
    yoe = p['years_of_experience']
    company = p['current_company']
    industry = p['current_industry']

    # Use raw scores for generating reasoning to ensure correct logic
    r_title = scores.get('raw_title', scores.get('title', 0.0))
    r_career = scores.get('raw_career', scores.get('career', 0.0))
    r_skills = scores.get('raw_skills', scores.get('skills', 0.0))

    if r_title >= 30:
        positives.append(f"{title} at {company}")
    elif r_title >= 12:
        positives.append(f"{title} ({yoe:.0f}yr exp)")
    else:
        concerns.append(f"non-ML title ({title})")

    if r_career >= 20:
        positives.append(f"strong production ML history in {industry}")
    elif r_career >= 10:
        positives.append(f"some production ML experience")
    elif r_career > 0:
        concerns.append("limited production ML depth in career history")

    top_skills = []
    for sk in c.get('skills', []):
        sname = sk['name'].lower()
        for kw in ['embeddings', 'faiss', 'pinecone', 'qdrant', 'weaviate', 'milvus',
                   'sentence transformers', 'information retrieval', 'learning to rank',
                   'nlp', 'elasticsearch', 'opensearch', 'pytorch', 'fine-tuning', 'ndcg']:
            if kw in sname and sk['proficiency'] in ('advanced', 'expert'):
                top_skills.append(sk['name'])
                break
    if top_skills:
        positives.append(f"core skills: {', '.join(top_skills[:3])}")

    github = sig.get('github_activity_score', -1)
    if github >= 50:
        positives.append(f"active on GitHub (score {github})")

    notice = sig.get('notice_period_days', 90)
    if notice > 90:
        concerns.append(f"{notice}d notice period")
    elif notice <= 30:
        positives.append(f"sub-30d notice")

    try:
        last_active = datetime.strptime(sig['last_active_date'], '%Y-%m-%d')
        days_since = (TODAY - last_active).days
        if days_since > 90:
            concerns.append(f"inactive {days_since}d")
        elif days_since <= 7:
            positives.append("active this week")
    except (ValueError, KeyError):
        pass

    if p['country'] == 'India':
        loc = p['location']
        if any(pref in loc.lower() for pref in PREF_LOCATIONS):
            positives.append(f"based in {loc}")
        else:
            positives.append(f"India-based ({loc})")
    elif not sig.get('willing_to_relocate', False):
        concerns.append(f"outside India ({p['country']}), not willing to relocate")

    sentence1 = "; ".join(positives) + "." if positives else "Limited fit on core criteria."

    # Build varied sentence2 based on actual signals
    open_flag = sig.get('open_to_work_flag', False)
    rr = sig.get('recruiter_response_rate', 0)
    views = sig.get('profile_views_received_30d', 0)
    saved = sig.get('saved_by_recruiters_30d', 0)

    if concerns:
        sentence2 = f"Concerns: {'; '.join(concerns)}."
    else:
        if open_flag and rr >= 0.7:
            sentence2 = f"Open to work; {rr:.0%} response rate; {views} profile views/30d."
        elif github >= 50:
            sentence2 = f"Active on GitHub (score {github:.0f}); open to work."
        elif saved > 10:
            sentence2 = f"Saved by {saved} recruiters in last 30 days; strong interest."
        else:
            sentence2 = f"Notice: {notice}d; response rate {rr:.0%}; willing to relocate."
    return f"{sentence1} {sentence2}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(c: dict, config: dict = None) -> tuple[float, dict]:
    hp, reason = is_honeypot(c)
    if hp:
        return -9999.0, {'honeypot': reason}

    p = c['profile']
    sig = c['redrob_signals']

    if config is None:
        config = {}

    w_title = config.get('weight_title', 1.0)
    w_career = config.get('weight_career', 1.0)
    w_skills = config.get('weight_skills', 1.0)
    w_experience = config.get('weight_experience', 1.0)
    w_location = config.get('weight_location', 1.0)

    t_score = score_title(p['current_title'], config)
    if t_score < 0:
        t_score = -50.0

    c_score = score_career(c['career_history'], config, current_company=p.get('current_company'))
    sk_score = score_skills(c.get('skills', []), sig.get('skill_assessment_scores', {}), config)
    e_score = score_experience(p['years_of_experience'], config)
    l_score = score_location(p['country'], p['location'], sig.get('willing_to_relocate', False), config)
    b_mult = compute_behavioral_multiplier(sig, config)

    base = (t_score * w_title) + (c_score * w_career) + (sk_score * w_skills) + (e_score * w_experience) + (l_score * w_location)
    total = max(0.0, base) * b_mult
    total = min(total, 100.0)

    components = {
        'title': t_score * w_title,
        'career': c_score * w_career,
        'skills': sk_score * w_skills,
        'experience': e_score * w_experience,
        'location': l_score * w_location,
        'behavioral_mult': b_mult,
        'base': base,
        'total': total,
        'raw_title': t_score,
        'raw_career': c_score,
        'raw_skills': sk_score,
        'raw_experience': e_score,
        'raw_location': l_score,
    }
    return total, components


# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    opener = gzip.open if p.suffix == '.gz' else open
    mode = 'rt'
    print(f"Loading candidates from {path}...", file=sys.stderr)
    with opener(p, mode, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {i+1}: {e}", file=sys.stderr)


def write_submission(ranked: list, output_path: str) -> None:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for row in ranked:
            writer.writerow([
                row['candidate_id'],
                row['rank'],
                f"{row['score']:.6f}",
                row['reasoning'],
            ])
    print(f"Submission written to {output_path} ({len(ranked)} rows)", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Redrob Hackathon — Candidate Ranker')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl or .jsonl.gz')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--top', type=int, default=100, help='Number of candidates to output (default: 100)')
    parser.add_argument('--debug', action='store_true', help='Print score breakdown for top candidates')
    args = parser.parse_args()

    start = datetime.now()
    scored = []
    honeypot_count = 0
    total_count = 0

    for c in load_candidates(args.candidates):
        total_count += 1
        total_score, components = score_candidate(c)

        if components.get('honeypot'):
            honeypot_count += 1
            continue

        scored.append({
            'candidate_id': c['candidate_id'],
            'score': total_score,
            'components': components,
            '_candidate': c,
        })

        if total_count % 10000 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  Processed {total_count:,} candidates in {elapsed:.1f}s...", file=sys.stderr)

    print(f"\nTotal candidates: {total_count:,}", file=sys.stderr)
    print(f"Honeypots detected: {honeypot_count}", file=sys.stderr)
    print(f"Eligible candidates: {len(scored):,}", file=sys.stderr)

    # Sort by score descending (rounded to 6 decimals), then by candidate_id ascending
    scored.sort(key=lambda x: (-round(x['score'], 6), x['candidate_id']))

    top_n = scored[:args.top]

    output_rows = []
    for rank, entry in enumerate(top_n, start=1):
        reasoning = build_reasoning(entry['_candidate'], entry['components'], rank)
        output_rows.append({
            'candidate_id': entry['candidate_id'],
            'rank': rank,
            'score': entry['score'],
            'reasoning': reasoning,
        })

    if args.debug:
        print("\n=== TOP 20 SCORE BREAKDOWN ===", file=sys.stderr)
        for row in output_rows[:20]:
            comp = next(e['components'] for e in top_n if e['candidate_id'] == row['candidate_id'])
            print(
                f"  #{row['rank']:3d} {row['candidate_id']} score={row['score']:7.2f} | "
                f"T:{comp['title']:+5.0f} C:{comp['career']:5.1f} Sk:{comp['skills']:5.1f} "
                f"E:{comp['experience']:3.0f} L:{comp['location']:2.0f} Bx:{comp['behavioral_mult']:.3f}",
                file=sys.stderr
            )
            print(f"       {row['reasoning'][:120]}", file=sys.stderr)

    write_submission(output_rows, args.out)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    if elapsed > 240:
        print("WARNING: approaching 5-minute time limit!", file=sys.stderr)


if __name__ == '__main__':
    main()