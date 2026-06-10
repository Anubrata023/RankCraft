# Redrob Hackathon — Intelligent Candidate Ranker

> Submission for the **Intelligent Candidate Discovery & Ranking Challenge**

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Anubrata023/redrob-hackathon-final.git
cd redrob-ranker
pip install -r requirements.txt

# 2. Rank the full 100K pool
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. Validate before uploading
python validate_submission.py submission.csv
```

The reproduce command runs in **< 5 minutes on CPU, no GPU, no network**.

---

## Architecture

The ranker is a **five-component weighted scorer** with a behavioral availability multiplier.

### Why not embeddings or an LLM?

The compute constraint (5 min, CPU-only, no network, 100K candidates) rules out:
- Per-candidate LLM calls (too slow)
- Dense embedding similarity (sentence-transformers inference on 100K × 1 = ~20 min on CPU)

The JD itself tells you the scoring signal that matters: **title and career history beat skills keywords**. A keyword-stuffer with perfect skills and a "Marketing Manager" title is a trap. Our title/career components catch this automatically.

### Scoring components

| Component | Max pts | Key logic |
|---|---|---|
| **Title fit** | 30 | Core ML titles → 30; partial engineering → 12; disqualified titles → -20 |
| **Career history** | 30 | Production ML months at product companies; consulting-only → -15 |
| **Skills (trusted)** | 25 | JD-relevant skills × proficiency × (endorsements + duration) trust weight |
| **Experience years** | 10 | Peak at 6–8 yrs; hard penalty < 2 yrs or > 15 yrs |
| **Location** | 5 | India: +3; preferred city (Pune/Noida/Hyd/Blr): +5; willing to relocate: +2 |
| **Behavioral ×** | ×0.3–1.2 | last_active_date, notice_period, recruiter_response_rate, open_to_work |

**Final score = (T + C + Sk + E + L) × Behavioral_multiplier**

### Honeypot detection

Candidates are zeroed out if they exhibit impossible profiles:
- Expert skill with 0 months usage
- Career history months < 20% of stated years of experience
- 8+ simultaneous expert skills
- All behavioral signals simultaneously maxed

### Reasoning generation

Each candidate gets a 1–2 sentence reasoning referencing specific facts: title, company, top matched skills, notice period, location, activity recency. Concerns are explicitly stated.

---

## Files

```
rank.py                    — Main ranker (the reproduce command runs this)
app.py                     — Streamlit sandbox (HuggingFace Spaces)
requirements.txt           — Python dependencies
submission_metadata.yaml   — Hackathon metadata
validate_submission.py     — Format validator (from hackathon bundle)
README.md                  — This file
```

---

## Compute environment

- Python 3.11+
- CPU only (no GPU used at any stage)
- No network calls during ranking
- Peak memory: ~4–6 GB for 100K candidates
- Runtime: ~60–120 seconds for full 100K pool

---

## Sandbox

Live demo: [HuggingFace Spaces link here]

Upload a candidate JSON/JSONL sample (up to 500 candidates) and download the ranked CSV.

---

## Development log

See git history for incremental development:
1. Initial scoring skeleton with title/career components
2. Skills trust multiplier (endorsements × duration weighting)
3. Behavioral availability multiplier
4. Honeypot detection rules
5. Reasoning generator
6. Performance optimisation for 100K scale
7. Streamlit sandbox

---

## AI tools declaration

As declared in `submission_metadata.yaml`: Claude was used for architecture discussion and code review. All scoring logic, weight calibration, and engineering decisions were made by the team after analysing the JD, sample candidates, and signals documentation.
