"""
AI Resume Analyzer and Job Recommendation System
================================================
A production-grade Streamlit application for analyzing resume content,
extracting technical skills, evaluating match scores via TF-IDF & Cosine Similarity,
and generating tailored skill-gap learning roadmaps.

Author: Senior Software Engineer & UI/UX Specialist
"""

import io
import re
from typing import Dict, List, Tuple

import docx
import pandas as pd
import plotly.express as px
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Compass — AI Resume Analyzer & Career Guidance",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Container */
.main-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}

.header-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #38BDF8;
    background-color: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.header-sub {
    color: #94A3B8;
    font-size: 1.0rem;
    line-height: 1.5;
    margin: 0;
    max-width: 750px;
}

/* Metric Cards */
.metric-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    border-color: #38BDF8;
    transform: translateY(-2px);
}

.metric-label {
    font-size: 0.825rem;
    font-weight: 500;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #38BDF8;
}

/* Recommendation Cards */
.top-role-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.25rem;
    height: 100%;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.role-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #F59E0B;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.role-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 0.75rem;
}

.score-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

/* Skill Chips */
.skill-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    margin: 0.25rem 0.4rem 0.25rem 0;
}

.chip-found {
    background-color: rgba(16, 185, 129, 0.12);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.25);
}

.chip-missing {
    background-color: rgba(239, 68, 68, 0.12);
    color: #F87171;
    border: 1px solid rgba(248, 113, 113, 0.25);
}

.category-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #CBD5E1;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Section Containers */
.section-box {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Sidebar Modifications */
section[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] h2 {
    font-size: 1.0rem;
    font-weight: 600;
    color: #F8FAFC;
    letter-spacing: 0.02em;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------
# Data Loading & Initialization
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and validates the job roles and skill dictionary CSV files.
    """
    try:
        job_roles_df = pd.read_csv("job_roles.csv")
        skill_dict_df = pd.read_csv("skill_dictionary.csv")

        # Validate required columns
        if "job_role" not in job_roles_df.columns or "required_skills" not in job_roles_df.columns:
            st.error("Error: 'job_roles.csv' must contain 'job_role' and 'required_skills' columns.")
            st.stop()

        if "skill" not in skill_dict_df.columns or "category" not in skill_dict_df.columns:
            st.error("Error: 'skill_dictionary.csv' must contain 'skill' and 'category' columns.")
            st.stop()

        # Clean and parse required skills list
        job_roles_df["required_skills"] = job_roles_df["required_skills"].fillna("").apply(
            lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
        )

        skill_dict_df["skill"] = skill_dict_df["skill"].astype(str).str.strip()
        skill_dict_df["category"] = skill_dict_df["category"].fillna("General").astype(str).str.strip()

        return job_roles_df, skill_dict_df

    except FileNotFoundError as e:
        st.error(f"Missing essential dataset file: {e.filename}. Please check your deployment directory.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset files: {str(e)}")
        st.stop()


# ---------------------------------------------------------------
# Parsing & NLP Processing Utilities
# ---------------------------------------------------------------
def extract_text(uploaded_file) -> str:
    """
    Extracts raw text content from uploaded PDF or DOCX file streams with error checking.
    """
    filename = uploaded_file.name.lower()
    extracted_text = ""

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    pages_text.append(txt)
            extracted_text = "\n".join(pages_text)

        elif filename.endswith(".docx"):
            document = docx.Document(uploaded_file)
            extracted_text = "\n".join([p.text for p in document.paragraphs if p.text])

        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

    except Exception as e:
        raise RuntimeError(f"Could not read uploaded file: {str(e)}")

    if not extracted_text.strip():
        raise ValueError("No readable text found in file. If uploading a PDF, ensure it is not a scanned image.")

    return extracted_text


def clean_text(text: str) -> str:
    """
    Normalizes text for NLP parsing while preserving specific technical language markers.
    """
    text = text.lower()

    # Retain standard technical symbols prior to regex stripping
    replacements = {
        "c++": " cplusplus ",
        "c#": " csharp ",
        ".net": " dotnet ",
        "node.js": " nodejs ",
        "react.js": " reactjs ",
        "vue.js": " vuejs ",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # Sanitize non-alphanumeric characters except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Restore converted tech markers
    restorations = {
        "cplusplus": "c++",
        "csharp": "c#",
        "dotnet": ".net",
        "nodejs": "node.js",
        "reactjs": "react.js",
        "vuejs": "vue.js",
    }

    for key, val in restorations.items():
        text = text.replace(key, val)

    return text


def extract_skills(cleaned_resume_text: str, known_skills: List[str]) -> List[str]:
    """
    Matches known skills against normalized resume text using word-boundary regular expressions.
    """
    found_skills = []
    for skill in known_skills:
        # Regex boundary check to avoid partial word collisions (e.g., 'Java' matching 'JavaScript')
        escaped_skill = re.escape(skill)
        pattern = r"(?<!\w)" + escaped_skill + r"(?!\w)"
        if re.search(pattern, cleaned_resume_text, flags=re.IGNORECASE):
            found_skills.append(skill)
    return found_skills


def match_roles(found_skills: List[str], job_roles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cosine similarity between extracted skills and required skills per job role using TF-IDF.
    """
    results_df = job_roles_df.copy()

    # If no skills are detected, return zero scores to avoid vectorizer errors
    if not found_skills:
        results_df["match_score"] = 0.0
        return results_df.sort_values("match_score", ascending=False).reset_index(drop=True)

    resume_skill_text = " ".join(found_skills)
    role_texts = [" ".join(skills) for skills in job_roles_df["required_skills"]]

    corpus = [resume_skill_text] + role_texts

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b|\+\+|#|\.net")
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        results_df["match_score"] = (similarities * 100).round(1)
    except Exception:
        results_df["match_score"] = 0.0

    return results_df.sort_values("match_score", ascending=False).reset_index(drop=True)


def generate_roadmap(missing_skills: List[str]) -> List[str]:
    """
    Generates a structured weekly study plan for targeted skill acquisition.
    """
    if not missing_skills:
        return ["Your profile meets or exceeds all explicit skill requirements for this target role."]

    roadmap = []
    for idx, skill in enumerate(missing_skills, start=1):
        roadmap.append(
            f"**Week {idx}:** Master core concepts of **{skill}**, review documentation, and build a hands-on project module."
        )
    return roadmap


# ---------------------------------------------------------------
# Main Application Header & State Initialization
# ---------------------------------------------------------------
job_roles_df, skill_dict_df = load_datasets()
all_known_skills = sorted(list(set(skill_dict_df["skill"].str.lower().tolist())))

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "found_skills" not in st.session_state:
    st.session_state.found_skills = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# Header Banner
st.markdown(
    """
<div class="main-header">
    <div class="header-badge">NLP Powered · Skill Matching Engine</div>
    <div class="header-title">🧭 Compass AI Resume Analyzer</div>
    <div class="header-sub">
        Upload your resume to receive instantaneous matching against industry job roles, pinpoint missing skill gaps, and access customized learning roadmaps.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Resume Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=["pdf", "docx"],
        help="Upload a text-based resume file for automated evaluation.",
    )

    analyze_clicked = st.button("🚀 Analyze Resume", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("## ℹ️ Methodical Overview")
    st.caption(
        """
        **System Architecture:**
        - **Extraction:** Standard PDF/DOCX text parsing.
        - **Skill Normalization:** Match via boundary-checked skill dictionary.
        - **Matching:** TF-IDF Vector Space Model & Cosine Similarity distance.
        - **Privacy:** Operations run locally in-memory; no resume data stored.
        """
    )


# ---------------------------------------------------------------
# Core Analysis Workflow
# ---------------------------------------------------------------
if analyze_clicked:
    if not uploaded_file:
        st.sidebar.error("⚠️ Please select a valid PDF or DOCX file before proceeding.")
    else:
        with st.spinner("Processing document, extracting skills, and calculating match metrics..."):
            try:
                raw_text = extract_text(uploaded_file)
                cleaned = clean_text(raw_text)
                detected = extract_skills(cleaned, all_known_skills)
                scored_df = match_roles(detected, job_roles_df)

                st.session_state.raw_text = raw_text
                st.session_state.found_skills = detected
                st.session_state.results_df = scored_df

                st.sidebar.success("Analysis Completed Successfully!")

            except Exception as err:
                st.error(f"An error occurred during analysis: {str(err)}")


# ---------------------------------------------------------------
# Dashboard Results Renderer
# ---------------------------------------------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    found_skills = st.session_state.found_skills

    # High-level Metrics Row
    top_role_name = results_df.iloc[0]["job_role"]
    top_role_score = results_df.iloc[0]["match_score"]
    total_skills_count = len(found_skills)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Extracted Skills</div>
            <div class="metric-value">{total_skills_count}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Top Target Role</div>
            <div class="metric-value" style="font-size: 1.2rem; line-height: 2.2rem;">{top_role_name}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Peak Match Score</div>
            <div class="metric-value">{top_role_score}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 3 Recommended Roles
    st.markdown("### 🏆 Top 3 Recommended Roles")
    c1, c2, c3 = st.columns(3)
    top_3 = results_df.head(3)

    cols = [c1, c2, c3]
    ranks = ["#1 Top Match", "#2 Strong Match", "#3 Good Match"]

    for idx, (_, row) in enumerate(top_3.iterrows()):
        with cols[idx]:
            st.markdown(
                f"""
            <div class="top-role-card">
                <div>
                    <div class="role-rank">{ranks[idx]}</div>
                    <div class="role-title">{row['job_role']}</div>
                </div>
                <div>
                    <span class="score-badge">{row['match_score']}% Overall Match</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Skill Visualizer Section
    st.markdown("### 🔍 Identified Skills by Category")
    if not found_skills:
        st.warning("No skills matching the system dictionary were detected in the resume text.")
    else:
        # Match identified skills against categories
        matched_dict = skill_dict_df[skill_dict_df["skill"].str.lower().isin([s.lower() for s in found_skills])]

        if matched_dict.empty:
            chips = "".join([f'<span class="skill-chip chip-found">{s}</span>' for s in found_skills])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            grouped = matched_dict.groupby("category")
            for category_name, group in grouped:
                formatted_cat = category_name.replace("_", " ").title()
                cat_skills = group["skill"].tolist()
                chips_html = "".join([f'<span class="skill-chip chip-found">{s}</span>' for s in cat_skills])
                st.markdown(f'<div class="category-header">{formatted_cat}</div>', unsafe_allow_html=True)
                st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overall Match Scores Chart
    st.markdown("### 📊 Role Compatibility Breakdown")
    fig = px.bar(
        results_df,
        x="match_score",
        y="job_role",
        orientation="h",
        text="match_score",
        labels={"match_score": "Match Score (%)", "job_role": "Job Role"},
        color="match_score",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_range=[0, 100],
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"),
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Skill Gap & Roadmap Section
    st.markdown("### 🎯 Skill Gap Analysis & Roadmap")
    selected_role = st.selectbox("Select a target role for detailed gap evaluation:", results_df["job_role"])

    selected_row = results_df[results_df["job_role"] == selected_role].iloc[0]
    required_skills = selected_row["required_skills"]

    # Match normalized skills
    found_set = set([s.lower() for s in found_skills])
    skills_possessed = [s for s in required_skills if s.lower() in found_set]
    skills_missing = [s for s in required_skills if s.lower() not in found_set]

    col_pos, col_missing = st.columns(2)

    with col_pos:
        st.markdown("**Skills You Possess**")
        if skills_possessed:
            chips = "".join([f'<span class="skill-chip chip-found">✓ {s}</span>' for s in skills_possessed])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("No matching required skills currently found in resume.")

    with col_missing:
        st.markdown("**Skills Recommended to Acquire**")
        if skills_missing:
            chips = "".join([f'<span class="skill-chip chip-missing">+ {s}</span>' for s in skills_missing])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.success("You possess 100% of the specified required skills for this role!")

    st.markdown("<br>", unsafe_allow_html=True)

    # Learning Roadmap Display
    st.markdown("#### 🗺️ Suggested Learning Plan")
    roadmap_steps = generate_roadmap(skills_missing)
    for step in roadmap_steps:
        st.markdown(f"- {step}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Downloadable Text Report Generation
    st.markdown("### 📄 Export Analysis")
    report_content = [
        "==================================================",
        "          RESUME ANALYSIS REPORT                  ",
        "==================================================",
        f"Target Role Evaluated: {selected_role}",
        f"Role Match Compatibility: {selected_row['match_score']}%",
        "",
        "--------------------------------------------------",
        "1. SKILLS POSSESSED",
        "--------------------------------------------------",
        *(f" - {s}" for s in skills_possessed) if skills_possessed else [" - None identified"],
        "",
        "--------------------------------------------------",
        "2. SKILL GAPS IDENTIFIED",
        "--------------------------------------------------",
        *(f" - {s}" for s in skills_missing) if skills_missing else [" - None (Fully Matched)"],
        "",
        "--------------------------------------------------",
        "3. TOP RECOMMENDED ROLES",
        "--------------------------------------------------",
        *(
            f" {idx+1}. {r['job_role']}: {r['match_score']}% Match"
            for idx, (_, r) in enumerate(results_df.head(5).iterrows())
        ),
        "",
        "--------------------------------------------------",
        "4. ACTIONABLE LEARNING ROADMAP",
        "--------------------------------------------------",
        *(f" {step}" for step in roadmap_steps),
        "==================================================",
    ]

    st.download_button(
        label="📥 Download Detailed Analysis Report (.txt)",
        data="\n".join(report_content),
        file_name=f"resume_analysis_{selected_role.lower().replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=False,
    )

else:
    # Default State Message
    st.info("👈 Please select and upload your resume in PDF or DOCX format from the sidebar to start.")
