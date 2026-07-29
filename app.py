"""
AI Resume Analyzer and Job Recommendation System
A Streamlit app that extracts resume content, identifies skills,
matches against job roles via TF-IDF + cosine similarity, and
generates a skill-gap learning roadmap.
"""

import re
import pandas as pd
import streamlit as st
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Compass — Resume Analyzer", page_icon="🧭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #12211C; }

.compass-header { padding: 1.6rem 0 0.4rem 0; border-bottom: 1px solid #2A3F35; margin-bottom: 1.6rem; }
.compass-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.18em;
    color: #C97C4B; text-transform: uppercase; margin-bottom: 0.3rem;
}
.compass-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.3rem; color: #EDEAE3; margin: 0; }
.compass-sub { color: #9BAFA5; font-size: 0.95rem; margin-top: 0.5rem; max-width: 620px; }

section[data-testid="stSidebar"] { background-color: #0E1A15; border-right: 1px solid #2A3F35; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.14em;
    color: #C97C4B; text-transform: uppercase;
}

.skill-chip {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    color: #EDEAE3; border-radius: 3px; padding: 3px 9px; margin: 3px 6px 3px 0;
}
.chip-found { background-color: #1C3428; border: 1px solid #4E8266; }
.chip-missing { background-color: #3A2420; border: 1px solid #8A4A38; }

.role-card {
    border: 1px solid #2A3F35; border-radius: 6px; padding: 0.9rem 1.1rem;
    background-color: #16261F; margin-bottom: 0.6rem;
}
.role-name { font-family: 'Fraunces', serif; font-size: 1.05rem; color: #EDEAE3; }
.role-score { font-family: 'JetBrains Mono', monospace; color: #C97C4B; font-size: 0.9rem; }

.stButton button {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    border: 1px solid #C97C4B; color: #C97C4B; background-color: transparent;
}
.stButton button:hover { background-color: #C97C4B; color: #12211C; }
</style>

<div class="compass-header">
    <div class="compass-eyebrow">Skill Matching · TF-IDF & Cosine Similarity</div>
    <div class="compass-title">🧭 Compass</div>
    <div class="compass-sub">Upload your resume to see how it matches real job roles, which skills you're missing, and where to focus next.</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------
@st.cache_data
def load_datasets():
    job_roles_df = pd.read_csv("job_roles.csv")
    skill_dict_df = pd.read_csv("skill_dictionary.csv")
    job_roles_df["required_skills"] = job_roles_df["required_skills"].apply(
        lambda s: [x.strip() for x in s.split(",")]
    )
    return job_roles_df, skill_dict_df


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        d = docx.Document(uploaded_file)
        return "\n".join(p.text for p in d.paragraphs)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")


def clean_text(text):
    text = text.lower()
    text = text.replace("c++", "cplusplus").replace("c#", "csharp").replace(".net", "dotnet")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("cplusplus", "c++").replace("csharp", "c#").replace("dotnet", ".net")
    return text


def extract_skills(text, known_skills):
    found = []
    for skill in known_skills:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)
    return found


def match_roles(found_skills, job_roles_df):
    resume_skill_text = " ".join(found_skills)
    role_texts = [" ".join(skills) for skills in job_roles_df["required_skills"]]
    corpus = [resume_skill_text] + role_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

    results_df = job_roles_df.copy()
    results_df["match_score"] = (similarities * 100).round(1)
    return results_df.sort_values("match_score", ascending=False).reset_index(drop=True)


def generate_roadmap(missing_skills):
    return [f"Week {i}: Learn the fundamentals of **{s}** and build one small practice project."
            for i, s in enumerate(missing_skills, start=1)]


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## Your Resume")
    resume_upload = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    analyze_clicked = st.button("Analyze Resume", use_container_width=True)

    st.divider()
    st.markdown("## About")
    st.caption(
        "Skills are matched against a controlled dictionary. "
        "Roles are ranked using TF-IDF vectors and cosine similarity — "
        "no personal attributes (age, gender, photo, etc.) are ever scored."
    )

job_roles_df, skill_dict_df = load_datasets()
all_known_skills = sorted(skill_dict_df["skill"].str.lower().tolist())

if "results_df" not in st.session_state:
    st.session_state.results_df = None
    st.session_state.found_skills = None

# ---------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------
if analyze_clicked:
    if not resume_upload:
        st.sidebar.error("Please upload a resume first.")
    else:
        with st.spinner("Extracting text, identifying skills, and matching roles..."):
            raw_text = extract_text(resume_upload)
            cleaned_text = clean_text(raw_text)
            found_skills = extract_skills(cleaned_text, all_known_skills)
            results_df = match_roles(found_skills, job_roles_df)

        st.session_state.results_df = results_df
        st.session_state.found_skills = found_skills

# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    found_skills = st.session_state.found_skills

    skills_by_cat = skill_dict_df[skill_dict_df["skill"].str.lower().isin(found_skills)]

    st.markdown("### Skills Found in Your Resume")
    if skills_by_cat.empty:
        st.info("No known skills detected. Try a resume with more technical keywords.")
    else:
        for cat, group in skills_by_cat.groupby("category"):
            chips = "".join(f'<span class="skill-chip chip-found">{s}</span>' for s in group["skill"])
            st.markdown(f"**{cat.replace('_', ' ').title()}**  \n{chips}", unsafe_allow_html=True)

    st.markdown("### Match Score by Role")
    chart_df = results_df.set_index("job_role")[["match_score"]]
    st.bar_chart(chart_df, color="#C97C4B")

    st.markdown("### Top 3 Recommended Roles")
    cols = st.columns(3)
    for i, row in results_df.head(3).iterrows():
        with cols[i]:
            st.markdown(f"""
            <div class="role-card">
                <div class="role-name">{row['job_role']}</div>
                <div class="role-score">{row['match_score']}% match</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Skill-Gap Analysis")
    target_role = st.selectbox("Choose a target role for detailed gap analysis:", results_df["job_role"])
    row = results_df[results_df["job_role"] == target_role].iloc[0]
    required = row["required_skills"]
    have = [s for s in required if s in found_skills]
    missing = [s for s in required if s not in found_skills]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Skills you have:**")
        chips = "".join(f'<span class="skill-chip chip-found">✓ {s}</span>' for s in have) or "None yet"
        st.markdown(chips, unsafe_allow_html=True)
    with c2:
        st.markdown("**Skills to build:**")
        chips = "".join(f'<span class="skill-chip chip-missing">+ {s}</span>' for s in missing) or "None — full match!"
        st.markdown(chips, unsafe_allow_html=True)

    if missing:
        st.markdown("### Suggested Learning Roadmap")
        for line in generate_roadmap(missing):
            st.markdown(f"- {line}")

    # Downloadable report
    report_lines = [
        "RESUME ANALYSIS REPORT",
        "=" * 40,
        f"Target Role: {target_role}",
        f"Match Score: {row['match_score']}%",
        "",
        "Skills Found:",
        *[f"  - {s}" for s in have],
        "",
        "Missing Skills:",
        *[f"  - {s}" for s in missing],
        "",
        "Top 3 Recommended Roles:",
        *[f"  {i+1}. {r['job_role']} - {r['match_score']}%" for i, r in results_df.head(3).iterrows()],
        "",
        "Suggested Roadmap:",
        *[f"  {l}" for l in generate_roadmap(missing)],
    ]
    st.download_button(
        "📥 Download Analysis Report",
        data="\n".join(report_lines),
        file_name="resume_analysis_report.txt",
        mime="text/plain",
    )
else:
    st.info("🧭 Upload your resume in the sidebar and click **Analyze Resume** to get started.")
