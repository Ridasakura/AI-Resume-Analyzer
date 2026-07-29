# 🧭 AI Resume Analyzer & Job Recommendation System

A modern AI-powered Resume Analyzer built using **Python**, **Streamlit**, **Natural Language Processing (NLP)**, **TF-IDF**, and **Cosine Similarity** to evaluate resumes, recommend suitable job roles, identify missing skills, and generate personalized learning roadmaps.

---

## 📌 Project Overview

This project helps users evaluate how well their resume matches different job roles by extracting technical skills from uploaded resumes and comparing them with predefined job requirements.

The application provides:

* 📄 Resume parsing from PDF and DOCX files
* 🧠 Automatic skill extraction
* 📊 Resume-to-job matching using TF-IDF and Cosine Similarity
* 🎯 Top job role recommendations
* 📈 Match percentage visualization
* 🚀 Skill gap analysis
* 🗺️ Personalized learning roadmap
* 📥 Downloadable analysis report

---

# 🖥️ Technologies Used

* Python 3.10+
* Streamlit
* Pandas
* Scikit-learn
* PyPDF
* python-docx
* Plotly
* Regular Expressions (Regex)

---

# 🧠 Machine Learning Concepts

This project uses the following techniques:

* TF-IDF (Term Frequency–Inverse Document Frequency)
* Cosine Similarity
* Resume Text Normalization
* Rule-Based Skill Extraction
* NLP Text Processing

---

# 📂 Project Structure

```text
AI_Resume_Analyzer/
│
├── app.py
├── requirements.txt
├── job_roles.csv
├── skill_dictionary.csv
├── sample_resume.docx
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project folder:

```bash
cd YOUR_REPOSITORY
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 📋 Features

✅ Upload PDF Resume

✅ Upload DOCX Resume

✅ Automatic Resume Parsing

✅ Technical Skill Extraction

✅ TF-IDF Based Resume Analysis

✅ Cosine Similarity Matching

✅ Top 3 Job Recommendations

✅ Skill Gap Detection

✅ Learning Roadmap Generation

✅ Interactive Dashboard

✅ Downloadable Resume Analysis Report

---

# 📊 Workflow

1. Upload Resume
2. Extract Resume Text
3. Clean and Normalize Text
4. Detect Technical Skills
5. Compare Skills with Job Roles
6. Calculate TF-IDF Scores
7. Compute Cosine Similarity
8. Rank Job Roles
9. Identify Missing Skills
10. Generate Learning Roadmap
11. Export Analysis Report

---

# 🎯 Future Improvements

* AI-powered resume suggestions using Large Language Models (LLMs)
* ATS compatibility scoring
* Resume keyword optimization
* More job role datasets
* Authentication and user profiles
* PDF report generation
* Cloud database integration

---

# 👨‍💻 Author

**Rida Sahrin**

---

# 📜 License

This project is developed for educational and academic purposes.
