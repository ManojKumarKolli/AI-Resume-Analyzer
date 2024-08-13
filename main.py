import streamlit as st
import requests
from PyPDF2 import PdfReader
import pandas as pd
import json
import re


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to get alignment data from Gemini API
def get_alignment_data(resume_text, jd_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    prompt = f"""
    Analyze the following resume and job description. Provide the technologies present in the resume, technologies not present in the resume, and an alignment score between 0 and 100. Output the results in the following JSON format:

    {{
      "technologies_present": ["<list of technologies present>"],
      "technologies_not_present": ["<list of technologies not present>"],
      "alignment_score": <int between 0 and 100>
    }}

    Resume: {resume_text}

    Job Description: {jd_text}
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()


# Function to get resume score and suggestions from Gemini API
def get_resume_analysis(resume_text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    prompt = f"""
    Analyze the following resume. Provide a resume score between 0 and 100 and suggestions for improvement. Output the results in the following JSON format:

    {{
      "resume_score": <int between 0 and 100>,
      "suggestions": "<suggestions for improvement>"
    }}

    Resume: {resume_text}
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()


# Function to extract JSON from a string containing a JSON block
def extract_json_block(text):
    match = re.search(r"```json\n({.*})\n```", text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError("No JSON block found")


# Function to display resume analysis
def display_resume_analysis(analysis):
    try:
        analysis_data = analysis["candidates"][0]["content"]["parts"][0]["text"]
        analysis_data_json = extract_json_block(analysis_data)
        analysis_data = json.loads(analysis_data_json)

        resume_score = analysis_data.get("resume_score", 0)
        suggestions = analysis_data.get("suggestions", "")

        st.write(f"**Resume Score:** {resume_score}")
        st.write(f"**Suggestions for Improvement:**")
        st.write(suggestions)

        st.write("**Score Indicator:**")
        st.progress(resume_score / 100)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        st.error(f"Error processing the analysis data: {e}. Please check the API response format.")


# Load the dataset for visualization
@st.cache_data
def load_data():
    df = pd.read_csv('us_salaries.csv')
    return df


# Set page title and logo
st.set_page_config(page_title="Job Companion", page_icon="🧑‍💼")

# Sidebar for navigation
st.sidebar.title("AI RESUME HELPER")
# st.sidebar.image(
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/No_image_3x4.svg/2048px-No_image_3x4.svg.png", width=150)
st.sidebar.header("Navigation")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
option = st.sidebar.selectbox("Choose the App",
                              ["Resume Score Checker", "Resume and JD Alignment Checker", "Job Trends in Data Science"])

if option == "Resume Score Checker":
    st.title("Resume Score Checker")
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")

    if uploaded_file is not None and api_key:
        resume_text = extract_text_from_pdf(uploaded_file)
        analysis = get_resume_analysis(resume_text, api_key)

        # Debugging: Print the analysis response
        st.write("API Response:", analysis)
        display_resume_analysis(analysis)
    else:
        st.write("Please upload a PDF resume and enter your API key.")

elif option == "Resume and JD Alignment Checker":
    st.title("Resume and JD Alignment Checker")
    resume_file = st.file_uploader("Upload your resume (PDF format)", type="pdf")
    jd_text = st.text_area("Paste the job description",
                           placeholder="6-8 years of experience as a Python Developer with a strong portfolio of projects. • Bachelor’s degree in computer science, Software Engineering or a related field. • In-depth understanding of the Python software development stacks, ecosystems, frameworks, and tools such as Numpy, Pandas, Dask, spaCy, NLTK, sci-kit-learn and Py-Torch. • Experience with front-end development using HTML, CSS, and JavaScript. • Familiarity with database technologies such as SQL and NoSQL. • Experience with leading RDBMS. • Experience with Snowflake is an added advantage. • Excellent problem-solving ability with solid communication and collaboration skills.")
    analyze_button = st.button("Analyze")

    if analyze_button and resume_file is not None and jd_text and api_key:
        resume_text = extract_text_from_pdf(resume_file)

        alignment_response = get_alignment_data(resume_text, jd_text, api_key)

        # Debugging: Print the full API response
        st.write("API Response:", alignment_response)

        # Extract data from the API response
        try:
            alignment_data = alignment_response["candidates"][0]["content"]["parts"][0]["text"]
            st.write("Extracted alignment data:", alignment_data)  # Debugging

            # Extract JSON block
            alignment_data_json = extract_json_block(alignment_data)
            alignment_data = json.loads(alignment_data_json)

            technologies_present = alignment_data.get("technologies_present", [])
            technologies_not_present = alignment_data.get("technologies_not_present", [])
            alignment_score = alignment_data.get("alignment_score", 0)

            st.write("**Technologies Present in Resume:**")
            st.write(", ".join(technologies_present) if technologies_present else "None")

            st.write("**Technologies Not Present in Resume:**")
            st.write(", ".join(technologies_not_present) if technologies_not_present else "None")

            st.write(f"**Alignment Score:** {alignment_score}%")

            # Use the original progress bar indicator
            st.write("**Alignment Indicator:**")
            st.progress(alignment_score / 100)

            if alignment_score >= 75:
                st.success("Your resume has a high chance of being shortlisted!")
            else:
                st.warning("Consider improving your resume to increase your chances of being shortlisted.")
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            st.error(f"Error processing the alignment data: {e}. Please check the API response format.")

elif option == "Job Trends in Data Science":
    st.title("Job Trends in Data Science")

    # Load data
    df = load_data()

    st.write("### Select options to visualize job trends")
    job_role = st.selectbox("Job Role", df['job_title'].unique())
    city = st.selectbox("City", df['company_location'].unique())

    filtered_data = df[(df['job_title'] == job_role) & (df['company_location'] == city)]

    st.write(f"### Salary Trends for {job_role} in {city}")
    st.bar_chart(filtered_data.set_index('experience_level')['salary_in_usd'])

    st.write("### Detailed Data")
    st.write(filtered_data)

# Sidebar README section
st.sidebar.header("README")
st.sidebar.write("""
### How to Create an API Key

1. Go to the [Gemini API](https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent) website.
2. Sign up or log in to your account.
3. Navigate to the API section.
4. Generate a new API key.
5. Copy the API key and paste it into the input field above.

### Highlights

- **Resume Score Checker:** Upload your resume and get a score along with suggestions for improvement.
- **Resume and JD Alignment Checker:** Check how well your resume aligns with a job description.
- **Job Trends in Data Science:** Visualize job trends based on categories like job role and city.
""")
