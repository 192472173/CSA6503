import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(page_title="College Admission Chatbot")

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini model
model = genai.GenerativeModel("gemini-3.6-flash")

# UI
st.title("🎓 College Admission Chatbot")

question = st.text_input("Ask your admission-related question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        prompt = f"""
You are a college admission assistant.

Answer ONLY questions about:
- Admissions
- Courses
- Eligibility
- Fees
- Scholarships
- Hostel
- Placements
- Required Documents

If the question is unrelated, reply:
"Sorry, I can only answer college admission-related questions."

Question:
{question}
"""

        response = model.generate_content(prompt)
        st.write(response.text)