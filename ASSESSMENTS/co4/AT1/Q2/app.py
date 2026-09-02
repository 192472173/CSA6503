import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Library Assistant Chatbot",
    page_icon="📚"
)

st.title("📚 College Library Assistant Chatbot")
st.write("Ask me about books, borrowing, membership, working hours, and return policies.")

if not api_key:
    st.error("API Key not found.")
    st.stop()

try:
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-3.6-flash")

    question = st.text_input(
        "Enter your library question:"
    )

    if st.button("Ask Library Assistant"):

        if not question.strip():
            st.warning("Please enter a library-related question.")
            st.stop()

        prompt = f"""
You are a College Library Assistant chatbot.

Answer ONLY questions related to:
- Book availability
- Borrowing rules
- Library working hours
- Library membership
- Book return policies
- Book renewal
- Book reservation
- Library services

Library information:
- Working hours: 8:00 AM to 6:00 PM
- Maximum books: 3
- Borrowing period: 14 days
- Membership: Available to registered students
- Renewal: Allowed if the book is not reserved by another student
- Books must be returned before the due date.

If the question is unrelated to the library, respond:
"Sorry, I can assist only with college library-related questions."

If the required information is not available, respond:
"I don't have that information. Please contact the library staff or check the official library catalogue."

Student Question:
{question}
"""

        try:
            response = model.generate_content(prompt)

            if response.text:
                st.success("Library Assistant Response")
                st.write(response.text)
            else:
                st.warning("No response received.")

        except Exception as e:
            st.error("Error while communicating with Gemini.")
            st.exception(e)

except Exception as e:
    st.error("Gemini configuration error.")
    st.exception(e)