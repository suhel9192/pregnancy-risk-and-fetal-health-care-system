import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from patient.chat_db import save_message, load_chat, clear_chat
from groq import Groq

load_dotenv()

# ---------- LOAD GROQ ---------- #
try:
    groq_client = Groq(api_key=os.getenv("API_KEY"))
except Exception:
    groq_client = None


# ---------- LOCAL EMBEDDINGS ---------- #
class LocalEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        return self.embed_query(text)


# ---------- LOAD RAG ---------- #
@st.cache_resource
def load_rag():
    with open("doctor/maternal_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    embeddings = LocalEmbeddings()
    db = FAISS.from_texts(chunks, embedding=embeddings)

    return db


# ---------- QUERY IMPROVEMENT ---------- #
def improve_query(user_input):
    q = user_input.lower()

    if "avoid" in q:
        return "FOOD_AVOID pregnancy diet avoid"
    elif "eat" in q:
        return "FOOD_EAT pregnancy diet eat"
    elif "bp" in q or "pressure" in q:
        return "BP blood pressure pregnancy"
    elif "sugar" in q or "diabetes" in q:
        return "SUGAR pregnancy diabetes"
    elif "exercise" in q:
        return "EXERCISE pregnancy"
    elif "sleep" in q:
        return "SLEEP pregnancy"
    elif "vitamin" in q:
        return "VITAMINS pregnancy"
    elif "symptom" in q:
        return "SYMPTOMS pregnancy danger signs"
    else:
        return user_input


# ---------- GENERATE RESPONSE USING GROQ ---------- #
def ask_groq(user_question, context):
    if not groq_client:
        return "Groq API not configured."

    try:
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful maternal healthcare assistant.
Answer only based on the given context.
If answer is not in context, say consult doctor.
Give short, clear and safe answers.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{user_question}
                    """
                }
            ],
            temperature=0.3
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Groq Error: {str(e)}"


# ---------- CHATBOT PAGE ---------- #
def chatbot_page():

    if st.session_state.get("role") != "patient":
        st.error("Chatbot is only for patients")
        return

    st.markdown("<h2 style='text-align:center;'>🤖 Maternal Care Chatbot</h2>", unsafe_allow_html=True)

    db = load_rag()
    user_id = st.session_state.get("user_id")

    # ---------- LOAD CHAT ---------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

        db_data = load_chat(user_id)
        for role, msg in db_data:
            st.session_state.messages.append({"role": role, "content": msg})

    # ---------- CLEAR CHAT ---------- #
    col1, col2 = st.columns([8, 2])

    with col2:
        if st.button("🗑 Clear Chat"):
            clear_chat(user_id)
            st.session_state.messages = []
            st.rerun()

    # ---------- SHOW CHAT ---------- #
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ---------- INPUT ---------- #
    user_input = st.chat_input("Ask your question...")

    if user_input:

        # Show user message
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(user_id, "user", user_input)

        # Improve search query
        improved_query = improve_query(user_input)

        docs = db.similarity_search(improved_query, k=2)

        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])

            # Use Groq here
            response = ask_groq(user_input, context)

        else:
            response = "No matching information found. Please consult your doctor."

        # Show response
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message(user_id, "assistant", response)