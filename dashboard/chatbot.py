"""
Dashboard — AI Chat page: RAG-powered chatbot for querying
project documentation, data insights, and model details.
The API key is loaded from .env automatically — no user input needed.
"""

import streamlit as st
import os
from dotenv import load_dotenv

from config import THEME_COLORS

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"


def render(df) -> None:
    """Render the AI Chatbot dashboard page."""
    st.markdown("## 🤖 AI Energy Assistant")
    st.markdown(
        "Ask me anything about this platform — the data, models, energy insights, "
        "how things work, or what the results mean."
    )

    if not GROQ_API_KEY:
        st.error(
            "⚠️ Chatbot is currently unavailable. "
            "The server administrator needs to configure the GROQ_API_KEY."
        )
        return

    if "vectorstore" not in st.session_state:
        with st.spinner("🔍 Building knowledge base from project docs..."):
            from rag.chatbot import build_knowledge_base
            st.session_state.vectorstore = build_knowledge_base(df)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi! I'm the Energy AI Assistant. Ask me anything about:\n\n"
                    "- 📊 **Data** — dataset details, household stats, seasonal patterns\n"
                    "- 🔮 **Models** — forecasting, clustering, anomaly detection\n"
                    "- 📈 **Results** — performance metrics, R² scores, MAE\n"
                    "- 💡 **Insights** — energy savings tips, peak consumption analysis\n"
                    "- 🛠️ **Architecture** — how the platform is built"
                ),
            }
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about energy data, models, or insights..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from rag.chatbot import ask
                    answer, sources = ask(
                        question=prompt,
                        vectorstore=st.session_state.vectorstore,
                        api_key=GROQ_API_KEY,
                        chat_history=st.session_state.chat_messages[:-1],
                        model_name=MODEL_NAME,
                    )
                except Exception as e:
                    answer = f"Sorry, I encountered an error: {str(e)}"
                    sources = []

            st.markdown(answer)

            if sources:
                unique_sources = list(
                    {doc.metadata.get("source", "Unknown") for doc in sources}
                )
                source_tags = " · ".join([f"`{s}`" for s in unique_sources])
                st.markdown(
                    f"<p style='color: rgba(255,255,255,0.35); font-size: 0.75rem; "
                    f"margin-top: 0.5rem;'>"
                    f"📚 Sources: {source_tags}</p>",
                    unsafe_allow_html=True,
                )

                with st.expander("🔎 View retrieved context chunks"):
                    for i, doc in enumerate(sources):
                        st.markdown(
                            f"**Chunk {i + 1}** — "
                            f"*{doc.metadata.get('source', 'Unknown')}*"
                        )
                        st.code(doc.page_content[:500], language="markdown")

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )
