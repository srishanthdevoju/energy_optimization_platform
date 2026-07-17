"""
Dashboard — AI Chat page: RAG-powered chatbot for querying
project documentation, data insights, and model details.
"""

import streamlit as st
import os

from config import THEME_COLORS


def render(df) -> None:
    """Render the AI Chatbot dashboard page."""
    st.markdown("## 🤖 AI Energy Assistant")
    st.markdown(
        "Ask questions about the platform, data, models, or energy insights. "
        "Powered by RAG (Retrieval-Augmented Generation) with Groq."
    )

    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        api_key = st.text_input(
            "Enter your Groq API key",
            value=os.environ.get("GROQ_API_KEY", ""),
            type="password",
            help="Get a free key at https://console.groq.com",
        )

        model_name = st.selectbox(
            "🧠 LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )

    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to start chatting.")
        st.markdown("""
            <div style="
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin-top: 2rem;
            ">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🔐</p>
                <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem;">
                    Enter your Groq API key in the sidebar to unlock the AI Assistant.
                </p>
                <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">
                    Get a free API key at
                    <a href="https://console.groq.com" target="_blank"
                       style="color: #00D4AA;">console.groq.com</a>
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    if "vectorstore" not in st.session_state:
        with st.spinner("🔍 Building knowledge base from project docs..."):
            from rag.chatbot import build_knowledge_base
            st.session_state.vectorstore = build_knowledge_base(df)

    if "rag_chain" not in st.session_state or st.session_state.get("current_model") != model_name:
        with st.spinner("⚡ Initializing AI model..."):
            from rag.chatbot import get_rag_chain
            st.session_state.rag_chain = get_rag_chain(
                api_key=api_key,
                vectorstore=st.session_state.vectorstore,
                model_name=model_name,
            )
            st.session_state.current_model = model_name

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "👋 Hi! I'm your Energy AI Assistant. Ask me anything about the platform, data, models, or energy insights!"}
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
                from rag.chatbot import ask
                answer, sources = ask(prompt, st.session_state.rag_chain)

            st.markdown(answer)

            if sources:
                unique_sources = list({doc.metadata.get("source", "Unknown") for doc in sources})
                source_tags = " · ".join([f"`{s}`" for s in unique_sources])
                st.markdown(
                    f"<p style='color: rgba(255,255,255,0.35); font-size: 0.75rem; margin-top: 0.5rem;'>"
                    f"📚 Sources: {source_tags}</p>",
                    unsafe_allow_html=True,
                )

                with st.expander("🔎 View retrieved context chunks"):
                    for i, doc in enumerate(sources):
                        st.markdown(
                            f"**Chunk {i+1}** — *{doc.metadata.get('source', 'Unknown')}*"
                        )
                        st.code(doc.page_content[:500], language="markdown")

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
