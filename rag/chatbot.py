"""
RAG (Retrieval-Augmented Generation) chatbot engine for the Energy Analytics platform.
Uses FAISS for vector search and Groq LLM for answer generation.
Direct Groq API approach — no deprecated langchain chains.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """You are an AI Energy Assistant for the AI-Powered Energy Analytics System.
You answer questions about the platform, its data, ML models, energy insights, and architecture.
Use ONLY the provided context to answer. If the context doesn't contain the answer, say so honestly.
Keep answers clear, concise, and helpful. Use bullet points and formatting when appropriate."""


def _load_documents() -> List[Document]:
    """Load project documentation files and return as LangChain Documents."""
    docs = []

    doc_files = [
        ("explain.md", "Project Documentation"),
        ("README.md", "Project README"),
    ]

    for filename, source_label in doc_files:
        filepath = PROJECT_ROOT / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=content,
                metadata={"source": source_label, "file": filename},
            ))
            logger.info("Loaded %s (%d chars)", filename, len(content))

    return docs


def _generate_data_summary(df) -> Document:
    """Generate a live data summary document from the loaded DataFrame."""
    summary_parts = [
        "## Live Dataset Summary",
        f"- Total records: {len(df):,}",
        f"- Unique households: {df['LCLid'].nunique()}",
        f"- Date range: {df['day'].min().date()} to {df['day'].max().date()}",
        f"- Average daily energy (kWh): {df['energy_sum'].mean():.2f}",
        f"- Median daily energy (kWh): {df['energy_sum'].median():.2f}",
        f"- Max daily energy (kWh): {df['energy_sum'].max():.2f}",
        f"- Min daily energy (kWh): {df['energy_sum'].min():.2f}",
        f"- Std deviation of daily energy (kWh): {df['energy_sum'].std():.2f}",
    ]

    if "season" in df.columns:
        season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
        seasonal = df.groupby("season")["energy_sum"].mean()
        summary_parts.append("\n## Seasonal Average Energy Consumption")
        for season_id, avg in seasonal.items():
            name = season_names.get(season_id, f"Season {season_id}")
            summary_parts.append(f"- {name}: {avg:.2f} kWh")

    if "is_weekend" in df.columns:
        weekend_avg = df[df["is_weekend"] == 1]["energy_sum"].mean()
        weekday_avg = df[df["is_weekend"] == 0]["energy_sum"].mean()
        summary_parts.append(f"\n## Weekend vs Weekday")
        summary_parts.append(f"- Weekend average: {weekend_avg:.2f} kWh")
        summary_parts.append(f"- Weekday average: {weekday_avg:.2f} kWh")

    model_comparison_path = PROJECT_ROOT / "saved_models" / "model_comparison.csv"
    if model_comparison_path.exists():
        import pandas as pd
        metrics_df = pd.read_csv(str(model_comparison_path))
        summary_parts.append("\n## Model Performance (Baseline)")
        for _, row in metrics_df.iterrows():
            summary_parts.append(
                f"- {row['Model']}: MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, R²={row['R2']:.4f}"
            )

    best_model_path = PROJECT_ROOT / "saved_models" / "best_model.txt"
    if best_model_path.exists():
        best = best_model_path.read_text().strip()
        summary_parts.append(f"\n## Best Model: {best}")

    content = "\n".join(summary_parts)
    return Document(
        page_content=content,
        metadata={"source": "Live Data Summary", "file": "runtime"},
    )


def build_knowledge_base(df=None) -> FAISS:
    """Build a FAISS vector store from project docs and optional live data."""
    logger.info("Building RAG knowledge base...")

    docs = _load_documents()

    if df is not None:
        docs.append(_generate_data_summary(df))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n#### ", "\n---", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS vector store built with %d vectors", vectorstore.index.ntotal)
    return vectorstore


def ask(
    question: str,
    vectorstore: FAISS,
    api_key: str,
    chat_history: List[dict],
    model_name: str = "llama-3.3-70b-versatile",
) -> Tuple[str, List[Document]]:
    """
    Query the RAG system: retrieve relevant chunks, then ask Groq LLM.
    Returns (answer, source_documents).
    """
    retrieved_docs = vectorstore.similarity_search(question, k=4)

    context = "\n\n---\n\n".join(
        [f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
         for doc in retrieved_docs]
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = (
        f"Context from knowledge base:\n\n{context}\n\n---\n\n"
        f"User question: {question}"
    )
    messages.append({"role": "user", "content": user_message})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    return answer, retrieved_docs
