"""
RAG (Retrieval-Augmented Generation) chatbot engine for the Energy Analytics platform.
Uses FAISS for vector search and Groq LLM for answer generation.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    import numpy as np

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


def get_rag_chain(
    api_key: str,
    vectorstore: FAISS,
    model_name: str = "llama-3.3-70b-versatile",
) -> ConversationalRetrievalChain:
    """Create a conversational RAG chain backed by Groq."""
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=1024,
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

    logger.info("RAG chain initialized with model=%s", model_name)
    return chain


def ask(question: str, chain: ConversationalRetrievalChain) -> Tuple[str, List[Document]]:
    """Query the RAG chain and return (answer, source_documents)."""
    result = chain.invoke({"question": question})
    answer = result.get("answer", "I couldn't generate a response.")
    sources = result.get("source_documents", [])
    return answer, sources
