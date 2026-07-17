"""
Lightweight RAG chatbot engine for the Energy Analytics platform.
Uses simple TF-IDF similarity for retrieval and Groq LLM for answers.
No PyTorch, no FAISS, no sentence-transformers — Streamlit/Render Cloud friendly.
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = """You are an AI Energy Assistant for the AI-Powered Energy Analytics System.
You answer questions about the platform, its data, ML models, energy insights, and architecture.
Use ONLY the provided context to answer. If the context doesn't contain the answer, say so honestly.
Keep answers clear, concise, and helpful. Use bullet points and formatting when appropriate."""


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    source: str


def _load_documents() -> List[Chunk]:
    """Load project documentation files and return as Chunks."""
    chunks = []

    doc_files = [
        ("explain.md", "Project Documentation"),
        ("README.md", "Project README"),
    ]

    for filename, source_label in doc_files:
        filepath = PROJECT_ROOT / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(_split_text(content, source_label))
            logger.info("Loaded %s (%d chars)", filename, len(content))

    return chunks


def _split_text(text: str, source: str, chunk_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """Split text into overlapping chunks by paragraphs/sections."""
    sections = text.split("\n## ")

    result = []
    for i, section in enumerate(sections):
        if i > 0:
            section = "## " + section

        if len(section) <= chunk_size:
            result.append(Chunk(content=section.strip(), source=source))
        else:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > chunk_size and current:
                    result.append(Chunk(content=current.strip(), source=source))
                    current = current[-overlap:] if len(current) > overlap else current
                current += "\n\n" + para
            if current.strip():
                result.append(Chunk(content=current.strip(), source=source))

    return result


def _generate_data_summary(df) -> List[Chunk]:
    """Generate live data summary chunks from the loaded DataFrame."""
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
    return [Chunk(content=content, source="Live Data Summary")]


def build_knowledge_base(df=None) -> List[Chunk]:
    """Build the knowledge base from project docs and optional live data."""
    logger.info("Building RAG knowledge base...")

    chunks = _load_documents()

    if df is not None:
        chunks.extend(_generate_data_summary(df))

    logger.info("Knowledge base built with %d chunks", len(chunks))
    return chunks


def _find_relevant_chunks(question: str, knowledge_base: List[Chunk], top_k: int = 4) -> List[Chunk]:
    """Find the most relevant chunks using keyword matching."""
    question_lower = question.lower()
    keywords = set(question_lower.split())

    stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
                 "when", "where", "which", "who", "do", "does", "did", "can", "could",
                 "will", "would", "should", "in", "on", "at", "to", "for", "of", "with",
                 "and", "or", "not", "this", "that", "it", "i", "me", "my", "you", "your",
                 "we", "they", "he", "she", "be", "have", "has", "had"}
    keywords = keywords - stopwords

    if not keywords:
        keywords = set(question_lower.split())

    scored = []
    for chunk in knowledge_base:
        chunk_lower = chunk.content.lower()
        score = 0

        for kw in keywords:
            count = chunk_lower.count(kw)
            score += count

        if question_lower in chunk_lower:
            score += 10

        for kw in keywords:
            if kw in chunk_lower:
                score += 2

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k] if _ > 0]


def ask(
    question: str,
    knowledge_base: List[Chunk],
    api_key: str,
    chat_history: List[dict],
    model_name: str = "llama-3.3-70b-versatile",
) -> Tuple[str, List[Chunk]]:
    """
    Query the RAG system: find relevant chunks, then ask Groq LLM.
    Returns (answer, source_chunks).
    """
    relevant_chunks = _find_relevant_chunks(question, knowledge_base)

    if not relevant_chunks:
        relevant_chunks = knowledge_base[:3]

    context = "\n\n---\n\n".join(
        [f"[Source: {chunk.source}]\n{chunk.content}" for chunk in relevant_chunks]
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
    return answer, relevant_chunks
