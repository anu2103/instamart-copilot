
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"


def chunk_documents(chunk_size=40, overlap=10):
    """Split each policy into word chunks, preserving its source filename."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")

    chunks = []
    step = chunk_size - overlap

    for path in sorted(POLICY_DIR.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "source": path.name,
                "text": " ".join(words[start:end]),
            })
            if end >= len(words):
                break
            start += step

    return chunks


def build_retriever(chunk_size=40, overlap=10):
    """Fit TF-IDF once on the selected policy chunks."""
    chunks = chunk_documents(chunk_size, overlap)
    if not chunks:
        raise RuntimeError(f"No policy text files found in {POLICY_DIR}")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return chunks, vectorizer, matrix


def retrieve(query, k=3, retriever=None):
    """Return the top-k chunks ranked by cosine similarity."""
    if retriever is None:
        retriever = build_retriever()

    chunks, vectorizer, matrix = retriever
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    # Stable tie-breaking: original chunk order.
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [
        {
            "source": chunks[i]["source"],
            "score": float(scores[i]),
            "text": chunks[i]["text"],
        }
        for i in ranked[:k]
    ]


def answer(question, retriever=None):
    """Ask the mock model to answer only from the retrieved policy context."""
    results = retrieve(question, k=3, retriever=retriever)

    context_parts = []
    for result in results:
        context_parts.append(f"[{result['source']}]\n{result['text']}")

    user_content = (
        "CONTEXT:\n"
        + "\n".join(context_parts)
        + "\nQUESTION:\n"
        + question
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only using the supplied policy context. "
                "Do not invent or assume policy details. If the context does not "
                "contain the answer, reply beginning exactly with "
                "'I could not find' and explain that the information was not found."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    response = mock_llm.chat(messages)
    return response.choices[0].message.content, results



def evaluate(retriever, label):
    """Evaluate retrieval, exact expected facts and unsupported questions."""
    questions = pd.read_csv(EVAL_FILE).fillna("")
    rows = []

    for _, row in questions.iterrows():
        question = str(row["question"]).strip()
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        # The two unsupported questions have no expected fact.
        answerable = bool(expected_fact)

        results = retrieve(question, k=3, retriever=retriever)
        text_answer, _ = answer(question, retriever=retriever)
        sources = [r["source"] for r in results]

        rows.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answerable": answerable,
            "top3_sources": ";".join(sources),
            "retrieval_hit": (
                expected_doc in sources if answerable else None
            ),
            "expected_fact_found": (
                expected_fact in text_answer if answerable else None
            ),
            "unanswerable_fallback": (
                text_answer.startswith("I could not find")
                if not answerable else None
            ),
            "answer": text_answer,
            "chunking": label,
        })

    result_df = pd.DataFrame(rows)

    answerable_rows = result_df[result_df["answerable"]]
    unanswerable_rows = result_df[~result_df["answerable"]]

    summary = {
        "chunking": label,
        "chunks": len(retriever[0]),
        "answerable_questions": len(answerable_rows),
        "retrieval_hits": int(answerable_rows["retrieval_hit"].sum()),
        "expected_facts_found": int(
            answerable_rows["expected_fact_found"].sum()
        ),
        "unanswerable_questions": len(unanswerable_rows),
        "fallback_answers": int(
            unanswerable_rows["unanswerable_fallback"].sum()
        ),
    }

    return result_df, summary

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Normal configuration: 40-word chunks with 10-word overlap.
    standard = build_retriever(chunk_size=40, overlap=10)
    print(f"Chunks with 10-word overlap: {len(standard[0])}")

    if len(standard[0]) != 15:
        print("WARNING: expected 15 chunks; inspect policy text lengths.")

    print("\nTop three results for: refund policy late delivery")
    for result in retrieve("refund policy late delivery", retriever=standard):
        print(
            f"{result['source']} | score={result['score']:.4f} | "
            f"{result['text']}"
        )

    result_df, summary = evaluate(standard, "overlap_10")
    result_df.to_csv(OUTPUT_DIR / "policy_eval_overlap10.csv", index=False)

    # Repeat the complete evaluation without overlap.
    no_overlap = build_retriever(chunk_size=40, overlap=0)
    zero_df, zero_summary = evaluate(no_overlap, "overlap_0")
    zero_df.to_csv(OUTPUT_DIR / "policy_eval_overlap0.csv", index=False)

    summaries = pd.DataFrame([summary, zero_summary])
    summaries.to_csv(OUTPUT_DIR / "policy_eval_summary.csv", index=False)

    print("\nEvaluation summary:")
    print(summaries.to_string(index=False))

    print("\nQuestions that fail with zero overlap:")
    failed = zero_df[
        (zero_df["expected_doc"] != "")
        & (
            (~zero_df["retrieval_hit"].fillna(False))
            | (~zero_df["expected_fact_found"].fillna(False))
        )
    ]
    if failed.empty:
        print("No failures found; inspect the chunk boundaries and policy text.")
    else:
        for _, row in failed.iterrows():
            print(f"- {row['question']}")
            print(f"  Expected fact: {row['expected_fact']}")
            print(f"  Answer: {row['answer']}")

    # Show one grounded answer for a manual check.
    question = "What delivery time makes a completed order eligible for a full refund?"
    text_answer, sources = answer(question, retriever=standard)
    print("\nExample answer:")
    print(text_answer)
    print("Sources:", ", ".join(dict.fromkeys(s["source"] for s in sources)))


if __name__ == "__main__":
    main()
