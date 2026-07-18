import os
import json
import re
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.query_cache = {}

        abstracts = df["abstract"].fillna("").tolist()

        cache_path = "data/rag_embeddings.npy"
        index_path = "data/rag_faiss.index"

        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path).astype("float32")
        else:
            embeddings = self.model.encode(
                abstracts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self.embeddings = np.array(embeddings).astype("float32")
            np.save(cache_path, self.embeddings)

        dim = self.embeddings.shape[1]

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            nlist = 100
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            if not self.index.is_trained:
                self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = 10
            faiss.write_index(self.index, index_path)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[WARNING] GROQ_API_KEY not found. RAG LLM will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=api_key)

    def retrieve(self, query: str, k: int = 5):
        if query in self.query_cache:
            query_vec = self.query_cache[query]
        else:
            query_vec = self.model.encode([query], normalize_embeddings=True)
            query_vec = np.array(query_vec).astype("float32")
            self.query_cache[query] = query_vec

        if not self.index.is_trained:
            self.index.train(self.embeddings)

        distances, indices = self.index.search(query_vec, k)

        papers = []
        for idx in indices[0]:
            if idx < len(self.df):
                row = self.df.iloc[idx]
                papers.append(
                    {"journal": row["journal_name"], "abstract": row["abstract"]}
                )
        return papers

    def generate(self, user_abstract: str, top_journals, extra_context=None):
        """
        LLM-as-validator pattern.

        Instead of asking Groq to explain why the pipeline's top pick is correct
        (yes-man behavior), we give Groq all 3 candidates and ask it to
        independently decide which is the best fit and why.

        This allows the LLM to catch retrieval errors — cases where the pipeline
        ranked a journal highly due to corpus bias or embedding noise, but the
        actual domain match is weaker than a lower-ranked candidate.

        The validator response includes:
        - best_journal: Groq's independent pick (may differ from pipeline's top)
        - reason: specific explanation grounded in topic + method + domain
        - pipeline_agreement: whether Groq agrees with the pipeline's top pick
        - override: True if Groq picked a different journal than pipeline rank 1
        """
        if not top_journals:
            return {
                "best_journal": "Unknown",
                "reason": "No journal candidates provided",
                "pipeline_agreement": True,
                "override": False,
            }

        extra_context = extra_context or {}
        pipeline_top = top_journals[0].get("journal_name", "Unknown")

        # Build candidate context with scores and scope descriptions
        context_lines = []
        for i, j in enumerate(top_journals[:3]):
            journal_name = j.get("journal_name", "Unknown Journal")
            similarity = round(j.get("similarity", 0.0), 3)
            confidence = round(j.get("confidence", 0.0), 3)
            context_lines.append(
                f"{i + 1}. {journal_name}\n"
                f"   Retrieval similarity: {similarity}\n"
                f"   Pipeline confidence: {confidence}"
            )

        context = "\n".join(context_lines)

        prompt = f"""You are an expert academic journal reviewer with deep knowledge of computer science research domains.

TASK:
You are given a research abstract and THREE candidate journals ranked by an automated retrieval pipeline.
Your job is to independently evaluate which journal is the BEST fit for this paper.
You are NOT required to agree with the pipeline's top-ranked journal.

ABSTRACT:
{user_abstract}

CANDIDATE JOURNALS (ranked by pipeline):
{context}

EVALUATION CRITERIA:
1. Research topic alignment — does the journal publish papers on this topic?
2. Methodology fit — does the journal cover these techniques/methods?
3. Application domain — does the journal serve this application area?
4. If none of the candidates are a strong fit, say so explicitly.

STRICT RULES:
- Be specific — mention the actual topic, method, and domain from the abstract
- Do NOT use generic phrases like "high similarity" or "good match"
- Maximum 2-3 sentences for the reason
- If you believe a different journal (not in the list) would be better, name it in best_journal
- Sound like a reviewer, not a chatbot

Return ONLY valid JSON with no text outside it:
{{
  "best_journal": "<your independent pick — can differ from pipeline rank 1>",
  "reason": "<specific explanation: topic + method + domain>",
  "pipeline_agreement": <true if your pick matches pipeline rank 1, false otherwise>,
  "override": <true if you picked a journal NOT in the candidate list>
}}"""

        if self.client is None:
            # Fallback when no API key
            best = top_journals[0]
            journal_name = best.get("journal_name", "Unknown")
            sim = round(best.get("similarity", 0.0), 3)

            if sim >= 0.65:
                reason = f"{journal_name} strongly aligns with the research focus in both methodology and application domain."
            elif sim >= 0.5:
                reason = f"{journal_name} shows moderate alignment with the research topic and applied techniques."
            else:
                reason = f"{journal_name} has limited alignment, indicating only partial thematic overlap."

            return {
                "best_journal": journal_name,
                "reason": reason,
                "pipeline_agreement": True,
                "override": False,
            }

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[RAG LLM ERROR] {e}")
            return {
                "best_journal": pipeline_top,
                "reason": "LLM validation unavailable",
                "pipeline_agreement": True,
                "override": False,
            }

        try:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in LLM output")

            parsed = json.loads(match.group(0))

            # Ensure required fields exist
            parsed.setdefault("best_journal", pipeline_top)
            parsed.setdefault("reason", "Reason not provided")
            parsed.setdefault(
                "pipeline_agreement", parsed["best_journal"] == pipeline_top
            )
            parsed.setdefault("override", False)

            # Auto-detect agreement based on journal name match
            parsed["pipeline_agreement"] = (
                parsed["best_journal"].strip().lower() == pipeline_top.strip().lower()
            )

            return parsed

        except Exception:
            print("[RAG PARSE ERROR]")
            print(output)

            best = top_journals[0]
            journal_name = best.get("journal_name", "Unknown")
            sim = round(best.get("similarity", 0.0), 3)

            if sim >= 0.65:
                reason = f"{journal_name} strongly matches the paper's focus in methodology and application domain."
            elif sim >= 0.5:
                reason = f"{journal_name} shows moderate relevance in research topic and techniques."
            else:
                reason = f"{journal_name} shows weak relevance, suggesting limited suitability."

            return {
                "best_journal": journal_name,
                "reason": reason,
                "pipeline_agreement": True,
                "override": False,
            }
