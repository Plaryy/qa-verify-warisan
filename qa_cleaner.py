#!/usr/bin/env python3
"""
QA Data Cleaner and Validator
Optimised for AMD Ryzen AI Max+ 395 / Radeon 8060S / 64 GB RAM.

Key optimisations vs original:
  1. Batch SSUN  — all embeddings encoded in a single GPU pass before the loop.
  2. Parallel LLM — ThreadPoolExecutor sends concurrent requests to Ollama
                    (set OLLAMA_NUM_PARALLEL env var to match --workers).
  3. Single prompt — validation + cleaning merged into one Ollama call per record.
  4. Connection pool — per-thread requests.Session keeps TCP connections alive.
  5. ROCm/CUDA auto-detect — sentence-transformers uses the Radeon iGPU when available.
"""

import argparse
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from prompt_manager import PromptManager

# ---------------------------------------------------------------------------
# Sentence-transformer setup — prefers GPU (ROCm exposes as CUDA on Windows)
# ---------------------------------------------------------------------------
try:
    import torch
    from sentence_transformers import SentenceTransformer

    _st_device = "cuda" if torch.cuda.is_available() else "cpu"
    SEMANTIC_ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device=_st_device)
    print(f"[INFO] Semantic encoder loaded on: {_st_device}", file=sys.stderr)
except ImportError:
    SEMANTIC_ENCODER = None
    _st_device = "cpu"
    print("[WARN] sentence-transformers not installed — SSUN disabled.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Per-thread HTTP session (connection pooling)
# ---------------------------------------------------------------------------
_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=16)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return _thread_local.session


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class QARecord:
    question: str
    answer: str
    chunk: str
    similarity_score: float = 0.0
    has_mixed_language: bool = False
    has_noise: bool = False
    noise_percentage: float = 0.0
    is_too_short: bool = False
    cleaned_answer: str = ""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class QAValidator:
    def __init__(
        self,
        prompts_dir: str = "prompts",
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        workers: int = 4,
        context_size: int = 32768,
        debug: bool = False,
    ):
        self.ollama_host = ollama_host
        self.model = model
        self.workers = workers
        self.context_size = context_size
        self.debug = debug
        self.prompt_manager = PromptManager(prompts_dir=prompts_dir)
        self.results: List[QARecord] = []
        self.semantic_encoder = SEMANTIC_ENCODER
        self._verify_ollama_connection()

    # ------------------------------------------------------------------
    # Connectivity check
    # ------------------------------------------------------------------
    def _verify_ollama_connection(self):
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama status {resp.status_code}")
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[INFO] Connected to Ollama at {self.ollama_host}")
            print(f"[INFO] Available models: {models}")
            if not any(self.model in m for m in models):
                print(f"[WARN] Model '{self.model}' not found. Run: ollama pull {self.model}")
        except requests.exceptions.ConnectionError:
            print(f"[ERR]  Cannot connect to Ollama at {self.ollama_host}", file=sys.stderr)
            print("       Make sure Ollama is running: ollama serve", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # SSUN helpers
    # ------------------------------------------------------------------
    def _chunk_size_factor(self, chunk_words: int, answer_words: int) -> float:
        ratio = chunk_words / max(answer_words, 1)
        adj = 1.0 / (1.0 + math.log10(max(ratio, 1)) * 0.1)
        return max(0.5, min(1.0, adj))

    def _batch_compute_ssun(self, answers: List[str], chunks: List[str]) -> List[float]:
        """
        Encode all answers and chunks in two batched GPU passes, then compute
        per-record cosine similarity via a single element-wise dot product.
        For 250 records this is ~100x faster than encoding one-by-one.
        """
        if self.semantic_encoder is None:
            return [0.5] * len(answers)

        try:
            print(f"[INFO] Batch-encoding {len(answers)} answers…", file=sys.stderr)
            ans_embs = self.semantic_encoder.encode(
                answers,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            print(f"[INFO] Batch-encoding {len(chunks)} chunks…", file=sys.stderr)
            chk_embs = self.semantic_encoder.encode(
                chunks,
                batch_size=16,  # large texts — smaller batch to stay within VRAM
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            # Normalised embeddings: dot product == cosine similarity
            sims = (ans_embs * chk_embs).sum(dim=1).cpu().numpy()

            scores = []
            for i, (ans, chk) in enumerate(zip(answers, chunks)):
                factor = self._chunk_size_factor(len(chk.split()), len(ans.split()))
                scores.append(float(sims[i]) * factor)
            return scores

        except Exception as e:
            print(f"[ERR]  Batch SSUN failed: {e}", file=sys.stderr)
            return [0.5] * len(answers)

    # ------------------------------------------------------------------
    # Ollama call (uses per-thread session)
    # ------------------------------------------------------------------
    def _call_ollama(self, prompt: str, temperature: float = 0.3, timeout: int = 300) -> str:
        session = _get_session()
        try:
            resp = session.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temperature,
                    "options": {"num_ctx": self.context_size},
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return ""
        except Exception as e:
            print(f"[ERR]  Ollama call failed: {e}", file=sys.stderr)
            return ""

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> Dict[str, str]:
        result = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                result[key.strip().upper()] = val.strip()
        return result or {"STATUS": "error", "CLEANED_ANSWER": ""}

    # ------------------------------------------------------------------
    # Per-record processing (called from thread pool)
    # ------------------------------------------------------------------
    def _process_one(
        self,
        idx: int,
        question: str,
        answer: str,
        chunk: str,
        ssun_risk: float,
        total: int,
    ) -> QARecord:
        record = QARecord(question=question, answer=answer, chunk=chunk)
        record.similarity_score = ssun_risk
        word_count = len(answer.split())

        try:
            prompt = self.prompt_manager.format_prompt(
                "combined_cleaner",
                question=question,
                answer=answer,
                chunk=chunk,
                length=word_count,
                word_count=word_count,
            )
            response_text = self._call_ollama(prompt, temperature=0.3)

            if self.debug:
                print(f"[DEBUG] Record {idx} raw response:\n{response_text}\n", file=sys.stderr)

            parsed = self._parse_response(response_text)

            record.is_too_short = parsed.get("IS_TOO_SHORT", "false").lower() == "true"
            record.has_noise = parsed.get("HAS_NOISE", "false").lower() == "true"
            try:
                record.noise_percentage = float(parsed.get("NOISE_PERCENTAGE", "0.0"))
            except ValueError:
                record.noise_percentage = 0.0
            record.has_mixed_language = parsed.get("HAS_MIXED_LANGUAGE", "false").lower() == "true"

            cleaned = parsed.get("CLEANED_ANSWER", answer)
            record.cleaned_answer = str(cleaned).strip() if cleaned else answer

        except Exception as e:
            print(f"[ERR]  Record {idx} failed: {e}", file=sys.stderr)
            record.cleaned_answer = answer
            record.is_too_short = word_count < 10

        print(f"  [{idx + 1}/{total}] processed", file=sys.stderr)
        return record

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def process_csv(self, input_path: str) -> List[QARecord]:
        # --- Read CSV with auto-delimiter detection ---
        df = None
        for delim in [",", "\t", ";", "|"]:
            try:
                tmp = pd.read_csv(input_path, delimiter=delim, on_bad_lines="skip")
                tmp.columns = tmp.columns.str.lower().str.strip()
                matches = sum(
                    1 for c in tmp.columns
                    if any(x in c for x in ["soalan", "jawapan", "potongan"])
                )
                if matches >= 2:
                    df = tmp
                    print(f"[INFO] Using delimiter: {repr(delim)}")
                    break
            except Exception:
                pass

        if df is None:
            raise ValueError("Could not parse CSV with any common delimiter")

        df.columns = df.columns.str.lower().str.strip()
        print(f"[DEBUG] Columns: {list(df.columns)}", file=sys.stderr)

        col_map: Dict[str, str] = {}
        for col in df.columns:
            c = col.lower().replace(" ", "_")
            if "soalan" in c or "question" in c:
                col_map["soalan"] = col
            elif "jawapan" in c or "answer" in c:
                col_map["jawapan"] = col
            elif "potongan" in c or "chunk" in c or "text" in c:
                col_map["potongan_teks"] = col

        missing = [k for k in ("soalan", "jawapan", "potongan_teks") if k not in col_map]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

        # Drop rows where required fields are NaN
        df = df.dropna(subset=list(col_map.values())).reset_index(drop=True)

        questions = df[col_map["soalan"]].astype(str).str.strip().tolist()
        answers   = df[col_map["jawapan"]].astype(str).str.strip().tolist()
        chunks    = df[col_map["potongan_teks"]].astype(str).str.strip().tolist()
        total     = len(questions)

        print(f"[INFO] {total} records | {self.workers} Ollama workers")

        # --- Step 1: Batch SSUN (single GPU pass) ---
        ssun_risks = self._batch_compute_ssun(answers, chunks)

        # --- Step 2: Parallel Ollama calls ---
        results_map: Dict[int, QARecord] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self._process_one,
                    i, questions[i], answers[i], chunks[i], ssun_risks[i], total,
                ): i
                for i in range(total)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results_map[i] = future.result()
                except Exception as e:
                    print(f"[ERR]  Future {i} raised: {e}", file=sys.stderr)
                    results_map[i] = QARecord(
                        question=questions[i],
                        answer=answers[i],
                        chunk=chunks[i],
                        cleaned_answer=answers[i],
                        similarity_score=ssun_risks[i],
                        is_too_short=len(answers[i].split()) < 10,
                    )

        # Preserve original CSV order
        self.results = [results_map[i] for i in range(total)]
        return self.results

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_results(self, output_path: str):
        rows = [
            {
                "soalan":            r.question,
                "jawapan_original":  r.answer,
                "jawapan_cleaned":   r.cleaned_answer,
                "potongan_teks":     r.chunk,
                "similarity_score":  round(r.similarity_score, 3),
                "has_noise":         r.has_noise,
                "noise_percentage":  round(r.noise_percentage, 3),
                "is_too_short":      r.is_too_short,
                "has_mixed_language": r.has_mixed_language,
            }
            for r in self.results
        ]
        out = pd.DataFrame(rows)
        out.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n[OK] Results exported to: {output_path}")
        print(f"Total records processed: {len(rows)}")
        self._print_statistics(out)

    def _print_statistics(self, df: pd.DataFrame):
        n = len(df)
        print("\n=== STATISTICS ===")
        print(f"Records with noise:          {df['has_noise'].sum()} ({df['has_noise'].sum()/n*100:.1f}%)")
        print(f"Records too short:           {df['is_too_short'].sum()} ({df['is_too_short'].sum()/n*100:.1f}%)")
        print(f"Records with mixed language: {df['has_mixed_language'].sum()} ({df['has_mixed_language'].sum()/n*100:.1f}%)")
        print(f"Avg grounding score:         {df['similarity_score'].mean():.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QA Data Cleaner — optimised")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: <input>_cleaned.csv)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("-m", "--model", default="qwen3:8b",
                        help="Ollama model name")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent Ollama workers (set OLLAMA_NUM_PARALLEL to match)")
    parser.add_argument("--context-size", type=int, default=32768,
                        help="LLM context window (tokens). Raise if chunks are very large.")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw LLM responses to stderr")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(input_path.parent / f"{input_path.stem}_cleaned.csv")

    try:
        validator = QAValidator(
            ollama_host=args.ollama_host,
            model=args.model,
            workers=args.workers,
            context_size=args.context_size,
            debug=args.debug,
        )
        validator.process_csv(str(input_path))
        validator.export_results(output_path)
        print("\n[OK] Processing complete!")
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
