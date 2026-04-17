# QA Data Cleaner & Validator

A Python tool for validating and cleaning Q&A datasets by detecting hallucinations, mixed languages, noise, and improving answer quality using **Ollama with Qwen 3 8B** (local LLM).

## Features

1. **Semantic Hallucination Detection (SSUN Algorithm)** — Uses Sentence Transformers embeddings to detect semantic gaps between answers and reference chunks
2. **Answer Length Validation** — Identifies answers below the minimum threshold and rewrites them using the LLM
3. **Mixed Language Detection** — Detects and fixes improper Bahasa Malaysia / non-Malay mixing
4. **Noise Detection** — Identifies random characters, excessive punctuation, and garbled text
5. **Automatic Cleaning** — Single combined LLM prompt handles validation and cleaning in one pass
6. **External Prompt Management** — Centralised prompt configuration via `prompts/config.yaml`
7. **Parallel Processing** — ThreadPoolExecutor + batch embeddings for fast throughput on modern hardware

## Project Structure

```
qa_cleaner.py          Main validation and cleaning script
prompt_manager.py      Loads and caches LLM prompts
prompts/
  config.yaml          Maps prompt names to files
  combined_cleaner.txt Combined validation + cleaning prompt (used by default)
  validate_all.txt     Standalone validation prompt
  unified_qa_cleaner.txt  Standalone cleaning prompt
  examples/            Reference templates for customisation
requirements.txt       Python dependencies
```

## Installation

### 1. Install Ollama
Download from https://ollama.ai and install.

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the model
```bash
ollama pull qwen3:8b
```

### 4. Run
Ollama starts automatically on Windows. If it is not running, start it:
```bash
ollama serve
```
If you see `bind: Only one usage of each socket address` it is already running — proceed directly to step 5.

### 5. For maximum throughput, set parallel workers before starting Ollama (PowerShell)
```powershell
$env:OLLAMA_NUM_PARALLEL = "4"
ollama serve
```

## Usage

### Basic
```bash
python qa_cleaner.py input.csv
```

### With options
```bash
python qa_cleaner.py input.csv -o cleaned.csv --workers 4 --model qwen3:8b
```

### All flags
| Flag | Default | Description |
|------|---------|-------------|
| `input_file` | — | Input CSV path (required) |
| `-o / --output` | `<input>_cleaned.csv` | Output CSV path |
| `-m / --model` | `qwen3:8b` | Ollama model name |
| `--ollama-host` | `http://localhost:11434` | Ollama server URL |
| `--workers` | `4` | Concurrent Ollama requests (match `OLLAMA_NUM_PARALLEL`) |
| `--context-size` | `32768` | LLM context window in tokens |
| `--debug` | off | Print raw LLM responses to stderr |

### Python API
```python
from qa_cleaner import QAValidator

validator = QAValidator(
    ollama_host="http://localhost:11434",
    model="qwen3:8b",
    workers=4,
    context_size=32768,
)

results = validator.process_csv("input.csv")
validator.export_results("output.csv")

for r in results:
    print(f"Grounding score: {r.similarity_score:.2f}")  # 1.0 = fully grounded
    print(f"Has noise:       {r.has_noise}")
    print(f"Cleaned answer:  {r.cleaned_answer}")
```

## Input CSV Format

| Column | Description |
|--------|-------------|
| `soalan` | Question in Bahasa Malaysia |
| `jawapan` | Answer / response |
| `potongan_teks` | Reference source chunk |

Column names are matched flexibly (case-insensitive, partial match).

## Output CSV Format

| Column | Description |
|--------|-------------|
| `soalan` | Original question |
| `jawapan_original` | Original answer |
| `jawapan_cleaned` | Cleaned / improved answer |
| `potongan_teks` | Source chunk |
| `similarity_score` | SSUN grounding score (see below) |
| `has_noise` | Noise detected |
| `noise_percentage` | Fraction of answer that is noise |
| `is_too_short` | Answer below 10-word threshold |
| `has_mixed_language` | Improper language mixing detected |

## How It Works

### SSUN Grounding Score

The `similarity_score` column measures how well the answer is **grounded in the source chunk** on a `0.0 – 1.0` scale.

| Score | Meaning |
|-------|---------|
| `0.85 – 1.00` | Strongly grounded — answer is semantically consistent with the chunk |
| `0.65 – 0.84` | Well grounded — minor divergence, likely safe |
| `0.45 – 0.64` | Moderate — answer may paraphrase beyond what the chunk supports |
| `0.00 – 0.44` | Weakly grounded — answer is likely hallucinated or off-topic |

> **Note for this dataset:** Chunks are 4000–8000 words and answers are 20–50 words. The chunk-size adjustment factor (≈ 0.64–0.70) is applied automatically, so expected grounding scores for valid answers typically fall in the **0.45–0.70** range. Scores below **0.35** are a meaningful red flag.

**Algorithm steps:**

```
1. Encode all answers and chunks as 384-dim embeddings (batched, GPU-accelerated)
2. Compute cosine similarity for each answer–chunk pair
3. Apply chunk-size adjustment factor:
      factor = 1 / (1 + log10(chunk_words / answer_words) × 0.1)   clamped [0.5, 1.0]
4. grounding_score = raw_cosine_similarity × factor
```

The chunk-size adjustment compensates for embedding dilution in large chunks — without it, a 5000-word chunk would appear superficially similar to almost any answer.

### Processing Pipeline

```
Input CSV
  │
  ├─ Batch-encode all answers + chunks (single GPU pass)
  ├─ Compute all grounding scores at once
  │
  └─ ThreadPoolExecutor (N workers)
       └─ Per record: one combined Ollama call
            → validation flags (noise, length, language)
            → cleaned answer
  │
Output CSV
```

## Performance

Optimised for AMD Ryzen AI Max+ 395 / Radeon 8060S / 64 GB RAM, but benefits any modern machine.

| Step | Time (250 records) |
|------|--------------------|
| Batch SSUN encoding (GPU) | ~5–15 s |
| Parallel Ollama calls (`--workers 4`) | ~10–20 min |
| **Total** | **~10–20 min** |

Compared to the original sequential approach (~60–125 min for 250 records).

**Tips:**
- Increase `--workers` and `OLLAMA_NUM_PARALLEL` together for more throughput
- The Radeon 8060S uses unified memory, so Ollama can allocate more VRAM without a hard limit

## Troubleshooting

**"Cannot connect to Ollama"**  
→ Run `ollama serve` in a terminal, or check that Ollama is installed

**"Only one usage of each socket address"**  
→ Ollama is already running — this is fine, just run the script

**"Model not found"**  
→ Run `ollama pull qwen3:8b`

**"Out of memory"**  
→ Use a smaller model (`qwen3:4b`) or reduce `--workers`

**Slow processing**  
→ Make sure `OLLAMA_NUM_PARALLEL` matches `--workers` before starting Ollama  
→ Ensure the Radeon iGPU is being used by Ollama (check `ollama ps`)

## Benefits of Local LLM

- No API keys or costs
- Data stays on your machine
- Works offline
- Full control over the model and prompts
