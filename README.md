# QA Data Cleaner & Validator

A Python tool for validating and cleaning Q&A datasets by detecting hallucinations, mixed languages, noise, and improving answer quality using **Ollama with Qwen 2.5 27B** (local LLM).

## Features

1. **Hallucination Detection** - Calculates similarity scores between answers and reference chunks to detect answers that don't align with source material
2. **Answer Length Validation** - Identifies answers below minimum threshold and paraphrases them using LLM
3. **Mixed Language Detection** - Detects and fixes mixed Bahasa Malaysia/Indonesian usage
4. **Noise Detection** - Identifies random characters, excessive punctuation, and garbled text
5. **Automatic Cleaning** - Uses Ollama to paraphrase short answers and fix language issues
6. **Comprehensive Output** - Exports both original and cleaned versions for comparison

## Benefits of Local LLM (Ollama)
✅ **No API Keys** - Works completely locally  
✅ **Privacy** - Your data never leaves your computer  
✅ **Cost-Free** - No per-token charges  
✅ **Offline** - Works without internet  
✅ **Fast** - Optimized for local hardware  

## Installation

### 1. Install Ollama
Download and install from: https://ollama.ai

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama
```bash
ollama serve
```

### 4. Pull the Model (First Time)
In another terminal:
```bash
ollama pull qwen3:8b
```

## Setup

Make sure Ollama is running before using the script:

```bash
# Terminal 1: Keep Ollama running
ollama serve

# Terminal 2: Use the script
python qa_cleaner.py your_data.csv -o output.csv
```

## Input CSV Format

Your input CSV must contain exactly these columns:

| Column | Description |
|--------|-------------|
| `soalan` | Question in Bahasa Malaysia |
| `jawapan` | Answer/Response |
| `potongan teks` | Reference chunk/text segment |

### Example:
```csv
soalan,jawapan,potongan teks
"Apa itu Python?","Python adalah bahasa programming","Python adalah bahasa pemrograman tingkat tinggi..."
```

## Usage

### Basic Usage
```bash
python qa_cleaner.py sample_data.csv
```

### With Custom Output
```bash
python qa_cleaner.py sample_data.csv -o cleaned_results.csv
```

### With Custom Model
```bash
python qa_cleaner.py sample_data.csv -m mistral:7b
```

### With Remote Ollama Server
```bash
python qa_cleaner.py sample_data.csv --ollama-host http://192.168.1.50:11434
```

### Full Example
```bash
python qa_cleaner.py data.csv -o results.csv --ollama-host http://localhost:11434 -m qwen2.5:27b
```

### Full Help
```bash
python qa_cleaner.py --help
```

## Output

The script generates a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `soalan` | Original question |
| `jawapan_original` | Original answer |
| `jawapan_cleaned` | Cleaned/improved answer |
| `potongan_teks` | Reference chunk |
| `similarity_score` | Hallucination risk score (0-1)<br>Higher = more likely hallucinated |
| `has_noise` | Whether noise was detected |
| `noise_percentage` | Percentage of noise characters |
| `is_too_short` | Whether answer was below minimum length |
| `has_mixed_language` | Whether mixed BM/Indonesian detected |
| `language_details` | Details about language detection |

## Configuration

Edit constants at the top of `qa_cleaner.py`:

```python
MIN_ANSWER_LENGTH = 20        # Minimum answer character length
SIMILARITY_THRESHOLD = 0.5    # Hallucination risk threshold
NOISE_THRESHOLD = 0.3         # Noise percentage threshold (%)
```

### Model Selection

Default: `qwen2.5:27b` (~15GB, very accurate)

Fast alternatives:
```bash
ollama pull mistral:7b        # ~4GB, faster, good quality
ollama pull neural-chat:7b    # ~4GB, good for Q&A
```

Large/Accurate:
```bash
ollama pull llama2:70b        # ~39GB, very accurate
ollama pull mistral:large     # ~26GB, accurate
```

## How It Works

### 1. Hallucination Detection
- Compares answer text with reference chunk using substring matching and similarity scoring
- Returns score 0-1 (higher = more likely hallucinated)
- Uses SequenceMatcher for text similarity

### 2. Length Validation
- Checks if answer is below `MIN_ANSWER_LENGTH` (default: 20 chars)
- If too short, Ollama is used to paraphrase and expand using chunk information

### 3. Noise Detection
- Counts non-alphanumeric characters (excluding spaces)
- Detects repeated characters (4+ consecutive same character)
- Combined noise score determines if cleaning needed

### 4. Mixed Language Detection
- Uses language markers to identify Bahasa Malaysia and Indonesian mixing
- Uses balanced marker ratios to reduce domain bias
- Ollama fixes detected language mixing

### 5. Cleaning & Paraphrasing
- Random character noise is cleaned with regex
- Short answers are expanded preserving original meaning
- Language mixing is corrected by Ollama for consistency

## Architecture

```
┌─────────────────────────┐
│   qa_cleaner.py         │
└────────────┬────────────┘
             │ HTTP requests
             ↓
┌─────────────────────────┐
│   Ollama Server         │
│ (localhost:11434)       │
└────────────┬────────────┘
             │ LLM processing
             ↓
┌─────────────────────────┐
│   Local Model           │
│   (qwen2.5:27b)         │
└─────────────────────────┘
```

## Performance

First run:
- Model loads into memory: ~5-30s
- Processing per record: ~10-30s
- Total for 100 records: 15-60 minutes (depends on hardware)

Subsequent runs:
- Faster (model cached): ~5-15s per record
- Total for 100 records: 8-30 minutes

**Machine with GPU:** 2-3x faster  
**Machine with RAM:** Important (8GB minimum, 16GB+ recommended)

## Example Output

```csv
soalan,jawapan_original,jawapan_cleaned,similarity_score,has_noise,is_too_short,has_mixed_language
"Apa itu Python?","Python adalah bahasa programming yang mudah dipelajari","Python adalah bahasa pemrograman tingkat tinggi yang diinterpretasikan, diciptakan oleh Guido van Rossum, dengan sintaks yang sederhana dan mudah dibaca.",0.782,False,False,False
```

## Troubleshooting

**"Cannot connect to Ollama"**
- Make sure `ollama serve` is running in another terminal
- Check Ollama is on the right host/port

**"Model not found"**
- Run: `ollama pull qwen2.5:27b`
- Wait for download to complete

**"Out of memory"**
- Try smaller model: `ollama pull mistral:7b`
- Close other applications
- Add more RAM

**"Request timeout"**
- Increase timeout (not configurable yet, modify source)
- Use faster model
- Wait for Ollama to finish previous requests

## Language Support

Current language-mix detection focuses on:
- **Bahasa Malaysia (BM)**
- **Indonesian**

The paraphrasing flow is now domain-agnostic (not technical-themed by default).  
Language detection markers can be adjusted by editing `BM_MARKERS` and `INDONESIAN_MARKERS` in `qa_cleaner.py`.

