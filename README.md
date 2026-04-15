# QA Data Cleaner & Validator

A Python tool for validating and cleaning Q&A datasets by detecting hallucinations, mixed languages, noise, and improving answer quality using **Ollama with Qwen 2.5 27B** (local LLM).

## Features

1. **Semantic Hallucination Detection (SSUN Algorithm)** - Uses Sentence Transformers embeddings to detect semantic gaps between answers and reference chunks (not just string matching)
2. **Answer Length Validation** - Identifies answers below minimum threshold and paraphrases them using Ollama LLM
3. **Mixed Language Detection** - Detects and fixes mixed Bahasa Malaysia/Indonesian usage
4. **Noise Detection** - Identifies random characters, excessive punctuation, and garbled text
5. **Automatic Cleaning** - Uses Ollama unified prompts to paraphrase short answers and fix language issues
6. **External Prompt Management** - Centralized prompt configuration via `prompts/config.yaml` for easy customization
7. **Comprehensive Output** - Exports both original and cleaned versions for comparison

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
This includes:
- `sentence-transformers` (for SSUN semantic similarity)
- `pandas` (for CSV handling)
- `pyyaml` (for prompt configuration)
- `requests` (for Ollama API calls)

### 3. Start Ollama
```bash
ollama serve
```

### 4. Pull the Model (First Time)
In another terminal:
```bash
ollama pull qwen3:8b
```
Note: You can use other models like `qwen2.5:7b`, `mistral:7b`, or `qwen3:8b` depending on your hardware

## Setup

### Before Running

1. **Start Ollama** (in one terminal):
   ```bash
   ollama serve
   ```

2. **Configure Prompts** (optional):
   - Edit `prompts/config.yaml` to customize prompt files
   - Modify prompt files in `prompts/` folder for custom validation behavior
   - See `PROMPTS_STRUCTURE.txt` for prompt file locations

3. **Run the QA Cleaner** (in another terminal):
   ```bash
   python qa_cleaner.py your_data.csv -o output.csv
   ```

### How Prompts Work

The tool uses a **Prompt Manager** system that centralizes all LLM prompts:
- **config.yaml** - Lists which prompt files to use
- **unified_qa_cleaner.txt** - Main prompt for all validation tasks
- **examples/** - Reference prompt templates you can copy and customize

## How It Works

### SSUN Algorithm (Semantic Similarity)

Instead of basic string matching, the tool uses **SSUN (Semantic Similarity Understanding Network)** to understand meaning:

```
1. Convert answer & chunk to semantic embeddings
2. Calculate cosine similarity between embeddings  
3. Return hallucination risk (inverse of similarity)
```

**Example:**
- Question: "What's equivalent to list() in Python?"
- Answer: "You can use [] brackets"
- Chunk: "Python lists use square brackets []"
- **Result:** ✓ High semantic match (~95%), even though words differ

This handles paraphrases, synonyms, and meaning-based matches that simple string comparison would miss.

See [SSUN_ALGORITHM.md](SSUN_ALGORITHM.md) for technical details.

### Processing Steps

1. **Load Prompts** - Loads unified validation prompt from `prompts/config.yaml`
2. **Semantic Analysis** - Encodes answer + chunk to embeddings
3. **Validation Checks** - Runs hallucination, noise, language, and length checks
4. **LLM Cleaning** - Uses Ollama to fix issues (paraphrase, language fixes)
5. **Export Results** - Saves cleaned data alongside original

## Input CSV Format

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

### With Custom Prompts Directory
```bash
python qa_cleaner.py sample_data.csv --prompts-dir ./custom_prompts
```

### Full Example
```bash
python qa_cleaner.py data.csv -o results.csv --ollama-host http://localhost:11434 -m qwen2.5:27b
```

### Python API Usage
```python
from qa_cleaner import QAValidator

# Initialize with custom settings
validator = QAValidator(
    prompts_dir="prompts",
    ollama_host="http://localhost:11434",
    model="qwen2.5:27b"
)

# Process single record
record = validator.process_record(
    question="Sample question",
    answer="Sample answer",
    chunk="Reference chunk"
)

# Access results
print(f"Hallucination Risk: {record.similarity_score:.2%}")
print(f"Has Noise: {record.has_noise}")
print(f"Cleaned Answer: {record.cleaned_answer}")

# Process CSV file
results = validator.process_csv("data.csv")
validator.export_results("output.csv")
```

### Full Help
```bash
python qa_cleaner.py --help
```

## Prompt Management System

### Overview

The tool uses a centralized **Prompt Manager** (`prompt_manager.py`) to load and manage all LLM prompts from external files. This makes it easy to customize validation behavior without changing code.

### Configuration

**File: `prompts/config.yaml`**
```yaml
prompts:
  qa_cleaner: unified_qa_cleaner.txt
  # Add more prompts as needed
```

### Prompt Files

- **Location:** `prompts/` folder
- **unified_qa_cleaner.txt** - Main prompt used for all QA validation and cleaning
- **examples/** - Reference templates you can copy to customize

### Customizing Prompts

1. **Edit existing prompts:**
   ```bash
   # Edit the main validation prompt
   nano prompts/unified_qa_cleaner.txt
   ```

2. **Use example templates:**
   ```bash
   # Copy an example and customize it
   cp prompts/examples/hallucination_strict.txt prompts/hallucination_custom.txt
   nano prompts/hallucination_custom.txt
   
   # Update config.yaml to use it
   ```

3. **Change prompt in code:**
   ```python
   validator = QAValidator(prompts_dir="prompts")
   # Now uses prompts/config.yaml for prompt configuration
   ```

For details on prompt file structure, see [PROMPTS_STRUCTURE.txt](PROMPTS_STRUCTURE.txt)

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

