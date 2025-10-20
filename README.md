# 🧠 R-Debater: Retrieval-Augmented Multi-Agent Debate Generation System

R-Debater is a retrieval-augmented debate generation framework designed for research on **computational argumentation**, **dialogue reasoning**, and **LLM-based debate modeling**.  
It integrates retrieval, logic summarization, and generation agents into a coherent pipeline for producing high-quality debate speeches and argument-level analysis.

---

## 📦 Project Structure

> ⚠️ **Important:**  
> This repository **must be cloned or placed under** the following absolute path for the scripts to run correctly:
> ```
> D:\conversational_rag
> ```

### Core Modules

| Folder | Description |
|:--------|:-------------|
| `rag_for_longchain/retriever/` | Argumentative memory retrieval pipelines |
| `rag_for_longchain/generator/` | LLM-based counterargument and rebuttal generation |
| `rag_for_longchain/utils/agents/` | Logic Summarization and Debate Optimization Agents |
| `rag_for_longchain/data/` | Processed debate transcripts and experimental data |
| `rag_for_longchain/config/` | Configuration files, including API keys and model settings |

---

## ⚙️ Environment Setup

```bash
conda create -n Conrag python=3.12
conda activate Conrag
pip install -r requirements.txt
Ensure you have the following dependencies:
openai,langchain,pyyaml,numpy, pandas,tqdm, rich,faiss-cpu (for retrieval experiments)
```



## 🔑 Configuration
All API and model-related settings are stored in:
```arduino
D:\conversational_rag\rag_for_longchain\config\config.yaml
```

Example structure:
```yaml
openai:
  api_key: ""
  api_base: ""
  model_name:""
```

## 🚀 Run the Debate System
Example (batch debate generation):
```bash
python -m rag_for_longchain.demo_test \
    --input_dir "D:\conversational_rag\rag_for_longchain\data\processed_input" \
    --output_file "D:\conversational_rag\rag_for_longchain\data\output\debate_output.json" \
    --side pro
```
Supported arguments:
--input_dir: directory containing processed debate transcripts
--output_file: output file path
--side: choose "pro" or "con" for the desired stancede.


🧩 Research Context

This repository accompanies ongoing research on Retrieval-Augmented Argument Generation (RAG) and multi-agent debate modeling, aiming to enhance the logical, factual, and persuasive quality of generated arguments.
The system is designed for reproducibility and clarity to support  submissions.




