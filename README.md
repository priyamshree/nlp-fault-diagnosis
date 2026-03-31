---
title: NLP Fault Diagnosis System
emoji: ⚙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
---

# NLP Fault Diagnosis System

**Course:** 21CSE356T — Natural Language Processing  
**Institution:** SRM Institute of Science and Technology  
**Department:** Computational Intelligence  

---

## What This Project Does

Industrial machines generate free-text maintenance logs written by human
operators. This system reads those logs, extracts fault signals using NLP,
and diagnoses the root cause using A\* search over a fault state graph —
running entirely without cloud or GPU dependency.

---

## The Problem Statement

> *A machine fails at 2 AM at a remote site. No internet. No GPU. No expert
> on-site. Only a typed maintenance log and a small embedded chip. Can the
> system read that text and diagnose the fault in under 2 seconds?*

Current NLP tools (OpenIE, REBEL) require cloud GPUs and cannot run on
edge hardware. Standard NLP metrics (F1, BERTScore) measure linguistic
quality — not whether the diagnosis is actually correct. This project
addresses both gaps.

---

## Research Gaps Addressed

**Gap 1 — Architecture Void**  
No existing pipeline converts natural language fault descriptions into
edge-runnable state representations. This project builds that pipeline
using lightweight spaCy PhraseMatcher instead of heavy transformer models.

**Gap 2 — Metric Mismatch**  
NLP quality metrics do not measure diagnostic accuracy. This project
demonstrates the gap by running two methods side by side — A\* search
(causal reasoning) and TF-IDF + Logistic Regression (pattern matching)
— and showing where they agree and disagree.

---

## Pipeline — 5 Stages
```
Raw Maintenance Log
       ↓
NLP Signal Extraction       (spaCy PhraseMatcher — 7 fault signals)
       ↓
Fault State Graph           (NetworkX — 13 nodes, 16 edges)
       ↓
A* Search                   (heuristic traversal → root cause)
       ↓
Explainable Diagnosis       (fault + confidence + evidence trail)
```

Runs in parallel with a **TF-IDF + Logistic Regression classifier**
(98.65% accuracy) for comparison.

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
Source: UCI Machine Learning Repository  
- 10,000 rows, 5 failure types  
- Sensor columns: air temperature, process temperature, rotational speed,
  torque, tool wear  
- Numeric sensor values converted to maintenance log text via
  template-based generation  

Failure types: Tool Wear Failure · Heat Dissipation Failure ·
Power Failure · Overstrain Failure · Random Failure

---

## How to Use

1. Select a sample from the dropdown **or** type your own maintenance log
2. Click **Run Diagnosis**
3. See the full reasoning chain — detected signals, A\* search path,
   ML classifier probabilities, and engineer recommendation

---

## Tech Stack

| Component | Library |
|---|---|
| NLP Extraction | spaCy 3.8 |
| Fault Graph | NetworkX |
| A\* Search | NetworkX `astar_path` |
| ML Classifier | scikit-learn (TF-IDF + Logistic Regression) |
| Web UI | Gradio |
| Dataset | pandas |

---

## Project Structure
```
├── app.py                  # Gradio UI — entry point
├── src/
│   ├── text_generator.py   # Sensor data → maintenance log text
│   ├── nlp_extractor.py    # spaCy signal extraction
│   ├── fault_graph.py      # Fault state graph + A* search
│   └── classifier.py       # TF-IDF + Logistic Regression
├── data/
│   └── ai4i2020.csv        # AI4I 2020 dataset
└── requirements.txt
```

---

## Faculty

* **[Dr. Kanmani P](https://www.srmist.edu.in/faculty/dr-p-kanmani/)**
  *Department of Data Science and Business Systems* 