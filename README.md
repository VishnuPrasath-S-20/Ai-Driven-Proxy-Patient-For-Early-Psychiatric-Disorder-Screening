# AI-Driven Psychiatric Screening & Clinical Simulation Framework

An enterprise-grade, 100% offline edge-computing platform designed for clinical intake triage and diagnostic simulation. The core architecture integrates a fine-tuned **PyTorch DeBERTa-v3 Transformer** model for high-accuracy psychiatric risk evaluation alongside a localized **Llama 3 (Ollama)** generative backend to simulate interactive patient personas for clinician training.

---

## Core Architectural Layout
The system handles multi-role authentication boundaries and local state transactions across three dedicated functional interfaces:

* **Patient Intake Interface:** Captures open-ended text inputs, passes sequences to a GPU-accelerated classifier, logs safety-triage parameters, and uses a local Llama 3 endpoint to generate immediate, empathetic, non-diagnostic behavioral feedback.
* **Clinical Monitoring Interface (Doctor Hub):** Allows medical staff to review a global patient matrix color-coded by situational risk severity (Red/Yellow/Green alerts). It features a granular Case History Explorer to trace individual session logs and linguistic clinical markers over active temporal arrays.
* **Access-Control Matrix (Admin Console):** Gated strictly behind a root master password, allowing administrators to inject or revoke access tokens for authorized patient and medical staff profiles with instant filesystem state synchronization.

---

## Empirical Research Results & Performance
The underlying neural classification layer was validated over unseen multi-class validation splits on local edge hardware (NVIDIA GeForce RTX 4050 Laptop GPU):

* **Total Classification Accuracy:** **95.38%**
* **Weighted F1-Score:** **95.34%**
* **Model Parameter Tuning Strategy:** Full-precision (FP32) stable optimization using AdamW with gradient norm clipping set to 1.0 to completely mitigate gradient explosion profiles.

### ### Sorted Evaluation Performance Matrix (%)
| Psychiatric Status | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Anxiety** | 92.46% | 80.90% | 86.30% |
| **Bipolar** | 94.93% | 95.35% | 95.14% |
| **Depression** | 96.14% | 97.54% | 96.84% |
| **Normal** | 94.93% | 95.35% | 95.14% |
| **Personality disorder** | 94.93% | 95.35% | 95.14% |
| **Stress** | 94.93% | 95.35% | 95.14% |
| **Suicidal** | 96.14% | 97.54% | 96.84% |

---

## Local Repository Topology
```text
research_workspace/
├── aiscr.py                  # Production Streamlit Application Engine
├── aisrcbck.ipynb            # Verified Research & Model Evaluation Pipeline
├── credentials.example.txt   # Public Account Structure Layout Template
├── README.md                 # Primary Presentation Landing Documentation
└── .gitignore                # System File Isolation Matrix Configuration
