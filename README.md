# Federated Learning with Explainability for U.S. State-Level Financial Distress Modeling

> **📄 Research Paper:** [Federated Learning with Explainability for U.S. State-Level Financial Distress Modeling](XAI_FIN25_FINRA_FL.pdf)

This repository contains the code and resources for the research paper, "Federated Learning with Explainability for U.S. State-Level Financial Distress Modeling." [cite_start]We present 3 frameworks that uses cross-silo Federated Learning (FL), Centralized Learning, and Local modeling to predict consumer financial distress across all 50 U.S. states and the District of Columbia[cite: 23]. [cite_start]The system leverages the U.S. National Financial Capability Study (NFCS) dataset without centralizing sensitive personal data, treating each state as a distinct data silo[cite: 23, 24].

[cite_start]Our approach integrates an 8-layer Highway Network, specifically tailored for imbalanced and highly categorical survey data, with advanced explainable AI (XAI) techniques like SHAP and Owen values. This allows for the identification of both nationwide (global) and state-specific (local) predictors of financial hardship, providing a scalable, regulation-compliant blueprint for early warning systems in finance.

---

## Setting up the Environment

### Prerequisites
- Python 3.9+
- Git
- tmux (for managing long-running training sessions)

### Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    ```

2.  **Create and Activate Virtual Environment**
    From the root of the project folder, create and activate a Python virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Start a `tmux` Session (Recommended)**
    For long training runs, it's best to use a terminal multiplexer like `tmux`.
    ```bash
    tmux new -s dca
    ```
    You can safely detach from this session (`Ctrl+b`, then `d`) and re-attach later (`tmux attach -t fl_finance`).

4.  **Install Dependencies**
    Upgrade pip and install all required Python packages:
    ```bash
    pip install --upgrade pip
    pip install torch flwr[simulation] ray scikit-learn matplotlib pytorch-tabnet shap
    ```

### Acquiring the Dataset

This project uses the **2021 National Financial Capability Study (NFCS)** dataset from the FINRA Investor Education Foundation. Due to licensing, you must acquire the dataset directly from the [official FINRA website](https://www.finrafoundation.org/national-financial-capability-study). Once downloaded, place the CSV file in a local `data` directory.

---

## Codebase Organization

The project structure is organized as follows:

```
.
├── data/                          # Dataset directory (contains raw CSV and processed figures)
│
├── FLT_Loader2.py                 # Data loader for Federated Learning model
├── FLT_LoaderC.py                 # Data loader for Centralized Learning model
├── FLT_LoaderC2.py                # Data loader for Local Learning model
│
├── FLT_Server2.py                 # Federated Learning server (Flower-based simulation)
│                                   # Handles client-server interaction, model evaluation,
│                                   # INFO logging, and function-calling
├── FLT_ServerC.py                 # Centralized Learning server
├── FLT_ServerC2.py                # Local Learning server
│
├── utils2.py                      # Utility functions for FL model
│                                   # Includes Highway Network, figure creation, and evaluation
├── utilsC.py                      # Utility functions for Centralized model
│                                   # Simplified version with updated attributes
├── utilsC2.py                     # Utility functions for Local model
│                                   # Simplified version with updated attributes
│
├── XAI_FIN25_FINRA_FL.pdf         # Research paper
├── README.md                      # This file
└── LICENSE                        # License file
```

### File Descriptions

**Data Loaders:**
- `FLT_Loader2.py` - Loads and prepares data for federated learning training
- `FLT_LoaderC.py` - Loads and prepares data for centralized learning training
- `FLT_LoaderC2.py` - Loads and prepares data for local learning training

**Server Scripts:**
- `FLT_Server2.py` - Simulates federated learning with Flower framework
- `FLT_ServerC.py` - Simulates centralized learning approach
- `FLT_ServerC2.py` - Simulates localized learning approach

**Utilities:**
- `utils2.py` - Core utilities for FL model (Highway Network, evaluation, visualization)
- `utilsC.py` - Utilities for centralized model (simplified)
- `utilsC2.py` - Utilities for local model (simplified)

---

## How to Run the Experiments

Ensure your virtual environment is activated (`source .venv/bin/activate`) before running any commands. Scripts should be run as modules from the project root to handle imports correctly.

### 1. Run Federated Learning
Initiate the federated training simulation by calling `FLT_Server2.py` via python3. This will train the global model with default settings (200 rounds, 51 clients, etc.). The final model parameters are not saved. The same can be done for `FLT_ServerC.py` and `FLT_ServerC2.py` to train their respective models.

```bash
python3 FLT_Server2.py    # Federated Learning
python3 FLT_ServerC.py     # Centralized Learning
python3 FLT_ServerC2.py   # Local Learning
```

### 2. Evaluate Results
All figures and results are automatically stored in the `data/` directory (same location as the CSV file). To view or change result file names, check the respective utility files (`utils2.py`, `utilsC.py`, `utilsC2.py`).