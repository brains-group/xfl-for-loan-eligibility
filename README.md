# Federated Learning with Explainability for U.S. State-Level Financial Distress Modeling

This repository contains the code and resources for the research paper, "Federated Learning with Explainability for U.S. State-Level Financial Distress Modeling." [cite_start]We present a novel framework that uses cross-silo Federated Learning (FL) to predict consumer financial distress across all 50 U.S. states and the District of Columbia[cite: 23]. [cite_start]The system leverages the U.S. National Financial Capability Study (NFCS) dataset without centralizing sensitive personal data, treating each state as a distinct data silo[cite: 23, 24].

[cite_start]Our approach integrates an 8-layer Highway Network, specifically tailored for imbalanced and highly categorical survey data, with advanced explainable AI (XAI) techniques like SHAP and Owen values[cite: 25, 27, 132, 177]. [cite_start]This allows for the identification of both nationwide (global) and state-specific (local) predictors of financial hardship, providing a scalable, regulation-compliant blueprint for early warning systems in finance[cite: 25, 26, 28].

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
    cd financial-distress-fl
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
    tmux new -s fl_finance
    ```
    You can safely detach from this session (`Ctrl+b`, then `d`) and re-attach later (`tmux attach -t fl_finance`).

4.  **Install System and Python Dependencies**
    First, install necessary system libraries. Google Chrome is required by the Kaleido library for saving high-resolution static plots.
    ```bash
    # Install Google Chrome (for Kaleido)
    sudo apt-get install -y google-chrome-stable
    ```
    Next, upgrade pip and install all the required Python packages from `requirements.txt`.
    ```bash
    # Upgrade pip and install Python packages
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes key packages such as `torch`, `flwr[simulation]`, `ray`, `scikit-learn`, `matplotlib`, `pytorch-tabnet`, and `shap`.

### Acquiring the Dataset

This project uses the **2021 National Financial Capability Study (NFCS)** dataset from the FINRA Investor Education Foundation. Due to licensing, you must acquire the dataset directly from the [official FINRA website](https://www.finrafoundation.org/national-financial-capability-study). Once downloaded, place the CSV file in a local `data` directory.

---

## Codebase Organization

root_directory
├── data/                       # Dataset directory, use to contain both the raw CSV file as well as all the processed figures.
├── FLT_Loader2.py 				# Script to load CSV file and store data.
├── FLT_Server2.py 				# Script to simulate Flower-based client-server interaction, as well as model evaluation, INFO logging, and function-calling.
├── README.mb 					# this README file.
├── utils2.py 					# General utility functions, including the Highway Network, figure creation, and evaluation.

---

## How to Run the Experiments

Ensure your virtual environment is activated (`source .venv/bin/activate`) before running any commands. Scripts should be run as modules from the project root to handle imports correctly.

### 1. Run Federated Learning
Initiate the federated training simulation by calling FLT_Server2.py via python3. This will train the global model with default settings (200 rounds, 51 clients, ect). The final model parameters are not saved.

### 2. Evaluate Results
All figures and results are automatically stored into the same directory as the CSV (which by default is assumed to be data\)