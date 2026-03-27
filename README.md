# 🔍 AI-Powered Silent Failure Detection

This machine learning pipeline focuses on analyzing deep execution trace data from distributed computer systems (e.g., Google Borg architecture) to predict when software tasks will silently fail. 

By analyzing deeply imbalanced metrics like CPU Allocation, Memory Scaling constraints, and Processing Priorities, the AI correlates hardware resource starvation to process termination risks.

---

## 📸 The Tech Stack
* **Language:** Python
* **Models:** `RandomForest` (Primary Engine) and `XGBoost` (Comparative Engine)
* **Frontend Analytics:** `Streamlit` and `Plotly`
* **Data Processing:** `Pandas` and `scikit-learn`

---

## 🛠️ How It Works (The Core Scripts)

The architecture is explicitly broken into 5 main components. You can run them sequentially:

### 1. Data Cleaning (`preprocess.py`)
Because raw structural traces natively log lists and unparsed dictionaries for CPU/Memory stats, this script breaks them open. It calculates `cpu_usage_distribution_mean`, `maximum_usage_memory`, etc. Crucially, we scrub high-cardinality IDs (`machine_id`, `user`) out of the feature set before sending it to the AI to prevent the model from "cheating" by memorizing User IDs instead of learning failure symptoms.
**Run:** `python preprocess.py --nrows 100000 --output processed_traces_100k.csv`

### 2. Model Training (`train.py`)
This script natively feeds the cleaned execution traces into a profoundly robust `RandomForestClassifier`. Since "Silent Failures" only happen ~20% of the time in the dataset, we inject the `class_weight='balanced'` parameter. This mathematically forces the algorithm to penalize itself much harder for missing a failure than giving a false alarm, granting us heavily optimized Recall.
**Run:** `python train.py --input processed_traces_100k.csv` *(outputs `random_forest_model.pkl`)*

### 3. Analytics Simulation / Benchmarking (`compare_models.py`)
We built a sandbox script that loads the exact same data split and pits **Random Forest** against an **XGBoost Classifier**. It evaluates their Precision, Recall, and F1-Scores. Fun fact: **Random Forest won** with a ~98.2% F1-score!
**Run:** `python compare_models.py` *(outputs `model_comparison_metrics.csv`)*

### 4. Inference Engine (`predict.py`)
This takes live incoming traces and instantly classifies them through `random_forest_model.pkl`. The magic here is the algorithm actively stitches the `machine_id` and `collection_id` back onto the final output logs organically, so your operations team can precisely locate the failing tasks!
**Run:** `python predict.py` *(outputs `predictions_enriched.csv`)*

### 5. Web Dashboard (`dashboard.py`)
A gorgeous, dark-themed local website built using `Streamlit`. It maps all of the machine learning output into human-readable, reactive graphics for Site Reliability Engineers.
**Run:** `streamlit run dashboard.py`

---

## 🚀 Installation & Local Deployment

### 1. Requirements
Ensure you have the virtual environment booted (`python -m venv .venv` and source it) and install the core pipeline architecture:
```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly
```

### 2. Start The Dashboard
Launch the pipeline to instantly visualize your trace analytics and the comparative detection algorithm stats!
```bash
streamlit run dashboard.py
```
This application will automatically bind precisely to `http://localhost:8501`.
