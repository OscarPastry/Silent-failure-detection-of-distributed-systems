import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def run_comparison(input_csv="processed_traces_100k.csv", output_csv="model_comparison_metrics.csv"):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Exclude identical data-leakage columns to ensure pure baseline
    leakage_cols = ['failed', 'instance_events_type', 'collections_events_type', 'event', 'time', 'collection_id', 'machine_id', 'user']
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df['failed']
    
    # Stratified split ensures exact identical ratio of failures for both
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate XGBoost weights natively
    majority_class_count = len(y_train[y_train == 0])
    minority_class_count = len(y_train[y_train == 1])
    scale_pos_weight = majority_class_count / max(minority_class_count, 1)

    print("\nTraining XGBoost...")
    start_time = time.time()
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight, random_state=42, 
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    
    xgb_preds = xgb.predict(X_test)
    xgb_metrics = precision_recall_fscore_support(y_test, xgb_preds, zero_division=0)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    print("\nTraining Random Forest baseline...")
    start_time = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    rf_preds = rf.predict(X_test)
    rf_metrics = precision_recall_fscore_support(y_test, rf_preds, zero_division=0)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start_time
    lr_preds = lr.predict(X_test)
    lr_metrics = precision_recall_fscore_support(y_test, lr_preds, zero_division=0)
    lr_acc = accuracy_score(y_test, lr_preds)

    print("\nTraining Decision Tree...")
    start_time = time.time()
    dt = DecisionTreeClassifier(class_weight='balanced', max_depth=15, random_state=42)
    dt.fit(X_train, y_train)
    dt_time = time.time() - start_time
    dt_preds = dt.predict(X_test)
    dt_metrics = precision_recall_fscore_support(y_test, dt_preds, zero_division=0)
    dt_acc = accuracy_score(y_test, dt_preds)

    print("\nTraining Support Vector Machine (Linear)...")
    start_time = time.time()
    svm = LinearSVC(class_weight='balanced', dual=False, max_iter=2000, random_state=42)
    svm.fit(X_train, y_train)
    svm_time = time.time() - start_time
    svm_preds = svm.predict(X_test)
    svm_metrics = precision_recall_fscore_support(y_test, svm_preds, zero_division=0)
    svm_acc = accuracy_score(y_test, svm_preds)

    print("\nCompiling cross-analytics...")
    
    # Structure into clean DataFrame for Streamlit to instantly ingest
    results = []
    
    for model_name, metrics, train_time, acc in [
        ('XGBoost', xgb_metrics, xgb_time, xgb_acc),
        ('Random Forest', rf_metrics, rf_time, rf_acc),
        ('Logistic Regression', lr_metrics, lr_time, lr_acc),
        ('Decision Tree', dt_metrics, dt_time, dt_acc),
        ('SVM', svm_metrics, svm_time, svm_acc)
    ]:
        results.append({
            'Model': model_name,
            'Class': 'Healthy Trace (0)',
            'Precision': float(metrics[0][0]),
            'Recall': float(metrics[1][0]),
            'F1-Score': float(metrics[2][0]),
            'Overall Accuracy': float(acc),
            'Training Time (s)': float(train_time)
        })
        results.append({
            'Model': model_name,
            'Class': 'Silent Failure (1)',
            'Precision': float(metrics[0][1]),
            'Recall': float(metrics[1][1]),
            'F1-Score': float(metrics[2][1]),
            'Overall Accuracy': float(acc),
            'Training Time (s)': float(train_time)
        })
        
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved metrics to {output_csv}")

if __name__ == "__main__":
    run_comparison()
