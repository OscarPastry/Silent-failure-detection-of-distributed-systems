import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
import argparse

def train_model(data_path, model_output):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if 'failed' not in df.columns:
        raise ValueError("Target column 'failed' not found in dataset")

    # The target might be continuous or have NaNs, ensure it's clean binary (0 or 1)
    df = df.dropna(subset=['failed'])
    df['failed'] = df['failed'].astype(int)

    # Some events might be labeled 'failed' if they are 1. We consider 1 to be the rare 'failure' class.
    # Check class distribution
    class_counts = df['failed'].value_counts()
    print("Class distribution:")
    print(class_counts)

    if len(class_counts) < 2:
        print("Warning: Only one class present in the data. Model cannot learn properly.")

    majority_class_count = class_counts.get(0, 1)
    minority_class_count = class_counts.get(1, 1)
    
    # Calculate scale factor for unbalanced dataset handling
    scale_pos_weight = majority_class_count / max(minority_class_count, 1)

    # Drop the target and any columns that cause data leakage (e.g. event ids)
    leakage_cols = ['failed', 'instance_events_type', 'collections_events_type', 'event', 'time', 'collection_id', 'machine_id', 'user']
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['failed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if minority_class_count > 1 else None
    )

    print(f"Training XGBoost classifier with scale_pos_weight={scale_pos_weight:.2f}")
    # Initialize XGBClassifier
    clf = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        scale_pos_weight=scale_pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Saving model to {model_output}...")
    joblib.dump(clf, model_output)
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='processed_traces.csv')
    parser.add_argument('--output', type=str, default='xgboost_model.pkl')
    args = parser.parse_args()
    
    train_model(args.input, args.output)
