import pandas as pd
import joblib
import argparse
from preprocess import run_preprocessing

def predict_failures(model_path, input_traces, output_path, nrows=None):
    print(f"Loading raw data for predictions from {input_traces}...")
    
    # We will run the preprocessing steps directly inline to memory, and then predict
    # However run_preprocessing saves it to a file, so let's import the preprocessing dynamically
    
    # Instead, let's just use run_preprocessing to create a temporary processed file
    temp_processed = 'temp_inference_preprocessed.csv'
    run_preprocessing(input_traces, temp_processed, nrows=nrows)
    
    processed_df = pd.read_csv(temp_processed)
    
    print(f"Loading ML Model from {model_path}...")
    clf = joblib.load(model_path)
    
    # Drop target and known data leakage columns before predicting
    leakage_cols = ['failed', 'instance_events_type', 'collections_events_type', 'event', 'time', 'collection_id', 'machine_id', 'user']
    cols_to_drop_pred = [c for c in leakage_cols if c in processed_df.columns]
    X_pred = processed_df.drop(columns=cols_to_drop_pred)
        
    print("Making Predictions...")
    preds = clf.predict(X_pred)
    probs = clf.predict_proba(X_pred)[:, 1] if hasattr(clf, "predict_proba") else preds
    
    print(f"Identified {sum(preds)} potential silent failures across {len(preds)} traces.")
    
    # Outputting entirely enriched predictions so the dashboard has context
    processed_df['silent_failure_pred'] = preds
    processed_df['silent_failure_prob'] = probs
    processed_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='borg_traces_data.csv')
    parser.add_argument('--model', type=str, default='random_forest_model.pkl')
    parser.add_argument('--output', type=str, default='predictions_enriched.csv')
    parser.add_argument('--nrows', type=int, default=10000, help="Number of rows to predict on for testing")
    args = parser.parse_args()
    
    predict_failures(args.model, args.input, args.output, args.nrows)
