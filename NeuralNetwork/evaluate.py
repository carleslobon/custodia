#!/usr/bin/env python3
"""
NetFlow Classification Evaluation Script

This script loads a trained Keras model, along with the LabelEncoder and StandardScaler,
preprocesses a new CSV file, and evaluates the model on the new data.

Usage:
    python evaluate_model.py --csv_path /path/to/new_data.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# 1. CONFIGURATION
###############################################################################
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths to saved objects
MODEL_PATH = "netflow_classification_model_conditional_epochs.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"

# Columns
TARGET_COLUMN = "Attack"
DROP_COLUMNS = ["Dataset", "Label"]

# Preprocessing parameters
CLAMP_VALUE = 1e5  # Should match the training clamp value
CHUNK_SIZE = 1024  # Must match the training chunk size


###############################################################################
# 2. ARGUMENT PARSER
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NetFlow Classification Model on New CSV"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the new CSV file to evaluate.",
    )
    return parser.parse_args()


###############################################################################
# 3. PREPROCESSING FUNCTIONS
###############################################################################
def preprocess_chunk(
    chunk,
    drop_columns,
    target_column,
    scaler,
    label_encoder,
    clamp_value,
    is_train=False,
):
    """
    Preprocesses a single chunk of data:
      - Drop unnecessary columns
      - Handle object columns by hashing
      - Clip numerical values
      - Scale features
      - Encode labels
    """
    # Drop columns not needed
    if drop_columns:
        chunk.drop(
            columns=[c for c in drop_columns if c in chunk.columns],
            inplace=True,
            errors="ignore",
        )

    # Drop rows with missing target
    chunk.dropna(subset=[target_column], inplace=True)
    if chunk.empty:
        return None, None

    # Extract labels
    labels = chunk.pop(target_column)

    # Convert object columns -> numeric (hashing)
    for col in chunk.columns:
        if chunk[col].dtype == object:
            chunk[col] = chunk[col].apply(
                lambda x: hash(x) % (2**31) if pd.notna(x) else 0
            )

    # Select numeric columns
    X_chunk = chunk.select_dtypes(include=[np.number]).astype(np.float64)
    X_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_chunk = X_chunk.clip(-clamp_value, clamp_value)
    X_chunk.fillna(0, inplace=True)  # Assuming scaler was fitted with no NaNs

    # Scale features
    if scaler is not None:
        X_chunk = scaler.transform(X_chunk)
    X_chunk = X_chunk.astype(np.float32)

    # Encode labels
    y_encoded = label_encoder.transform(labels)

    return X_chunk, y_encoded


###############################################################################
# 4. EVALUATION FUNCTIONS
###############################################################################
def evaluate_model_on_csv(
    model,
    csv_path,
    scaler,
    label_encoder,
    target_column=TARGET_COLUMN,
    drop_columns=DROP_COLUMNS,
    clamp_value=CLAMP_VALUE,
    chunk_size=CHUNK_SIZE,
):
    """
    Evaluates the model on the provided CSV file and returns predictions and true labels.
    """
    print(f"[INFO] Evaluating model on '{csv_path}'...")
    all_preds = []
    all_labels = []
    all_probs = []

    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True)

    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        X, y = preprocess_chunk(
            chunk,
            drop_columns=drop_columns,
            target_column=target_column,
            scaler=scaler,
            label_encoder=label_encoder,
            clamp_value=clamp_value,
            is_train=False,
        )
        if X is None or y is None:
            print(
                f"  [WARNING] Chunk #{chunk_idx} is empty after preprocessing. Skipping."
            )
            continue

        # Predict
        preds_probs = model.predict(X, batch_size=256)
        preds_class = np.argmax(preds_probs, axis=1)

        # Collect results
        all_preds.extend(preds_class)
        all_labels.extend(y)
        all_probs.append(preds_probs)

        if chunk_idx % 10 == 0:
            print(f"  [INFO] Processed chunk #{chunk_idx}")

    # Concatenate all probabilities
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else None

    print("[INFO] Evaluation complete.")
    return np.array(all_labels), np.array(all_preds), all_probs


###############################################################################
# 5. PLOTTING FUNCTIONS
###############################################################################
def plot_confusion_matrix(cm, labels, normalize=False, log_norm=False, title_suffix=""):
    """
    Plots a confusion matrix.
      - If normalize=True, row-normalized percentages are shown.
      - If log_norm=True, uses a log scale for color mapping.
    """
    if normalize:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm = cm / (cm_sum + 1e-9)

    plt.figure(figsize=(10, 8))

    if log_norm:
        import matplotlib

        sns.heatmap(
            cm,
            annot=False,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            norm=matplotlib.colors.LogNorm(),
        )
    else:
        sns.heatmap(
            cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels
        )

    plt.title(
        ("Normalized " if normalize else "")
        + "Confusion Matrix"
        + (" (Log Scale)" if log_norm else "")
        + title_suffix
    )
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    plt.show()


def plot_f1_scores_bar(report_dict, classes):
    """
    Given a classification_report dict, plots a bar chart of F1 scores for each class.
    """
    class_f1 = []
    class_names = []

    for cls_name in classes:
        metrics = report_dict.get(cls_name, {})
        # metrics should contain 'precision', 'recall', 'f1-score', 'support'
        if "f1-score" in metrics:
            class_f1.append(metrics["f1-score"])
            class_names.append(cls_name)

    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_f1, color="skyblue")
    plt.title("F1 Score by Class")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true, y_probs, classes):
    """
    Plots One-vs-Rest ROC curves for each class (for multi-class classification).
    """
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curves(y_true, y_probs, classes):
    """
    Plots One-vs-Rest Precision-Recall curves for each class (for multi-class classification).
    """
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        plt.plot(recall, precision, label=f"{class_name}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend(loc="lower left")
    plt.show()


###############################################################################
# 6. MAIN FUNCTION
###############################################################################
def main():
    args = parse_args()
    csv_path = args.csv_path

    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
        return
    if not os.path.exists(LABEL_ENCODER_PATH):
        print(f"[ERROR] LabelEncoder file '{LABEL_ENCODER_PATH}' not found.")
        return
    if not os.path.exists(SCALER_PATH):
        print(f"[ERROR] StandardScaler file '{SCALER_PATH}' not found.")
        return
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file '{csv_path}' not found.")
        return

    # Load LabelEncoder and StandardScaler
    print("[INFO] Loading LabelEncoder and StandardScaler...")
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Loaded LabelEncoder and StandardScaler.")

    # Load Keras model
    print("[INFO] Loading Keras model...")
    model = keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    # Evaluate model on new CSV
    y_true, y_pred, y_probs = evaluate_model_on_csv(
        model=model,
        csv_path=csv_path,
        scaler=scaler,
        label_encoder=label_encoder,
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS,
        clamp_value=CLAMP_VALUE,
        chunk_size=CHUNK_SIZE,
    )

    if len(y_true) == 0:
        print("[WARNING] No data to evaluate.")
        return

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n[INFO] Confusion Matrix:\n", cm)

    # Plot raw (non-normalized) confusion matrix
    plot_confusion_matrix(
        cm, labels=label_encoder.classes_, normalize=False, title_suffix=" (Raw)"
    )
    # Plot normalized confusion matrix
    plot_confusion_matrix(
        cm, labels=label_encoder.classes_, normalize=True, title_suffix=" (Normalized)"
    )

    # Generate classification report
    print("\n[INFO] Classification Report:")
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    print(report)

    # Plot F1 scores
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    plot_f1_scores_bar(report_dict, label_encoder.classes_)

    # Plot ROC curves if probabilities are available
    if y_probs is not None:
        plot_roc_curves(y_true, y_probs, label_encoder.classes_)
        plot_precision_recall_curves(y_true, y_probs, label_encoder.classes_)

    print("[INFO] Evaluation completed successfully.")


###############################################################################
# 7. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
