import warnings
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import matplotlib.pyplot as plt

# Constants
MAX_THREADS = 8

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# === Utility Functions ===
def collect_unique_labels(
    dataset_path, chunk_size=1000, max_chunks=None, max_workers=MAX_THREADS
):
    """
    Collect unique labels from the dataset in chunks using multithreading.
    """

    def gather_labels_from_chunk(chunk):
        return set(chunk["Attack"].unique())

    unique_labels = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for chunk_idx, chunk in enumerate(
            pd.read_csv(dataset_path, chunksize=chunk_size, usecols=["Attack"])
        ):
            if max_chunks and chunk_idx >= max_chunks:
                break
            futures.append(executor.submit(gather_labels_from_chunk, chunk))
        for future in futures:
            unique_labels.update(future.result())
    return sorted(unique_labels)


def preprocess_chunk(chunk, encoder):
    """
    Preprocess a chunk: encode IPs, labels, and handle invalid values.
    """
    chunk.drop(["Dataset", "Label"], axis=1, inplace=True)
    chunk["IPV4_SRC_ADDR"] = chunk["IPV4_SRC_ADDR"].apply(
        lambda x: (
            int("".join([f"{int(octet):03}" for octet in x.split(".")]))
            if pd.notnull(x)
            else 0
        )
    )
    chunk["IPV4_DST_ADDR"] = chunk["IPV4_DST_ADDR"].apply(
        lambda x: (
            int("".join([f"{int(octet):03}" for octet in x.split(".")]))
            if pd.notnull(x)
            else 0
        )
    )
    chunk["Attack"] = encoder.transform(chunk["Attack"])
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.fillna(0, inplace=True)
    chunk = chunk.clip(lower=-1e10, upper=1e10)
    X = chunk.drop("Attack", axis=1)
    y = chunk["Attack"]
    return X, y


# === Model Training ===
def train_model(
    dataset_path, encoder, chunk_size=1000, max_chunks=None, max_workers=MAX_THREADS
):
    """
    Incrementally train an XGBoost model using chunks of the dataset.
    """
    model = XGBClassifier(eval_metric="mlogloss")
    lock = Lock()
    is_first_chunk = True
    all_classes = np.arange(len(encoder.classes_))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_idx, chunk in enumerate(
            pd.read_csv(dataset_path, chunksize=chunk_size)
        ):
            if max_chunks and chunk_idx >= max_chunks:
                break
            print(f"Processing chunk {chunk_idx + 1} for training...")
            future = executor.submit(preprocess_chunk, chunk, encoder)
            X, y = future.result()
            missing_classes = set(all_classes) - set(y.unique())
            if missing_classes:
                print(f"Handling missing classes: {missing_classes}")
                synthetic_X = pd.DataFrame(
                    np.zeros((len(missing_classes), X.shape[1])), columns=X.columns
                )
                synthetic_y = pd.Series(list(missing_classes))
                X = pd.concat([X, synthetic_X], ignore_index=True)
                y = pd.concat([y, synthetic_y], ignore_index=True)
            with lock:
                if is_first_chunk:
                    model.fit(X, y)
                    is_first_chunk = False
                else:
                    model.fit(X, y, xgb_model=model.get_booster())
    return model


# === Model Evaluation ===
def evaluate_model(
    model,
    dataset_path,
    encoder,
    chunk_size=1000,
    max_chunks=None,
    max_workers=MAX_THREADS,
):
    """
    Evaluate the trained model on unseen data and display metrics.
    """
    all_y_true = []
    all_y_pred = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_idx, chunk in enumerate(
            pd.read_csv(dataset_path, chunksize=chunk_size)
        ):
            if max_chunks and chunk_idx >= max_chunks:
                break
            print(f"Evaluating chunk {chunk_idx + 1}...")
            future = executor.submit(preprocess_chunk, chunk, encoder)
            X, y_true = future.result()
            y_pred = model.predict(X)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    # Classification report
    print("\nClassification Report:\n")
    print(classification_report(all_y_true, all_y_pred, target_names=encoder.classes_))

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        all_y_true,
        all_y_pred,
        display_labels=encoder.classes_,
        xticks_rotation="vertical",
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Accuracy
    accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"Final Accuracy: {accuracy:.4f}")


# === Main Execution ===
if __name__ == "__main__":
    dataset_path = "/mnt/c/Users/mikig/Desktop/UPC/PAE/Datasets/9810e03bba4983da_MOHANAD_A4706/9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv"
    max_chunks = 150  # For proof of concept
    chunk_size = 100
    max_threads = 8

    # Step 1: Collect unique labels
    print("Collecting unique labels...")
    all_labels = collect_unique_labels(
        dataset_path,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        max_workers=max_threads,
    )
    encoder = LabelEncoder()
    encoder.fit(all_labels)

    # Step 2: Train the model
    print("Training the model...")
    model = train_model(
        dataset_path,
        encoder,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        max_workers=max_threads,
    )

    # Step 3: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(
        model,
        dataset_path,
        encoder,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        max_workers=max_threads,
    )
