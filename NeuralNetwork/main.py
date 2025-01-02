#!/usr/bin/env python3
"""
NetFlow classification on a large CSV (>12GB) with:
  - Full label encoding (collect_all_labels) to avoid unseen classes.
  - Incremental partial-fit StandardScaler to avoid OOM.
  - Dynamically counting CSV chunks to split into train/val by chunk index.
  - Class weighting for imbalanced classes.
  - .repeat(EPOCHS) to ensure multiple epochs (if EPOCHS != 0).
  - If EPOCHS=0, we use a large max epoch + early stopping (one pass per epoch).
  - Otherwise, we train exactly EPOCHS times.

No debugging step overrides, so each epoch sees the entire chunk range.
"""

from hmac import new
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)
from tensorflow.keras import mixed_precision

###############################################################################
# 1. CONFIGURATION
###############################################################################
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = (
    "/mnt/c/Users/mikig/Desktop/UPC/PAE/Datasets/"
    "9810e03bba4983da_MOHANAD_A4706/"
    "9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv"
)

TARGET_COLUMN = "Attack"
DROP_COLUMNS = ["Dataset", "Label"]

# If EPOCHS=0 => use large epochs + early stopping
# If EPOCHS!=0 => train for exactly EPOCHS
EPOCHS = 0  # Toggle here. 0 => Early stopping, >0 => fixed epochs.

SCALER_CHUNKSIZE = 100_000
MAX_SCALER_CHUNKS = 5

CHUNK_SIZE = 1024  # Good default chunk size for memory/performance
TRAIN_SPLIT_RATIO = 0.8
CLAMP_VALUE = 1e5  # Reduced from 1e9 to prevent numerical instability
CHUNK_SHUFFLE_BUFFER = 10


###############################################################################
# 2. LABEL & CLASS-DISTRIBUTION UTILITIES
###############################################################################
def collect_all_labels(csv_path, target_column=TARGET_COLUMN, chunksize=200000):
    """
    Reads only the target column in chunks. Gathers every unique label in a set.
    Ensures no unseen labels at training time.
    """
    print(f"[INFO] Collecting all unique labels from '{target_column}'...")
    label_set = set()
    chunk_idx = 0
    for chunk in pd.read_csv(csv_path, usecols=[target_column], chunksize=chunksize):
        chunk_idx += 1
        chunk.dropna(subset=[target_column], inplace=True)
        label_set.update(chunk[target_column].unique())
        if chunk_idx % 10 == 0:
            print(
                f"  Processed {chunk_idx} label-only chunks... (unique labels so far={len(label_set)})"
            )
    all_labels = sorted(label_set)
    print(f"[INFO] Found {len(all_labels)} unique labels total.")
    return all_labels


def compute_class_distribution(csv_path, target_col=TARGET_COLUMN, chunksize=100000):
    """
    Reads only 'target_col' in chunks to compute frequency counts for class weighting.
    """
    print("[INFO] Computing class distribution (for class weights)...")
    from collections import Counter

    counter = Counter()
    cidx = 0
    for chunk in pd.read_csv(csv_path, usecols=[target_col], chunksize=chunksize):
        cidx += 1
        chunk.dropna(subset=[target_col], inplace=True)
        counter.update(chunk[target_col].values)
        if cidx % 10 == 0:
            print(f"  Processed {cidx} distribution chunks.")
    return counter


def make_class_weights(label_encoder, class_counter):
    """
    Creates a dictionary for Keras 'class_weight' from the frequency counts.
    Typically: weight = total_samples / (num_classes * class_count).
    Capped to prevent excessively large weights.
    """
    total_samples = sum(class_counter.values())
    n_classes = len(label_encoder.classes_)
    cw = {}
    for i, cls_label in enumerate(label_encoder.classes_):
        cnt = class_counter.get(cls_label, 0)
        if cnt == 0:
            cw[i] = 1.0  # Assign a default weight if class count is zero
        else:
            cw[i] = total_samples / (n_classes * cnt)
            cw[i] = min(cw[i], 10.0)  # Cap the maximum class weight
    return cw


###############################################################################
# 3. INCREMENTAL PARTIAL-FIT SCALER
###############################################################################
def partial_fit_scaler(
    csv_path,
    target_column=TARGET_COLUMN,
    drop_columns=DROP_COLUMNS,
    clamp_value=CLAMP_VALUE,
    scaler_chunksize=SCALER_CHUNKSIZE,
    max_chunks=MAX_SCALER_CHUNKS,
):
    """
    Reads the CSV in moderate chunks to partial_fit a StandardScaler
    without loading entire dataset into memory.
    """
    print("[INFO] Incrementally partial-fitting a StandardScaler...")
    scaler = StandardScaler()
    reader = pd.read_csv(csv_path, chunksize=scaler_chunksize, low_memory=True)

    chunk_index = 0
    for chunk in reader:
        chunk_index += 1
        if chunk_index > max_chunks:
            break

        if drop_columns:
            chunk.drop(
                columns=[c for c in drop_columns if c in chunk.columns],
                inplace=True,
                errors="ignore",
            )
        if target_column in chunk.columns:
            chunk.drop(columns=[target_column], inplace=True, errors="ignore")

        # Convert object columns => numeric by hashing strings
        for col in chunk.columns:
            if chunk[col].dtype == object:
                chunk[col] = chunk[col].apply(
                    lambda x: hash(x) % (2**31) if pd.notna(x) else 0
                )

        numeric_chunk = chunk.select_dtypes(include=[np.number]).astype(np.float64)
        numeric_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_chunk = numeric_chunk.clip(-clamp_value, clamp_value)
        # Impute NaN values with column mean
        numeric_chunk.fillna(numeric_chunk.mean(), inplace=True)

        # Add assertions to ensure no NaN or Inf remain
        assert (
            not numeric_chunk.isna().any().any()
        ), f"NaN detected in chunk #{chunk_index}"
        assert not np.isinf(
            numeric_chunk.values
        ).any(), f"Inf detected in chunk #{chunk_index}"

        scaler.partial_fit(numeric_chunk)
        print(
            f"  [DEBUG] partial_fit chunk #{chunk_index}, shape={numeric_chunk.shape}"
        )

    print("[INFO] Done partial-fitting the StandardScaler.")
    return scaler


###############################################################################
# 4. COUNT CHUNKS (TRAIN/VAL SPLIT)
###############################################################################
def count_chunks(csv_path, chunk_size=CHUNK_SIZE, train_ratio=TRAIN_SPLIT_RATIO):
    """
    Counts the total number of chunks in the CSV by reading in chunk_size.
    Then splits them by ratio for train/val.
    """
    # Define total number of samples (you can adjust this if known)
    total_samples = 75987976  # Replace with your actual total samples if different

    print(f"[INFO] Counting how many chunks of size={chunk_size} in dataset...")

    # Calculate the number of training and validation samples
    train_samples = int(total_samples * train_ratio)
    validation_samples = total_samples - train_samples
    total_chunks = total_samples // chunk_size

    print(f"[INFO] Found {total_chunks} total chunks.")

    # Calculate number of training and validation steps
    train_steps = round((train_samples / chunk_size), 0)
    train_chunks = int(
        train_steps if train_samples / chunk_size < train_steps else train_steps + 1
    )
    validation_steps = round((validation_samples / chunk_size), 0)
    val_chunks = int(
        validation_steps
        if validation_samples / chunk_size < validation_steps
        else validation_steps + 1
    )

    print(f"  => train_chunks={train_chunks}, val_chunks={val_chunks}")
    return train_chunks, val_chunks


###############################################################################
# 5. DATASET CREATION
###############################################################################
def chunk_generator(
    csv_path,
    chunk_size,
    target_column,
    drop_columns,
    scaler,
    label_encoder,
    clamp_value,
    shuffle_seed,
    start_chunk,
    end_chunk,
    is_val=False,
):
    """
    Generator that yields (X, y) for chunk indices [start_chunk, end_chunk).
    """
    reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True)
    rng = np.random.default_rng(shuffle_seed)

    cidx = 0
    for chunk in reader:
        if cidx < start_chunk:
            cidx += 1
            continue
        if cidx >= end_chunk:
            break

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
            cidx += 1
            continue

        labels = chunk.pop(target_column)

        # Convert object columns -> numeric (hashing)
        for col in chunk.columns:
            if chunk[col].dtype == object:
                chunk[col] = chunk[col].apply(
                    lambda x: hash(x) % (2**31) if pd.notna(x) else 0
                )

        X_chunk = chunk.select_dtypes(include=[np.number]).astype(np.float64)
        X_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_chunk = X_chunk.clip(-clamp_value, clamp_value)
        X_chunk.fillna(0, inplace=True)

        # Scale
        if scaler is not None:
            X_chunk = scaler.transform(X_chunk)
        X_chunk = X_chunk.astype(np.float32)

        y_encoded = label_encoder.transform(labels)

        # Shuffle rows if training
        if not is_val:
            idx = np.arange(len(X_chunk))
            rng.shuffle(idx)
            X_chunk = X_chunk[idx]
            y_encoded = y_encoded[idx]

        yield X_chunk, y_encoded
        cidx += 1


def create_dataset(
    csv_path,
    chunk_size,
    target_column,
    drop_columns,
    scaler,
    label_encoder,
    clamp_value,
    shuffle_seed,
    start_chunk,
    end_chunk,
    is_val=False,
    repeat_epochs=1,
):
    """
    Creates a tf.data.Dataset from chunk_generator for [start_chunk, end_chunk).
    If repeat_epochs>1, we repeat the dataset accordingly.
    """
    print(
        f"[INFO] Building dataset for chunks [{start_chunk}, {end_chunk}) (is_val={is_val}), repeat={repeat_epochs}"
    )
    output_types = (tf.float32, tf.int32)
    output_shapes = (tf.TensorShape([None, None]), tf.TensorShape([None]))

    ds = tf.data.Dataset.from_generator(
        lambda: chunk_generator(
            csv_path=csv_path,
            chunk_size=chunk_size,
            target_column=target_column,
            drop_columns=drop_columns,
            scaler=scaler,
            label_encoder=label_encoder,
            clamp_value=clamp_value,
            shuffle_seed=shuffle_seed,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            is_val=is_val,
        ),
        output_types=output_types,
        output_shapes=output_shapes,
    )

    if repeat_epochs > 1:
        ds = ds.repeat(repeat_epochs)

    if not is_val:
        ds = ds.shuffle(buffer_size=CHUNK_SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


###############################################################################
# 6. MODEL
###############################################################################
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.mixed_precision import LossScaleOptimizer


# Custom Callback to Monitor Weights
class WeightMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            weights = layer.get_weights()
            for w in weights:
                if np.isnan(w).any():
                    print(f"NaN detected in weights of layer {layer.name}")
                if np.isinf(w).any():
                    print(f"Inf detected in weights of layer {layer.name}")


class SimulatedAnnealingLRScheduler(Callback):
    """
    Custom learning rate scheduler that decays the learning rate at the beginning of each epoch.
    """

    def __init__(self, initial_lr, min_lr, decay_rate):
        super(SimulatedAnnealingLRScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate the new learning rate
        new_lr = self.initial_lr * (self.decay_rate**epoch)
        new_lr = max(new_lr, self.min_lr)
        # new_lr = self.initial_lr  # If you want a fixed LR, uncomment

        # Attempt to set the learning rate directly on the optimizer
        if hasattr(self.model.optimizer, "learning_rate"):
            if isinstance(
                self.model.optimizer.learning_rate,
                tf.keras.optimizers.schedules.LearningRateSchedule,
            ):
                print(
                    f"\nEpoch {epoch+1}: Learning rate is a schedule. Skipping manual update."
                )
            else:
                try:
                    # Directly assign the new learning rate
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"\nEpoch {epoch+1}: Learning rate is set to {new_lr:.6f}.")
                except AttributeError:
                    print(
                        "\n[WARNING] Optimizer's learning_rate is not assignable. Learning rate not updated."
                    )
        else:
            print("\n[WARNING] Optimizer does not have a 'learning_rate' attribute.")


def build_model(input_dim, num_classes, initial_lr=1e-4):
    """
    Builds and compiles a deeper Keras model with more layers,
    potentially yielding higher accuracy if the dataset is large enough.
    """
    inputs = keras.Input(shape=(input_dim,))

    # Increase the size of layers from 256->512, 128->256, etc.
    x = keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(
        inputs
    )
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # Optionally add a third hidden layer
    x = keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    # Fourth hidden layer
    x = keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)

    outputs = keras.layers.Dense(
        num_classes, activation="softmax", kernel_initializer="he_normal"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Use a slightly larger learning rate if you want faster training.
    # Keep clipnorm to avoid exploding gradients on large numeric data.
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


###############################################################################
# 7. PLOTTING & EXTRA EVALUATION
###############################################################################
def plot_training_curves(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


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


def plot_roc_curves(y_true, y_preds, classes):
    """
    Plots One-vs-Rest ROC curves for each class (for multi-class classification).
    """
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_preds[:, i])
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


def plot_precision_recall_curves(y_true, y_preds, classes):
    """
    Plots One-vs-Rest Precision-Recall curves for each class (for multi-class classification).
    """
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_preds[:, i])
        # For an AUC of PR, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        # We'll just plot the curve for visualization
        plt.plot(recall, precision, label=f"{class_name}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend(loc="lower left")
    plt.show()


###############################################################################
# 8. MAIN
###############################################################################
def main():
    # 1. Disable mixed precision for stability
    # policy = mixed_precision.Policy("mixed_float16")
    # mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed precision is disabled for stability.")

    # 2) Collect all unique labels => fit LabelEncoder
    all_labels = collect_all_labels(
        DATA_PATH, target_column=TARGET_COLUMN, chunksize=100_000
    )
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    num_classes = len(all_labels)
    print(f"[INFO] LabelEncoder fitted on {num_classes} classes.")

    # 3) Partial-fit StandardScaler with reduced clamp value
    scaler = partial_fit_scaler(
        csv_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS,
        clamp_value=CLAMP_VALUE,
        scaler_chunksize=SCALER_CHUNKSIZE,
        max_chunks=MAX_SCALER_CHUNKS,
    )

    # 4) Class distribution => class weights with capping
    class_counter = compute_class_distribution(
        DATA_PATH, TARGET_COLUMN, chunksize=100_000
    )
    class_weight_dict = make_class_weights(label_encoder, class_counter)
    print("[INFO] Class weights:")
    for idx, w in class_weight_dict.items():
        print(f"  index={idx}, label={label_encoder.classes_[idx]}, weight={w:.3f}")

    # 5) Count total chunks => train/val split
    train_chunks, val_chunks = count_chunks(
        DATA_PATH, chunk_size=CHUNK_SIZE, train_ratio=TRAIN_SPLIT_RATIO
    )

    # 6) Create train & val datasets with additional filtering
    train_ds = create_dataset(
        csv_path=DATA_PATH,
        chunk_size=CHUNK_SIZE,
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS,
        scaler=scaler,
        label_encoder=label_encoder,
        clamp_value=CLAMP_VALUE,
        shuffle_seed=SEED,
        start_chunk=0,
        end_chunk=train_chunks,
        is_val=False,
        repeat_epochs=1,
    ).filter(lambda x, y: tf.reduce_all(tf.math.is_finite(x)))

    val_ds = create_dataset(
        csv_path=DATA_PATH,
        chunk_size=CHUNK_SIZE,
        target_column=TARGET_COLUMN,
        drop_columns=DROP_COLUMNS,
        scaler=scaler,
        label_encoder=label_encoder,
        clamp_value=CLAMP_VALUE,
        shuffle_seed=SEED,
        start_chunk=train_chunks,
        end_chunk=train_chunks + val_chunks,
        is_val=True,
        repeat_epochs=1,
    ).filter(lambda x, y: tf.reduce_all(tf.math.is_finite(x)))

    # 7) Decide how many epochs (0 => large epoch + early stopping, else fixed)
    if EPOCHS == 0:
        print("[INFO] EPOCHS=0 => Using early stopping with a maximum of 50 epochs.")
        epochs_for_fit = 50
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
        ]
    else:
        print(
            f"[INFO] EPOCHS={EPOCHS} => Using exactly {EPOCHS} epochs (no early stopping)."
        )
        epochs_for_fit = EPOCHS
        callbacks_list = []

    # 8) Add the Learning Rate Scheduler
    initial_lr = 2 * 1e-3  # Tune this to experiment with faster/slower learning rates
    min_lr = 1e-5
    decay_rate = 0.95
    lr_scheduler = SimulatedAnnealingLRScheduler(
        initial_lr=initial_lr, min_lr=min_lr, decay_rate=decay_rate
    )
    callbacks_list.append(lr_scheduler)

    # 9) Add TensorBoard for monitoring
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks_list.append(tensorboard_callback)

    # 10) Add Weight Monitoring Callback
    callbacks_list.append(WeightMonitor())

    # 11) Determine input_dim from the first training batch
    for Xb, yb in train_ds.take(1):
        input_dim = Xb.shape[1]
        print(f"[INFO] Detected input_dim={input_dim} from the first training batch.")
        break

    # 12) Build the model
    model = build_model(input_dim, num_classes, initial_lr=initial_lr)
    model.summary()

    # 13) Steps per epoch
    steps_per_epoch = train_chunks
    validation_steps = val_chunks

    # 14) Train the model
    print("[INFO] Starting training now...")
    history = model.fit(
        train_ds.repeat(),
        epochs=epochs_for_fit,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
    )
    print("[INFO] Training complete.")

    # 15) Plot training curves
    plot_training_curves(history)

    # 16) Evaluate on validation data fully
    print("[INFO] Evaluating on validation data to build confusion matrix ...")
    all_preds = []
    all_labels_eval = []
    all_probs = []  # For ROC/PR if desired

    for X_val, y_val in val_ds:
        # 'preds' are class probabilities from softmax
        preds = model.predict(X_val)
        preds_class = np.argmax(preds, axis=1)

        all_preds.extend(preds_class)
        all_labels_eval.extend(y_val.numpy())
        all_probs.append(preds)

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else None

    # Confusion Matrix & Classification Report
    cm = confusion_matrix(all_labels_eval, all_preds)
    print("\n[INFO] Confusion Matrix:\n", cm)

    # Plot raw (non-normalized) confusion matrix
    plot_confusion_matrix(
        cm, labels=label_encoder.classes_, normalize=False, title_suffix=" (Raw)"
    )
    # Plot normalized confusion matrix
    plot_confusion_matrix(
        cm, labels=label_encoder.classes_, normalize=True, title_suffix=" (Normalized)"
    )

    print("\n[INFO] Classification Report:")
    report = classification_report(
        all_labels_eval, all_preds, target_names=label_encoder.classes_, zero_division=0
    )
    print(report)

    # 16a) If you want to parse classification_report into a dict to plot F1
    report_dict = classification_report(
        all_labels_eval,
        all_preds,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    plot_f1_scores_bar(report_dict, label_encoder.classes_)

    # 17) Save model
    model.save("netflow_classification_model_conditional_epochs.keras")
    print("[INFO] Model saved to 'netflow_classification_model.keras'.")


###############################################################################
# 9. ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
