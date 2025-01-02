import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time
import os
import ipaddress


def ip_to_long(ip):
    """
    Convert an IP address to its long integer representation.

    Args:
        ip (str): IP address in string format.

    Returns:
        int: Integer representation of the IP address.
    """
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        return 0  # Handle invalid IPs (if any) by returning 0


def process_ip_columns(chunk, ip_columns):
    """
    Convert IP address columns in a DataFrame to numeric values.

    Args:
        chunk (pd.DataFrame): Data chunk containing IP address columns.
        ip_columns (list): List of IP address column names.

    Returns:
        pd.DataFrame: Updated DataFrame with IP addresses converted to integers.
    """
    for col in ip_columns:
        if col in chunk.columns:
            chunk[col] = chunk[col].apply(ip_to_long)
    return chunk


def data_generator(
    file_path,
    chunk_size,
    target_column,
    drop_columns,
    ip_columns,
    scaler=None,
    is_training=True,
):
    """
    A generator that yields batches of data for training or evaluation dynamically.

    Args:
        file_path (str): Path to the large CSV file.
        chunk_size (int): Number of rows per chunk to read.
        target_column (str): Name of the target column (label).
        drop_columns (list): List of columns to drop from the dataset.
        ip_columns (list): List of IP address columns to convert.
        scaler (StandardScaler): Pre-fitted scaler for consistent feature scaling.
        is_training (bool): Whether the generator is for training (fit the scaler) or testing.

    Yields:
        tuple: A tuple of (X_batch, y_batch).
    """
    print("Starting on-the-fly data generation...")

    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        print(f"Processing batch {i + 1}...")
        chunk = chunk.drop(
            columns=drop_columns, errors="ignore"
        )  # Drop unnecessary columns

        # Convert IP address columns to numeric
        chunk = process_ip_columns(chunk, ip_columns)

        # Extract features and target
        if target_column not in chunk.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data!")

        X = chunk.drop(columns=[target_column]).values
        y = chunk[target_column].values

        # Normalize features (fit scaler only during training on first chunk)
        if scaler is None:
            if is_training:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            else:
                raise ValueError("Scaler must be pre-fitted before use in testing.")
        else:
            X = scaler.transform(X)

        # Map categorical labels to numerical
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y])

        print(f"Batch {i + 1} ready for use.")
        yield X, y


def create_model(input_shape, num_classes):
    """
    Create and compile a neural network model.

    Args:
        input_shape (int): Number of features in the input data.
        num_classes (int): Number of target classes.

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    print("Creating and compiling the model...")
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_shape,)),  # Input layer
            Dense(32, activation="relu"),  # Hidden layer
            Dense(num_classes, activation="softmax"),  # Output layer
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # For multi-class classification
        metrics=["accuracy"],
    )
    print("Model created and compiled.")
    return model


def plot_metrics(history):
    """
    Plot the evolution of training/validation loss and accuracy.

    Args:
        history (tf.keras.callbacks.History): History object returned by model.fit().
    """
    print("Plotting training metrics...")
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("Metrics plotted.")


def main():
    print("Starting program...")
    # Path to your CSV file
    file_path = "/mnt/c/Users/mikig/Desktop/UPC/PAE/Datasets/9810e03bba4983da_MOHANAD_A4706/9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv"

    print("Ensuring dataset file path...")
    if os.path.exists(file_path):
        print("Datafile found!")
    else:
        print("No datafile found!")
        return

    # Define target column, columns to drop, and IP columns
    target_column = "Attack"
    drop_columns = ["Label", "Dataset"]
    ip_columns = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

    print("Reading sample chunk to determine feature and class count...")
    sample_chunk = pd.read_csv(file_path, nrows=1).drop(columns=drop_columns)
    sample_chunk = process_ip_columns(sample_chunk, ip_columns)
    input_shape = len(sample_chunk.columns) - 1  # Exclude the target column
    num_classes = len(
        pd.read_csv(file_path, usecols=[target_column])[target_column].unique()
    )
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")

    # Create the model
    model = create_model(input_shape, num_classes)

    # Define chunk size and steps per epoch
    chunk_size = 10000
    steps_per_epoch = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))
    print(f"Total steps per epoch: {steps_per_epoch}")

    # Initialize the scaler with a small subset of the data
    print("Fitting scaler on a small subset...")
    scaler = StandardScaler()
    subset = pd.read_csv(file_path, nrows=chunk_size).drop(columns=drop_columns)
    subset = process_ip_columns(subset, ip_columns)
    subset_X = subset.drop(columns=[target_column]).values
    scaler.fit(subset_X)
    print("Scaler fitted.")

    # Start training
    print("Starting training...")
    train_generator = data_generator(
        file_path,
        chunk_size,
        target_column,
        drop_columns,
        ip_columns,
        scaler,
        is_training=True,
    )
    history = model.fit(
        train_generator, steps_per_epoch=steps_per_epoch, epochs=10, verbose=1
    )
    print("Training completed.")

    # Plot metrics
    plot_metrics(history)

    # Evaluate the model
    print("Starting evaluation...")
    test_generator = data_generator(
        file_path,
        chunk_size,
        target_column,
        drop_columns,
        ip_columns,
        scaler,
        is_training=False,
    )
    y_true, y_pred = [], []

    for i, (X_batch, y_batch) in enumerate(test_generator):
        print(f"Evaluating batch {i + 1}...")
        predictions = model.predict(X_batch)
        y_true.extend(y_batch)
        y_pred.extend(np.argmax(predictions, axis=1))

    print("Evaluation completed.")
    print("Generating confusion matrix...")
    print(confusion_matrix(y_true, y_pred))

    print("\nGenerating classification report...")
    print(classification_report(y_true, y_pred))

    print("Program completed.")


if __name__ == "__main__":
    main()
