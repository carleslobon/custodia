import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def reduce_memory_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    return df


def main():
    # Set the data path
    data_path = "data/netTraffic"

    # List all CSV files in the directory
    csv_files = [file for file in os.listdir(data_path) if file.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified directory.")
        return

    # Process data in chunks and append to a list
    data_frames = []
    for csv_file in csv_files:
        file_path = os.path.join(data_path, csv_file)
        try:
            # Load CSV in smaller chunks to limit memory usage
            chunk_list = []
            for chunk in pd.read_csv(
                file_path, chunksize=1000
            ):  # Smaller chunks to avoid memory overload
                # Standardize column names
                chunk.columns = chunk.columns.str.strip().str.lower()

                # Ensure that 'label' column is present, otherwise skip
                if "label" not in chunk.columns:
                    print(f"Warning: 'label' column missing in file: {csv_file}")
                    continue

                chunk_list.append(chunk)

            # Concatenate all chunks
            if chunk_list:
                full_df = pd.concat(chunk_list, ignore_index=True)
                data_frames.append(full_df)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    # Combine all data_frames into one joint dataset
    if not data_frames:
        print("No valid dataframes to combine.")
        return

    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reduce memory usage by downcasting data types
    combined_df = reduce_memory_usage(combined_df)

    # Check for missing values
    missing_values = combined_df.isnull().sum()
    print("\nMissing Values Summary:")
    print(missing_values[missing_values > 0])

    # Drop rows with missing values in columns of interest
    combined_df.dropna(subset=["label"], inplace=True)

    # Encode the label column
    combined_df["label"] = combined_df["label"].apply(
        lambda x: 1 if x.strip().upper() != "BENIGN" else 0
    )

    # Feature selection
    X = combined_df.drop("label", axis=1)
    y = combined_df["label"]

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace inf/-inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows that have NaN due to conversion issues or infinity values
    X.dropna(inplace=True)
    y = y.loc[X.index]  # Align labels with remaining data

    # ** Subsample to reduce memory requirements **
    subsample_fraction = 0.2  # Adjust the fraction to control memory usage
    combined_sample = X.sample(frac=subsample_fraction, random_state=42)
    y_sample = y.loc[combined_sample.index]

    # Optional: Normalize/Standardize the data
    scaler = StandardScaler()
    try:
        X = scaler.fit_transform(combined_sample)
    except ValueError as e:
        print(f"Error during scaling: {e}")
        return

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sample, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    # Create the RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model to the training data
    rf_model.fit(X_train, y_train)

    # Predict the labels on the test data
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    main()
