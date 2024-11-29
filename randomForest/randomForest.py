import numpy as np
import pandas as pd
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(file_path: str, target_column: str, nrows=None):
    """
    Load and preprocess the dataset.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        target_column (str): Name of the target column.
        nrows (int): Number of rows to read from the dataset.

    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y) datasets.
    """
    data = pd.read_csv(file_path, nrows=nrows)
    print("Dataset loaded successfully.")
    print(data.info())

    # Drop unnecessary columns
    columns_to_drop = ["Label", "Dataset"]
    if target_column in columns_to_drop:
        columns_to_drop.remove(target_column)

    data = data.drop(columns=columns_to_drop, axis=1)

    # Handle IP addresses: Convert to integers
    for column in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]:
        if column in data.columns:
            data[column] = data[column].apply(lambda x: int(ipaddress.ip_address(x)))

    # Convert non-numeric columns to numeric values
    for column in data.select_dtypes(include=["object"]).columns:
        if column != target_column:
            data[column] = pd.Categorical(data[column]).codes

    # Separate features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    print(f"Non-numeric columns converted to numeric.")
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts()}")
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
):
    """
    Train a Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the trained model using test data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", class_report)
    print("Confusion Matrix:\n", conf_matrix)

    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
    }


def feature_importances(model, feature_names: list):
    """
    Extract and display feature importances from the trained model.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    print("Feature importances extracted.")
    print(importance_df)
    return importance_df


if __name__ == "__main__":
    # Configurations
    DATA_FILE_PATH = "/mnt/c/Users/mikig/Desktop/UPC/PAE/Datasets/9810e03bba4983da_MOHANAD_A4706/9810e03bba4983da_MOHANAD_A4706/data/NF-UQ-NIDS-v2.csv"
    TARGET_COLUMN = "Attack"  # Adjust based on dataset
    N_ROWS = 1000  # None to read full dataset

    # Step 1: Load the data
    X, y = load_data(DATA_FILE_PATH, TARGET_COLUMN, N_ROWS)

    # Step 2: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Train the Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate the model
    evaluation_metrics = evaluate_model(rf_model, X_test, y_test)

    # Step 5: Analyze feature importance
    feature_importances_df = feature_importances(rf_model, X.columns.tolist())
