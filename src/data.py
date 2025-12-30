"""
Data loading module - Simple example for MLOps workflow testing
"""

def load_data():
    """
    Load sample data

    Returns:
        dict: Sample dataset information
    """
    print("=== Loading data ===")
    print("Dataset: Iris")
    print("Samples: 150")
    print("Features: 4")

    result = {
        "dataset_name": "iris",
        "num_samples": 150,
        "num_features": 4,
        "status": "success"
    }

    print(f"Data loaded successfully: {result}")
    return result


def preprocess_data():
    """
    Preprocess data

    Returns:
        dict: Preprocessing results
    """
    print("=== Preprocessing data ===")
    print("Removing duplicates...")
    print("Normalizing features...")
    print("Splitting train/test...")

    result = {
        "train_samples": 120,
        "test_samples": 30,
        "preprocessing_steps": ["remove_duplicates", "normalize", "split"],
        "status": "success"
    }

    print(f"Preprocessing completed: {result}")
    return result
