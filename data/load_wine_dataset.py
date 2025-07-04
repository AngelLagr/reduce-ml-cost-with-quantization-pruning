import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_wine_dataset(test_size=0.2, random_state=42):
    """
    Load and preprocess the breast cancer dataset.

    Parameters:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Return:
        tuple: A tuple containing the training and testing data and labels.
    """
    # Load the dataset from sklearn
    from sklearn.datasets import load_wine
    data = load_wine()
    
    # Create a DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split the data into features and labels
    X = df.drop('target', axis=1)
    y = df['target']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test