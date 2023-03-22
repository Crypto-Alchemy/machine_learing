import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, test_size=0.2):
    """
    Splits the preprocessed data into training and testing sets.

    Args:
        df (pd.DataFrame): Preprocessed OHLCV data with additional features.
        test_size (float): The proportion of data to be used for testing. Defaults to 0.2.

    Returns:
        A tuple of train_set and test_set.
    """
    train_set, test_set = train_test_split(
        df, test_size=test_size, shuffle=False)
    return train_set, test_set
