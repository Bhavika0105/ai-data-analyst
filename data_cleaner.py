import pandas as pd

def clean_data(df):
    """
    Fills missing values in the DataFrame for numeric and categorical columns.
    """
    # Fill numeric columns (Age, Fare) with median
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Fill categorical columns (Cabin) with 'Unknown'
    if 'Cabin' in df.columns:
        df['Cabin'].fillna('Unknown', inplace=True)

    return df
