# analysis/data_cleaning.py

import pandas as pd

def check_missing_values(df):
    """
    Check the percentage of missing values for each column in the dataframe.
    Returns a Series with column names and missing value percentages.
    """
    return df.isna().mean()

def check_missing_values_columns(df, columns):
    """
    Check the percentage of missing values for specific columns.
    Returns a Series with column names and missing value percentages for the specified columns.
    """
    return df[columns].isna().mean()

def drop_na_in_columns(df, columns):
    """
    Drop rows that have missing values in specified columns.
    Returns the cleaned dataframe.
    """
    df.dropna(subset=columns, inplace=True)
    return df

def find_duplicates(df):
    """
    Find and return duplicate rows in the dataframe.
    """
    return df[df.duplicated()]
