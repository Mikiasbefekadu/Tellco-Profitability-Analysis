# analysis/eda.py

import pandas as pd

def get_top_handsets_by_type(df):
    """
    Groups the data by 'Handset Type', counts occurrences, removes missing 'Handset Type' values,
    and returns the top 10 handsets sorted by user count in descending order.
    """
    # Remove rows where 'Handset Type' is missing
    df_cleaned = df.dropna(subset=['Handset Type'])
    
    # Group by 'Handset Type' and count occurrences
    handset_grouping = df_cleaned.groupby('Handset Type').size().reset_index(name='User Count')
    
    # Sort by 'User Count' in descending order and get the top 10
    top_handsets = handset_grouping.sort_values(by='User Count', ascending=False).head(10)
    
    return top_handsets



def get_top_manufacturers(df):
    """
    Groups the data by 'Handset Manufacturer', counts occurrences, and returns the top 3 manufacturers sorted by user count.
    """
    manufacturer_group = df.groupby('Handset Manufacturer').size().reset_index(name='User Count')
    top_manufacturers = manufacturer_group.sort_values(by='User Count', ascending=False).head(3)
    return top_manufacturers

def get_top_handsets_per_manufacturer(df):
    """
    Groups the data by 'Handset Manufacturer' and 'Handset Type', counts occurrences, and then filters out the top 3 manufacturers. 
    For each of the top manufacturers, returns the top 5 handsets sorted by count.
    """
    handset_counts = df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='Count')
    top_3_manufacturers = handset_counts.groupby('Handset Manufacturer')['Count'].sum().nlargest(3).index
    filtered_handsets = handset_counts[handset_counts['Handset Manufacturer'].isin(top_3_manufacturers)]
    top_handsets_per_manufacturer = filtered_handsets.groupby('Handset Manufacturer').apply(
        lambda x: x.nlargest(5, 'Count')
    ).reset_index(drop=True)
    return top_handsets_per_manufacturer
