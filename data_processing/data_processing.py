import pandas as pd


def split_and_truncate(df, start_date, end_date):
    truncated_dfs = {}

    for column in df.columns[1:]:
        truncated_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

        truncated_dfs[column] = truncated_df[['Timestamp', column]]

    return truncated_dfs


def load_dataset(path):
    df = pd.read_csv(path)
    # print(df.head())

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df
