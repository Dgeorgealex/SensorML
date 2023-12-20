import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt
from cross_validation import print_cross_validation


def split_and_truncate(df, start_date, end_date):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    truncated_dfs = {}

    for column in df.columns[1:]:
        truncated_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

        truncated_dfs[column] = truncated_df[['Timestamp', column]]

    return truncated_dfs


def do_stuff(df):
    start_date = df['Timestamp'].min()
    end_date = start_date + pd.Timedelta(weeks=1)

    split_dfs = split_and_truncate(df, start_date, end_date)

    actual = split_and_truncate(df, end_date, end_date + pd.Timedelta(days=2))
    for column, data in actual.items():
        data.rename(columns={'Timestamp': 'ds', column: 'y'})

    for column, data in split_dfs.items():
        print(column)
        print(data)

        m = Prophet()
        m.fit(data.rename(columns={'Timestamp': 'ds', column: 'y'}))

        future = m.make_future_dataframe(periods=48, freq='H')

        forecast = m.predict(future)

        print(f"Forecast for {column}:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(48))

        plt.figure(figsize=(10, 6))

        col = actual[column].rename(columns={'Timestamp': 'ds', column: 'y'})

        plt.plot(col['ds'], col['y'], label='Actual')
        plt.plot(forecast['ds'].tail(48), forecast['yhat'].tail(48), label='Predicted')
        plt.fill_between(forecast['ds'].tail(48), forecast['yhat_lower'].tail(48), forecast['yhat_upper'].tail(48),
                         alpha=0.3)
        plt.title(f"Actual vs Predicted for {column}")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


def main():
    df = pd.read_csv('dataset.csv')
    # print(df.head())

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # do_stuff(df)

    print_cross_validation(df)


if __name__ == "__main__":
    main()
