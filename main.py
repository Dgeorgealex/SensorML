import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

from graphs import show_correlation_matrix, show_variable_distributions, show_calendar_plots
from data_processing import load_dataset, split_and_truncate
from cross_validation import print_cross_validation


def show_prophet_data(df):
    start_date = df['Timestamp'].min()
    end_date = start_date + pd.Timedelta(weeks=2)

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
    df = load_dataset('dataset.csv')

    # show_correlation_matrix(df)

    # show_variable_distributions(df)

    # show_calendar_plots(df)

    # do_stuff(df)

    print_cross_validation(df)


if __name__ == "__main__":
    main()
