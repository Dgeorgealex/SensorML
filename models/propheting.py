from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.data_processing import split_and_truncate, truncate


def print_column_forcast(forecast, column):
    print(f"Forecast for {column}:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(48))


def plot_forecast(forecast, actual, column):
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


def prophet_uni_variable(df):
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

        print_column_forcast(forecast, column)
        plot_forecast(forecast, actual, column)


def prophet_uni_regressor(df):
    start_date = df['Timestamp'].min()
    end_date = start_date + pd.Timedelta(weeks=2)

    data = truncate(df, start_date, end_date)

    actual = split_and_truncate(df, end_date, end_date + pd.Timedelta(days=2))

    for column in data.columns[1:]:
        model = Prophet()

        new_data = data.rename(columns={'Timestamp': 'ds', column: 'y'})
        for column2 in new_data.columns[1:]:
            if column2 != column and column2 != 'y':
                model.add_regressor(column2)
        model.fit(new_data)

        future = model.make_future_dataframe(periods=48, freq='H')
        for column2 in new_data.columns[1:]:
            if column2 != column and column2 != 'y':
                future[column2] = df[column2]
        forecast = model.predict(future)

        print_column_forcast(forecast, column)
        plot_forecast(forecast, actual, column)
