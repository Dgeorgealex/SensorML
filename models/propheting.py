import os
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.data_processing import split_and_truncate, truncate
from data_processing.graphs import plot_forecast


def print_column_forcast(forecast, column):
    print(f"Forecast for {column}:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(48))


def prophet_uni_variable(df, start_date, end_date):
    split_dfs = split_and_truncate(df, start_date, end_date)

    actual = split_and_truncate(df, end_date, end_date + pd.Timedelta(days=2))
    for column, data in actual.items():
        data.rename(columns={'Timestamp': 'ds', column: 'y'})

    for column, data in split_dfs.items():
        m = Prophet()
        m.fit(data.rename(columns={'Timestamp': 'ds', column: 'y'}))

        future = m.make_future_dataframe(periods=48, freq='H')

        forecast = m.predict(future)

        # print_column_forcast(forecast, column)
        plot_forecast(forecast, actual, column)


def prophet_uni_regressor(df, start_date, end_date, directory=None):
    # start_date = df['Timestamp'].min()
    # end_date = start_date + pd.Timedelta(weeks=2)
    # If start_date is not a pd timestamp, make it one
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

    data = truncate(df, start_date, end_date)

    actual = split_and_truncate(df, end_date, end_date + pd.Timedelta(days=2))

    result = {}

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

        # print_column_forcast(forecast, column)
        file_path = os.path.join(directory, f'{column}.png')
        if column == "temp1":
            result['temperature'] = forecast['yhat'].tail(48)
        if column == "umid":
            result['humidity'] = forecast['yhat'].tail(48)
            result['timestamp'] = forecast['ds'].tail(48)
        plot_forecast(forecast, actual, column, file_path)
    return zip(result['temperature'], result['humidity'], result['timestamp'])
