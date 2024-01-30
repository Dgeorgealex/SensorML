import os
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.data_processing import split_and_truncate, truncate
from data_processing.graphs import plot_forecast, show_correlation_matrix


def print_column_forcast(forecast, column):
    print(f"Forecast for {column}:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(48))


def compute_error(forecast, actual, column):
    error = 0
    for target, output in zip(actual[column], forecast['yhat']):
        error += (target - output) ** 2
    error = error / len(actual)
    return error


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


def prophet_uni_regressor(df, start_date, end_date, date_difference, directory=None, regressors=False):
    # start_date = df['Timestamp'].min()
    # end_date = start_date + pd.Timedelta(weeks=2)
    # If start_date is not a pd timestamp, make it one

    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

    data = truncate(df, start_date, end_date)
    if regressors:
        regressString = "regressors"
    else:
        regressString = "no_regressors"

    subdirectory = os.path.join(directory, regressString)
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    corr_matrix = data.corr()
    show_correlation_matrix(df, os.path.join(directory, "correlation_matrix"))
    actual = split_and_truncate(df, end_date, end_date + pd.Timedelta(days=date_difference))
    result = {}
    errors = {}
    for column in data.columns[1:]:
        model = Prophet()

        new_data = data.rename(columns={'Timestamp': 'ds', column: 'y'})
        if regressors:
            for column2 in new_data.columns[1:]:
                if column2 != column and column2 != 'y' and corr_matrix[column][column2] >= 0.9:
                    model.add_regressor(column2)
        model.fit(new_data)

        future = model.make_future_dataframe(periods=date_difference * 24, freq='H')
        if regressors:
            for column2 in new_data.columns[1:]:
                if column2 != column and column2 != 'y' and corr_matrix[column][column2] >= 0.9:
                    future[column2] = df[column2]
        forecast = model.predict(future)

        file_path = os.path.join(subdirectory, f'{column}.png') if directory else None
        if column == "temp1":
            result['temperature'] = forecast['yhat'].tail(date_difference * 24)
        if column == "umid":
            result['humidity'] = forecast['yhat'].tail(date_difference * 24)
            result['timestamp'] = forecast['ds'].tail(date_difference * 24)
        plot_forecast(forecast, actual, column, date_difference, file_path)
        errors[column] = round(compute_error(forecast.tail(date_difference * 24), actual[column].tail(date_difference * 24), column), 4)
    return zip(result['temperature'], result['humidity'], result['timestamp']), errors
