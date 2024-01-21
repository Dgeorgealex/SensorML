import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def perform_cross_validation(df, column_name):
    df_prophet = pd.DataFrame({
        'ds': df['Timestamp'],
        'y': df[column_name]
    })

    m = Prophet()
    m.fit(df_prophet)

    df_cv = cross_validation(m, initial='30 days', period='10 days', horizon='10 days')

    df_p = performance_metrics(df_cv)
    return df_p


def print_cross_validation(df):
    print("We are using Prophet")
    for column in df.columns[1:]:
        print(f"Performing cross-validation for {column}")

        performance_metrics_df = perform_cross_validation(df, column)

        print(f"Training error for {column}:")
        print(performance_metrics_df[['horizon', 'rmse', 'mae']])
        print("\n")
