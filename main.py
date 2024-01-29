import pandas as pd
import os
from models.propheting import prophet_uni_variable
from models.propheting import prophet_uni_regressor
from data_processing.data_processing import load_dataset
from datetime import datetime


def main():
    df = load_dataset('assets/SensorMLDataset.csv')

    # show_correlation_matrix(df)

    # show_variable_distributions(df)

    # show_calendar_plots(df)

    start_date = df['Timestamp'].min() + pd.Timedelta(weeks=4)
    end_date = start_date + pd.Timedelta(days=2)

    # prophet_uni_variable(df, start_date, end_date)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, "blabla")
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    prophet_uni_regressor(df, start_date, 2, directory=subdirectory, regressors=False)

    # print_cross_validation(df)


if __name__ == "__main__":
    main()
