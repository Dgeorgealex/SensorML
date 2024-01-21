import pandas as pd

from models.propheting import prophet_uni_variable
from models.propheting import prophet_uni_regressor
from data_processing.data_processing import load_dataset


def main():
    df = load_dataset('assets/dataset.csv')

    # show_correlation_matrix(df)

    # show_variable_distributions(df)

    # show_calendar_plots(df)

    start_date = df['Timestamp'].min() + pd.Timedelta(weeks=4)
    end_date = start_date + pd.Timedelta(weeks=10)

    # prophet_uni_variable(df, start_date, end_date)

    prophet_uni_regressor(df, start_date, end_date)

    # print_cross_validation(df)


if __name__ == "__main__":
    main()
