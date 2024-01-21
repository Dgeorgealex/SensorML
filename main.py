from models.propheting import prophet_uni_variable
from data_processing.data_processing import load_dataset


def main():
    df = load_dataset('assets/dataset.csv')

    # show_correlation_matrix(df)

    # show_variable_distributions(df)

    # show_calendar_plots(df)

    prophet_uni_variable(df)

    # print_cross_validation(df)


if __name__ == "__main__":
    main()
