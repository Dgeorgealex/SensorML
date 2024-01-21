from graphs import show_correlation_matrix, show_variable_distributions, show_calendar_plots
from propheting import prophet_uni_variable
from data_processing import load_dataset, split_and_truncate
from cross_validation import print_cross_validation


def main():
    df = load_dataset('dataset.csv')

    # show_correlation_matrix(df)

    # show_variable_distributions(df)

    # show_calendar_plots(df)

    prophet_uni_variable(df)

    # print_cross_validation(df)


if __name__ == "__main__":
    main()
