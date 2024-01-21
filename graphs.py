import os
from threading import Lock

import pandas as pd
import matplotlib.pyplot as plt
import calplot
import seaborn as sns

mutex = Lock()  # Only generate one plot at a time


def show_correlation_matrix(df, file_path=None):
    with mutex:
        if hasattr(show_correlation_matrix, 'is_generated') and file_path:
            return
        if file_path:
            show_correlation_matrix.is_generated = True

        # d1 = df.groupby(df['Timestamp'].dt.date).median()
        # d2 = df.groupby(df['Timestamp'].dt.date).mean()
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Matrix')
        if file_path:
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()


def show_calendar_plots(df, directory=None):
    with mutex:
        if hasattr(show_calendar_plots, 'is_generated') and directory:
            return
        if directory:
            show_calendar_plots.is_generated = True

        numerical_columns = df.columns[1:]
        for column in numerical_columns:
            df_aux = df.groupby(df['Timestamp'].dt.date)[column].mean()
            df_aux.index = pd.to_datetime(df_aux.index)

            calplot.calplot(df_aux, cmap='YlGnBu', colorbar=True, figsize=(10, 6))

            plt.title(f"Calendar heatmap of {column}")

            if directory:
                file_path = os.path.join(directory, f'{column}_calendar_plot.png')
                plt.savefig(file_path)
                plt.close()
            else:
                plt.show()


def show_variable_distributions(df, directory=None):
    with mutex:
        if hasattr(show_variable_distributions, 'is_generated') and directory:
            return
        if directory:
            show_variable_distributions.is_generated = True

        numerical_columns = df.columns[1:]

        for column in numerical_columns:
            plt.figure(figsize=(10, 6))

            # Distribution
            plt.subplot(1, 2, 1)
            sns.histplot(df[column], kde=False, bins=50)
            plt.title(f'Distribution of {column}')

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')

            if directory:
                file_path = os.path.join(directory, f'{column}_distribution.png')
                plt.savefig(file_path)
                plt.close()
            else:
                plt.show()
