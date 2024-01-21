import os

from flask import Flask, send_file, render_template, request

from main import load_dataset
from data_processing.graphs import show_correlation_matrix, show_variable_distributions, show_calendar_plots
from models.inferencer import make_inference
app = Flask(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_file_path = '../assets/dataset.csv'
absolute_file_path = os.path.join(script_directory, relative_file_path)
df = load_dataset(absolute_file_path)


def main():
    app.run(debug=False)


@app.route('/')
def index():
    return render_template('index.html')


# CORRELATION MATRIX

@app.route('/correlation_matrix')
def correlation_matrix():
    directory = 'img'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'correlation_matrix.png')
    show_correlation_matrix(df, file_path)
    return send_file(file_path, mimetype='image/png')


# DISTRIBUTIONS


@app.route('/show_distributions')
def show_distributions():
    columns = df.columns[1:]
    return render_template('var_distro.html', columns=columns)


@app.route('/variable_distribution/<column>')
def variable_distribution(column):
    directory = 'variable_distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)

    show_variable_distributions(df, directory)
    file_path = os.path.join(directory, f'{column}_distribution.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Image not found", 404


# CALENDAR PLOTS

@app.route('/show_calendar_plots')
def show_calendar():
    columns = df.columns[1:]
    return render_template('calendar_plots.html', columns=columns)


@app.route('/calendar_plot/<column>')
def calendar_plot(column):
    directory = 'calendar_plots'
    if not os.path.exists(directory):
        os.makedirs(directory)

    show_calendar_plots(df, directory)
    file_path = os.path.join(directory, f'{column}_calendar_plot.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Image not found", 404

# DISEASE INFORMATION


@app.route('/disease_information')
def show_disease():
    columns = df.columns[1:]
    return render_template('analyze_disease.html', columns=columns)


@app.route('/disease_information/disease', methods=['POST'])
def find_disease():
    columns = df.columns[1:]
    form_list = ["part", "intensity", "texture", "color", "pattern", "anatomicalRegion", "shape", "borderColor"]
    parameters = {}
    for element in form_list:
        if request.form[element]:
            parameters[element] = request.form[element]
    print(make_inference(parameters))
    return render_template('analyze_disease.html', columns=columns)


if __name__ == '__main__':
    main()
