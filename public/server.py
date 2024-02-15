import os
from datetime import datetime

import pandas as pd

from flask import Flask, send_file, render_template, request, jsonify

from main import load_dataset
from models.inferencer import make_inference
from data_processing.data_processing import truncate
from data_processing.graphs import (show_correlation_matrix, show_variable_distributions,
                                    show_calendar_plots)
from models.propheting import prophet_uni_regressor
from models.lstm import show_lstm, lstm_predict
from models.seq2seq import show_seq2seq, seq2seq_predict
from models.seq2seq_atn import ask_model_seq2seq_attention

app = Flask(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
relative_file_path = '../assets/SensorMLDataset.csv'
absolute_file_path = os.path.join(script_directory, relative_file_path)
df = load_dataset(absolute_file_path)

lstm_dir = 'lstm'
seq2seq_dir = 'seq2seq'


def main():
    if not os.path.exists(lstm_dir):
        os.makedirs(lstm_dir)
    if not os.path.exists(seq2seq_dir):
        os.makedirs(seq2seq_dir)

    # Train models, could save and just load them to build the images
    # show_lstm(df, lstm_dir, 0)
    # show_seq2seq(df, seq2seq_dir, 0)
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


@app.route('/correlation_matrix/<start_date>/<end_date>')
def correlation_matrix_by_date(start_date, end_date):
    start_date_dir = datetime.strptime(start_date, '%m/%d/%Y').strftime('%Y%m%d')
    end_date_dir = datetime.strptime(end_date, '%m/%d/%Y').strftime('%Y%m%d')
    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, start_date_dir + end_date_dir)
    file_path = os.path.join(subdirectory, "correlation_matrix.png")
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


@app.route('/lstm-sliders')
def lstm_sliders():
    return render_template('lstm-sliders.html')


@app.route('/seq2seq-sliders')
def seq2seq_sliders():
    return render_template('seq2seq-sliders.html')


@app.route('/seq2seq-atn-sliders')
def seq2seq_atn_sliders():
    return render_template('seq2seq-atn-sliders.html')


@app.route('/generate-graph-lstm')
def lstm_graphs():
    start_date = request.args.get('start')
    start_date_dt = datetime.strptime(start_date, '%m/%d/%Y')

    # Extract year, month, and day
    year = start_date_dt.year
    month = start_date_dt.month
    day = start_date_dt.day

    # Create a new datetime object
    date = datetime(year, month, day)

    subdirectory = os.path.join(script_directory, 'lstm_images')

    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    result_data, mse_error = lstm_predict(df, date, subdirectory)
    disease_threats = []
    for temp, humid, date in result_data:
        print(temp, humid, date)
        if 24 <= int(temp) <= 29 and 90 <= int(humid) <= 100:
            disease_threats.append(("Early Blight", date))
        if 17 <= int(temp) <= 23 and 90 <= int(humid) <= 100:
            disease_threats.append(("Gray Mold", date))
        if 10 <= int(temp) <= 24 and 90 <= int(humid) <= 100:
            disease_threats.append(("Late Blight", date))
        if 21 <= int(temp) <= 24 and 85 <= int(humid) <= 100:
            disease_threats.append(("Leaf Mold", date))
        if 22 <= int(temp) <= 30 and 50 <= int(humid) <= 75:
            disease_threats.append(("Powdery Mildew", date))

    columns = df.columns[1:]
    return render_template('lstm.html', columns=columns, diseases=disease_threats, mse_error=mse_error)



@app.route('/generate-graph-seq2seq')
def seq2seq_graphs():
    start_date = request.args.get('start')
    days = int(request.args.get('num_days'))
    start_date_dt = datetime.strptime(start_date, '%m/%d/%Y')

    # Extract year, month, and day
    year = start_date_dt.year
    month = start_date_dt.month
    day = start_date_dt.day

    # Create a new datetime object
    date = datetime(year, month, day)

    subdirectory = os.path.join(script_directory, 'seq2seq_images')

    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    if days == 2:
        learning_size = 72
        predict_size = 48
        hidden_size = 100
    elif days == 5:
        learning_size = 168
        predict_size = 120
        hidden_size = 100
    else:
        learning_size = 336
        predict_size = 168
        hidden_size = 128

    result_data, mse_error = seq2seq_predict(df, date, subdirectory, learning_size, predict_size, hidden_size)

    disease_threats = []
    for temp, humid, date in result_data:
        print(temp, humid, date)
        if 24 <= int(temp) <= 29 and 90 <= int(humid) <= 100:
            disease_threats.append(("Early Blight", date))
        if 17 <= int(temp) <= 23 and 90 <= int(humid) <= 100:
            disease_threats.append(("Gray Mold", date))
        if 10 <= int(temp) <= 24 and 90 <= int(humid) <= 100:
            disease_threats.append(("Late Blight", date))
        if 21 <= int(temp) <= 24 and 85 <= int(humid) <= 100:
            disease_threats.append(("Leaf Mold", date))
        if 22 <= int(temp) <= 30 and 50 <= int(humid) <= 75:
            disease_threats.append(("Powdery Mildew", date))

    columns = df.columns[1:]
    return render_template('seq2seq.html', columns=columns, diseases=disease_threats, mse_error=mse_error)


@app.route('/generate-graph-seq2seq-atn')
def seq2seq_atn_graphs():
    start_date = request.args.get('start')
    days = int(request.args.get('num_days'))
    start_date_dt = datetime.strptime(start_date, '%m/%d/%Y')

    # Extract year, month, and day
    year = start_date_dt.year
    month = start_date_dt.month
    day = start_date_dt.day

    # Create a new datetime object
    date = datetime(year, month, day)

    subdirectory = os.path.join(script_directory, 'seq2seq_atn_images')

    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    result_data, mse_error = ask_model_seq2seq_attention(df, 'seq2seq210.pth',
                                                         date, 120, days*24, subdirectory)
    disease_threats = []
    for temp, humid, date in result_data:
        print(temp, humid, date)
        if 24 <= int(temp) <= 29 and 90 <= int(humid) <= 100:
            disease_threats.append(("Early Blight", date))
        if 17 <= int(temp) <= 23 and 90 <= int(humid) <= 100:
            disease_threats.append(("Gray Mold", date))
        if 10 <= int(temp) <= 24 and 90 <= int(humid) <= 100:
            disease_threats.append(("Late Blight", date))
        if 21 <= int(temp) <= 24 and 85 <= int(humid) <= 100:
            disease_threats.append(("Leaf Mold", date))
        if 22 <= int(temp) <= 30 and 50 <= int(humid) <= 75:
            disease_threats.append(("Powdery Mildew", date))

    columns = df.columns[1:]
    return render_template('seq2seq_atn.html', columns=columns, diseases=disease_threats, mse_error=mse_error)


@app.route('/prophet-sliders')
def prophet_sliders():
    return render_template('prophet_sliders.html')


@app.route('/generate-graph')
def prophet_image():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    num_days = int(request.args.get('num_days'))

    start_date_dir = datetime.strptime(start_date, '%m/%d/%Y').strftime('%Y%m%d')
    end_date_dir = datetime.strptime(end_date, '%m/%d/%Y').strftime('%Y%m%d')

    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, start_date_dir + end_date_dir)
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    result_data, regressors_errors = prophet_uni_regressor(df, start_date, end_date, num_days, subdirectory, regressors=True)
    _, uni_errors = prophet_uni_regressor(df, start_date, end_date, num_days, subdirectory, regressors=False)
    disease_threats = []
    for temp, humid, date in result_data:
        if 24 <= int(temp) <= 29 and 90 <= int(humid) <= 100:
            disease_threats.append(("Early Blight", date))
        if 17 <= int(temp) <= 23 and 90 <= int(humid) <= 100:
            disease_threats.append(("Gray Mold", date))
        if 10 <= int(temp) <= 24 and 90 <= int(humid) <= 100:
            disease_threats.append(("Late Blight", date))
        if 21 <= int(temp) <= 24 and 85 <= int(humid) <= 100:
            disease_threats.append(("Leaf Mold", date))
        if 22 <= int(temp) <= 30 and 50 <= int(humid) <= 75:
            disease_threats.append(("Powdery Mildew", date))
    return render_template('prophet_graph.html', columns=df.columns[1:],
                           start_date=start_date_dir, end_date=end_date_dir, diseases=disease_threats,
                            regressors_errors=regressors_errors, uni_errors=uni_errors)


@app.route('/prophet_images/<column>/<start_date>/<end_date>/<regressor>')
def prophet_images(column, start_date, end_date, regressor):
    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, start_date + end_date, regressor)

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


@app.route('/lstm_good/<column>')
def lstm_good(column):
    subdirectory = os.path.join(script_directory, 'lstm_images')

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


@app.route('/seq2seq_good/<column>')
def seq2seq_good(column):
    subdirectory = os.path.join(script_directory, 'seq2seq_images')

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


@app.route('/seq2seq_atn_good/<column>')
def seq2seq_atn_good(column):
    subdirectory = os.path.join(script_directory, 'seq2seq_atn_images')

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


# DISEASE INFORMATION
@app.route('/disease_information')
def show_disease():
    return render_template('analyze_disease.html')


@app.route('/disease_information/disease', methods=['POST'])
def find_disease():
    form_list = ["part", "intensity", "texture", "color", "pattern", "anatomicalRegion", "shape", "borderColor"]
    parameters = {}
    for element in form_list:
        if request.form[element]:
            parameters[element] = request.form[element]
    return render_template('diseases.html', diseases=make_inference(parameters))


# GET ALL DATA (FOR NEW CHARTS)
@app.route('/getdata')
def get_add_data():
    return jsonify(df.to_dict(orient='records'))


if __name__ == '__main__':
    main()
