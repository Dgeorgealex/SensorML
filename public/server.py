import os
from datetime import datetime

from flask import Flask, send_file, render_template, request

from main import load_dataset
from models.inferencer import make_inference
from data_processing.graphs import (show_correlation_matrix, show_variable_distributions,
                                    show_calendar_plots)
from models.propheting import prophet_uni_regressor
from models.lstm import show_lstm, lstm_predict
from models.seq2seq import show_seq2seq

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
    show_lstm(df, lstm_dir, 10)
    show_seq2seq(df, seq2seq_dir, 10)
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


# PROPHET PLOTS

@app.route('/prophet-sliders')
def prophet_sliders():
    return render_template('prophet_sliders.html')


@app.route('/generate-graph')
def prophet_image():
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    start_date_dir = datetime.strptime(start_date, '%m/%d/%Y').strftime('%Y%m%d')
    end_date_dir = datetime.strptime(end_date, '%m/%d/%Y').strftime('%Y%m%d')

    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, start_date_dir + end_date_dir)
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    result_data = prophet_uni_regressor(df, start_date, end_date, subdirectory)
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
    print(disease_threats)
    return render_template('prophet_graph.html', columns=df.columns[1:],
                           start_date=start_date_dir, end_date=end_date_dir, diseases=disease_threats)


@app.route('/prophet_images/<column>/<start_date>/<end_date>')
def prophet_images(column, start_date, end_date):
    directory = os.path.join(script_directory, 'prophet_images')
    subdirectory = os.path.join(directory, start_date + end_date)

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


# LSTM & SEQ2SEQ
@app.route('/lstm-date-selector')
def lstm_select():
    return render_template('lstm_date_selector.html')


@app.route('/generate-prediction-lstm')
def lstm_prediction():
    year = int(request.args.get('year'))
    month = int(request.args.get('month'))
    day = int(request.args.get('day'))

    date = datetime(year, month, day)

    date_string = date.strftime("'%Y%m%d")

    directory = os.path.join(script_directory, 'lstm_images')
    subdirectory = os.path.join(directory, date_string)

    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)

    lstm_predict(df, date, subdirectory)

    columns = df.columns[1:]
    return render_template('lstm.html', columns=columns, date=date_string)


@app.route('/lstm_page_a/<column>/<date>')
def lstm_page_a(column, date):
    directory = os.path.join(script_directory, 'lstm_images')
    subdirectory = os.path.join(directory, date)

    file_path = os.path.join(subdirectory, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404


@app.route('/show_lstm')
def lstm_page():
    columns = df.columns[1:]
    return render_template('lscm.html', columns=columns)


@app.route('/lstm/<column>')
def lstm(column):
    file_path = os.path.join(lstm_dir, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
        return "Image not found", 404


@app.route('/show_seq2seq')
def seq2seq_page():
    columns = df.columns[1:]
    return render_template('seq2seq.html', columns=columns)


@app.route('/seq2seq/<column>')
def seq2seq(column):
    file_path = os.path.join(seq2seq_dir, f'{column}.png')
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    else:
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


if __name__ == '__main__':
    main()
