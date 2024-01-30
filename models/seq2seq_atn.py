import os.path
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


LEARNING_RATE = 0.0001
NUM_EPOCHS = 100

SEQUENCE_LENGTH = 72
PREDICTION_LENGTH = 48
BATCH_SIZE = 32

HIDDEN_LAYER = 32
NUM_LAYERS = 1


class Encoder(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input_seq):
        output, hidden = self.gru(input_seq)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(2*hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, curr_hidden, enc_outputs):
        hidden = curr_hidden.unsqueeze(1).repeat(1, enc_outputs.shape[1], 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        weightings = nn.functional.softmax(attention, dim=1)

        return weightings


class DecoderWithAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.attention_model = Attention(hidden_size)
        self.gru = nn.GRU(feature_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, feature_size)

    def forward(self, dec_input, hidden, enc_outputs, dec_output_len, expected_outputs, teacher_force_prob=None):
        curr_input = dec_input.clone()

        batch_size = dec_input.shape[0]

        outputs = torch.zeros(batch_size, dec_output_len, self.feature_size)

        for t in range(dec_output_len):
            # for attention
            weightings = self.attention_model(hidden[-1], enc_outputs)

            weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)

            dec_output, hidden = self.gru(torch.cat((curr_input.unsqueeze(1), weighted_sum), dim=2), hidden)

            curr_output = self.out(hidden).squeeze(0)

            outputs[:, t:t+1, :] = curr_output.unsqueeze(1)

            teacher_force = random.random() < teacher_force_prob if teacher_force_prob is not None else False

            if teacher_force:
                curr_input = expected_outputs[:, t:t+1, :].squeeze(1)
            else:
                curr_input = curr_output

        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers):
        super().__init__()

        self.encoder = Encoder(feature_size, hidden_size, num_layers)

        self.decoder = DecoderWithAttention(feature_size, hidden_size, num_layers)

        self.opt = torch.optim.Adam(self.parameters(), LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def forward(self, inputs, expected_outputs=None, teacher_force_prob=None):
        enc_outputs, hidden = self.encoder(inputs)

        # dec_output_len = expected_outputs.shape[1]

        dec_output_len = PREDICTION_LENGTH

        outputs = self.decoder(inputs[:, -1, :], hidden, enc_outputs, dec_output_len, expected_outputs, teacher_force_prob)

        return outputs


def preprocess_train_data(df, sequence_length, prediction_length, scaler):
    df_scaled = scaler.transform(df.drop(columns=["Timestamp"]))

    x, y = [], []
    for i in range(len(df_scaled) - sequence_length - prediction_length):
        x.append(df_scaled[i:i+sequence_length])
        y.append(df_scaled[i + sequence_length:i + sequence_length + prediction_length])

    return np.array(x), np.array(y)


def fit_scaler(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df.drop(columns=["Timestamp"]))
    return scaler


def create_train_dataset(x, y, batch_size):
    train_data = TensorDataset(torch.Tensor(x), torch.Tensor(y))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)

    return train_loader


def train_model(model, train_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(num_epochs):
        model.train()
        err = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch, y_batch, 0.2)

            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            err += loss.item()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'../trained_models/seq2seq_atn/seq2seq{epoch}.pth')

        print(err)

    return model


def get_trained_model_x_y(df, scaler, name=None, sequence_length=SEQUENCE_LENGTH, prediction_length=PREDICTION_LENGTH):
    x, y = preprocess_train_data(df, sequence_length, prediction_length, scaler)

    train_loader = create_train_dataset(x, y, BATCH_SIZE)

    model = Seq2Seq(x.shape[2], HIDDEN_LAYER, NUM_LAYERS)

    if name is not None:
        file_path = os.path.join("../trained_models/seq2seq_atn", name)
        model.load_state_dict(torch.load(file_path))
    else:
        model = train_model(model, train_loader, NUM_EPOCHS)
        torch.save(model.state_dict(), '../trained_models/seq2seq_atn/seq2seq.pth')

    return model, x, y


# for testing without ui
def test_model(model, x_values, y_values, scaler, feature_names):

    with torch.no_grad():
        x_value = torch.from_numpy(x_values).to(torch.float32)
        y_pred = model(x_value.unsqueeze(0))
        prediction = y_pred.cpu().numpy()

    prediction = prediction[0]
    actual = y_values

    prediction = np.vstack(prediction).reshape(-1, len(feature_names))
    actual = np.vstack(actual).reshape(-1, len(feature_names))

    prediction = scaler.inverse_transform(prediction)
    actual = scaler.inverse_transform(actual)

    for i in range(prediction.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(actual[:, i], label='Actual')
        plt.plot(prediction[:, i], label='Predicted')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        plt.show()

    mse = ((prediction - actual) ** 2)
    print(mse)


def main():
    df = pd.read_csv("../assets/SensorMLDataset.csv")
    scaler = fit_scaler(df)

    model, x, y = get_trained_model_x_y(df, scaler, )

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    test_date = datetime(2023, 4, 10)
    condition = df['Timestamp'] == test_date
    index_of_first_row = condition.idxmax() - SEQUENCE_LENGTH

    x_values = x[index_of_first_row]
    y_values = y[index_of_first_row]

    feature_names = df.columns.tolist()[1:]

    test_model(model, x_values, y_values, scaler, feature_names)


if __name__ == "__main__":
    main()


# function that will be called from user interface
# df = dataframe
# name = name of model
# date = date from which the model will predict
def ask_model_seq2seq_attention(df, name, date, sequence_length, prediction_length):
    scaler = fit_scaler(df)

    model, x, y = get_trained_model_x_y(df, scaler, name, sequence_length, prediction_length)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    condition = df['Timestamp'] == date
    index_of_first_row = condition.idxmax() - sequence_length

    x_values = x[index_of_first_row]
    y_values = y[index_of_first_row]

    feature_names = df.columns.tolist()[1:]

    with torch.no_grad():
        x_value = torch.from_numpy(x_values).to(torch.float32)
        y_pred = model(x_value.unsqueeze(0))
        prediction = y_pred.cpu().numpy()

    prediction = prediction[0]
    actual = y_values

    prediction = np.vstack(prediction).reshape(-1, len(feature_names))
    actual = np.vstack(actual).reshape(-1, len(feature_names))

    # here you save the photos
    for i in range(prediction.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(actual[:, i], label='Actual')
        plt.plot(prediction[:, i], label='Predicted')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        plt.show()

    mse = np.mean((prediction - actual)**2)