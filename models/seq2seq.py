import os
import random

import numpy as np
import torch
import torch.nn as nn

from models.lstm import preprocess_data, create_datasets, plot_predictions
import matplotlib.pyplot as plt


class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, num_layers, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_layer_size = hidden_layer_size

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(output_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq, target_seq=None, teaching_forcing_epsilon=0.15):
        # Encoder
        _, (hidden_state, cell_state) = self.encoder_lstm(input_seq)

        if target_seq is not None:
            decoder_input = target_seq[:, 0, :].unsqueeze(1)
        else: decoder_input = input_seq[:, -1, :].unsqueeze(1)
        # Decoder
        outputs = []
        for t in range(self.sequence_length):
            out, (hidden_state, cell_state) = self.decoder_lstm(decoder_input, (hidden_state, cell_state))
            out = self.linear(out.squeeze(1))
            outputs.append(out.unsqueeze(1))

            if t + 1 < self.sequence_length and target_seq is not None and random.random() < teaching_forcing_epsilon:
                decoder_input = target_seq[:, t + 1, :].unsqueeze(1)
            else: decoder_input = out.unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def train_model(model, train_loader, test_loader, num_epochs=50, learning_size=72, predict_size=48):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if os.path.isfile(f'./trained_models/{learning_size}_seq2seq_best_loss_{predict_size}.txt'):
        with open(f'./trained_models/{learning_size}_seq2seq_best_loss_{predict_size}.txt', 'r') as file:
            content = file.read()
            best_loss = float(content)
    else: best_loss = 1.0

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch, target_seq=y_batch)  # Enable teaching forcing during training by passing the target sequence
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        test_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_test_pred = model(X_batch)  # During evaluation, only pass the input sequence (no teaching forcing)
                test_loss = criterion(y_test_pred, y_batch)
                test_losses.append(test_loss.item())
        print(f'Epoch {epoch} Test Loss: {np.mean(test_losses)}')
        if np.mean(test_losses) < best_loss:
            best_loss = np.mean(test_losses)
            torch.save(model.state_dict(), f'./trained_models/{learning_size}_seq2seq_{predict_size}.pth')
            with open(f'./trained_models/{learning_size}_seq2seq_best_loss_{predict_size}.txt', 'w') as file:
                file.write(str(best_loss))


    return model


def show_seq2seq(df, directory=None, num_epochs=10, learning_size=168, predict_size=120):
    X, y, scaler = preprocess_data(df, sequence_length=learning_size, prediction_length=predict_size)

    # Create datasets
    train_loader, test_loader = create_datasets(X, y)

    # Initialize and train the model
    model = Seq2SeqModel(input_size=X.shape[2], output_size=y.shape[2], hidden_layer_size=128, num_layers=2, sequence_length=predict_size)

    if os.path.isfile(f'./trained_models/{learning_size}_seq2seq_{predict_size}.pth'):
        model.load_state_dict(torch.load(f'./trained_models/{learning_size}_seq2seq_{predict_size}.pth'))
    else:
        model = train_model(model, train_loader, test_loader, num_epochs, learning_size=learning_size, predict_size=predict_size)

    feature_names = df.columns.tolist()[1:]
    plot_predictions(model, test_loader, scaler, feature_names, directory)


#"""
def seq2seq_predict(df, date, subdirectory, learning_size=336, predict_size=168, hidden_size=128):
    model = Seq2SeqModel(input_size=13, output_size=13, hidden_layer_size=hidden_size, num_layers=2, sequence_length=predict_size)  # hidden_layer_size = 128, pentru 336->168, 100 in rest
    model.load_state_dict(torch.load(f'../trained_models/{learning_size}_seq2seq_{predict_size}.pth'))

    X, y, scaler = preprocess_data(df, sequence_length=learning_size, prediction_length=predict_size)

    condition = df['Timestamp'] == date

    index_of_first_row = condition.idxmax() - learning_size
    x_value = X[index_of_first_row]
    y_value = y[index_of_first_row]

    feature_names = df.columns.tolist()[1:]

    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        x_value = torch.from_numpy(x_value).to(torch.float32)
        y_pred = model(x_value.unsqueeze(0))
        predictions = y_pred.cpu().numpy()

    predictions = predictions[0]
    actuals = y_value

    predictions = np.vstack(predictions).reshape(-1, len(feature_names))
    actuals = np.vstack(actuals).reshape(-1, len(feature_names))

    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    mse_error = {}
    for i in range(predictions.shape[1]):
        plt.figure(figsize=(10, 4))
        actuals = actuals[:predict_size]
        predictions = predictions[:predict_size]

        plt.plot(actuals[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        file_path = os.path.join(subdirectory, f'{feature_names[i]}')
        plt.savefig(file_path)
        plt.close()

        mse_error[feature_names[i]] = np.mean((predictions[:, i] - actuals[:, i])**2)

    return zip(predictions[:, 1], predictions[:, 2], range(1, 41)), mse_error