import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_processing.data_processing import load_dataset


def preprocess_data(df, sequence_length, prediction_length):
    # Normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.drop(columns=["Timestamp"]))

    # Create sequences
    X, y = [], []
    for i in range(len(df_scaled) - sequence_length - prediction_length):
        X.append(df_scaled[i:(i + sequence_length)])
        y.append(df_scaled[i + sequence_length:i + sequence_length + prediction_length])

    return np.array(X), np.array(y), scaler


def create_datasets(X, y, test_size=0.2, batch_size=32):
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Convert to PyTorch tensors
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    # Create Data Loaders
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    return train_loader, test_loader


class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_layer_size = hidden_layer_size

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(output_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Encoder
        _, (hidden_state, cell_state) = self.encoder_lstm(input_seq)

        # Prepare decoder input (initially zeros)
        decoder_input = torch.zeros(input_seq.size(0), self.sequence_length, input_seq.size(2))

        # Decoder
        outputs = []
        for t in range(self.sequence_length):
            # At each time step, use hidden_state from the last step as the hidden state for the decoder
            out, (hidden_state, cell_state) = self.decoder_lstm(decoder_input[:, [t], :], (hidden_state, cell_state))
            out = self.linear(out.squeeze(1))
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs


def train_model(model, train_loader, test_loader, num_epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # batch_size hours features
            # 32          72     13
            # print(X_batch.shape)
            # 48          48     13
            # print(y_pred.shape)
            # 32          48     13
            # print(y_batch.shape)
            # print(y_pred)
            # print(y_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate on the test set
        model.eval()
        test_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_test_pred = model(X_batch)
                test_loss = criterion(y_test_pred, y_batch)
                test_losses.append(test_loss.item())
        print(f'Epoch {epoch} Test Loss: {np.mean(test_losses)}')

    return model


def main():
    print('Running Seq2Seq')
    df = load_dataset('../assets/SensorMLDataset.csv')

    X, y, scaler = preprocess_data(df, sequence_length=72, prediction_length=48)  # 72 hours for 3 days input

    # Create datasets
    train_loader, test_loader = create_datasets(X, y)
    print(f'Input size is {X.shape[2]}')
    print(f'Output size is {y.shape[2]}')

    # Initialize and train the model
    model = Seq2SeqModel(input_size=X.shape[2], hidden_layer_size=100, output_size=y.shape[2], num_layers=2,
                         sequence_length=48)
    model = train_model(model, train_loader, test_loader, 3)

    feature_names = df.columns.tolist()[1:]
    plot_predictions(model, test_loader, scaler, feature_names)


def plot_predictions(model, test_loader, scaler, feature_names):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_test_pred = model(X_batch)
            predictions.append(y_test_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = predictions[0]
    actuals = actuals[0]

    # Convert lists to numpy arrays
    print(predictions)
    # predictions = np.vstack(predictions)
    # actuals = np.vstack(actuals)
    predictions = np.vstack(predictions).reshape(-1, len(feature_names))
    actuals = np.vstack(actuals).reshape(-1, len(feature_names))

    print(f'Predictions shape is {predictions.shape}')
    print(f'Actual shape is {actuals.shape}')

    # Inverse transform the predictions and actuals
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)

    # Plot for each feature
    for i in range(predictions.shape[1]):
        plt.figure(figsize=(10, 4))
        # print(actuals[:, i])
        # print(predictions[:, i])
        actuals = actuals[:48]
        predictions = predictions[:48]

        plt.plot(actuals[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
