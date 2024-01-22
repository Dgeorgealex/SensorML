import os.path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, sequence_length):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # predictions = self.linear(lstm_out)
        # Reshape the predictions to match the target format
        # predictions = predictions.reshape(-1, self.sequence_length, predictions.size(-1))
        # predictions = self.linear(lstm_out[:, -1, :]) ##
        predictions = self.linear(lstm_out[:, -self.sequence_length:, :])
        return predictions


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
        # model.eval()
        # test_losses = []
        # with torch.no_grad():
        #     for X_batch, y_batch in test_loader:
        #         y_test_pred = model(X_batch)
        #         test_loss = criterion(y_test_pred, y_batch)
        #         test_losses.append(test_loss.item())
        # print(f'Epoch {epoch} Test Loss: {np.mean(test_losses)}')

    return model


def show_lstm(df, directory=None, num_epochs=3):
    X, y, scaler = preprocess_data(df, sequence_length=72, prediction_length=48)  # 72 hours for 3 days input

    # Create datasets
    train_loader, test_loader = create_datasets(X, y)

    # Initialize and train the model
    model = LSTMModel(input_size=X.shape[2], hidden_layer_size=100, output_size=y.shape[2], num_layers=2,
                      sequence_length=48)
    model = train_model(model, train_loader, test_loader, num_epochs)

    feature_names = df.columns.tolist()[1:]
    plot_predictions(model, test_loader, scaler, feature_names, directory)


def plot_predictions(model, test_loader, scaler, feature_names, directory=None):
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
    predictions = np.vstack(predictions).reshape(-1, len(feature_names))
    actuals = np.vstack(actuals).reshape(-1, len(feature_names))

    # print(f'Predictions shape is {predictions.shape}')
    # print(f'Actual shape is {actuals.shape}')

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
        if directory:
            file_path = os.path.join(directory, f'{feature_names[i]}')
            plt.savefig(file_path)
            plt.close()
        else:
            plt.show()
