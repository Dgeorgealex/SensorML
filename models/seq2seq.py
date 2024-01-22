import os

import numpy as np
import torch
import torch.nn as nn

from models.lstm import preprocess_data, create_datasets, plot_predictions
import matplotlib.pyplot as plt

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
        decoder_input = input_seq[:, -1, :].unsqueeze(1)

        # Decoder
        outputs = []
        for t in range(self.sequence_length):
            # At each time step, use hidden_state from the last step as the hidden state for the decoder
            out, (hidden_state, cell_state) = self.decoder_lstm(decoder_input, (hidden_state, cell_state))
            out = self.linear(out.squeeze(1))
            outputs.append(out.unsqueeze(1))
            decoder_input = out.unsqueeze(1)

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

    torch.save(model.state_dict(), '../trained_models/seq2seq.pth')
    return model


def seq2seq_predict(df, date, subdirectory):
    model = Seq2SeqModel(input_size=13, hidden_layer_size=100, output_size=13, num_layers=2,
                      sequence_length=48)
    model.load_state_dict(torch.load('../trained_models/seq2seq.pth'))

    X, y, scaler = preprocess_data(df, sequence_length=72, prediction_length=48)

    condition = df['Timestamp'] == date

    index_of_first_row = condition.idxmax() - 72
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

    for i in range(predictions.shape[1]):
        plt.figure(figsize=(10, 4))
        actuals = actuals[:48]
        predictions = predictions[:48]

        plt.plot(actuals[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted')
        plt.title(f'Feature: {feature_names[i]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        file_path = os.path.join(subdirectory, f'{feature_names[i]}')
        plt.savefig(file_path)
        plt.close()


def show_seq2seq(df, directory=None, num_epochs=3):
    X, y, scaler = preprocess_data(df, sequence_length=72, prediction_length=48)

    # Create datasets
    train_loader, test_loader = create_datasets(X, y)

    # Initialize and train the model
    model = Seq2SeqModel(input_size=X.shape[2], hidden_layer_size=100, output_size=y.shape[2], num_layers=2,
                         sequence_length=48)

    if os.path.isfile('../trained_models/seq2seq.pth'):
        model.load_state_dict(torch.load('../trained_models/seq2seq.pth'))
    else:
        model = train_model(model, train_loader, test_loader, num_epochs)

    feature_names = df.columns.tolist()[1:]
    plot_predictions(model, test_loader, scaler, feature_names, directory)
