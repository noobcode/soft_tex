import torch as th
import numpy as np
import torch.nn as nn
import torch.optim as opt
import time
from tqdm import tqdm


class SoftSensingLSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.0, 
                 device=th.device('cpu'),
                 input_scaler=None, output_scaler=None):
        """
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        """
        super(SoftSensingLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.h_x = None # (h_n, c_n) or None

        # network architecture
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size, device=device)

        # input and output scalers (remember statistics of the training data)
        self.input_size = input_scaler
        self.output_scaler = output_scaler


    def forward(self, x):
        """
        h_x: None or (h_0, c_0)
        """
        if self.h_x == None:
            hx = None
        else:
            hx = (self.h_x[0].detach(), self.h_x[1].detach())

        y, self.h_x = self.lstm(x, hx=hx)
        y = self.linear(y)

        return y, self.h_x
    
    def reset_states(self):
        self.h_x = None

    def fit(self, X, Y, loss_fn, optimizer, n_epochs, validation_data=None, history=None,
            X_noise_scale=0.0, Y_noise_scale=0.0):
        if history == None:
            history = {'training_losses': [], 'validation_losses': []}

        progress_epoch_bar = tqdm(range(n_epochs))
        for epoch in progress_epoch_bar:
            self.train() # sets the module in training mode
            self.reset_states() # reset network hidden states at the beginning of each time-series

            # apply noise to input
            X_noise = th.Tensor(np.random.normal(loc=0, scale=X_noise_scale, size=(X.shape))).detach()
            Y_noise = th.Tensor(np.random.normal(loc=0, scale=Y_noise_scale, size=(Y.shape))).detach()

            # learn on one time series
            series_losses = self._train_series(X + X_noise, Y + Y_noise, loss_fn, optimizer)
        
            history['training_losses'].append(np.mean(series_losses))

            # Validation set
            if validation_data != None:
                X_val_series, Y_val_series = validation_data

                self.eval()
                self.reset_states()
                val_loss, _ = self.validate(X_val_series, Y_val_series, loss_fn)

                history['validation_losses'].append(val_loss)
            else:
                history['validation_losses'].append(None)

            # add progress after progress bar
            progress_epoch_bar.set_postfix({'n_sequences': X.shape[0],
                                            'seq_len': X.shape[1],
                                            'TR loss': history['training_losses'][-1],
                                            'VL loss': history['validation_losses'][-1]})
        progress_epoch_bar.close()    

        return history
    
    def _train_series(self, X, Y, loss_fn, optimizer):
        n_sequences = X.shape[0]
        series_losses = np.zeros(n_sequences)  # store losses

        for i, (x, y) in enumerate(zip(X, Y)):
            # x is (seq_len, 3)
            # y is (seq_len, 3)
            # Clear accumulated gradient before each instance
            self.zero_grad()

            # Forward pass
            y_hat, _ = self(x) # y_hat is (sequence_len, 3)

            # Compute the loss, the loss gradients, and update the parameters with the optimizer
            loss = loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()

            # detach() returns a new Tensor detached from the current graph (it will not require gradient) 
            series_losses[i] = loss.detach().numpy()
        
        return series_losses
    
    def predict(self, X_series):
        n_samples = X_series.shape[0]
        Y_hats = th.zeros((n_samples, 1, self.output_size)) # store predictions

        self.eval() # Set the module in evaluation mode (equivalent self.train(False))
        with th.no_grad():
            for i, x in tqdm(enumerate(X_series)): 
                y_hat, _ = self(x) # (1, 3)

                Y_hats[i] = y_hat

        return Y_hats

    def validate(self, X_series, Y_series, loss_fn):
        Y_hats = self.predict(X_series)

        assert Y_hats.shape == Y_series.shape

        with th.no_grad():
            loss = loss_fn(Y_series, Y_hats).numpy()

        return loss, Y_hats
    
    def unfold_dataset(self, X, Y, sequence_len, sequence_shift):
        """
        Prepares dataset for training the recurrent neural networks from time series X and Y
        """
        # dimension - dimension in which unfolding happens
        # size - the size of each slice that is unfolded
        # step - the step between each slice
        X_unfold = X.unfold(dimension=0, size=sequence_len, step=sequence_shift) # (time, 3, seq_len)
        X_unfold = X_unfold.transpose(-2, -1).to(self.device) # (time, seq_len, 3)

        Y_unfold = Y.unfold(dimension=0, size=sequence_len, step=sequence_shift) # (time, 3, seq_len)
        Y_unfold = Y_unfold.transpose(-2, -1).to(self.device) # (time, seq_len, 3)

        return X_unfold, Y_unfold
    








""""
LSTM-CNN-LINEAR
"""

class SoftSensingLSTMCNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.0, 
                 device=th.device('cpu'),
                 input_scaler=None, output_scaler=None):
        """
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        """
        super(SoftSensingLSTMCNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.h_x = None # (h_n, c_n) or None

        # network architecture
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=3, padding=3), 
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=3, padding=3), 
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2),
                                  nn.Flatten())
        
        with th.no_grad():
            dummy_x = th.ones((1, 1, hidden_size)) # batch_size, in_channels, n_features 
            dummy_y = self.conv(dummy_x)
            n_flatten = dummy_y.shape[1]

        self.linear = nn.Linear(in_features=n_flatten, out_features=output_size, device=device)

        # input and output scalers (remember statistics of the training data)
        self.input_size = input_scaler
        self.output_scaler = output_scaler
        

    def forward(self, x):
        """
        h_x: None or (h_0, c_0)
        """
        if self.h_x == None:
            hx = None
        else:
            hx = (self.h_x[0].detach(), self.h_x[1].detach())

        y, self.h_x = self.lstm(x, hx=hx)
        y = self.conv(y.view(-1, 1, self.hidden_size)) # batch_size, in_channels, n_features
        y = self.linear(y)

        return y, self.h_x
    
    def reset_states(self):
        self.h_x = None

    def fit(self, X, Y, loss_fn, optimizer, n_epochs, validation_data=None, history=None,
            X_noise_scale=0.0, Y_noise_scale=0.0):
        if history == None:
            history = {'training_losses': [], 'validation_losses': []}

        progress_epoch_bar = tqdm(range(n_epochs))
        for epoch in progress_epoch_bar:
            self.train() # sets the module in training mode
            self.reset_states() # reset network hidden states at the beginning of each time-series

            # apply noise to input
            X_noise = th.Tensor(np.random.normal(loc=0, scale=X_noise_scale, size=(X.shape))).detach()
            Y_noise = th.Tensor(np.random.normal(loc=0, scale=Y_noise_scale, size=(Y.shape))).detach()

            # learn on one time series
            series_losses = self._train_series(X + X_noise, Y + Y_noise, loss_fn, optimizer)
        
            history['training_losses'].append(np.mean(series_losses))

            # Validation set
            if validation_data != None:
                X_val_series, Y_val_series = validation_data

                self.eval()
                self.reset_states()
                val_loss, _ = self.validate(X_val_series, Y_val_series, loss_fn)

                history['validation_losses'].append(val_loss)
            else:
                history['validation_losses'].append(None)

            # add progress after progress bar
            progress_epoch_bar.set_postfix({'n_sequences': X.shape[0],
                                            'seq_len': X.shape[1],
                                            'TR loss': history['training_losses'][-1],
                                            'VL loss': history['validation_losses'][-1]})
        progress_epoch_bar.close()    

        return history
    
    def _train_series(self, X, Y, loss_fn, optimizer):
        n_sequences = X.shape[0]
        series_losses = np.zeros(n_sequences)  # store losses

        for i, (x, y) in enumerate(zip(X, Y)):
            # x is (seq_len, 3)
            # y is (seq_len, 3)
            # Clear accumulated gradient before each instance
            self.zero_grad()

            # Forward pass
            y_hat, _ = self(x) # y_hat is (sequence_len, 3)

            # Compute the loss, the loss gradients, and update the parameters with the optimizer
            loss = loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()

            # detach() returns a new Tensor detached from the current graph (it will not require gradient) 
            series_losses[i] = loss.detach().numpy()
        
        return series_losses
    
    def predict(self, X_series):
        n_samples = X_series.shape[0]
        Y_hats = th.zeros((n_samples, 1, self.output_size)) # store predictions

        self.eval() # Set the module in evaluation mode (equivalent self.train(False))
        with th.no_grad():
            for i, x in tqdm(enumerate(X_series)): 
                y_hat, _ = self(x) # (1, 3)

                Y_hats[i] = y_hat

        return Y_hats

    def validate(self, X_series, Y_series, loss_fn):
        Y_hats = self.predict(X_series)

        assert Y_hats.shape == Y_series.shape

        with th.no_grad():
            loss = loss_fn(Y_series, Y_hats).numpy()

        return loss, Y_hats
    
    def unfold_dataset(self, X, Y, sequence_len, sequence_shift):
        """
        Prepares dataset for training the recurrent neural networks from time series X and Y
        """
        # dimension - dimension in which unfolding happens
        # size - the size of each slice that is unfolded
        # step - the step between each slice
        X_unfold = X.unfold(dimension=0, size=sequence_len, step=sequence_shift) # (time, 3, seq_len)
        X_unfold = X_unfold.transpose(-2, -1).to(self.device) # (time, seq_len, 3)

        Y_unfold = Y.unfold(dimension=0, size=sequence_len, step=sequence_shift) # (time, 3, seq_len)
        Y_unfold = Y_unfold.transpose(-2, -1).to(self.device) # (time, seq_len, 3)

        return X_unfold, Y_unfold