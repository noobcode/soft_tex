import torch as th
import torch.nn as nn
import torch.optim as opt

import numpy as np
import time
from tqdm import tqdm


"""
PARALLEL SOFT SENSING NETWORK
"""
class ParallelSoftSensingLSTM(nn.Module):

    def __init__(self, 
                 input_size_1, 
                 input_size_2, 
                 output_size, 
                 hidden_size_1, 
                 hidden_size_2, 
                 num_layers, 
                 dropout=0.0, 
                 bidirectional=False,
                 input_dropout_1=0.0,
                 input_dropout_2=0.0,
                 device=th.device('cpu')):
        """
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        """
        super(ParallelSoftSensingLSTM, self).__init__()
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.output_size = output_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.device = device
        self.h_x_1 = None # (h_n, c_n) or None
        self.h_x_2 = None

        self.input_dropout_1 = nn.Dropout(p=input_dropout_1)
        self.input_dropout_2 = nn.Dropout(p=input_dropout_2)

        # network architecture
        self.lstm_1 = nn.LSTM(input_size=input_size_1, hidden_size=hidden_size_1, num_layers=num_layers, 
                              dropout=dropout, bidirectional=bidirectional, device=device)
        self.lstm_2 = nn.LSTM(input_size=input_size_2, hidden_size=hidden_size_2, num_layers=num_layers, 
                              dropout=dropout, bidirectional=bidirectional, device=device)

        in_features_1 = 2 * hidden_size_1 if bidirectional == True else hidden_size_1
        in_features_2 = 2 * hidden_size_2 if bidirectional == True else hidden_size_2
        self.linear_1 = nn.Linear(in_features=in_features_1, out_features=int(hidden_size_1/2), device=device)
        self.linear_2 = nn.Linear(in_features=in_features_2, out_features=int(hidden_size_2/2), device=device)

        in_features_fusion = int((hidden_size_1 + hidden_size_2)/2)
        self.linear = nn.Linear(in_features=in_features_fusion, out_features=output_size, device=device)

    def forward(self, x):
        """
        h_x: None or (h_0, c_0)
        """
        if self.h_x_1 == None and self.h_x_2 == None:
            hx_1 = None
            hx_2 = None
        else:
            hx_1 = (self.h_x_1[0].detach(), self.h_x_1[1].detach())
            hx_2 = (self.h_x_2[0].detach(), self.h_x_2[1].detach())

        # extract the input features for the two (features are at the last dimension)
        x_1, x_2 = th.split(x, split_size_or_sections=self.input_size_1, dim=-1)

        x_1 = self.input_dropout_1(x_1)
        x_2 = self.input_dropout_2(x_2)

        y_1, self.h_x_1 = self.lstm_1(x_1, hx=hx_1)
        y_2, self.h_x_2 = self.lstm_2(x_2, hx=hx_2)

        # concatenate outputs of the two LSTMs (features are at the last dimension)
        y_1 = self.linear_1(y_1)
        y_2 = self.linear_2(y_2)
        
        y = th.concatenate((y_1, y_2), axis=-1)
        y = self.linear(y)

        return y, (self.h_x_1, self.h_x_2)
    
    def reset_states(self):
        self.h_x_1 = None
        self.h_x_2 = None

    def fit(self, X, Y, loss_fn, optimizer, n_epochs, validation_data=None, history=None,
            X_noise_scale=0.0, Y_noise_scale=0.0):
        if history == None:
            history = {'training_losses': [], 'validation_losses': []}

        progress_epoch_bar = tqdm(range(n_epochs), desc='training', leave=False)
        for epoch in progress_epoch_bar:
            self.train() # sets the module in training mode
            self.reset_states() # reset network hidden states at the beginning of each time-series

            # apply noise to input
            X_noise = th.Tensor(np.random.normal(loc=0, scale=X_noise_scale, size=(X.shape))).detach().to(self.device)
            Y_noise = th.Tensor(np.random.normal(loc=0, scale=Y_noise_scale, size=(Y.shape))).detach().to(self.device)

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
                                            'seq_len': X.shape[-2],
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
            series_losses[i] = loss.cpu().detach().numpy()
        
        return series_losses
    
    def predict(self, X_series):
        n_samples = X_series.shape[0]
        Y_hats = th.zeros((n_samples, 1, self.output_size)) # store predictions

        self.eval() # Set the module in evaluation mode (equivalent self.train(False))
        with th.no_grad():
            for i, x in enumerate(X_series): 
                y_hat, _ = self(x) # (1, 3)

                Y_hats[i] = y_hat

        return Y_hats

    def validate(self, X_series, Y_series, loss_fn):
        Y_hats = self.predict(X_series).to(self.device)

        assert Y_hats.shape == Y_series.shape

        with th.no_grad():
            loss = loss_fn(Y_series, Y_hats).cpu().numpy()

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
    

class ParallelSoftSensingGRU(nn.Module):

    def __init__(self, 
                 input_size_1, 
                 input_size_2, 
                 output_size, 
                 hidden_size_1, 
                 hidden_size_2, 
                 num_layers, 
                 dropout=0.0, 
                 bidirectional=False,
                 input_dropout_1=0.0,
                 input_dropout_2=0.0,
                 device=th.device('cpu')):
        """
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        """
        super(ParallelSoftSensingGRU, self).__init__()
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.output_size = output_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_layers = num_layers
        self.device = device
        self.h_x_1 = None # (h_n, c_n) or None
        self.h_x_2 = None

        self.input_dropout_1 = nn.Dropout(p=input_dropout_1)
        self.input_dropout_2 = nn.Dropout(p=input_dropout_2)

        # network architecture
        self.gru_1 = nn.GRU(input_size=input_size_1, hidden_size=hidden_size_1, num_layers=num_layers, 
                              dropout=dropout, bidirectional=bidirectional, device=device)
        self.gru_2 = nn.GRU(input_size=input_size_2, hidden_size=hidden_size_2, num_layers=num_layers, 
                              dropout=dropout, bidirectional=bidirectional, device=device)

        in_features_1 = 2 * hidden_size_1 if bidirectional == True else hidden_size_1
        in_features_2 = 2 * hidden_size_2 if bidirectional == True else hidden_size_2
        self.linear_1 = nn.Linear(in_features=in_features_1, out_features=int(hidden_size_1/2), device=device)
        self.linear_2 = nn.Linear(in_features=in_features_2, out_features=int(hidden_size_2/2), device=device)

        in_features_fusion = int((hidden_size_1 + hidden_size_2)/2)
        self.linear = nn.Linear(in_features=in_features_fusion, out_features=output_size, device=device)

    def forward(self, x):
        """
        h_x: None or (h_0, c_0)
        """
        if self.h_x_1 == None and self.h_x_2 == None:
            hx_1 = None
            hx_2 = None
        else:
            hx_1 = self.h_x_1.detach()
            hx_2 = self.h_x_2.detach()

        # extract the input features for the two (features are at the last dimension)
        x_1, x_2 = th.split(x, split_size_or_sections=self.input_size_1, dim=-1)

        x_1 = self.input_dropout_1(x_1)
        x_2 = self.input_dropout_2(x_2)

        y_1, self.h_x_1 = self.gru_1(x_1, hx=hx_1)
        y_2, self.h_x_2 = self.gru_2(x_2, hx=hx_2)

        # concatenate outputs of the two LSTMs (features are at the last dimension)
        y_1 = self.linear_1(y_1)
        y_2 = self.linear_2(y_2)
        
        y = th.concatenate((y_1, y_2), axis=-1)
        y = self.linear(y)

        return y, (self.h_x_1, self.h_x_2)
    
    def reset_states(self):
        self.h_x_1 = None
        self.h_x_2 = None

    def fit(self, X, Y, loss_fn, optimizer, n_epochs, validation_data=None, history=None,
            X_noise_scale=0.0, Y_noise_scale=0.0):
        if history == None:
            history = {'training_losses': [], 'validation_losses': []}

        progress_epoch_bar = tqdm(range(n_epochs), desc='training', leave=False)
        for epoch in progress_epoch_bar:
            self.train() # sets the module in training mode
            self.reset_states() # reset network hidden states at the beginning of each time-series

            # apply noise to input
            X_noise = th.Tensor(np.random.normal(loc=0, scale=X_noise_scale, size=(X.shape))).detach().to(self.device)
            Y_noise = th.Tensor(np.random.normal(loc=0, scale=Y_noise_scale, size=(Y.shape))).detach().to(self.device)

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
                                            'seq_len': X.shape[-2],
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
            series_losses[i] = loss.cpu().detach().numpy()
        
        return series_losses
    
    def predict(self, X_series):
        n_samples = X_series.shape[0]
        Y_hats = th.zeros((n_samples, 1, self.output_size)) # store predictions

        self.eval() # Set the module in evaluation mode (equivalent self.train(False))
        with th.no_grad():
            for i, x in enumerate(X_series): 
                y_hat, _ = self(x) # (1, 3)

                Y_hats[i] = y_hat

        return Y_hats

    def validate(self, X_series, Y_series, loss_fn):
        Y_hats = self.predict(X_series).to(self.device)

        assert Y_hats.shape == Y_series.shape

        with th.no_grad():
            loss = loss_fn(Y_series, Y_hats).cpu().numpy()

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
    