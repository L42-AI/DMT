import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


class MLPFeatureImputation(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLPFeatureImputation, self).__init__()

        self.W = Parameter(torch.Tensor(input_size, hidden_size, input_size))
        self.b = Parameter(torch.Tensor(input_size, hidden_size))

        self.nonlinear_regression = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        """ Applied after the linear transformation above, projects the representation into a scalar,
        which is the imputation estimate."""

        m = torch.ones(input_size, hidden_size, input_size)
        stdv = 1. / math.sqrt(input_size)
        for i in range(input_size):
            m[i, :, i] = 0
            """ The mask is to prevent the imputation of a variable to use it's own value. This forces each 
            variable's value to be imputed from other variables, not from itself (otherwise it can just learn
            the identity function).
            """
        self.register_buffer('m', m)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        hidden = torch.cat(tuple(F.linear(x, self.W[i] * Variable(self.m[i]), self.b[i]).unsqueeze(2)
                            for i in range(len(self.W))), dim=2)
        """ Linear transformation of x (batch_size, T_max, vars) with W[i] (T_max, vars), resulting in the shape
        (batch_size, T_max, vars, hidden_size). Four-dimensional, because we add dimension 2, and concatanate 
        along that dimension all variables. 
        """

        z_h = self.nonlinear_regression(hidden)
        """ The non-linear regression projects down to shape (batch_size, T_max, vars, 1), which is then squeezed to 
        the imputation output (batch_size, T_max, vars)
        """

        return z_h.squeeze(-1)


class InputTemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)


class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

        """ This is a different way from the MLP to compute context vector. The author uses
        a GRU instead of the non-linear regression. Additionally, he uses a GRUCell instead 
        of GRU, in order to iterate through each time step. 
        """

    def forward(self, input, seq_lengths):
        T_max = input.shape[1]  # batch x time x dims
        """ Interestingly, the input is 3-dimensional. It is likely that batch is the patient 
        dimension. T_max is then just the longest observed sequence of all patients, assuming
        shorter patient sequences are padded. 
        """

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)

        for t in range(T_max):
            # slice all patients data at timepoint t, feed it into GRU cell with previous hidden 
            # state h
            h = self.rnn_cell(input[:, t, :], h) # (batch size, hidden_size)
            padding_mask = ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device) # (batch size, 1)
            hn = padding_mask * h + (1-padding_mask) * hn # broadcast (batch size, 1) (batch size, hidden size)
            """ Within a timestep, select only the hidden states of patients who still have observations 
            (so not padded). Interestingly, the padding seems to only be at the end of sequences, otherwise
            the masking logic wouldn't work. So for patients who still have observations at timestep t, h is 
            appended to hn, while for hidden_sizepatients who have no more observations, hn stays the same. At the end
            we have the hidden-states of all patients, outputted from the GRU cell at their last observed 
            time-step.
            """

        return hn # batch_size x context_hidden


class CATSI(nn.Module):
    def __init__(self, num_vars, hidden_size=64, context_hidden=32):
        """ 
        num_vars (int):         Number of variables
        hidden_size(int):       Dimension of the LSTM hidden states and cell states
        context_hidden(int):    Dimension of the context vector returned by MLP
        """
        super().__init__()

        # Attribute parameters to be accessed in the forward method
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.context_mlp = nn.Sequential(
            nn.Linear(3*self.num_vars+1, 2*context_hidden),
            nn.ReLU(),
            nn.Linear(2*context_hidden, context_hidden)
        )
        """ The purpose of context_mlp is to compress summary statistics of an individual into
        a context vector. The input size is 3 x num_vars + 1, because the input includes:
        - means (for every variable)
        - stds (for every variable)
        - missing rates (for every variable)
        - seq_lengths (number of timestamps, a single scalar)"""

        self.context_rnn = RNNContext(2*self.num_vars, context_hidden)
        """ Summarizes complex temporal dynamics of time series into another context vector"""

        self.initial_hidden = nn.Linear(2*context_hidden, 2*hidden_size)
        """ This takes both context vectors concatenated (batch size, context hidden * 2), and projects it to 
        (batch size, 2 * hidden_size). This is to initialize the hidden states for both the forward and 
        backward LSTM 
        """
        self.initial_cell_state = nn.Tanh()
        """
        Will be applied to to the output of initial_hidden to transform values of the hidden state to the 
        expected values (-1, 1) of the cell state.
        """

        self.rnn_cell_forward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)
        self.rnn_cell_backward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)
        """ LSTM inputs include: data tensor, mask, and both context vectors. Both output hidden states at 
        each timepoint t
        """
        self.decay_inputs = InputTemporalDecay(input_size=num_vars)

        self.recurrent_impute = nn.Linear(2*hidden_size, num_vars)
        """ This produces RNN-based imputation estimates from the hidden states of the forward and backward
        LSTM
        """
        self.feature_impute = MLPFeatureImputation(num_vars)
        """ Produces a second imputation estimation based solely on the relationship between variables at
        timepoint t, ignoring temporal dependencies
        """

        self.fuse_imputations = nn.Linear(2*num_vars, num_vars)
        """ Fuses the two imputation estimates for a final estimate
        """

    def forward(self, data):

        # number of time-steps
        seq_lengths = data['lengths'] # (pts, ): Duration of observation for each patient
        values = data['values']  # pts x time_stamps x vars
        masks = data['masks'] 
        deltas = data['deltas']

        """ The data here is an entire batch, as given by the dataloader iteration. """

        # compute context vector, h0 and c0
        T_max = values.shape[1]
        """ Extracts the maximum duration in the batch, this is the padded length that all sequences in the 
        batch have been padded to. 
        """

        padding_masks = torch.cat(tuple(((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                                   for t in range(T_max)), dim=1)
        """ This essentially marks which time-points for each patient is actual observation and which are 
        padded. 
        For timestep t, (t + 1) <= seq_lengths will give a boolean vector of shape (batch_size, ) denoting for 
        all patients whether they have a valid observation at t. This is concatenated to produce a matrix denoting
        at which timesteps which patient has padded values
        """

        padding_masks = padding_masks.unsqueeze(2).repeat(1, 1, values.shape[2])  # pts x time_stamps x vars
        """ This just expands the mask to cover every variable for each patient, thereby aligning it with the 
        data tensor in shape!
        """

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars
        """ The author smartly has previously replaced NaNs with the mean column/variable value before. Hence,
        when we sum the values, the mean replacements won't affect the mean computation, as they are already the 
        mean, haha. We then divide by the count of non-missing values, to obtain the average. He could have also 
        just used the mask...
        """

        data_variance = ((values - data_means.unsqueeze(1)) ** 2).sum(dim=1) / (masks.sum(dim=1) - 1) # pts x var
        """ Again, interesting that he replaced NaNs by column means... The mean subtracted from the mean = 0. 
        """

        data_stdev = data_variance ** 0.5 # pts x var
        data_missing_rate = 1 - masks.sum(dim=1) / padding_masks.sum(dim=1) # pts x var
        """ masks.sum(dim=1) gives the number of observations per variable, padding_masks.sum(dim=1) gives the 
        number of valid timesteps per patient (same for all variables). Dividing gives the observation rate
        per variable, and subtracting from 1 gives the missing rate per variable. 
        """

        data_stats = torch.cat((seq_lengths.unsqueeze(1).float(), data_means, data_stdev, data_missing_rate), dim=1)
        """ All stats are concatenated into shape (batch_size, 3 * vars + 1). This is for computing the first context 
        vector
        """

        # normalization
        min_max_norm = (data['max_vals'] - data['min_vals']).clamp(min=1e-8)
        normalized_values = (values - data['min_vals']) / min_max_norm
        normalized_means = (data_means - data['min_vals'].squeeze(1)) / min_max_norm.squeeze(1)

        x_prime = torch.zeros_like(normalized_values)
        x_prime[:, 0, :] = normalized_values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = normalized_values[:, t-1, :]
        """ Shifts all values by one timestep, for the decay computation
        """

        gamma = self.decay_inputs(deltas)
        x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        x_complement = (masks * normalized_values + (1-masks) * x_decay) * padding_masks
        """ Pre-imputation values using time-decay. The closer the last observed value, the more the imputed value
        will be similar to the last observed. If the last observed value is distant (smaller gamma), the imputed 
        value is closer to the mean. 
        """

        context_mlp = self.context_mlp(data_stats)
        context_rnn = self.context_rnn(torch.cat((x_complement, deltas), dim=-1), seq_lengths)
        context_vec = torch.cat((context_mlp, context_rnn), dim=1)
        """ The summary statistics of each patient (mean, std, missingrate, seq_lengths) are made into context vector 1,
        the pre-imputed data and the deltas are made into context vector 2, and both are concatenated. 
        We are inputing the deltas because this allows the context_rnn to differentiate between pre-imputed and real observations
        """
        h = self.initial_hidden(context_vec)
        c = self.initial_cell_state(h)
        """ Projected the context into hidden states and cell states for the two LSTM. Shape (batch_size, 2 * hidden_size)"""

        inputs = torch.cat([x_complement, masks, context_vec.unsqueeze(1).repeat(1, T_max, 1)], dim=-1)
        """ Result is shape: (batch_size, T_max, 2 * num_vars + 2 * context_hidden)"""

        h_forward, c_forward = h[:, :self.hidden_size], c[:, :self.hidden_size]
        h_backward, c_backward = h[:, self.hidden_size:], c[:, self.hidden_size:]
        """ These are the running hidden states for forward and backward (batch_size, hidden_size). Start with initial hidden states
        from the context vectors
        """

        hiddens_forward = h[:, :self.hidden_size].unsqueeze(1)
        hiddens_backward = h[:, self.hidden_size:].unsqueeze(1)
        """ These are the collection of hidden states, which will be appended iteratively across time(batch_size, 1, hidden_size)
        """

        for t in range(T_max-1):
            h_forward, c_forward = self.rnn_cell_forward(inputs[:, t, :],
                                                         (h_forward, c_forward))
            h_backward, c_backward = self.rnn_cell_backward(inputs[:, T_max-1-t, :],
                                                            (h_backward, c_backward))
            hiddens_forward = torch.cat((hiddens_forward, h_forward.unsqueeze(1)), dim=1)
            hiddens_backward = torch.cat((h_backward.unsqueeze(1), hiddens_backward), dim=1)
        
        """ The bi-directional processes the inputs and hidden states from both directions, computing
        forward and backward hidden states for each timestep, in the same (and correct) temporal order. Each time-step now has access
        to both past and future information.
        """

        rnn_imp = self.recurrent_impute(torch.cat((hiddens_forward, hiddens_backward), dim=2))
        """ Now, both sets of hidden states are concatenated and projected down to the data dimensions (batch_size, T_max, vars) and result
        thereby in the temporal imputation estimates
        """
        feat_imp = self.feature_impute(x_complement).squeeze(-1)
        """ Get cross-sectional imputation estimates
        """

        # imputation fusion
        beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imp = masks * normalized_values + (1-masks) * imp_fusion
        """ This is very cool, we use the time-deltas to compute weights for the imputation components for each missing value. If the 
        last observation was long ago, the model tends to use cross-sectional information, but if it is recent then we use the temporal
        information!
        """

        rnn_loss = F.mse_loss(rnn_imp * masks, normalized_values * masks, reduction='sum')
        feat_loss = F.mse_loss(feat_imp * masks, normalized_values * masks, reduction='sum')
        fusion_loss = F.mse_loss(imp_fusion * masks, normalized_values * masks, reduction='sum')
        total_loss = rnn_loss + feat_loss + fusion_loss
        """ Again, very cool. Instead of only training on the fusion loss, we train on the loss of all components. This makes it so that
        the model cannot just rely on a single component, all components must be strong performers and develop meaningful representations. 
        """

        def rescale(x):
            return torch.where(padding_masks==1, x * min_max_norm + data['min_vals'], padding_masks)

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imp)
        """ Rescale after training for interpretability and downstream use (f.e. imputing in our project!!!)"""

        out_dict = {
            'loss': total_loss / masks.sum(),
            'verbose_loss': [
                ('rnn_loss', rnn_loss / masks.sum(), masks.sum()),
                ('feat_loss', feat_loss / masks.sum(), masks.sum()),
                ('fusion_loss', fusion_loss / masks.sum(), masks.sum())
            ],
            'loss_count': masks.sum(), # number of imputations per batch
            'imputations': final_imp,
            'feat_imp': feat_imp,
            'hist_imp': rnn_imp,
            'orig': values
        }

        return out_dict

