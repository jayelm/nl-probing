import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class lstm_probing(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, lstm_hidden_size, device, output_size):
        
        super(lstm_probing, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        self.output_size = output_size

        self.h_projection = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)

        self.embedding = nn.Embedding(self.input_size, self.lstm_hidden_size)
        self.lstm = nn.LSTM(input_size = self.lstm_hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)           

    def forward(self, y_input, seq_lengths, encoder_hidden_states):
        inputs = y_input[:, :-1]  # Feed in tokens at time t. We don't need to feed in the last token since we don't predict afterwards.
        targets = y_input[:, 1:]  # Predict tokens at time t+1
        predict_lengths = seq_lengths.cpu() - 1

        embedded = self.embedding(inputs)  # (B, seq_len, hidden_size)
        
        packed_input = pack_padded_sequence(embedded, predict_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input, encoder_hidden_states)

        # Obtain logits for each next token.
        packed_hidden_states = packed_output.data  # (however_many_tokens_in_batch, hidden_size)
        packed_logits = self.linear(packed_hidden_states)  # (however_many_tokens_in_batch, vocab_size)

        # Pack the targets in the same way, so the targets and hidden states align.
        packed_targets = pack_padded_sequence(targets, predict_lengths, batch_first=True, enforce_sorted=False).data

        return packed_logits, packed_targets
        
    def init_hidden(self, encoder_hidden_states):
        return (self.h_projection(encoder_hidden_states),
                self.c_projection(encoder_hidden_states))

    def generate(self, word, hidden):
        embed = self.embedding(word)
        output, (hidden_states, cell_states) = self.lstm(embed, hidden)
        logits = self.linear(hidden_states)
        return output, (hidden_states, cell_states), logits
