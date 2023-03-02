import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SOS_token = 0

class lstm_probing(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, device, output_size):
        
        super(lstm_probing, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.output_size = output_size

        self.h_projection = nn.Linear(in_features=self.input_size, out_features=self.input_size, bias=False)
        self.c_projection = nn.Linear(in_features=self.input_size, out_features=self.input_size, bias=False)

        # Use number of tokens in GPT-2 tokenizer.
        self.embedding = nn.Embedding(50257, self.hidden_size)
        self.lstm = nn.LSTM(input_size = self.hidden_size, hidden_size=self.hidden_size, bias=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)           

    def forward(self, y_input, seq_lengths, encoder_hidden_states):
        inputs = y_input[:, :-1]  # Feed in tokens at time t. We don't need to feed in the last token since we don't predict afterwards.
        targets = y_input[:, 1:]  # Predict tokens at time t+1
        predict_lengths = seq_lengths.cpu() - 1

        embedded = self.embedding(inputs)  # (B, seq_len, hidden_size)
        #print(embedded)
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
            
'''
class Model(nn.Module):
    def __init__(self, dataset, input_size, hidden_size = 32, output_size = 32, num_layers = 1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.hidden_size,
        )
        self.lstm = nn.LSTMCell(
            input_size=self.input_size+self.hidden_size,
            hidden_size=self.hidden_size,
            bias=True
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs, explanation, state_h, state_c):
        probe_output = self.probe(encoder_outputs, explanation, state_h, state_c)
        logits = self.fc(probe_output)
        scores = F.log_softmax(logits)
        return scores


    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

    def probe(self, encoder_outputs, explanation, state_h, state_c):
        embed = self.embedding(encoder_outputs)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        explanation.insert(0, SOS_token)
        for di in range(len(explanation)):
            Ybar_t = torch.cat((explanation[di], o_prev), 1)
            y_pred, (state_h, state_c) = self.step(Ybar_t, (state_h, state_c), decoder_input)
            o_prev - 
            loss = criterion(y_pred.transpose(1, 2), y)
            decoder_input = y[di]

            embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)

    def step(self, ):
'''