import torch
from torch import nn
import torch.nn.functional as F

SOS_token = 0

class lstm_probing(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, device, num_layers = 1):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        
        super(lstm_probing, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = 1)
        self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        print(lstm_out)
        print(lstm_out.shape)
        output = self.linear(lstm_out.squeeze(0)).type(torch.LongTensor).to(self.device)     
        return output, self.hidden

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