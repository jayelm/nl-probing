import torch
from torch import nn
import torch.nn.functional as F

SOS_token = 0

class lstm_probing(nn.Module):
    ''' Decodes hidden state output by encoder '''
    
    def __init__(self, input_size, hidden_size, device, output_size, num_layers = 1):
        
        super(lstm_probing, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.output_size = output_size

        self.h_projection = nn.Linear(in_features=self.input_size, out_features=self.input_size, bias=False)
        self.c_projection = nn.Linear(in_features=self.input_size, out_features=self.input_size, bias=False)

        self.lstm = nn.LSTMCell(input_size = 1 + self.hidden_size, hidden_size = self.hidden_size, bias=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)           

    def forward(self, y_input, encoder_hidden_states):
        
        y_lengths = [len(s) for s in y_input]
        init_hideen = self.h_projection(encoder_hidden_states)
        init_cell = self.c_projection(encoder_hidden_states)
        #y_input = torch.nn.utils.rnn.pack_padded_sequence(y_input, lengths=y_lengths, batch_first=True)
        probe_output, enc_hiddens = self.probe(y_input, init_hideen, init_cell)
        #enc_hiddens, lens_enc_hiddens = torch.nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)
        logits = self.linear(probe_output)
        logits = torch.permute(logits, (1, 2, 0)).to(self.device)
        #scores = F.log_softmax(logits)
        return logits   
        #lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        #output = self.linear(lstm_out.squeeze(0)).type(torch.LongTensor).to(self.device) 
        #return output, enc_hiddens

    def probe(self, explanation, decoder_hidden, decoder_cell):
        output = []
        for t in range(explanation.shape[1]): 
            explanation_t = explanation[:, t].unsqueeze(1)
            Ybar_t = torch.cat((explanation_t, decoder_hidden), -1)
            decoder_hidden, decoder_cell = self.lstm(Ybar_t, (decoder_hidden, decoder_cell))
            output.append(decoder_hidden)
        output = torch.stack(output, dim=0)
        return output, decoder_hidden
            
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