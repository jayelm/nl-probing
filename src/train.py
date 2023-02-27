import argparse
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from lstm_probing_model import lstm_probing
import data
from evaluation import evaluate
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Model
import dataset

SOS_token = 0

def collate_fn_decode(data):
    hidden_states, explanation = zip(*data)
    return hidden_states, explanation

def train_iterations(encoder_states, exp_dataset, tokenizer_length, args):

    training_dataset = dataset.TrainDataset(encoder_states, exp_dataset)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    criterion = nn.CrossEntropyLoss()
    plot_loss_total = 0
    plot_losses = []

    for epoch in range(args.max_epochs):
        loss = train(dataloader, criterion, encoder_states, epoch, tokenizer_length, args)
        '''
        if epoch % args.plot_every == 0:
            plot_loss_avg = plot_loss_total / args.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        '''


def train(dataloader, criterion, encoder_states, epoch, tokenizer_length, args):
    model = lstm_probing(input_size=encoder_states.shape[2], hidden_size=encoder_states.shape[2], device=args.device, 
    output_size=tokenizer_length)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    # x dimension: (batch, seq_lenth=798)
    # y dimension: (batch, seq_lenth=55)
    for batch, (x, y) in enumerate(dataloader):
        loss = 0
        y_tensor = torch.tensor(np.asarray(y),  dtype = torch.long).to(args.device)
        optimizer.zero_grad()
        #outputs = torch.zeros(y_tensor.shape[1], args.batch_size, len(x[0]))
        #outputs = torch.zeros(len(y), len(x[0]))

        #decoder_input = torch.zeros(args.batch_size, 1, device=args.device)
        #decoder_input = torch.zeros(len(x[0]), device=args.device)
        x_tensor = torch.tensor(x).to(args.device)
        #x_tensor = torch.tensor(x).to(args.device)
        #decoder_hidden = (x_tensor, x_tensor)
        probing_model_output = model(y_tensor, x_tensor)
        '''
        print("### labels.long().min()", y_tensor.long().min())
        print("### labels.min()", y_tensor.min())
        print("### labels.long().max()", y_tensor.long().max())
        print("### labels.max()", y_tensor.max())
        print("### pred.long().min()", probing_model_output.long().min())
        print("### pred.min()", probing_model_output.min())
        print("### pred.long().max()", probing_model_output.long().max())
        print("### pred.max()", probing_model_output.max())
        '''
        loss = criterion(probing_model_output, y_tensor)
        batch_loss = loss.item()
        x_tensor.detach()
        y_tensor.detach()
        probing_model_output.detach()
        '''
        for t in range(y_tensor.shape[1]): 
            decoder_output, decoder_hidden = model(decoder_input, x_tensor)
            #decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
            #outputs[t] = decoder_output
            print(decoder_output.squeeze())
            print(y_tensor[:, t])
            loss += criterion(decoder_output.squeeze(), y_tensor[:, t])
            decoder_input = y_tensor[:, t]
        '''

        loss.backward()
        optimizer.step()

        print({ 'epoch': epoch, 'batch': batch, 'loss': batch_loss })
        return batch_loss / args.batch_size

@hydra.main(config_path="conf", config_name="train_probing_config")
def main(args: DictConfig) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_model = GPT2Model.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer_model.resize_token_embeddings(len(tokenizer))
    explanation_dataset = data.load_explanation(args, tokenizer)
    for split in args.data.splits:
        encoder_states = data.load_encoder_states(args, split)
        exp_dataset = explanation_dataset[split]
        #model = Model(encoder_states, input_size=encoder_states.shape[2])

        train_iterations(encoder_states, exp_dataset, len(tokenizer), args)
    #print(predict(dataset, model, text='Knock knock. Whos there?'))

if __name__ == "__main__":
    main()