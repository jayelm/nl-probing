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
from transformers import GPT2Tokenizer, AutoTokenizer
import dataset

SOS_token = 0

def collate_fn_decode(data):
    hidden_states, explanation = zip(*data)
    return hidden_states, explanation

def train_iterations(encoder_states, exp_dataset, model, args):
    model = model.to(args.device)
    model.train()

    training_dataset = dataset.TrainDataset(encoder_states, exp_dataset)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    plot_loss_total = 0
    plot_losses = []

    for epoch in range(args.max_epochs):
        loss = train(dataloader, model, criterion, optimizer, args)
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / args.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def train(dataloader, model, criterion, optimizer, args):
    #state_h, state_c = model.init_state(args.sequence_length)
    for batch, (x, y) in enumerate(dataloader):
        loss = 0
        y_tensor = torch.LongTensor(np.asarray(y)).to(args.device)
        optimizer.zero_grad()
        outputs = torch.zeros(y_tensor.shape[1], args.batch_size, len(x[0]))
        #outputs = torch.zeros(len(y), len(x[0]))

        decoder_input = torch.zeros(args.batch_size, 1, device=args.device)
        #decoder_input = torch.zeros(len(x[0]), device=args.device)
        x_tensor = torch.tensor(x).unsqueeze(0).to(args.device)
        #x_tensor = torch.tensor(x).to(args.device)
        x_tensor_size = x_tensor.size()
        decoder_hidden = (x_tensor, torch.zeros(x_tensor_size, device=args.device))

        for t in range(y_tensor.shape[1]): 
            decoder_output, decoder_hidden = model(decoder_input, decoder_hidden)
            #outputs[t] = decoder_output
            print(decoder_output.squeeze())
            print(y_tensor[:, t])
            loss += criterion(decoder_output.squeeze(), y_tensor[:, t])
            decoder_input = y_tensor[:, t]

        batch_loss += loss

        loss.backward()
        optimizer.step()

        print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        return batch_loss / batch_size

@hydra.main(config_path="conf", config_name="train_probing_config")
def main(args: DictConfig) -> None:
    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    explanation_dataset = data.load_explanation(args, tokenizer)
    for split in args.data.splits:
        encoder_states = data.load_encoder_states(args, split)
        exp_dataset = explanation_dataset[split]
        model = lstm_probing(input_size=1, hidden_size=args.hidden_size, device=args.device)
        #model = Model(encoder_states, input_size=encoder_states.shape[2])

        train_iterations(encoder_states, exp_dataset, model, args)
    #print(predict(dataset, model, text='Knock knock. Whos there?'))

if __name__ == "__main__":
    main()