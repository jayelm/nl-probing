import argparse
import torch
import time
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
import util

SOS_token = 0

def collate_fn_decode(data):
    hidden_states, explanation = zip(*data)
    return hidden_states, explanation

def train_iterations(encoder_states, exp_dataset, tokenizer_length, args):

    training_dataset = dataset.TrainDataset(encoder_states, exp_dataset)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model = lstm_probing(input_size=encoder_states.shape[2], hidden_size=encoder_states.shape[2], device=args.device, 
    output_size=tokenizer_length)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0 
    plot_loss_total = 0 
    plot_losses = []
    start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        loss = train(model, dataloader, criterion, encoder_states, epoch, optimizer, args)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (util.timeSince(start, epoch / args.max_epochs),
                                         epoch, epoch / args.max_epochs * 100, print_loss_avg))

        if epoch % args.plot_every == 0:
            plot_loss_avg = plot_loss_total / args.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    return model, dataloader


def train(model, dataloader, criterion, encoder_states, epoch, optimizer, args):
    # x dimension: (batch, 1, seq_lenth=768)
    # y dimension: (batch, seq_lenth=55)
    for batch, (x, y) in enumerate(dataloader):
        x_tensor = torch.stack(list(x), dim=0).to(args.device)  # (batch, 768)
        y_tensor = torch.tensor(np.asarray(y)).to(args.device)  # (batch, 55)
        optimizer.zero_grad()

        hidden = model.init_hidden(x_tensor)  # (batch, 768), (batch, 768)
        hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        y_lengths = (y_tensor != 50256).long().sum(-1)
        packed_logits, packed_targets = model(y_tensor, y_lengths, hidden)

        loss = criterion(packed_logits, packed_targets)

        loss.backward()
        optimizer.step()

        print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        return loss.item()

@hydra.main(config_path="conf", config_name="train_probing_config")
def main(args: DictConfig) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_model = GPT2Model.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer_model.resize_token_embeddings(len(tokenizer))
    #reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    explanation_dataset = data.load_explanation(args, tokenizer)
    for split in args.data.splits:
        encoder_states = data.load_encoder_states(args, split)
        exp_dataset = explanation_dataset[split]
        model, dataloader = train_iterations(encoder_states, exp_dataset, len(tokenizer), args)
        decoded_words, bleu_score = evaluate(tokenizer, model, dataloader, args)

if __name__ == "__main__":
    main()