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
import datetime

SOS_token = 0

def collate_fn_decode(data):
    hidden_states, explanation = zip(*data)
    return hidden_states, explanation

def train_iterations(encoder_states, exp_dataset, tokenizer_length, split, args):

    training_dataset = dataset.TrainDataset(encoder_states, exp_dataset)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model = lstm_probing(input_size=tokenizer_length, hidden_size=encoder_states.shape[2], device=args.device, 
    output_size=tokenizer_length)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0 
    plot_loss_total = 0 
    plot_losses = []
    plot_perplexity = []
    plot_epochs = []
    start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        loss, perplexity = train(model, dataloader, criterion, encoder_states, epoch, optimizer, args)
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
            plot_perplexity.append(np.exp(plot_loss_avg))
            plot_loss_total = 0
            plot_epochs.append(epoch)
    
    return model, dataloader


def train(model, dataloader, criterion, encoder_states, epoch, optimizer, args):
    model.train()
    # x dimension: (batch, 1, seq_lenth=768)
    # y dimension: (batch, seq_lenth=55)
    for batch, (x, y) in enumerate(dataloader):
        x_tensor = torch.stack(list(x), dim=0).to(args.device)  # (batch, 768)
        y_tensor = torch.stack(y, dim=0).to(args.device)  # (batch, 55)

        optimizer.zero_grad()

        hidden = model.init_hidden(x_tensor)  # (batch, 768), (batch, 768)
        hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        y_lengths = (y_tensor != 50256).long().sum(-1)
        packed_logits, packed_targets = model.forward(y_tensor, y_lengths, hidden)

        loss = criterion(packed_logits, packed_targets)
        
        loss.backward()
        optimizer.step()

        print({ 'epoch': epoch, 'loss': loss.item(), 'perplexity': torch.exp(loss) })
        return loss.item(), torch.exp(loss)

@hydra.main(config_path="conf", config_name="train_probing_config")
def main(args: DictConfig) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer_model = GPT2Model.from_pretrained("gpt2")
    explanation_dataset = data.load_explanation(args, tokenizer)
    for split in args.data.splits:
        print("======"+split+"======")
        encoder_states = data.load_encoder_states(args, split)
        exp_dataset = explanation_dataset[split]
        if split == "train":
            model, dataloader = train_iterations(encoder_states, exp_dataset, len(tokenizer), split, args)
            torch.save(model.state_dict(), 'train-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now()))
        if split == "validation" or split == "test":
            model = lstm_probing(input_size=len(tokenizer), hidden_size=encoder_states.shape[2], device=args.device, 
    output_size=len(tokenizer))
            model.load_state_dict(torch.load(args.training_model))
            model = model.to(args.device)
            name = split + '-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now())
            decoded_words, reference_words, bleu_score = evaluate(tokenizer, model, encoder_states, exp_dataset, args)
            np.savetxt(name+'_decoded_words.txt', decoded_words, fmt='%s')
            np.savetxt(name+'_reference_words.txt', reference_words, fmt='%s')
            print(len(exp_dataset))
            print(bleu_score / len(exp_dataset))

if __name__ == "__main__":
    main()