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
import wandb
from tqdm import tqdm

wandb.init(project='lstm-probing', entity='clairecheng')

def collate_fn_decode(data):
    hidden_states, explanation, explanation_2, explanation_3 = zip(*data)
    return hidden_states, explanation, explanation_2, explanation_3

def train_iterations(data, explanation_dataset, tokenizer, args):
    train_encoder_states = data.load_encoder_states(args, "train")
    training_dataset = dataset.CombinedDataset(train_encoder_states, explanation_dataset['train'])
    validation_dataset = dataset.CombinedDataset(data.load_encoder_states(args, "validation"), explanation_dataset['validation'])
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model = lstm_probing(input_size=len(tokenizer), hidden_size=train_encoder_states.shape[2], device=args.device, 
            output_size=len(tokenizer))
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all")
    print_loss_total = 0 
    plot_loss_total = 0 
    plot_losses = []
    plot_perplexity = []
    plot_epochs = []
    start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        loss, perplexity = train(model, dataloader, criterion, epoch, optimizer, args)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % args.evaluate_every == 0:
            decoded_words, reference_words, bleu_score, loss, perplexity = evaluate(tokenizer, model, validation_dataset, criterion, args)
            wandb.log({'eval_epoch': epoch, 'eval_loss': loss, 'eval_perplexity': perplexity, 'bleu_score': bleu_score / len(explanation_dataset['validation'])})
            print({'eval_epoch': epoch, 'eval_loss': loss, 'eval_perplexity': perplexity, 'bleu_score': bleu_score / len(explanation_dataset['validation'])})

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


def train(model, dataloader, criterion, epoch, optimizer, args):
    model.train()
    # x dimension: (batch, 1, seq_lenth=768)
    # y dimension: (batch, seq_lenth=55)
    for batch, (x, y, y_2, y_3) in enumerate(tqdm(dataloader)):
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
        train_perplexity = torch.exp(loss).item()

        wandb.log({'train_epoch': epoch, 'train_loss': loss.item(), 'train_perplexity': train_perplexity})
        print({ 'train_epoch': epoch, 'train_loss': loss.item(), 'train_perplexity': train_perplexity })
        return loss.item(), train_perplexity

@hydra.main(config_path="conf", config_name="train_probing_config")
def main(args: DictConfig) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    explanation_dataset = data.load_explanation(args, tokenizer)
    for split in args.data.splits:
        print("======"+split+"======")
        if split == "train":
            model, dataloader = train_iterations(data, explanation_dataset, tokenizer, args)
            torch.save(model.state_dict(), 'train-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now()))
        if split == "test":
            encoder_states = data.load_encoder_states(args, split)
            model = lstm_probing(input_size=len(tokenizer), hidden_size=encoder_states.shape[2], device=args.device, 
    output_size=len(tokenizer))
            model.load_state_dict(torch.load(args.training_model))
            model = model.to(args.device)
            name = split + '-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now())
            decoded_words, reference_words, bleu_score, loss, perplexity = evaluate(tokenizer, model, encoder_states, explanation_dataset[split], args)
            np.savetxt('../results/'+name+'_decoded_words.txt', decoded_words, fmt='%s')
            np.savetxt('../results/'+name+'_reference_words.txt', reference_words, fmt='%s')
            print({ 'bleu_score': bleu_score / len(exp_dataset), 'loss': loss, 'perplexity': torch.exp(loss) })

if __name__ == "__main__":
    main()