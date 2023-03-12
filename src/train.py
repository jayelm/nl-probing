import argparse
import torch
import time
import hydra
import pickle
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
    training_dataset = dataset.CombinedDataset(train_encoder_states, explanation_dataset['train'], args.layer)
    validation_dataset = dataset.CombinedDataset(data.load_encoder_states(args, "validation"), explanation_dataset['validation'], args.layer)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model = lstm_probing(input_size=len(tokenizer), hidden_size=train_encoder_states.shape[2], lstm_hidden_size=args.lstm_hidden_size,
    device=args.device, output_size=len(tokenizer))
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all")
    early_stopper = util.EarlyStopper(patience=3, min_delta=0)
    print_loss_total = 0 
    plot_loss_total = 0 
    start = time.time()
    best_bleu = 0

    for epoch in range(1, args.max_epochs + 1):
        loss, perplexity = train(model, dataloader, criterion, epoch, optimizer, args)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % args.evaluate_every == 0:
            decoded_words, reference_words, bleu_score, loss, perplexity = evaluate(tokenizer, model, validation_dataset, criterion, args)
            avg_bleu_score = bleu_score / len(explanation_dataset['validation'])
            wandb.log({'eval_epoch': epoch, 'eval_loss': loss, 'eval_perplexity': perplexity, 'bleu_score': avg_bleu_score})
            print({'eval_epoch': epoch, 'eval_loss': loss, 'eval_perplexity': perplexity, 'bleu_score': avg_bleu_score})
            if avg_bleu_score > best_bleu:
                torch.save(model.state_dict(), 'train-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now()))
                best_bleu = avg_bleu_score
                name = 'validation' + '-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now())
                #np.savetxt('../results/'+name+'_decoded_words.txt', decoded_words, fmt='%s')
                decoded_words_output = open('../results/'+name+'_decoded_words.pkl', 'wb')
                pickle.dump(decoded_words, decoded_words_output)
                decoded_words_output.close()
            if early_stopper.early_stop(avg_bleu_score):             
                break

        if epoch % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (util.timeSince(start, epoch / args.max_epochs),
                                         epoch, epoch / args.max_epochs * 100, print_loss_avg))
    
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
        if split == "test":
            #model = lstm_probing(input_size=len(tokenizer), hidden_size=encoder_states.shape[2], device=args.device, 
    #output_size=len(tokenizer))
            #model.load_state_dict(torch.load(args.training_model))
            #model = model.to(args.device)
            test_dataset = dataset.CombinedDataset(data.load_encoder_states(args, "test"), explanation_dataset['test'], args.layer)
            criterion = nn.CrossEntropyLoss()
            name = split + '-{date:%Y-%m-%d_%H}'.format( date=datetime.datetime.now())
            decoded_words, reference_words, bleu_score, loss, perplexity = evaluate(tokenizer, model, test_dataset, criterion, args)
            #np.savetxt('../results/'+name+'_decoded_words.txt', decoded_words, fmt='%s')
            decoded_words_output = open('../results/'+name+'_decoded_words.pkl', 'wb')
            pickle.dump(decoded_words, decoded_words_output)
            decoded_words_output.close()
            print({ 'bleu_score': bleu_score / len(explanation_dataset['test']), 'loss': loss, 'perplexity': perplexity })

if __name__ == "__main__":
    main()