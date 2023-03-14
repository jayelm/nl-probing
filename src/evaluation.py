from lstm_probing_model import lstm_probing
import torchtext
import torch
import copy
from torch import nn
import numpy as np
from torchtext.data.metrics import bleu_score
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

EOS_IDX = 2

def collate_fn_decode(data):
    hidden_states, explanation, explanation_2, explanation_3 = zip(*data)
    return hidden_states, explanation, explanation_2, explanation_3

def evaluate(tokenizer, model, dataset, criterion, args):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model.eval()
    overall_bleu = 0
    decoded_words = []
    reference_words = []
    with torch.no_grad():
        for batch, (x, y, y_2, y_3) in enumerate(tqdm(dataloader)):
            x_tensor = torch.stack(list(x), dim=0).to(args.device)  # (batch, 768)
            y_tensor = torch.stack(y, dim=0).to(args.device)  # (batch, 55)
            for i in range(len(x_tensor)):
                input_token = 1
                hidden = model.init_hidden(x_tensor[i].unsqueeze(0))  # (batch, 768), (batch, 768)
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                #hidden = (torch.permute(hidden[0], (1, 0, 2)), torch.permute(hidden[1], (1, 0, 2)))

                sequence = [1]
                for t in range(len(y[i])): 
                    output, hidden, logits = model.generate(torch.tensor([[input_token]]).to(args.device), hidden)
                    topv, topi = logits.topk(1)
                    if topi.item() == EOS_IDX: 
                        break
                    sequence.append(topi.item())
                    input_token = topi.item()
                sequence = sequence[1:]
                ref_1 = copy.deepcopy(list(y[i]))
                ref_1.remove(1)
                ref_1.remove(2)
                decoded_sequence = tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=True)
                reference_1 = tokenizer.convert_ids_to_tokens(ref_1, skip_special_tokens=True)
                reference_2 = tokenizer.convert_ids_to_tokens(y_2[i], skip_special_tokens=True)
                reference_3 = tokenizer.convert_ids_to_tokens(y_3[i], skip_special_tokens=True)
                overall_bleu += bleu_score([decoded_sequence], [[reference_1, reference_2, reference_3]], max_n=2, weights=[0.5, 0.5])
                decoded_words.append(decoded_sequence)
                reference_words.append([reference_1, reference_2, reference_3])
            #overall_bleu += bleu_score(decoded_words, reference_words, max_n=2, weights=[0.5, 0.5])
            loss, perplexity = calculate_loss_and_perplexity(model, criterion, x_tensor, y_tensor, args)
        return decoded_words, reference_words, overall_bleu, loss, perplexity

def calculate_loss_and_perplexity(model, criterion, x_tensor, y_tensor, args):
    hidden = model.init_hidden(x_tensor)  # (batch, 768), (batch, 768)
    hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
    #hidden = (torch.permute(hidden[0], (1, 0, 2)), torch.permute(hidden[1], (1, 0, 2)))
    y_lengths = (y_tensor != 50256).long().sum(-1)
    packed_logits, packed_targets = model.forward(y_tensor, y_lengths, hidden)

    loss = criterion(packed_logits, packed_targets)
    perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()

