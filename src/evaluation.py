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
import util

EOS_IDX = 2

def evaluate(tokenizer, model, dataset, criterion, args):
    if args.data.dataset_name == 'esnli':
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=util.collate_fn_decode)
    elif args.data.dataset_name == 'aqua_rat':
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=util.collate_fn_aqua_decode)
    model.eval()
    overall_bleu = 0
    decoded_words = []
    reference_words = []
    with torch.no_grad():
        # sample: (x, y, y_2, y_3)
        for batch, sample in enumerate(tqdm(dataloader)):
            x_tensor = torch.stack(list(sample[0]), dim=0).to(args.device)  # (batch, 768)
            y_tensor = torch.stack(sample[1], dim=0).to(args.device)  # (batch, 55)
            for i in range(len(x_tensor)):
                input_token = 1
                #input_token = torch.tensor([[1]]).to(args.device)
                hidden = model.init_hidden(x_tensor[i].unsqueeze(0))  # (batch, 768), (batch, 768)
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                #hidden = (torch.permute(hidden[0], (1, 0, 2)), torch.permute(hidden[1], (1, 0, 2)))

                sequence = [1]
                for t in range(len(sample[1][i])): 
                    output, hidden, logits = model.generate(torch.tensor([sequence]).to(args.device), hidden)
                    #output, hidden, logits = model.generate(torch.tensor([[input_token]]).to(args.device), hidden)
                    if t > len(sample[1][i])/2:
                        topv, topi = logits.topk(1)
                        topi = topi.item()
                    else:
                        topv, topi = logits.topk(3)
                        prediction_output = topi.squeeze().cpu().detach().numpy()
                        topi = np.random.choice(prediction_output)
                    #topi = topi.item()
                    if topi == EOS_IDX: 
                        break
                    sequence.append(topi)
                    input_token = topi
                sequence = sequence[1:]
                ref_1 = copy.deepcopy(list(sample[1][i]))
                ref_1.remove(1)
                ref_1.remove(2)
                decoded_sequence = tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=True)
                reference_1 = tokenizer.convert_ids_to_tokens(ref_1, skip_special_tokens=True)
                if args.data.dataset_name == 'esnli':
                    reference_2 = tokenizer.convert_ids_to_tokens(sample[2][i], skip_special_tokens=True)
                    reference_3 = tokenizer.convert_ids_to_tokens(sample[3][i], skip_special_tokens=True)
                    overall_bleu += bleu_score([decoded_sequence], [[reference_1, reference_2, reference_3]], max_n=2, weights=[0.5, 0.5])
                    reference_words.append([reference_1, reference_2, reference_3])
                elif args.data.dataset_name == 'aqua_rat':
                    overall_bleu += bleu_score([decoded_sequence], [[reference_1]], max_n=2, weights=[0.5, 0.5])
                    reference_words.append([reference_1])
                decoded_words.append(decoded_sequence)
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

