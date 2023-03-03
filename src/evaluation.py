from lstm_probing_model import lstm_probing
import torchtext
import torch
from torch import nn
import numpy as np
from torchtext.data.metrics import bleu_score
import dataset
from torch.utils.data import DataLoader

def collate_fn_decode(data):
    hidden_states, explanation = zip(*data)
    return hidden_states, explanation

def evaluate(tokenizer, model, encoder_states, exp_dataset, args):
    training_dataset = dataset.TrainDataset(encoder_states, exp_dataset)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, collate_fn=collate_fn_decode)
    model.eval()
    overall_bleu = 0
    decoded_words = []
    reference_words = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x_tensor = torch.stack(list(x), dim=0).to(args.device)  # (batch, 768)
            y_tensor = torch.stack(y, dim=0).to(args.device)  # (batch, 55)
            for i in range(len(x_tensor)):
                input_token = torch.tensor([[1]]).to(args.device)
                hidden = model.init_hidden(x_tensor[i].unsqueeze(0))  # (batch, 768), (batch, 768)
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))

                #output, hidden, logits = model.generate(input_token, hidden)

                #topv, topi = logits.topk(1)
                #input_token = topi.item()
                sequence = [1]
                for t in range(len(y[i])): 
                    output, hidden, logits = model.generate(torch.tensor([[input_token]]).to(args.device), hidden)
                    #output, hidden, logits = model.generate(torch.tensor([sequence]).to(args.device), hidden)
                    #prediction = torch.argmax(output, dim=0)
                    topv, topi = logits.topk(1)
                    if topi.item() == 2: 
                        break
                    sequence.append(topi.item())
                    input_token = topi.item()
                decoded_sequence = tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=True)
                reference = tokenizer.convert_ids_to_tokens(y[i], skip_special_tokens=True)
                overall_bleu += bleu_score([decoded_sequence], [[reference]], max_n=2, weights=[0.5, 0.5])
                decoded_words.append(decoded_sequence)
                reference_words.append(reference)
        print("=======decoded words=======")
        print(decoded_words)
        print("=======reference words=======")
        print(reference_words)
        return decoded_words, reference_words, overall_bleu

