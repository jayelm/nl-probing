from lstm_probing_model import lstm_probing
import torchtext
import torch
import numpy as np
from torchtext.data.metrics import bleu_score

def evaluate(model, tokenizer, dataloader, args):
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            decoded_words = []

            y_tensor = torch.tensor(np.asarray(y),  dtype = torch.long).to(args.device)

            x_tensor = torch.tensor(x).to(args.device)

            probing_model_output = model(y_tensor, x_tensor)
            probing_model_output = probing_model_output.argmax(dim=2)
            #topv, topi = probing_model_output.topk(1)

            for i in range(probing_model_output.shape[0]): # iterate over batch dimension
                decoded_words.append(tokenizer.decode(probing_model_output[i]))

            targets = [list(map(tokenizer.decode, record)) for record in y]

            bleu = bleu_score(decoded_words, targets)
            print(bleu)

        return decoded_words, bleu

