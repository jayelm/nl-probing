from lstm_probing_model import lstm_probing
import torchtext
import torch
import numpy as np
from torchtext.data.metrics import bleu_score

def evaluate(tokenizer, model, dataloader, args):
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            
            decoded_words = []
            x_tensor = torch.stack(list(x), dim=0).to(args.device)
            x_tensor = x_tensor.unsqueeze(1).permute((1, 0, 2))
            y_tensor = torch.from_numpy(np.asarray(y)).to(args.device)

            hidden = model.init_hidden(x_tensor)
            outputs = torch.zeros(y_tensor.shape[1], args.batch_size, model.output_size, device=args.device)
            target_input = torch.zeros(y_tensor[:, 0].unsqueeze(1).shape).to(args.device)

            # Run the model for each timestep using teacher forcing
            for t in range(y_tensor.shape[1]):
                current_timestep_tensor = y_tensor[:, t]
                output, hidden = model.forward(target_input, [y_tensor.shape[1]] * args.batch_size, hidden)
                outputs[t] = output
                target_input = current_timestep_tensor.unsqueeze(1)
            outputs = outputs.argmax(dim=-1)
            for i in range(args.batch_size):     
                decoded_words.append([tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True) for idx in outputs[:, i]])
            
            targets = [list(tokenizer.convert_ids_to_tokens(record, skip_special_tokens=True)) for record in y]

            bleu = bleu_score(decoded_words, targets, max_n=3, weights=[0.3, 0.3, 0.4])
            print(bleu)
            print(decoded_words)
            print(targets)
            
        return decoded_words, bleu

