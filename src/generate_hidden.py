"""Script to generate hidden features from a model on an NLI dataset."""

import os
import os.path as osp
from typing import Optional

import h5py
import hydra
import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
)

from . import data


def simple_basename(f: str):
    return osp.splitext(osp.basename(f))[0]


OmegaConf.register_new_resolver("simple_basename", simple_basename)


def masked_average(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Do a masked average over the sequence length

    Args:
        x: Tensor of shape (..., sequence_length, hidden_size)
        mask: Tensor of shape (..., sequence_length)

    Returns:
        x_avg: Tensor of shape (..., hidden_size)
    """
    num_items = mask.sum(dim=-1, keepdim=True)
    # Clamp to avoid divide by zero.
    num_items = torch.clamp(num_items, min=1)

    # Sum up and divide by num items.
    x_avg = x.sum(dim=-2) / num_items

    return x_avg


@torch.no_grad()
def save_hidden_features(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    split: str,
    hdf5_file: h5py.File,
    args: DictConfig,
    desc: Optional[str] = None,
) -> None:
    """Extract average hidden states from a model and save them to HDF5.

    Args:
        model: Model to extract hidden states from. Should return model_outputs
            with a hidden_states attribute.
        tokenizer: Tokenizer to use to tokenize the dataset.
        split: Name of the split. Used to name the dataset in the HDF5 file.
        hdf5_file: HDF5 file to save the hidden states to. Will create a new
            dataset with the name `split`.
        args: Arguments passed through hydra config.
        desc: Optional to use for tqdm progress bar.
    """
    # h5py output
    output_dataset = None
    output_i = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important that we iterate in order.
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    for batch in tqdm(dataloader, desc=desc):
        # Move to cuda, if applicable
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # Forward pass thru model
        model_outputs = model(**batch, output_hidden_states=True)

        # Here, model_outputs.hidden_states is a tuple of length "num_layers",
        # where each element is a tensor of shape (batch_size, sequence_length,
        # hidden_size). Note that many of the hidden states will be padding.
        hidden_states = torch.stack(
            model_outputs.hidden_states,
            dim=1,  # (batch_size, num_layers, sequence_length, hidden_size)
        )

        # TODO: For now let's average the hidden states across the sequence
        # length to save space. But later on we might want to save the full
        # sequence of tokens.
        hidden_states_avg = masked_average(
            hidden_states,
            batch["attention_mask"].unsqueeze(1),  # Add extra dim for num_layers
        )

        # Move back to CPU
        hidden_states_avg = hidden_states_avg.cpu()

        # Save to hdf5 file.
        if output_dataset is None:
            # Create dataset and infer size from first batch.
            output_dataset = hdf5_file.create_dataset(
                split, (len(dataset), *hidden_states_avg.shape[1:]), dtype=np.float32
            )
        output_dataset[output_i : output_i + len(hidden_states_avg)] = hidden_states_avg
        output_i += len(hidden_states_avg)

    assert output_i == len(dataset)


@hydra.main(config_path="conf", config_name="generate_hidden_config")
def main(args: DictConfig) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
    )
    model = model.eval()
    model = model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load data
    datasets = data.load(args, tokenizer)

    os.makedirs(osp.split(args.output_file)[0], exist_ok=True)
    f = h5py.File(args.output_file, "w")

    for split in args.data.splits:
        if split not in datasets:
            raise ValueError(f"Split {split} not found in datasets.")
        save_hidden_features(
            model, tokenizer, datasets[split], split, f, args, desc=split
        )


if __name__ == "__main__":
    main()
