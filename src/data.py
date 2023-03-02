"""Dataset loaders."""


from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, GPT2Tokenizer, AutoTokenizer
import h5py

def load_esnli_explanation(args: DictConfig, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    datasets = load_dataset(args.data.dataset_name, args.data.dataset_config_name)
    def tokenize_fn(examples):
        """Use the tokenizer to tokenize the premise and the hypothesis.

        Given a dictionary of examples where "premise" is a list of premises and
        "hypothesis" is a list of hypotheses, tokenizer returns a dictionary
        with keys:
        - "input_ids" corresponding to the tokenized explanation, separated by a [SEP] token.
        - "token_type_ids" which indicates which of the tokens in "input_ids"
            correspond to premise (0) and hypothesis (1).
        - "attention_mask" which indicates which of the tokens in "input_ids"
            actually exist (1) and which are padding (0).
        """
        features = tokenizer(
            examples["explanation_1"],
            padding=True,
            truncation=True,
        )
        return features

    datasets = datasets.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing",
        remove_columns=datasets["train"].column_names,
    )
    return datasets


def load_snli(args: DictConfig, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    """Load an SNLI-type dataset with a "premise" and "hypothesis" field."""
    datasets = load_dataset(args.data.dataset_name, args.data.dataset_config_name)

    def tokenize_fn(examples):
        """Use the tokenizer to tokenize the premise and the hypothesis.

        Given a dictionary of examples where "premise" is a list of premises and
        "hypothesis" is a list of hypotheses, tokenizer returns a dictionary
        with keys:
        - "input_ids" corresponding to the tokenized premise and tokenized
            hypothesis concatenated together, separated by a [SEP] token.
        - "token_type_ids" which indicates which of the tokens in "input_ids"
            correspond to premise (0) and hypothesis (1).
        - "attention_mask" which indicates which of the tokens in "input_ids"
            actually exist (1) and which are padding (0).
        """
        features = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=True,
            truncation=True,
        )
        return features

    datasets = datasets.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing",
        remove_columns=datasets["train"].column_names,
    )
    return datasets


def load_encoder_states(args: DictConfig, split):
    hf = h5py.File(args.encoder_states, "r")
    dataset = hf[split][:]
    hf.close()
    return dataset

def load_explanation(args: DictConfig, tokenizer: AutoTokenizer) -> DatasetDict:
    if "esnli" in args.data.dataset_name:
        return load_esnli_explanation(args, tokenizer)

def load(args: DictConfig, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    if "snli" in args.data.dataset_name:
        return load_snli(args, tokenizer)
    else:
        raise NotImplementedError(f"{args.data=}")
