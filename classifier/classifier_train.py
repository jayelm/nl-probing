from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import torch
from transformers import TrainingArguments
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def get_dataset():
    return load_dataset('esnli', None)

# take in single example and tokenize the first explanation
# TODO: use all three explanations
def tokenize_data(example):
    return tokenizer(example['explanation_1'], padding='max_length')

def reshape_data(dataset):
    remove_columns = ['premise', 'hypothesis', 'explanation_2', 'explanation_3']
    dataset = dataset.map(tokenize_data, batched=True, remove_columns = remove_columns)
    return dataset

def get_model():
    return AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)

def train(dataset, model):
    training_args = TrainingArguments("test_trainer", num_train_epochs=3)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    train_info = trainer.train()
    print(train_info)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)    

def evaluate(model, dataset, metric):
    training_args = TrainingArguments("test_trainer", num_train_epochs=3)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric,
    )
    eval_info = trainer.evaluate()
    print(eval_info)

def main():
    print("\n-------Setting up dataset-------\n")
    dataset = get_dataset()
    dataset = reshape_data(dataset)
    print("\n-------Setting up model-------\n")
    model = get_model()
    print("\n-------Training-------\n")
    train(dataset, model)
    print("\n-------Evaluating-------\n")
    evaluate(model, dataset, compute_metrics)

if __name__ == "__main__":
    main()