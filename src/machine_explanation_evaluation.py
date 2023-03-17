from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import torch
from transformers import TrainingArguments
import numpy as np
import pandas as pd
from classifier_dataset import ClassifierDataset
import argparse

parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-dp", "--Datapath", help = "Data Path")


tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def get_dataset(path, split):
    dataset = pd.read_pickle(path)
    string_dataset = [' '.join(record).encode('ascii', 'ignore').decode("utf-8") for record in dataset]
    tokenized_dataset = [tokenizer(sent, padding='max_length')['input_ids'] for sent in string_dataset]
    return tokenized_dataset

def get_label():
    label = load_dataset('esnli', split="validation")
    remove_columns = ['premise', 'hypothesis', 'explanation_1', 'explanation_2', 'explanation_3']
    label = label.map(remove_columns = remove_columns)
    return label
    
def get_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    model.load_state_dict(torch.load(model_path))
    return model

def train(dataset, model):
    training_args = TrainingArguments("test_trainer", num_train_epochs=3)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset
    )
    try:
        model.load_state_dict(torch.load('../saved_states/model_state.pt'))
        print("----Loaded Previously trained model----")
    except:
        print("----Training model from scratch----")
        train_info = trainer.train()
        torch.save(model.state_dict(), '../saved_states/model_state.pt')
        print(train_info)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)    

def evaluate(model, dataset, metric):
    training_args = TrainingArguments("test_trainer", num_train_epochs=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=metric,
    )
    eval_info = trainer.evaluate()
    print(eval_info)

def main():
    args = parser.parse_args()

    print("\n-------Setting up dataset-------\n")
    result_path = '/data5/cocolab/gecheng/results/'
    data_path = result_path + args.Datapath
    model_path = '/data/cocolab/gecheng/model_state.pt'
    valid_explanations = get_dataset(data_path, "validation")
    labels = get_label()
    valid_dataset = ClassifierDataset(valid_explanations, labels)

    print("\n-------Setting up model-------\n")
    model = get_model(model_path)
    print("\n-------Evaluating-------\n")
    evaluate(model, valid_dataset, compute_metrics)

if __name__ == "__main__":
    main()