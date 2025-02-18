'''
This script is used to prepare the dataset for the finetuning of the model.
It reads the data from the json files and creates the training and test datasets.
The training dataset is created by combining the instruction, prompt, and answer.
The test dataset is created by removing the answers from the combined text.
The datasets are then saved in the datasets folder.

'''

from datasets import Dataset, load_dataset
from count_tokens_finetune import FILE, INSTRUCTION
import pandas as pd
import json

df_instructions = pd.DataFrame(columns=['text'])
df_prompts = pd.DataFrame(columns=['text'])
df_answers = pd.DataFrame(columns=['text'])

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


EOS = tokenizer.eos_token

def combine_texts(instruction, prompt, answer):
    """Combines texts in LLaMA chat format for fine-tuning"""
    return {"text": f"""
{instruction}

### Prompt
{prompt}

### Answer
{answer}
""" + EOS}


def create_dataset():
    """
    Create the training and test datasets for the finetuning of the model
    """
    global df_instructions, df_prompts, df_answers
    
    with open(FILE, "r") as f:
        data = json.load(f)
        for item in data:
            prompt = item["prompt"]
            answer = item["answer"]
            
            df_instructions = pd.concat(
            [df_instructions, pd.DataFrame([{'text': INSTRUCTION}])],
            ignore_index=True
            )
            df_prompts = pd.concat(
            [df_prompts, pd.DataFrame([{'text': prompt}])],
            ignore_index=True
            )
            df_answers = pd.concat(
            [df_answers, pd.DataFrame([{'text': answer}])],
            ignore_index=True
            )

def clean_test_dataset(test_dataset):
    """
    Remove the answers from the test

    Args:
        test_dataset (dict): the test dataset

    Returns:
        test_dataset (dict): only question
    """
    for i in range(len(test_dataset)):
        test_dataset[i]['text'] = test_dataset[i]['text'].split("### Answer")[0].split("### Prompt")[1].strip().strip("\n")
        return test_dataset


def create_datasets():
    """
    Create the training and test datasets for the finetuning of the model

    Returns:
        Dataset, Dataset: the training and test datasets
    """
    create_dataset()
    combined_texts = [combine_texts(instruction, prompt, answer) for instruction, prompt, answer in zip(df_instructions["text"], df_prompts["text"], df_answers["text"])]
    finetuning_dataset = Dataset.from_dict({"text": [ct["text"] for ct in combined_texts]})

    print(finetuning_dataset[0]['text'])

    train_dataset, test_dataset = finetuning_dataset.train_test_split(test_size=0.1)['train']
    test_dataset = clean_test_dataset(test_dataset)
    
    return train_dataset, test_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
train_dataset, test_dataset = create_datasets()
print(train_dataset[0])
print(test_dataset[0])