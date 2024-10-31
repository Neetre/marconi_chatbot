'''
da modificare per adattarlo al nostro caso
'''

from datasets import Dataset
from count_tokens_finetune import FILE, INSTRUCTION
import pandas as pd
import json

df_instructions = pd.DataFrame(columns=['text'])
df_prompts = pd.DataFrame(columns=['text'])
df_answers = pd.DataFrame(columns=['text'])

def combine_texts(instruction, prompt, answer):
  return {
      "text": f"""
{instruction}

### Prompt
{prompt}

### Answer
{answer}
"""}

def create_datasets():
    for pair in FILE:
        with open(pair, "r") as f:
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
    for i in range(len(test_dataset)):
        test_dataset[i]['text'] = test_dataset[i]['text'].split("### Answer")[0]
        test_dataset[i]['text'] = test_dataset[i]['text'] + "### Answer\n" # add the answer section back but empty
        return test_dataset


    combined_texts = [combine_texts(instruction, prompt, answer) for instruction, prompt, answer in zip(df_instructions["text"], df_prompts["text"], df_answers["text"])]

    finetuning_dataset = Dataset.from_dict({"text": [ct["text"] for ct in combined_texts]})

    print(finetuning_dataset[0]['text'])

    train_dataset = finetuning_dataset.train_test_split(test_size=0.1)['train']
    test_dataset = finetuning_dataset.train_test_split(test_size=0.1)['test']  # should remove the asnwers from the combined_texts
    test_dataset = clean_test_dataset(test_dataset)
    
    return train_dataset, test_dataset