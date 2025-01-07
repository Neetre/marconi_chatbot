'''
da modificare per adattarlo al nostro caso
'''

from transformers import AutoTokenizer
import json

FILE = "../../data/training_data.json"
INSTRUCTION = """You are a helpful assistant on a school community home page. You are responsible for answering questions about the school's courses, events, announcements, and general information. Respond in Italian. Be concise, friendly, and engaging."""

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
)

print(f"{'Total':<12}{'Instruction':<12}{'Story':<12}{'Summary':<12}")

with open(FILE, "r") as f:
    data = json.load(f)
    for item in data:
        prompt = item["prompt"]
        answer = item["answer"]

        # Count tokens
        instruction_tokens = tokenizer(INSTRUCTION, return_tensors="pt")["input_ids"].shape[1]
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        answer_tokens = tokenizer(answer, return_tensors="pt")["input_ids"].shape[1]

        # Print table of tokens
        total_tokens = instruction_tokens + answer_tokens + prompt_tokens
        print(f"{total_tokens:<12}{instruction_tokens:<12}{answer_tokens:<12}{answer_tokens:<12}")
