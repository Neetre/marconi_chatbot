from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B"
)

def count_tokens(text: str):
    return tokenizer(text, return_tensors="pt")["input_ids"].shape[1]

def read_tokens(file_path: str):
    with open(file_path, "r") as file:
        return "".join(file.readlines())

if __name__ == "__main__":
    text = read_tokens("./bruh.txt")
    print(count_tokens(text))
