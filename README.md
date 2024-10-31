# marconi_chatbot

## Description

Collection of different programs for the Marconi Chatbot project.

### model.py

This program creates a chatbot that uses the [Groq API](https://groq.com) to return a text response from a model offered by the API. It will also run a simple chat interface that allows the user to interact with the chatbot.

### finetune programs

- prep_dataset.py
- count_tokens_finetune.py
- finetune_llama.py

These programs are used to fine-tune the Llama3.1 model, using [Unsloth](https://unsloth.ai) and the [Hugging Face](https://huggingface.co) library. The first program prepares the dataset, the second counts the tokens in the dataset, and the third fine-tunes the model.

### data_creation.py

Automated Training Data Generation Pipeline for Multilingual AI Models

Overview of the Data Processing Workflow

The pipeline transforms raw text documents into structured, multilingual training data for an AI model through a sophisticated, multi-stage process. Here's how it works:

1. **Text Summarization**: 
   - The input document is first condensed into key points
   - Uses advanced language models to extract essential information
   - Preserves critical details and policy nuances

2. **Question-Answer Pair Generation**:
   - The summary is used to create multiple (typically 3-5) relevant Q&A pairs
   - Ensures comprehensive coverage of the source material
   - Generates natural, conversational questions matching the source context

3. **Translation and Localization**:
   - Each question and answer is translated to the target language (in this case, Italian)
   - Maintains formal school communication style
   - Preserves technical terminology and contextual meaning

4. **Training Data Formatting**:
   - Transforms Q&A pairs into a structured prompt-response format
   - Adds system instructions to define the AI's role (school office assistant)
   - Prepares data in a format compatible with instruction-tuning models

5. **Output**:
   - Generates a JSON file containing structured, multilingual training data
   - Each entry includes a prompt and corresponding response
   - Ready to be used for fine-tuning language models

### count_tokens.py

This program counts the number of tokens in a dataset.

## Requirements

python >= 3.9

To run the project, you need to have Python installed on your machine. You can download Python from the [official website](https://www.python.org/downloads/)

Make sure you have a virtual environment running on your machine. If you don't have it, you can create it by running the following commands:

Liunx:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
```

Windows:

```bash
$ python -m venv .venv
$ .venv\Scripts\activate
```

Install dependencies:

```bash
$ pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

[Neetre](https://github.com/Neetre)
