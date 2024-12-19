# Marconi Chatbot

## Overview

A comprehensive collection of tools and programs for building and training the Marconi Chatbot, featuring multilingual capabilities and fine-tuning options using various AI models.

Look [here](https://drive.google.com/file/d/1OajZgjOleDU8ilX6bE9lP6QrDz9sIK7a/view?usp=sharing) for the project's graphical representation.

## Features

- Chat interface powered by Groq API
- Automated multilingual training data generation
- Model fine-tuning capabilities using Llama3.1
- Token counting utilities
- Support for Italian language localization

## Programs

### Chat Interface

#### `model.py`

- Implements a chatbot using the [Groq API](https://groq.com)
- Provides an interactive chat interface
- Supports multiple Groq-hosted models

### Fine-tuning Tools

#### `prep_dataset.py`

- Prepares training data for model fine-tuning
- Formats data according to model requirements

#### `count_tokens_finetune.py`

- Analyzes token count in fine-tuning datasets
- Ensures data compatibility with model limitations

#### `finetune_llama.py`

- Fine-tunes Llama3.1 model using [Unsloth](https://unsloth.ai)
- Integrates with [Hugging Face](https://huggingface.co) for model management

### Data Generation Pipeline

#### `data_creation.py`

An automated pipeline for generating multilingual training data:

1. **Text Summarization**
   - Condenses documents into key points
   - Preserves critical information and policy details

2. **Q&A Generation**
   - Creates 3-5 contextually relevant Q&A pairs
   - Ensures natural, conversational tone

3. **Translation & Localization**
   - Translates content to Italian
   - Maintains formal school communication style
   - Preserves technical terminology

4. **Data Formatting**
   - Structures data for instruction-tuning
   - Includes system role definitions
   - Outputs JSON-formatted training data

### Utilities

#### `count_tokens.py`

- Utility for analyzing token counts in datasets
- Helps optimize data for model training

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment tool

### Environment Setup

1. Create and activate a virtual environment:

   **Linux/macOS:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### API Configuration

Create a `.env` file in the project root with the following:

```bash
GROQ_API_KEY="your_groq_api_key"
HUGGINGFACE_TOKEN="your_huggingface_token"
```

- Get your Groq API key from the [Groq Console](https://console.groq.com/playground)
- Get your Hugging Face token from [Hugging Face Tokens](https://huggingface.co/docs/hub/security-tokens) (required for fine-tuning)

## Additional Resources

- [Project Budget Tracking](https://docs.google.com/spreadsheets/d/1lcTVjObpK_JuiXuOtysZtq38bzckFuMH4qnUXEXj2b8/edit?usp=sharing)
- [Unsloth Documentation](https://unsloth.ai)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Author

- [Neetre](https://github.com/Neetre)
