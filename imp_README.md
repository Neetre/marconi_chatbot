# Marconi Chatbot

## Overview
A multilingual chatbot project designed for educational institutions, specifically tailored for the Marconi school environment. The chatbot assists students, parents, and staff by answering questions about school policies, procedures, and general information in both English and Italian.

## What is this project for?
This chatbot serves as a digital assistant that can:
- Answer common questions about school procedures
- Provide information in both English and Italian
- Handle basic administrative queries
- Help reduce the workload on school office staff
- Provide 24/7 access to school information

## Project Structure
```
marconi_chatbot/
├── model.py                    # Main chatbot interface
├── data_creation.py           # Training data generator
├── finetune/
│   ├── prep_dataset.py        # Dataset preparation
│   ├── count_tokens_finetune.py  # Token analysis
│   └── finetune_llama.py      # Model fine-tuning
└── utils/
    └── count_tokens.py        # Token counting utility
```

## Features
- **Bilingual Support**: Handles conversations in both English and Italian
- **Custom-trained Model**: Fine-tuned on school-specific documentation
- **Automated Data Generation**: Creates training data from school documents
- **Interactive Interface**: Easy-to-use chat interface for users
- **Policy-aware Responses**: Trained on school policies and procedures

## Programs

### Chat Interface
#### `model.py`
The main chatbot application that users interact with.
- Uses the [Groq API](https://groq.com) for fast, reliable responses
- Provides a simple chat interface
- Handles both English and Italian inputs
- Maintains conversation context

### Training Data Generation
#### `data_creation.py`
Converts school documents into training data for the chatbot:

1. **Text Summarization**
   - Input: School documents (policies, procedures, announcements)
   - Output: Condensed summaries focusing on key information

2. **Q&A Generation**
   - Creates natural question-answer pairs based on document content
   - Example:
     ```
     Document: "School starts at 8:00 AM on weekdays."
     Generated Q&A:
     Q: "What time does school begin?"
     A: "School begins at 8:00 AM on weekdays."
     ```

3. **Translation & Localization**
   - Translates content between English and Italian
   - Maintains formal tone appropriate for school communication
   - Example:
     ```
     English: "When does school start?"
     Italian: "A che ora inizia la scuola?"
     ```

### Model Fine-tuning Tools
#### `prep_dataset.py`
Prepares training data for the model:
- Formats Q&A pairs for training
- Validates data quality
- Ensures proper formatting

#### `count_tokens_finetune.py`
Analyzes the training dataset:
- Counts tokens to prevent exceeding model limits
- Identifies potential data issues
- Helps optimize training costs

#### `finetune_llama.py`
Customizes the Llama3.1 model for school-specific responses:
- Uses [Unsloth](https://unsloth.ai) for efficient fine-tuning
- Integrates with [Hugging Face](https://huggingface.co) for model storage
- Optimizes model for school-related queries

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Basic understanding of Python virtual environments
- Groq API access
- Hugging Face account (for fine-tuning)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Neetre/marconi_chatbot.git
   cd marconi_chatbot
   ```

2. Create and activate a virtual environment:

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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### API Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY="your_groq_api_key"
HUGGINGFACE_TOKEN="your_huggingface_token"
```

Obtain your API keys from:
- Groq API key: [Groq Console](https://console.groq.com/playground)
- Hugging Face token: [Hugging Face Tokens](https://huggingface.co/docs/hub/security-tokens)

### Quick Start
1. Set up your environment variables
2. Run the chatbot:
   ```bash
   python model.py
   ```

## Common Questions

### How do I add new training data?
1. Place your document in the `data/raw` folder
2. Run the data creation pipeline:
   ```bash
   python data_creation.py
   ```

### How do I update the model with new information?
1. Generate new training data
2. Run the fine-tuning process:
   ```bash
   python finetune/finetune_llama.py
   ```

### How can I contribute to the project?
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

## Project Status
- Current Version: [Add version number]
- Last Updated: [Add date]
- Status: [Active/Maintenance/Beta]

## Additional Resources
- [Project Budget Tracking](https://docs.google.com/spreadsheets/d/1lcTVjObpK_JuiXuOtysZtq38bzckFuMH4qnUXEXj2b8/edit?usp=sharing)
- [Unsloth Documentation](https://unsloth.ai)
- [Project Wiki](Add link if available)

## Support
For support, please:
1. Check the [Common Questions](#common-questions) section
2. [Create an issue](Add link to issues page)
3. Contact the project maintainer

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Author
- [Neetre](https://github.com/Neetre)