import os
from os import walk
import json
from typing import List, Dict
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class DataPipelineProcessor:
    def __init__(self, use_local: bool = False, groq_api_key: str = None, target_language: str = "italian"):
        self.use_local = use_local
        self.target_language = target_language
        
        if not use_local:
            self.client = Groq(api_key=groq_api_key)
        else:
            # Initialize Phi-3-mini for local processing
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-mini-instruct",
                device_map="auto",
                torch_dtype=torch.float16
            )

    def generate_with_local_model(self, prompt: str, max_length: int = 1024) -> str:
        """Generate text using local Phi-3-mini model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_with_groq(self, messages: List[Dict[str, str]]) -> str:
        """Generate text using Groq API"""
        completion = self.client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",  # Groq's most capable model
            temperature=0.7,
        )
        return completion.choices[0].message.content

    def summarize_text(self, text: str) -> str:
        """Summarize long text into key points."""
        if self.use_local:
            prompt = f"Summarize the following text into key points:\n\n{text}\n\nSummary:"
            return self.generate_with_local_model(prompt)
        else:
            messages = [
                {"role": "system", "content": "Summarize the following text into key points."},
                {"role": "user", "content": text}
            ]
            return self.generate_with_groq(messages)

    def generate_qa_pairs(self, summary: str) -> List[Dict[str, str]]:
        """Generate relevant Q&A pairs from the summary."""
        if self.use_local:
            prompt = f"""Generate 3 relevant question-answer pairs from this text. Format as JSON array with 'question' and 'answer' fields.
            
            Text: {summary}
            
            Q&A pairs:"""
            response = self.generate_with_local_model(prompt)
        else:
            messages = [
                {"role": "system", "content": "Generate 3 relevant question-answer pairs from this text. Return as JSON array with 'question' and 'answer' fields."},
                {"role": "user", "content": summary}
            ]
            response = self.generate_with_groq(messages)
        
        try:
            # Clean up the response to ensure it's valid JSON
            response = response.strip()
            if not response.startswith('['):
                response = response[response.find('['):]
            if not response.endswith(']'):
                response = response[:response.rfind(']')+1]
            return json.loads(response)
        except json.JSONDecodeError:
            print("Warning: Could not parse JSON response. Returning empty list.")
            return []

    def translate_content(self, content: str) -> str:
        """Translate content to target language."""
        if self.target_language.lower() == "english":
            return content
            
        if self.use_local:
            prompt = f"Translate the following text to {self.target_language}, maintaining any technical terms appropriate for a school context:\n\n{content}\n\nTranslation:"
            return self.generate_with_local_model(prompt)
        else:
            messages = [
                {"role": "system", "content": f"Translate the following text to {self.target_language}, maintaining any technical terms appropriate for a school context."},
                {"role": "user", "content": content}
            ]
            return self.generate_with_groq(messages)

    def create_training_prompt(self, qa_pair: Dict[str, str]) -> Dict[str, str]:
        """Convert Q&A pair into training prompt format."""
        prompt = f"You are a school office assistant. A student asks: {qa_pair['question']}"
        return {
            "prompt": prompt,
            "answer": qa_pair['answer']
        }

    def process_document(self, text: str) -> List[Dict[str, str]]:
        """Process entire document through the pipeline."""
        # Step 1: Summarize
        summary = self.summarize_text(text)
        
        # Step 2: Generate Q&A pairs
        qa_pairs = self.generate_qa_pairs(summary)
        
        # Step 3: Translate and create prompts
        training_data = []
        for qa in qa_pairs:
            # Translate both question and answer
            translated_q = self.translate_content(qa['question'])
            translated_a = self.translate_content(qa['answer'])
            
            # Create prompt format
            training_pair = self.create_training_prompt({
                'question': translated_q,
                'answer': translated_a
            })
            training_data.append(training_pair)
        
        return training_data


def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_data_from_folder(folder_path: str) -> List[str]:
    data = []
    for root, dirs, files in walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                text = read_file(f"{folder_path}/{file}")
                data.append(text)
    return data


def get_sample_data() -> str:
    return """
    Students must submit absence notes within 3 days of returning to school.
    Notes should include the date of absence, reason, and parent signature.
    For extended absences of more than 3 days, a doctor's note is required.
    """


def main():
    
    USE_LOCAL = False  # Set to True to use local Phi-3-mini model
    USE_SAMPLE_DATA = True  # Set to True to use sample data instead of folder

    processor = DataPipelineProcessor(
        use_local=USE_LOCAL,
        groq_api_key=GROQ_API_KEY,
        target_language="italian"
    )

    text = get_sample_data() if USE_SAMPLE_DATA else get_data_from_folder('../data/')

    training_data = processor.process_document(text)

    # Save to JSON file
    with open('../data/training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
