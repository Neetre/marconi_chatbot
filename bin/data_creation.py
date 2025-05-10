import os
from os import walk
import json
import logging
from typing import List, Dict, Optional
from groq import Groq
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.translate.bleu_score import sentence_bleu
from huggingface_hub import login

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGING_FACE_CLI_API_KEY = os.getenv("HUGGING_FACE_CLI_API_KEY")
login(HUGGING_FACE_CLI_API_KEY)
USE_LOCAL = os.getenv("USE_LOCAL", "False").lower() == "true"

USE_SAMPLE_DATA = os.getenv("USE_SAMPLE_DATA", "False").lower() == "true"
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "italian")
DATA_FOLDER = os.getenv("DATA_FOLDER", "../data/raw")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "../data/training_data.json")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


class DataPipelineProcessor:
    def __init__(self, use_local: bool = USE_LOCAL, groq_api_key: str = GROQ_API_KEY, target_language: str = TARGET_LANGUAGE):
        self.use_local = use_local
        self.target_language = target_language
        
        if not use_local:
            if not groq_api_key:
                raise ValueError("Groq API key is required when using the Groq API.")
            self.client = Groq(api_key=groq_api_key)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                device_map="auto",
                torch_dtype=torch.float16
            )

    def generate_with_local_model(self, prompt: str, max_length: int = 1024) -> str:
        """Generate text using local Phi-3-mini model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text with local model: {e}")
            raise

    def generate_with_groq(self, messages: List[Dict[str, str]]) -> str:
        """Generate text using Groq API."""
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model="mixtral-8x7b-32768",  # Groq's most capable model
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with Groq API: {e}")
            raise

    def summarize_text(self, text: str) -> str:
        """Summarize long text into key points."""
        try:
            if self.use_local:
                prompt = f"Summarize the following text into key points:\n\n{text}\n\nSummary:"
                return self.generate_with_local_model(prompt)
            else:
                messages = [
                    {"role": "system", "content": "Summarize the following text into key points."},
                    {"role": "user", "content": text}
                ]
                return self.generate_with_groq(messages)
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            raise

    def generate_qa_pairs(self, summary: str) -> List[Dict[str, str]]:
        """Generate relevant Q&A pairs from the summary."""
        try:
            if self.use_local:
                prompt = f"""Generate 6 relevant question-answer pairs from this text. Format as JSON array with 'question' and 'answer' fields.
                
                Text: {summary}
                
                Q&A pairs:"""
                response = self.generate_with_local_model(prompt)
            else:
                messages = [
                    {"role": "system", "content": "Generate 6 relevant question-answer pairs from this text. Return as JSON array with 'question' and 'answer' fields."},
                    {"role": "user", "content": summary}
                ]
                response = self.generate_with_groq(messages)

            response = response.strip()
            if not response.startswith('['):
                response = response[response.find('['):]
            if not response.endswith(']'):
                response = response[:response.rfind(']')+1]
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON response. Returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Error generating Q&A pairs: {e}")
            raise

    def translate_content(self, content: str, reference: Optional[str] = None) -> str:
        """Translate content to target language and optionally evaluate translation quality."""
        if self.target_language.lower() == "english":
            return content
            
        try:
            if self.use_local:
                prompt = f"Translate the following text to {self.target_language}, maintaining any technical terms appropriate for a school context:\n\n{content}\n\nTranslation:"
                translation = self.generate_with_local_model(prompt)
            else:
                messages = [
                    {"role": "system", "content": f"Translate the following text to {self.target_language}, maintaining any technical terms appropriate for a school context."},
                    {"role": "user", "content": content}
                ]
                translation = self.generate_with_groq(messages)

            if reference:
                bleu_score = self.evaluate_translation_quality(translation, reference)
                logger.info(f"Translation BLEU score: {bleu_score:.2f}")

            return translation
        except Exception as e:
            logger.error(f"Error translating content: {e}")
            raise

    def evaluate_translation_quality(self, translation: str, reference: str) -> float:
        """Evaluate translation quality using BLEU score."""
        try:
            translation_tokens = translation.split()
            reference_tokens = reference.split()
            return sentence_bleu([reference_tokens], translation_tokens)
        except Exception as e:
            logger.error(f"Error evaluating translation quality: {e}")
            raise

    def create_training_prompt(self, qa_pair: Dict[str, str]) -> Dict[str, str]:
        """Convert Q&A pair into training prompt format."""
        prompt = f"You are a school office assistant. A student asks: {qa_pair['question']}"
        return {
            "prompt": prompt,
            "answer": qa_pair['answer']
        }

    def process_document(self, text: str) -> List[Dict[str, str]]:
        """Process entire document through the pipeline."""
        try:
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
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise


def main():
    """Main function to process documents and generate training data."""
    try:
        processor = DataPipelineProcessor(
            use_local=USE_LOCAL,
            groq_api_key=GROQ_API_KEY,
            target_language=TARGET_LANGUAGE
        )

        if USE_SAMPLE_DATA:
            text = [get_sample_data()]
        else:
            text = get_data_from_folder(DATA_FOLDER)

        training_data = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(processor.process_document, doc) for doc in text]
            for future in as_completed(futures):
                try:
                    training_data.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing document in parallel: {e}")

        if os.path.exists(OUTPUT_FILE):
            existing_data = json.load(open(OUTPUT_FILE, 'r', encoding='utf-8'))
            training_data.extend(existing_data)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Training data saved to {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()
