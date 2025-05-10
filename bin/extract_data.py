import os
import json
from typing import List
from os import walk
import PyPDF2
import logging

logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder

    def read_file(file_path: str) -> str:
        """Read and return the content of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def extract_data_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfFileReader(f)
                text = ''
                for i in range(reader.numPages):
                    text += reader.getPage(i).extract_text()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise

    def get_data_from_folder(self, folder_path: str = None) -> List[str]:
        """Read all .txt files from a folder and return their content."""
        data = []
        folder_path = folder_path or self.data_folder
        try:
            for root, dirs, files in walk(folder_path):
                for file in files:
                    if file.endswith('.txt'):
                        text = self.read_file(os.path.join(root, file))
                        data.append(text)
            return data
        except Exception as e:
            logger.error(f"Error reading data from folder {folder_path}: {e}")
            raise

    def get_sample_data() -> str:
        """Return sample data for testing."""
        return """
        Students must submit absence notes within 3 days of returning to school.
        Notes should include the date of absence, reason, and parent signature.
        For extended absences of more than 3 days, a doctor's note is required.
        """