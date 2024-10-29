import re
import pandas as pd
from openai import AzureOpenAI
from typing import List, Optional, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    A class to process medical notes and extract relevant information about MS patients.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize the TextProcessor with a default window size for text extraction.
        
        Args:
            window_size (int): Number of characters to capture before and after keywords
        """
        self.window_size = window_size

    def _extract_window(self, text: str, keyword: str, before: int, after: int) -> Optional[str]:
        """
        Extract a window of text around a keyword.
        
        Args:
            text (str): The input text to search
            keyword (str): The keyword to search for
            before (int): Characters to capture before the keyword
            after (int): Characters to capture after the keyword
            
        Returns:
            Optional[str]: The extracted window of text if keyword is found, None otherwise
        """
        if keyword in text:
            start_idx = text.index(keyword)
            start = max(0, start_idx - before)
            end = min(len(text), start_idx + after + len(keyword))
            return text[start:end]
        return None

    def process_progressive(self, row: Dict) -> Optional[str]:
        """
        Process text to find references to progressive MS types.
        
        Args:
            row (Dict): Dictionary containing 'note_text' key
            
        Returns:
            Optional[str]: Concatenated windows of text containing progressive MS references
        """
        text = row['note_text'].lower()
        search_terms = ['ppms', 'spms', 'progressive']
        pattern = '|'.join(map(re.escape, search_terms))
        
        matches = list(re.finditer(pattern, text))
        if not matches:
            return None
            
        windows = []
        for match in matches:
            start = max(0, match.start() - 250)
            end = min(len(text), match.start() + 250 + len('progressive'))
            windows.append(text[start:end])
            
        return ' | '.join(windows) if windows else None

    def process_mobility_aid(self, row: Dict, aid_type: str, window_size: Optional[int] = None) -> Optional[str]:
        """
        Process text to find references to mobility aids.
        
        Args:
            row (Dict): Dictionary containing 'note_text' key
            aid_type (str): Type of mobility aid to search for
            window_size (Optional[int]): Custom window size for this search
            
        Returns:
            Optional[str]: Window of text containing mobility aid reference
        """
        text = row['note_text'].lower()
        size = window_size or self.window_size
        return self._extract_window(text, aid_type, size, size)

    def process_edss(self, row: Dict) -> Optional[str]:
        """
        Process text to find EDSS scores and related information.
        
        Args:
            row (Dict): Dictionary containing 'note_text' key
            
        Returns:
            Optional[str]: Window of text containing EDSS information
        """
        text = row['note_text'].lower()
        edss_positions = [m.start() for m in re.finditer('edss', text)]
        
        if not edss_positions:
            return None
            
        windows = []
        for pos in edss_positions:
            start = max(0, pos - 50)
            end = min(len(text), pos + 200 + len('EDSS'))
            windows.append(text[start:end])
            
        return ' | '.join(windows) if windows else None

def process_notes(
    deployment: str,
    df: pd.DataFrame,
    message_example: str,
    column: str,
    api_key: str,
    api_version: str,
    resource_endpoint: str,
    max_retries: int = 5
) -> pd.DataFrame:
    """
    Process clinical notes using Azure OpenAI API.
    
    Args:
        deployment (str): Azure OpenAI deployment name
        df (pd.DataFrame): DataFrame containing clinical notes
        message_example (str): Template for system messages
        column (str): Column name containing the notes
        api_key (str): Azure OpenAI API key
        api_version (str): Azure OpenAI API version
        resource_endpoint (str): Azure OpenAI resource endpoint
        max_retries (int): Maximum number of retry attempts for API calls
        
    Returns:
        pd.DataFrame: DataFrame containing processed results
    """
    response_list = []
    list_of_keys = []
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=resource_endpoint,
    )

    for index, row in df.iterrows():
        extract = row[column]
        deid_note_key = row['deid_note_key']
        
        messages = [
            {"role": "system", "content": 'You are a Medical Doctor collecting information for a clinical study:'},
            {"role": "user", "content": message_example.format(clinical_note=extract)}
        ]
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=0,
                )
                response_list.append(response.choices[0].message.content)
                list_of_keys.append(deid_note_key)
                logger.info(f"Processed note {index + 1}/{len(df)}")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed with error: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max attempts reached, skipping note")
                    response_list.append(None)
                    list_of_keys.append(deid_note_key)

    return pd.DataFrame({'response': response_list, 'deid_note_key': list_of_keys})

# Example usage:
if __name__ == "__main__":
    processor = TextProcessor()
    
    # Example of processing a single note
    sample_note = {'note_text': 'Primary Progressive MS patient uses a wheelchair and has EDSS score of 6.5'}
    
    # Process different mobility aids
    wheelchair_text = processor.process_mobility_aid(sample_note, 'wheelchair')
    crutch_text = processor.process_mobility_aid(sample_note, 'crutch')
    cane_text = processor.process_mobility_aid(sample_note, 'cane', window_size=150)
    walker_text = processor.process_mobility_aid(sample_note, 'walker', window_size=150)
    
    # Process EDSS and progressive information
    edss_text = processor.process_edss(sample_note)
    progressive_text = processor.process_progressive(sample_note)