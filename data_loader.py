"""
Data Loader Module for Social Media Sentiment Analysis
Handles data loading, preprocessing and validation for climate change tweets dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataLoader:
    
    def __init__(self, file_path: str):
       
        self.file_path = file_path
        self.data = None
        self.setup_logging()
    
    def setup_logging(self):
       
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> pd.DataFrame:
       
        try:
            self.logger.info(f"Loading data from {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            
            if self.data.empty:
                raise pd.errors.EmptyDataError("The dataset is empty")
            
            self.logger.info(f"Successfully loaded {len(self.data)} records")
            return self.data
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty dataset: {e}")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
      
        if self.data is None:
            raise ValueError("Data must be loaded before preprocessing")
        
        try:
            self.logger.info("Starting data preprocessing")
            
            # Create a copy to avoid modifying original data
            processed_data = self.data.copy()
            
            # Clean column names
            processed_data.columns = processed_data.columns.str.strip().str.lower()
            
            # Remove duplicates based on tweetid
            initial_count = len(processed_data)
            processed_data = processed_data.drop_duplicates(subset=['tweetid'])
            if len(processed_data) < initial_count:
                self.logger.info(f"Removed {initial_count - len(processed_data)} duplicate tweets")
            
            # Clean text data
            processed_data['message'] = processed_data['message'].astype(str).str.strip()
            
            # Remove tweets with empty messages
            processed_data = processed_data[processed_data['message'].str.len() > 0]
            
            # Create sentiment labels for better interpretation
            sentiment_mapping = {
                -1: 'Anti',
                0: 'Neutral', 
                1: 'Pro',
                2: 'News'
            }
            processed_data['sentiment_label'] = processed_data['sentiment'].map(sentiment_mapping)
            
            # Add text length feature
            processed_data['text_length'] = processed_data['message'].str.len()
            
            # Add word count feature
            processed_data['word_count'] = processed_data['message'].str.split().str.len()
            
            # Create binary sentiment for some analyses
            processed_data['binary_sentiment'] = processed_data['sentiment'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral'
            )
            
            self.logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            raise
    
    def get_data_summary(self) -> dict:
       
        if self.data is None:
            raise ValueError("Data must be loaded before getting summary")
        
        try:
            summary = {
                'total_tweets': len(self.data),
                'sentiment_distribution': self.data['sentiment'].value_counts().to_dict(),
                'columns': list(self.data.columns),
                'data_types': self.data.dtypes.to_dict()
            }
            
            # Add sentiment label distribution if it exists
            if 'sentiment_label' in self.data.columns:
                summary['sentiment_label_distribution'] = self.data['sentiment_label'].value_counts().to_dict()
            
            # Add text length stats if they exist
            if 'text_length' in self.data.columns:
                summary['average_text_length'] = self.data['text_length'].mean()
            
            # Add word count stats if they exist
            if 'word_count' in self.data.columns:
                summary['average_word_count'] = self.data['word_count'].mean()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating data summary: {e}")
            raise
    
    def filter_by_sentiment(self, sentiment_value: int) -> pd.DataFrame:
       
        if self.data is None:
            raise ValueError("Data must be loaded before filtering")
        
        try:
            filtered_data = self.data[self.data['sentiment'] == sentiment_value]
            self.logger.info(f"Filtered {len(filtered_data)} tweets with sentiment {sentiment_value}")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error filtering data: {e}")
            raise
    
    def get_sample_data(self, n: int = 1000) -> pd.DataFrame:
      
        if self.data is None:
            raise ValueError("Data must be loaded before sampling")
        
        try:
            sample_size = min(n, len(self.data))
            sample_data = self.data.sample(n=sample_size, random_state=42)
            self.logger.info(f"Generated sample of {sample_size} tweets")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Error sampling data: {e}")
            raise
