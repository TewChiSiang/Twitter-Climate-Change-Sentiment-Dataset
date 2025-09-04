"""
Sentiment Analyzer Module for Social Media Sentiment Analysis
Performs sentiment analysis, theme identification and provides insights from climate change tweets.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    
    def __init__(self, data: pd.DataFrame):
       
        self.data = data
        self.setup_logging()
        self.climate_keywords = [
            'climate', 'change', 'global', 'warming', 'emission', 'carbon',
            'temperature', 'weather', 'environment', 'greenhouse', 'pollution',
            'renewable', 'energy', 'solar', 'wind', 'fossil', 'fuel',
            'sustainability', 'eco', 'green', 'earth', 'planet', 'ocean',
            'ice', 'melting', 'sea', 'level', 'drought', 'flood', 'storm'
        ]
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_sentiment_distribution(self) -> dict:
        
        try:
            self.logger.info("Analyzing sentiment distribution")
            
            # Basic sentiment counts
            sentiment_counts = self.data['sentiment'].value_counts().sort_index()
            sentiment_label_counts = self.data['sentiment_label'].value_counts()
            
            # Calculate percentages
            total_tweets = len(self.data)
            sentiment_percentages = (sentiment_counts / total_tweets * 100).round(2)
            
            # Sentiment analysis results
            analysis_results = {
                'total_tweets': total_tweets,
                'sentiment_counts': sentiment_counts.to_dict(),
                'sentiment_label_counts': sentiment_label_counts.to_dict(),
                'sentiment_percentages': sentiment_percentages.to_dict(),
                'dominant_sentiment': sentiment_counts.idxmax(),
                'dominant_sentiment_label': sentiment_label_counts.idxmax(),
                'sentiment_balance': {
                    'pro_climate': sentiment_counts.get(1, 0) + sentiment_counts.get(2, 0),
                    'anti_climate': sentiment_counts.get(-1, 0),
                    'neutral': sentiment_counts.get(0, 0)
                }
            }
            
            self.logger.info("Sentiment distribution analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment distribution: {e}")
            raise
    
    def identify_key_themes(self, top_n: int = 20) -> dict:
       
        try:
            self.logger.info("Identifying key themes in tweets")
            
            # Combine all messages
            all_text = ' '.join(self.data['message'].astype(str))
            
            # Clean text for analysis
            cleaned_text = re.sub(r'[^\w\s]', ' ', all_text.lower())
            words = cleaned_text.split()
            
            # Filter out common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'rt', 'via', 'http', 'https', 'com', 'www', 'co', 'amp'}
            
            # Filter words
            filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
            
            # Count word frequencies
            word_counts = Counter(filtered_words)
            
            # Get top themes
            top_themes = word_counts.most_common(top_n)
            
            # Analyze themes by sentiment
            theme_by_sentiment = {}
            for sentiment in [-1, 0, 1, 2]:
                sentiment_data = self.data[self.data['sentiment'] == sentiment]
                sentiment_text = ' '.join(sentiment_data['message'].astype(str))
                sentiment_words = re.sub(r'[^\w\s]', ' ', sentiment_text.lower()).split()
                sentiment_filtered = [word for word in sentiment_words if len(word) > 2 and word not in stop_words]
                sentiment_counts = Counter(sentiment_filtered)
                theme_by_sentiment[sentiment] = dict(sentiment_counts.most_common(10))
            
            theme_analysis = {
                'top_themes': top_themes,
                'total_unique_words': len(word_counts),
                'theme_by_sentiment': theme_by_sentiment,
                'climate_related_words': {word: count for word, count in word_counts.items() 
                                        if any(keyword in word for keyword in self.climate_keywords)}
            }
            
            self.logger.info("Theme identification completed")
            return theme_analysis
            
        except Exception as e:
            self.logger.error(f"Error identifying themes: {e}")
            raise
    
    def analyze_text_patterns(self) -> dict:
       
        try:
            self.logger.info("Analyzing text patterns")
            
            # Text length analysis
            text_length_stats = {
                'mean_length': self.data['text_length'].mean(),
                'median_length': self.data['text_length'].median(),
                'min_length': self.data['text_length'].min(),
                'max_length': self.data['text_length'].max(),
                'std_length': self.data['text_length'].std()
            }
            
            # Word count analysis
            word_count_stats = {
                'mean_words': self.data['word_count'].mean(),
                'median_words': self.data['word_count'].median(),
                'min_words': self.data['word_count'].min(),
                'max_words': self.data['word_count'].max(),
                'std_words': self.data['word_count'].std()
            }
            
            # Analyze patterns by sentiment
            patterns_by_sentiment = {}
            for sentiment in [-1, 0, 1, 2]:
                sentiment_data = self.data[self.data['sentiment'] == sentiment]
                patterns_by_sentiment[sentiment] = {
                    'avg_length': sentiment_data['text_length'].mean(),
                    'avg_words': sentiment_data['word_count'].mean(),
                    'count': len(sentiment_data)
                }
            
            # Identify common patterns
            common_patterns = {
                'retweets': len(self.data[self.data['message'].str.contains('RT @', case=False)]),
                'mentions': len(self.data[self.data['message'].str.contains('@', case=False)]),
                'hashtags': len(self.data[self.data['message'].str.contains('#', case=False)]),
                'urls': len(self.data[self.data['message'].str.contains('http', case=False)])
            }
            
            pattern_analysis = {
                'text_length_stats': text_length_stats,
                'word_count_stats': word_count_stats,
                'patterns_by_sentiment': patterns_by_sentiment,
                'common_patterns': common_patterns
            }
            
            self.logger.info("Text pattern analysis completed")
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing text patterns: {e}")
            raise
    
    def perform_advanced_sentiment_analysis(self) -> dict:
       
        try:
            self.logger.info("Performing advanced sentiment analysis")
            
            # Sample data for TextBlob analysis (to avoid memory issues with large datasets)
            sample_size = min(5000, len(self.data))
            sample_data = self.data.sample(n=sample_size, random_state=42)
            
            # Analyze with TextBlob
            textblob_sentiments = []
            textblob_subjectivities = []
            
            for message in sample_data['message']:
                try:
                    blob = TextBlob(str(message))
                    textblob_sentiments.append(blob.sentiment.polarity)
                    textblob_subjectivities.append(blob.sentiment.subjectivity)
                except:
                    textblob_sentiments.append(0)
                    textblob_subjectivities.append(0)
            
            # Calculate statistics
            advanced_analysis = {
                'textblob_polarity': {
                    'mean': np.mean(textblob_sentiments),
                    'median': np.median(textblob_sentiments),
                    'std': np.std(textblob_sentiments),
                    'min': np.min(textblob_sentiments),
                    'max': np.max(textblob_sentiments)
                },
                'textblob_subjectivity': {
                    'mean': np.mean(textblob_subjectivities),
                    'median': np.median(textblob_subjectivities),
                    'std': np.std(textblob_subjectivities),
                    'min': np.min(textblob_subjectivities),
                    'max': np.max(textblob_subjectivities)
                },
                'sample_size': sample_size
            }
            
            self.logger.info("Advanced sentiment analysis completed")
            return advanced_analysis
            
        except Exception as e:
            self.logger.error(f"Error in advanced sentiment analysis: {e}")
            raise
    
    def generate_insights_report(self) -> dict:
      
        try:
            self.logger.info("Generating comprehensive insights report")
            
            # Perform all analyses
            sentiment_distribution = self.analyze_sentiment_distribution()
            key_themes = self.identify_key_themes()
            text_patterns = self.analyze_text_patterns()
            advanced_analysis = self.perform_advanced_sentiment_analysis()
            
            # Generate insights
            insights = {
                'dataset_overview': {
                    'total_tweets': sentiment_distribution['total_tweets'],
                    'date_range': 'Apr 27, 2015 - Feb 21, 2018',
                    'topic': 'Climate Change'
                },
                'key_findings': {
                    'dominant_sentiment': sentiment_distribution['dominant_sentiment_label'],
                    'sentiment_balance': sentiment_distribution['sentiment_balance'],
                    'most_common_themes': key_themes['top_themes'][:5],
                    'average_tweet_length': round(text_patterns['text_length_stats']['mean_length'], 1)
                },
                'detailed_analysis': {
                    'sentiment_distribution': sentiment_distribution,
                    'theme_analysis': key_themes,
                    'text_patterns': text_patterns,
                    'advanced_sentiment': advanced_analysis
                },
                'recommendations': self._generate_recommendations(sentiment_distribution, key_themes)
            }
            
            self.logger.info("Insights report generated successfully")
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights report: {e}")
            raise
    
    def _generate_recommendations(self, sentiment_distribution: dict, key_themes: dict) -> list:
      
        recommendations = []
        
        # Analyze sentiment balance
        pro_count = sentiment_distribution['sentiment_balance']['pro_climate']
        anti_count = sentiment_distribution['sentiment_balance']['anti_climate']
        neutral_count = sentiment_distribution['sentiment_balance']['neutral']
        
        if anti_count > pro_count:
            recommendations.append("High anti-climate change sentiment detected. Consider targeted educational campaigns.")
        
        if neutral_count > (pro_count + anti_count):
            recommendations.append("High neutral sentiment suggests need for more engaging climate change content.")
        
        if pro_count > anti_count:
            recommendations.append("Pro-climate change sentiment is dominant. Leverage this for broader engagement.")
        
        # Theme-based recommendations
        top_themes = [theme[0] for theme in key_themes['top_themes'][:10]]
        climate_words = list(key_themes['climate_related_words'].keys())
        
        if len(climate_words) < 10:
            recommendations.append("Limited climate-specific terminology. Consider expanding climate change vocabulary.")
        
        recommendations.append(f"Focus on key themes: {', '.join(top_themes[:5])}")
        
        return recommendations
