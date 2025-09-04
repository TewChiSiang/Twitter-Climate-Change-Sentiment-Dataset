"""
Main Application for Social Media Sentiment Analysis
Orchestrates data loading, analysis, visualization and reporting for climate change tweets.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
import argparse

# Import our custom modules
from data_loader import DataLoader
from sentiment_analyzer import SentimentAnalyzer
from visualizer import DataVisualizer

class SentimentAnalysisApp:
    
    def __init__(self, data_file: str = 'twitter_sentiment_data.csv'):
       
        self.data_file = data_file
        self.setup_logging()
        self.data_loader = None
        self.analyzer = None
        self.visualizer = None
        self.processed_data = None
        self.analysis_results = None
        
    def setup_logging(self):
        """Setup comprehensive logging configuration."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Sentiment Analysis Application initialized")
    
    def load_and_preprocess_data(self) -> bool:
       
        try:
            self.logger.info("Starting data loading and preprocessing")
            
            # Initialize data loader
            self.data_loader = DataLoader(self.data_file)
            
            # Load data
            raw_data = self.data_loader.load_data()
            self.logger.info(f"Loaded {len(raw_data)} raw records")
            
            # Preprocess data
            self.processed_data = self.data_loader.preprocess_data()
            self.logger.info(f"Preprocessed {len(self.processed_data)} records")
            
            # Get data summary
            summary = self.data_loader.get_data_summary()
            self.logger.info(f"Data summary: {summary['total_tweets']} tweets, "
                           f"sentiment distribution: {summary['sentiment_distribution']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data loading and preprocessing: {e}")
            return False
    
    def perform_analysis(self) -> bool:
       
        try:
            if self.processed_data is None:
                raise ValueError("Data must be loaded before analysis")
            
            self.logger.info("Starting sentiment analysis")
            
            # Initialize analyzer
            self.analyzer = SentimentAnalyzer(self.processed_data)
            
            # Perform comprehensive analysis
            self.analysis_results = self.analyzer.generate_insights_report()
            
            self.logger.info("Sentiment analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return False
    
    def create_visualizations(self, save_dir: str = 'outputs') -> bool:
        
        try:
            if self.processed_data is None or self.analysis_results is None:
                raise ValueError("Data and analysis results must be available before visualization")
            
            self.logger.info("Starting visualization creation")
            
            # Initialize visualizer
            self.visualizer = DataVisualizer(self.processed_data)
            
            # Create comprehensive dashboard
            saved_plots = self.visualizer.create_comprehensive_dashboard(
                self.analysis_results, save_dir
            )
            
            self.logger.info(f"Visualizations created and saved to {save_dir}")
            self.logger.info(f"Saved plots: {list(saved_plots.keys())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in visualization creation: {e}")
            return False
    
    def generate_report(self, save_dir: str = 'outputs') -> bool:
       
        try:
            if self.analysis_results is None:
                raise ValueError("Analysis results must be available before report generation")
            
            self.logger.info("Generating comprehensive report")
            
            # Create outputs directory
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate text report
            report_path = os.path.join(save_dir, 'sentiment_analysis_report.txt')
            self._write_text_report(report_path)
            
            # Generate CSV summary
            csv_path = os.path.join(save_dir, 'sentiment_summary.csv')
            self._write_csv_summary(csv_path)
            
            
            self.logger.info(f"Reports generated and saved to {save_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in report generation: {e}")
            return False
    
    def _write_text_report(self, file_path: str):
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("CLIMATE CHANGE SENTIMENT ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.data_file}\n")
                f.write(f"Total tweets analyzed: {self.analysis_results['dataset_overview']['total_tweets']:,}\n")
                f.write(f"Date range: {self.analysis_results['dataset_overview']['date_range']}\n")
                f.write(f"Topic: {self.analysis_results['dataset_overview']['topic']}\n\n")
                
                # Key findings
                f.write("KEY FINDINGS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Dominant sentiment: {self.analysis_results['key_findings']['dominant_sentiment']}\n")
                f.write(f"Average tweet length: {self.analysis_results['key_findings']['average_tweet_length']} characters\n")
                f.write(f"Most common themes: {', '.join([theme[0] for theme in self.analysis_results['key_findings']['most_common_themes']])}\n\n")
                
                # Sentiment distribution
                f.write("SENTIMENT DISTRIBUTION\n")
                f.write("-" * 40 + "\n")
                sentiment_dist = self.analysis_results['detailed_analysis']['sentiment_distribution']
                for sentiment, count in sentiment_dist['sentiment_label_counts'].items():
                    percentage = sentiment_dist['sentiment_percentages'].get(
                        list(sentiment_dist['sentiment_counts'].keys())[
                            list(sentiment_dist['sentiment_label_counts'].keys()).index(sentiment)
                        ], 0
                    )
                    f.write(f"{sentiment}: {count:,} tweets ({percentage}%)\n")
                f.write("\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                for i, rec in enumerate(self.analysis_results['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
                
                # Detailed statistics
                f.write("DETAILED STATISTICS\n")
                f.write("-" * 40 + "\n")
                text_patterns = self.analysis_results['detailed_analysis']['text_patterns']
                f.write(f"Text length - Mean: {text_patterns['text_length_stats']['mean_length']:.1f}, "
                       f"Median: {text_patterns['text_length_stats']['median_length']:.1f}\n")
                f.write(f"Word count - Mean: {text_patterns['word_count_stats']['mean_words']:.1f}, "
                       f"Median: {text_patterns['word_count_stats']['median_words']:.1f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("Report generation completed successfully.\n")
                
        except Exception as e:
            self.logger.error(f"Error writing text report: {e}")
            raise
    
    def _write_csv_summary(self, file_path: str):
       
        try:
            # Create summary DataFrame
            summary_data = []
            
            # Sentiment distribution
            sentiment_dist = self.analysis_results['detailed_analysis']['sentiment_distribution']
            for sentiment, count in sentiment_dist['sentiment_counts'].items():
                summary_data.append({
                    'metric': f'sentiment_{sentiment}',
                    'value': count,
                    'category': 'sentiment_distribution'
                })
            
            # Text patterns
            text_patterns = self.analysis_results['detailed_analysis']['text_patterns']
            summary_data.append({
                'metric': 'avg_text_length',
                'value': text_patterns['text_length_stats']['mean_length'],
                'category': 'text_patterns'
            })
            summary_data.append({
                'metric': 'avg_word_count',
                'value': text_patterns['word_count_stats']['mean_words'],
                'category': 'text_patterns'
            })
            
            # Save to CSV
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(file_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error writing CSV summary: {e}")
            raise
    
    
    def run_complete_analysis(self, save_dir: str = 'outputs') -> bool:
       
        try:
            self.logger.info("Starting complete sentiment analysis pipeline")
            
            # Step 1: Load and preprocess data
            if not self.load_and_preprocess_data():
                return False
            
            # Step 2: Perform analysis
            if not self.perform_analysis():
                return False
            
            # Step 3: Create visualizations
            if not self.create_visualizations(save_dir):
                return False
            
            # Step 4: Generate reports
            if not self.generate_report(save_dir):
                return False
            
            self.logger.info("Complete sentiment analysis pipeline finished successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis pipeline: {e}")
            return False
    
    def display_summary(self):
       
        if self.analysis_results is None:
            print("No analysis results available. Please run the analysis first.")
            return
        
        print("\n" + "=" * 80)
        print("CLIMATE CHANGE SENTIMENT ANALYSIS - SUMMARY")
        print("=" * 80)
        
        # Dataset overview
        overview = self.analysis_results['dataset_overview']
        print(f"\nDataset Overview:")
        print(f"  Total tweets: {overview['total_tweets']:,}")
        print(f"  Date range: {overview['date_range']}")
        print(f"  Topic: {overview['topic']}")
        
        # Key findings
        findings = self.analysis_results['key_findings']
        print(f"\nKey Findings:")
        print(f"  Dominant sentiment: {findings['dominant_sentiment']}")
        print(f"  Average tweet length: {findings['average_tweet_length']} characters")
        print(f"  Most common themes: {', '.join([theme[0] for theme in findings['most_common_themes']])}")
        
        # Sentiment distribution
        sentiment_dist = self.analysis_results['detailed_analysis']['sentiment_distribution']
        print(f"\nSentiment Distribution:")
        for sentiment, count in sentiment_dist['sentiment_label_counts'].items():
            print(f"  {sentiment}: {count:,} tweets")
        
        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(self.analysis_results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)

def main():

    parser = argparse.ArgumentParser(description='Climate Change Sentiment Analysis')
    parser.add_argument('--data-file', default='twitter_sentiment_data.csv',
                       help='Path to the CSV data file')
    parser.add_argument('--output-dir', default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize application
    app = SentimentAnalysisApp(args.data_file)
    
    if args.interactive:
        # Interactive mode
        print("Climate Change Sentiment Analysis Application")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Load and preprocess data")
            print("2. Perform sentiment analysis")
            print("3. Create visualizations")
            print("4. Generate reports")
            print("5. Run complete pipeline")
            print("6. Display summary")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            try:
                if choice == '1':
                    if app.load_and_preprocess_data():
                        print("✓ Data loaded and preprocessed successfully")
                    else:
                        print("✗ Error in data loading/preprocessing")
                
                elif choice == '2':
                    if app.perform_analysis():
                        print("✓ Sentiment analysis completed successfully")
                    else:
                        print("✗ Error in sentiment analysis")
                
                elif choice == '3':
                    if app.create_visualizations(args.output_dir):
                        print(f"✓ Visualizations created and saved to {args.output_dir}")
                    else:
                        print("✗ Error in visualization creation")
                
                elif choice == '4':
                    if app.generate_report(args.output_dir):
                        print(f"✓ Reports generated and saved to {args.output_dir}")
                    else:
                        print("✗ Error in report generation")
                
                elif choice == '5':
                    if app.run_complete_analysis(args.output_dir):
                        print(f"✓ Complete pipeline finished successfully. Outputs saved to {args.output_dir}")
                    else:
                        print("✗ Error in complete pipeline")
                
                elif choice == '6':
                    app.display_summary()
                
                elif choice == '7':
                    print("Exiting application. Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1-7.")
                    
            except Exception as e:
                print(f"Error: {e}")
                app.logger.error(f"Interactive mode error: {e}")
    
    else:
        # Non-interactive mode - run complete pipeline
        print("Running complete sentiment analysis pipeline...")
        if app.run_complete_analysis(args.output_dir):
            print(f"✓ Analysis completed successfully. Outputs saved to {args.output_dir}")
            app.display_summary()
        else:
            print("✗ Analysis failed. Check logs for details.")
            sys.exit(1)

if __name__ == "__main__":
    main()
