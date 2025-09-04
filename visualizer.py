"""
Visualizer Module for Social Media Sentiment Analysis
Creates informative visualizations for climate change sentiment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizer:
   
    def __init__(self, data: pd.DataFrame):
       
        self.data = data
        self.setup_logging()
        self.setup_plotting()
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
    
    def setup_plotting(self):
        """Setup plotting configuration and style."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Set color palette for sentiment categories
        self.sentiment_colors = {
            -1: '#FF6B6B',  # Red for Anti
            0: '#4ECDC4',   # Teal for Neutral
            1: '#45B7D1',   # Blue for Pro
            2: '#96CEB4'    # Green for News
        }
        
        self.sentiment_labels = {
            -1: 'Anti',
            0: 'Neutral',
            1: 'Pro',
            2: 'News'
        }
    
    def create_sentiment_distribution_chart(self, save_path: str = None) -> plt.Figure:
       
        try:
            self.logger.info("Creating sentiment distribution chart")
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Climate Change Sentiment Analysis - Distribution Overview', fontsize=16, fontweight='bold')
            
            # 1. Pie chart of sentiment distribution
            sentiment_counts = self.data['sentiment'].value_counts()
            colors = [self.sentiment_colors[sentiment] for sentiment in sentiment_counts.index]
            labels = [self.sentiment_labels[sentiment] for sentiment in sentiment_counts.index]
            
            ax1.pie(sentiment_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Sentiment Distribution (Pie Chart)')
            
            # 2. Bar chart of sentiment counts
            bars = ax2.bar(range(len(sentiment_counts)), sentiment_counts.values, 
                          color=[self.sentiment_colors[sentiment] for sentiment in sentiment_counts.index])
            ax2.set_title('Sentiment Distribution (Bar Chart)')
            ax2.set_xlabel('Sentiment Categories')
            ax2.set_ylabel('Number of Tweets')
            ax2.set_xticks(range(len(sentiment_counts)))
            ax2.set_xticklabels(labels, rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, sentiment_counts.values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count:,}', ha='center', va='bottom')
            
            # 3. Horizontal bar chart with percentages
            percentages = (sentiment_counts / len(self.data) * 100).round(1)
            y_pos = np.arange(len(percentages))
            bars = ax3.barh(y_pos, percentages.values, 
                           color=[self.sentiment_colors[sentiment] for sentiment in percentages.index])
            ax3.set_title('Sentiment Distribution (Percentage)')
            ax3.set_xlabel('Percentage (%)')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(labels)
            
            # Add percentage labels
            for bar, percentage in zip(bars, percentages.values):
                width = bar.get_width()
                ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{percentage}%', ha='left', va='center')
            
            # 4. Stacked bar chart showing text length by sentiment
            sentiment_length_data = self.data.groupby('sentiment')['text_length'].agg(['mean', 'median', 'count'])
            x_pos = np.arange(len(sentiment_length_data))
            
            ax4.bar(x_pos, sentiment_length_data['mean'], 
                   color=[self.sentiment_colors[sentiment] for sentiment in sentiment_length_data.index],
                   alpha=0.7, label='Mean Length')
            ax4.bar(x_pos, sentiment_length_data['median'], 
                   color=[self.sentiment_colors[sentiment] for sentiment in sentiment_length_data.index],
                   alpha=0.9, label='Median Length')
            
            ax4.set_title('Text Length by Sentiment Category')
            ax4.set_xlabel('Sentiment Categories')
            ax4.set_ylabel('Text Length (characters)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(labels, rotation=45)
            ax4.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Sentiment distribution chart saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment distribution chart: {e}")
            raise
    
    def create_theme_analysis_charts(self, theme_data: dict, save_path: str = None) -> plt.Figure:
       
        try:
            self.logger.info("Creating theme analysis charts")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Climate Change Theme Analysis', fontsize=16, fontweight='bold')
            
            # 1. Top themes word cloud
            if theme_data['top_themes']:
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                   colormap='viridis', max_words=50).generate_from_frequencies(
                    dict(theme_data['top_themes']))
                ax1.imshow(wordcloud, interpolation='bilinear')
                ax1.set_title('Top Themes Word Cloud')
                ax1.axis('off')
            
            # 2. Top 15 themes bar chart
            top_15_themes = theme_data['top_themes'][:15]
            theme_words = [theme[0] for theme in top_15_themes]
            theme_counts = [theme[1] for theme in top_15_themes]
            
            bars = ax2.barh(range(len(theme_words)), theme_counts, color='skyblue')
            ax2.set_title('Top 15 Most Frequent Themes')
            ax2.set_xlabel('Frequency')
            ax2.set_yticks(range(len(theme_words)))
            ax2.set_yticklabels(theme_words)
            
            # Add value labels
            for bar, count in zip(bars, theme_counts):
                width = bar.get_width()
                ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center')
            
            # 3. Climate-related words frequency
            climate_words = theme_data.get('climate_related_words', {})
            if climate_words:
                top_climate = sorted(climate_words.items(), key=lambda x: x[1], reverse=True)[:10]
                climate_terms = [word[0] for word in top_climate]
                climate_freqs = [word[1] for word in top_climate]
                
                bars = ax3.bar(range(len(climate_terms)), climate_freqs, color='lightgreen')
                ax3.set_title('Top Climate-Related Terms')
                ax3.set_xlabel('Climate Terms')
                ax3.set_ylabel('Frequency')
                ax3.set_xticks(range(len(climate_terms)))
                ax3.set_xticklabels(climate_terms, rotation=45, ha='right')
                
                # Add value labels
                for bar, freq in zip(bars, climate_freqs):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{freq}', ha='center', va='bottom')
            
            # 4. Themes by sentiment category
            theme_by_sentiment = theme_data.get('theme_by_sentiment', {})
            if theme_by_sentiment:
                # Get top 5 themes across all sentiments
                all_themes = set()
                for sentiment_themes in theme_by_sentiment.values():
                    all_themes.update(list(sentiment_themes.keys())[:5])
                
                all_themes = list(all_themes)[:8]  # Limit to 8 for readability
                
                # Create comparison data
                sentiment_names = [self.sentiment_labels[sentiment] for sentiment in theme_by_sentiment.keys()]
                theme_data_matrix = []
                
                for theme in all_themes:
                    theme_row = []
                    for sentiment in theme_by_sentiment.keys():
                        theme_row.append(theme_by_sentiment[sentiment].get(theme, 0))
                    theme_data_matrix.append(theme_row)
                
                # Create heatmap
                im = ax4.imshow(theme_data_matrix, cmap='YlOrRd', aspect='auto')
                ax4.set_title('Theme Frequency by Sentiment Category')
                ax4.set_xlabel('Sentiment Categories')
                ax4.set_ylabel('Themes')
                ax4.set_xticks(range(len(sentiment_names)))
                ax4.set_xticklabels(sentiment_names, rotation=45)
                ax4.set_yticks(range(len(all_themes)))
                ax4.set_yticklabels(all_themes)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Frequency')
                
                # Add text annotations
                for i in range(len(all_themes)):
                    for j in range(len(sentiment_names)):
                        text = ax4.text(j, i, theme_data_matrix[i][j],
                                       ha="center", va="center", color="black" if theme_data_matrix[i][j] < 50 else "white")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Theme analysis charts saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating theme analysis charts: {e}")
            raise
    
    def create_text_pattern_charts(self, pattern_data: dict, save_path: str = None) -> plt.Figure:
       
        try:
            self.logger.info("Creating text pattern charts")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Text Pattern Analysis', fontsize=16, fontweight='bold')
            
            # 1. Text length distribution by sentiment
            for sentiment in [-1, 0, 1, 2]:
                sentiment_data = self.data[self.data['sentiment'] == sentiment]
                ax1.hist(sentiment_data['text_length'], bins=30, alpha=0.6, 
                        label=self.sentiment_labels[sentiment], color=self.sentiment_colors[sentiment])
            
            ax1.set_title('Text Length Distribution by Sentiment')
            ax1.set_xlabel('Text Length (characters)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Word count distribution by sentiment
            for sentiment in [-1, 0, 1, 2]:
                sentiment_data = self.data[self.data['sentiment'] == sentiment]
                ax2.hist(sentiment_data['word_count'], bins=30, alpha=0.6,
                        label=self.sentiment_labels[sentiment], color=self.sentiment_colors[sentiment])
            
            ax2.set_title('Word Count Distribution by Sentiment')
            ax2.set_xlabel('Word Count')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Box plot of text length by sentiment
            sentiment_data_list = [self.data[self.data['sentiment'] == sentiment]['text_length'] 
                                 for sentiment in [-1, 0, 1, 2]]
            labels = [self.sentiment_labels[sentiment] for sentiment in [-1, 0, 1, 2]]
            colors = [self.sentiment_colors[sentiment] for sentiment in [-1, 0, 1, 2]]
            
            bp = ax3.boxplot(sentiment_data_list, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_title('Text Length Distribution (Box Plot)')
            ax3.set_ylabel('Text Length (characters)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Common patterns analysis
            common_patterns = pattern_data.get('common_patterns', {})
            if common_patterns:
                pattern_names = list(common_patterns.keys())
                pattern_counts = list(common_patterns.values())
                
                bars = ax4.bar(pattern_names, pattern_counts, color='lightcoral')
                ax4.set_title('Common Text Patterns')
                ax4.set_ylabel('Count')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, pattern_counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{count:,}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Text pattern charts saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating text pattern charts: {e}")
            raise
    
    def create_interactive_plotly_charts(self, theme_data: dict, save_path: str = None) -> go.Figure:
       
        try:
            self.logger.info("Creating interactive Plotly charts")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sentiment Distribution', 'Top Themes', 'Text Length by Sentiment', 'Climate Terms'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "bar"}]]
            )
            
            # 1. Sentiment distribution pie chart
            sentiment_counts = self.data['sentiment'].value_counts()
            fig.add_trace(
                go.Pie(labels=[self.sentiment_labels[s] for s in sentiment_counts.index],
                      values=sentiment_counts.values,
                      marker_colors=[self.sentiment_colors[s] for s in sentiment_counts.index],
                      name="Sentiment"),
                row=1, col=1
            )
            
            # 2. Top themes bar chart
            top_10_themes = theme_data['top_themes'][:10]
            fig.add_trace(
                go.Bar(x=[theme[0] for theme in top_10_themes],
                      y=[theme[1] for theme in top_10_themes],
                      name="Themes",
                      marker_color='lightblue'),
                row=1, col=2
            )
            
            # 3. Text length box plot by sentiment
            for sentiment in [-1, 0, 1, 2]:
                sentiment_data = self.data[self.data['sentiment'] == sentiment]
                fig.add_trace(
                    go.Box(y=sentiment_data['text_length'],
                          name=self.sentiment_labels[sentiment],
                          marker_color=self.sentiment_colors[sentiment]),
                    row=2, col=1
                )
            
            # 4. Climate terms frequency
            climate_words = theme_data.get('climate_related_words', {})
            if climate_words:
                top_climate = sorted(climate_words.items(), key=lambda x: x[1], reverse=True)[:10]
                fig.add_trace(
                    go.Bar(x=[word[0] for word in top_climate],
                          y=[word[1] for word in top_climate],
                          name="Climate Terms",
                          marker_color='lightgreen'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Interactive Climate Change Sentiment Analysis Dashboard",
                showlegend=True,
                height=800
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Themes", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            fig.update_xaxes(title_text="Climate Terms", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            fig.update_yaxes(title_text="Text Length", row=2, col=1)
            
            if save_path:
                fig.write_html(save_path)
                self.logger.info(f"Interactive Plotly charts saved to {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating interactive Plotly charts: {e}")
            raise
    
    def create_comprehensive_dashboard(self, analyzer_results: dict, save_dir: str = None) -> dict:
       
        try:
            self.logger.info("Creating comprehensive visualization dashboard")
            
            saved_plots = {}
            
            # Create all visualizations
            fig1 = self.create_sentiment_distribution_chart()
            fig2 = self.create_theme_analysis_charts(analyzer_results['detailed_analysis']['theme_analysis'])
            fig3 = self.create_text_pattern_charts(analyzer_results['detailed_analysis']['text_patterns'])
            fig4 = self.create_interactive_plotly_charts(analyzer_results['detailed_analysis']['theme_analysis'])
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                
                # Save matplotlib figures
                fig1.savefig(os.path.join(save_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
                fig2.savefig(os.path.join(save_dir, 'theme_analysis.png'), dpi=300, bbox_inches='tight')
                fig3.savefig(os.path.join(save_dir, 'text_patterns.png'), dpi=300, bbox_inches='tight')
                
                # Save interactive plot
                fig4.write_html(os.path.join(save_dir, 'interactive_dashboard.html'))
                
                saved_plots = {
                    'sentiment_distribution': os.path.join(save_dir, 'sentiment_distribution.png'),
                    'theme_analysis': os.path.join(save_dir, 'theme_analysis.png'),
                    'text_patterns': os.path.join(save_dir, 'text_patterns.png'),
                    'interactive_dashboard': os.path.join(save_dir, 'interactive_dashboard.html')
                }
                
                self.logger.info(f"All plots saved to directory: {save_dir}")
            
            return saved_plots
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive dashboard: {e}")
            raise
