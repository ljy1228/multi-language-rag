"""
Visualization utilities for cross-lingual QA results.
Creates publication-quality plots and tables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ResultVisualizer:
    """Visualization utilities for cross-lingual QA results."""
    
    def __init__(self, output_dir: str = './results/plots', style: str = 'publication'):
        """
        Initialize result visualizer.
        
        Args:
            output_dir: Output directory for plots
            style: Plot style ('publication' or 'default')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        if style == 'publication':
            self._setup_publication_style()
        
        logger.info(f"Initialized result visualizer with output directory: {output_dir}")
    
    def _setup_publication_style(self):
        """Setup publication-quality plot style."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def plot_performance_by_language(self, 
                                   results: Dict[str, Any],
                                   metric: str = 'f1',
                                   title: str = 'Performance by Language',
                                   save: bool = True) -> str:
        """
        Plot performance by language.
        
        Args:
            results: Evaluation results
            metric: Metric to plot
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract data
        languages = []
        scores = []
        
        for lang, lang_results in results.items():
            if lang != 'summary' and 'metrics' in lang_results:
                languages.append(lang.upper())
                scores.append(lang_results['metrics'].get(metric, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(languages, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Language', fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontweight='bold')
        ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / f'performance_by_language_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance plot to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def plot_model_comparison(self, 
                            comparison_results: Dict[str, Any],
                            metric: str = 'f1',
                            title: str = 'Model Comparison',
                            save: bool = True) -> str:
        """
        Plot model comparison.
        
        Args:
            comparison_results: Model comparison results
            metric: Metric to plot
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract data
        models = []
        scores = []
        
        for model_name, model_results in comparison_results.items():
            if model_name != 'comparison' and 'summary' in model_results:
                models.append(model_name.upper())
                scores.append(model_results['summary'].get(f'avg_{metric}', 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        bars = ax.bar(models, scores, color=colors[:len(models)], 
                     edgecolor='black', alpha=0.7, linewidth=1.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(f'Average {metric.upper()} Score', fontweight='bold')
        ax.set_ylim(0, max(scores) * 1.1 if scores else 1)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / f'model_comparison_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison plot to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def plot_few_shot_learning_curve(self, 
                                   few_shot_results: Dict[str, Any],
                                   metric: str = 'f1',
                                   title: str = 'Few-Shot Learning Curve',
                                   save: bool = True) -> str:
        """
        Plot few-shot learning curve.
        
        Args:
            few_shot_results: Few-shot experiment results
            metric: Metric to plot
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract data
        shots = []
        mbert_scores = []
        mt5_scores = []
        mbert_stds = []
        mt5_stds = []
        
        for shot, shot_results in few_shot_results.items():
            if isinstance(shot, int) and 'mbert' in shot_results and 'mt5' in shot_results:
                shots.append(shot)
                
                # Average across seeds
                mbert_scores_list = []
                mt5_scores_list = []
                
                for seed_results in shot_results['mbert'].values():
                    if 'evaluation_results' in seed_results and 'summary' in seed_results['evaluation_results']:
                        score = seed_results['evaluation_results']['summary'].get(f'avg_{metric}', 0)
                        mbert_scores_list.append(score)
                
                for seed_results in shot_results['mt5'].values():
                    if 'evaluation_results' in seed_results and 'summary' in seed_results['evaluation_results']:
                        score = seed_results['evaluation_results']['summary'].get(f'avg_{metric}', 0)
                        mt5_scores_list.append(score)
                
                mbert_avg = np.mean(mbert_scores_list) if mbert_scores_list else 0
                mt5_avg = np.mean(mt5_scores_list) if mt5_scores_list else 0
                mbert_std = np.std(mbert_scores_list) if mbert_scores_list else 0
                mt5_std = np.std(mt5_scores_list) if mt5_scores_list else 0
                
                mbert_scores.append(mbert_avg)
                mt5_scores.append(mt5_avg)
                mbert_stds.append(mbert_std)
                mt5_stds.append(mt5_std)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with error bars
        ax.errorbar(shots, mbert_scores, yerr=mbert_stds, 
                   label='mBERT', marker='o', linewidth=2, markersize=8,
                   capsize=5, capthick=2)
        ax.errorbar(shots, mt5_scores, yerr=mt5_stds, 
                   label='mT5', marker='s', linewidth=2, markersize=8,
                   capsize=5, capthick=2)
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Number of Shots', fontweight='bold')
        ax.set_ylabel(f'Average {metric.upper()} Score', fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_xticks(shots)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / f'few_shot_learning_curve_{metric}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved few-shot learning curve to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def plot_transfer_matrix(self, 
                           transfer_matrix: pd.DataFrame,
                           title: str = 'Cross-Lingual Transfer Matrix',
                           save: bool = True) -> str:
        """
        Plot cross-lingual transfer matrix.
        
        Args:
            transfer_matrix: Transfer matrix DataFrame
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        if transfer_matrix.empty:
            logger.warning("Empty transfer matrix provided")
            return ""
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numeric
        numeric_matrix = transfer_matrix.astype(float)
        
        sns.heatmap(numeric_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'F1 Score'},
                   ax=ax)
        
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel('Target Language', fontweight='bold')
        ax.set_ylabel('Source Language', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / 'transfer_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved transfer matrix plot to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def plot_error_analysis(self, 
                          error_data: Dict[str, Any],
                          title: str = 'Error Analysis',
                          save: bool = True) -> str:
        """
        Plot error analysis.
        
        Args:
            error_data: Error analysis data
            title: Plot title
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        # Extract error categories
        categories = list(error_data.keys())
        counts = list(error_data.values())
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%', 
                                         startangle=90, colors=colors)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / 'error_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error analysis plot to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def create_performance_table(self, 
                               results: Dict[str, Any],
                               metric: str = 'f1',
                               save: bool = True) -> str:
        """
        Create performance table.
        
        Args:
            results: Evaluation results
            metric: Metric to include in table
            save: Whether to save the table
            
        Returns:
            Path to saved table
        """
        # Extract data
        table_data = []
        
        for lang, lang_results in results.items():
            if lang != 'summary' and 'metrics' in lang_results:
                metrics = lang_results['metrics']
                table_data.append({
                    'Language': lang.upper(),
                    'F1': metrics.get('f1', 0),
                    'EM': metrics.get('exact_match', 0),
                    'BLEU': metrics.get('bleu', 0),
                    'ROUGE': metrics.get('rouge', 0)
                })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        df = df.sort_values('Language')
        
        # Format numbers
        for col in ['F1', 'EM', 'BLEU', 'ROUGE']:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Save table
        if save:
            table_path = self.output_dir / f'performance_table_{metric}.csv'
            df.to_csv(table_path, index=False)
            logger.info(f"Saved performance table to {table_path}")
            return str(table_path)
        
        return ""
    
    def create_summary_plot(self, 
                          zero_shot_results: Dict[str, Any],
                          few_shot_results: Optional[Dict[str, Any]] = None,
                          save: bool = True) -> str:
        """
        Create summary plot combining multiple results.
        
        Args:
            zero_shot_results: Zero-shot experiment results
            few_shot_results: Few-shot experiment results (optional)
            save: Whether to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Zero-shot performance by language
        if zero_shot_results:
            self._plot_performance_subplot(axes[0, 0], zero_shot_results, 'Zero-Shot Performance')
        
        # Plot 2: Model comparison
        if zero_shot_results:
            self._plot_model_comparison_subplot(axes[0, 1], zero_shot_results, 'Model Comparison')
        
        # Plot 3: Few-shot learning curve
        if few_shot_results:
            self._plot_few_shot_subplot(axes[1, 0], few_shot_results, 'Few-Shot Learning')
        
        # Plot 4: Summary statistics
        self._plot_summary_stats_subplot(axes[1, 1], zero_shot_results, few_shot_results)
        
        plt.tight_layout()
        
        # Save plot
        if save:
            plot_path = self.output_dir / 'summary_plot.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary plot to {plot_path}")
            return str(plot_path)
        
        return ""
    
    def _plot_performance_subplot(self, ax, results, title):
        """Plot performance subplot."""
        languages = []
        mbert_scores = []
        mt5_scores = []
        
        for lang, lang_results in results.items():
            if lang in ['mbert', 'mt5'] and 'evaluation_results' in lang_results:
                for eval_lang, eval_results in lang_results['evaluation_results'].items():
                    if eval_lang != 'summary' and 'metrics' in eval_results:
                        if eval_lang not in languages:
                            languages.append(eval_lang.upper())
                            mbert_scores.append(0)
                            mt5_scores.append(0)
                        
                        idx = languages.index(eval_lang.upper())
                        if lang == 'mbert':
                            mbert_scores[idx] = eval_results['metrics'].get('f1', 0)
                        else:
                            mt5_scores[idx] = eval_results['metrics'].get('f1', 0)
        
        x = np.arange(len(languages))
        width = 0.35
        
        ax.bar(x - width/2, mbert_scores, width, label='mBERT', alpha=0.7)
        ax.bar(x + width/2, mt5_scores, width, label='mT5', alpha=0.7)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Language')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45)
        ax.legend()
    
    def _plot_model_comparison_subplot(self, ax, results, title):
        """Plot model comparison subplot."""
        models = []
        scores = []
        
        for model_name, model_results in results.items():
            if model_name in ['mbert', 'mt5'] and 'evaluation_results' in model_results:
                if 'summary' in model_results['evaluation_results']:
                    summary = model_results['evaluation_results']['summary']
                    models.append(model_name.upper())
                    scores.append(summary.get('avg_f1', 0))
        
        ax.bar(models, scores, alpha=0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Average F1 Score')
        
        # Add value labels
        for i, score in enumerate(scores):
            ax.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    def _plot_few_shot_subplot(self, ax, results, title):
        """Plot few-shot learning subplot."""
        shots = []
        mbert_scores = []
        mt5_scores = []
        
        for shot, shot_results in results.items():
            if isinstance(shot, int) and 'mbert' in shot_results and 'mt5' in shot_results:
                shots.append(shot)
                
                mbert_avg = np.mean([
                    seed_results.get('evaluation_results', {}).get('summary', {}).get('avg_f1', 0)
                    for seed_results in shot_results['mbert'].values()
                ])
                mt5_avg = np.mean([
                    seed_results.get('evaluation_results', {}).get('summary', {}).get('avg_f1', 0)
                    for seed_results in shot_results['mt5'].values()
                ])
                
                mbert_scores.append(mbert_avg)
                mt5_scores.append(mt5_avg)
        
        ax.plot(shots, mbert_scores, 'o-', label='mBERT', linewidth=2)
        ax.plot(shots, mt5_scores, 's-', label='mT5', linewidth=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Number of Shots')
        ax.set_ylabel('Average F1 Score')
        ax.legend()
        ax.set_xticks(shots)
    
    def _plot_summary_stats_subplot(self, ax, zero_shot_results, few_shot_results):
        """Plot summary statistics subplot."""
        ax.text(0.5, 0.5, 'Summary Statistics\\n\\n(Placeholder)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Summary Statistics', fontweight='bold')
        ax.axis('off')


def create_visualizer(output_dir: str = './results/plots', 
                     style: str = 'publication') -> ResultVisualizer:
    """
    Create a result visualizer instance.
    
    Args:
        output_dir: Output directory for plots
        style: Plot style
        
    Returns:
        ResultVisualizer instance
    """
    return ResultVisualizer(output_dir, style)
