"""
Statistical analysis and visualization for cross-lingual QA evaluation.
Handles significance testing, correlation analysis, and result visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
import warnings

logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StatisticalAnalyzer:
    """Statistical analysis for cross-lingual QA results."""
    
    def __init__(self, alpha: float = 0.05, correction: str = 'bonferroni'):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level
            correction: Multiple comparison correction method
        """
        self.alpha = alpha
        self.correction = correction
        
        logger.info(f"Initialized statistical analyzer with alpha={alpha}, correction={correction}")
    
    def significance_test(self, 
                         scores1: List[float], 
                         scores2: List[float],
                         test_type: str = 'wilcoxon') -> Dict[str, Any]:
        """
        Perform significance test between two sets of scores.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            test_type: Type of test ('wilcoxon' or 'ttest')
            
        Returns:
            Dictionary containing test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Scores must have the same length")
        
        if test_type == 'wilcoxon':
            statistic, p_value = wilcoxon(scores1, scores2)
        elif test_type == 'ttest':
            statistic, p_value = ttest_rel(scores1, scores2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Apply multiple comparison correction
        if self.correction == 'bonferroni':
            p_value_corrected = min(p_value * 2, 1.0)  # Simple Bonferroni for two groups
        else:
            p_value_corrected = p_value
        
        is_significant = p_value_corrected < self.alpha
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'p_value_corrected': p_value_corrected,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'correction': self.correction
        }
    
    def compare_models(self, 
                      model_results: Dict[str, List[float]],
                      test_type: str = 'wilcoxon') -> pd.DataFrame:
        """
        Compare multiple models using significance tests.
        
        Args:
            model_results: Dictionary mapping model names to score lists
            test_type: Type of significance test
            
        Returns:
            DataFrame containing comparison results
        """
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Create comparison matrix
        comparison_matrix = pd.DataFrame(
            index=model_names, 
            columns=model_names,
            dtype=object
        )
        
        # Fill diagonal with model names
        for i, model in enumerate(model_names):
            comparison_matrix.loc[model, model] = model
        
        # Perform pairwise comparisons
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1, model2 = model_names[i], model_names[j]
                scores1 = model_results[model1]
                scores2 = model_results[model2]
                
                # Perform significance test
                test_result = self.significance_test(scores1, scores2, test_type)
                
                # Store result
                result_str = f"p={test_result['p_value_corrected']:.4f}"
                if test_result['is_significant']:
                    result_str += "*"
                
                comparison_matrix.loc[model1, model2] = result_str
                comparison_matrix.loc[model2, model1] = result_str
        
        return comparison_matrix
    
    def correlation_analysis(self, 
                           scores: List[float], 
                           linguistic_features: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze correlation between scores and linguistic features.
        
        Args:
            scores: Performance scores
            linguistic_features: Dictionary mapping feature names to feature values
            
        Returns:
            Dictionary containing correlation results
        """
        correlations = {}
        
        for feature_name, feature_values in linguistic_features.items():
            if len(scores) != len(feature_values):
                logger.warning(f"Length mismatch for feature {feature_name}")
                continue
            
            # Compute correlation
            correlation, p_value = stats.pearsonr(scores, feature_values)
            
            correlations[feature_name] = {
                'correlation': correlation,
                'p_value': p_value,
                'is_significant': p_value < self.alpha
            }
        
        return correlations


class ResultVisualizer:
    """Visualization utilities for cross-lingual QA results."""
    
    def __init__(self, output_dir: str = './results/plots'):
        """
        Initialize result visualizer.
        
        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized result visualizer with output directory: {output_dir}")
    
    def plot_performance_by_language(self, 
                                   results: Dict[str, Any],
                                   metric: str = 'f1',
                                   title: str = 'Performance by Language') -> str:
        """
        Plot performance by language.
        
        Args:
            results: Evaluation results
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        # Extract data
        languages = []
        scores = []
        
        for lang, lang_results in results.items():
            if lang != 'summary' and 'metrics' in lang_results:
                languages.append(lang)
                scores.append(lang_results['metrics'].get(metric, 0))
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(languages, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Language', fontsize=12)
        plt.ylabel(f'{metric.upper()} Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'performance_by_language_{metric}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance plot to {plot_path}")
        return str(plot_path)
    
    def plot_model_comparison(self, 
                            comparison_results: Dict[str, Any],
                            metric: str = 'f1',
                            title: str = 'Model Comparison') -> str:
        """
        Plot model comparison.
        
        Args:
            comparison_results: Model comparison results
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        # Extract data
        models = []
        scores = []
        
        for model_name, model_results in comparison_results.items():
            if model_name != 'comparison' and 'summary' in model_results:
                models.append(model_name)
                scores.append(model_results['summary'].get(f'avg_{metric}', 0))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color=['lightcoral', 'lightblue'], 
                      edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(f'Average {metric.upper()} Score', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'model_comparison_{metric}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {plot_path}")
        return str(plot_path)
    
    def plot_few_shot_learning_curve(self, 
                                   few_shot_results: Dict[str, Any],
                                   metric: str = 'f1',
                                   title: str = 'Few-Shot Learning Curve') -> str:
        """
        Plot few-shot learning curve.
        
        Args:
            few_shot_results: Few-shot experiment results
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        # Extract data
        shots = []
        mbert_scores = []
        mt5_scores = []
        
        for shot, shot_results in few_shot_results.items():
            if isinstance(shot, int) and 'mbert' in shot_results and 'mt5' in shot_results:
                shots.append(shot)
                
                # Average across seeds
                mbert_avg = np.mean([
                    seed_results.get('evaluation_results', {}).get('summary', {}).get(f'avg_{metric}', 0)
                    for seed_results in shot_results['mbert'].values()
                ])
                mt5_avg = np.mean([
                    seed_results.get('evaluation_results', {}).get('summary', {}).get(f'avg_{metric}', 0)
                    for seed_results in shot_results['mt5'].values()
                ])
                
                mbert_scores.append(mbert_avg)
                mt5_scores.append(mt5_avg)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(shots, mbert_scores, 'o-', label='mBERT', linewidth=2, markersize=8)
        plt.plot(shots, mt5_scores, 's-', label='mT5', linewidth=2, markersize=8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Number of Shots', fontsize=12)
        plt.ylabel(f'Average {metric.upper()} Score', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'few_shot_learning_curve_{metric}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved few-shot learning curve to {plot_path}")
        return str(plot_path)
    
    def plot_transfer_matrix(self, 
                           transfer_matrix: pd.DataFrame,
                           title: str = 'Cross-Lingual Transfer Matrix') -> str:
        """
        Plot cross-lingual transfer matrix.
        
        Args:
            transfer_matrix: Transfer matrix DataFrame
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        if transfer_matrix.empty:
            logger.warning("Empty transfer matrix provided")
            return ""
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(transfer_matrix.astype(float), 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'F1 Score'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Target Language', fontsize=12)
        plt.ylabel('Source Language', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'transfer_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved transfer matrix plot to {plot_path}")
        return str(plot_path)
    
    def plot_error_analysis(self, 
                          error_data: Dict[str, Any],
                          title: str = 'Error Analysis') -> str:
        """
        Plot error analysis.
        
        Args:
            error_data: Error analysis data
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        # Extract error categories
        categories = list(error_data.keys())
        counts = list(error_data.values())
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        # Save plot
        plot_path = self.output_dir / 'error_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error analysis plot to {plot_path}")
        return str(plot_path)


class ResultAnalyzer:
    """Main result analyzer combining statistical analysis and visualization."""
    
    def __init__(self, output_dir: str = './results'):
        """
        Initialize result analyzer.
        
        Args:
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer(str(self.output_dir / 'plots'))
        
        logger.info(f"Initialized result analyzer with output directory: {output_dir}")
    
    def analyze_results(self, 
                       zero_shot_results: Dict[str, Any],
                       few_shot_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of results.
        
        Args:
            zero_shot_results: Zero-shot experiment results
            few_shot_results: Few-shot experiment results (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting comprehensive result analysis")
        
        analysis_results = {}
        
        # Analyze zero-shot results
        if zero_shot_results:
            zero_shot_analysis = self._analyze_zero_shot_results(zero_shot_results)
            analysis_results['zero_shot'] = zero_shot_analysis
        
        # Analyze few-shot results
        if few_shot_results:
            few_shot_analysis = self._analyze_few_shot_results(few_shot_results)
            analysis_results['few_shot'] = few_shot_analysis
        
        # Generate comparison analysis
        if zero_shot_results and few_shot_results:
            comparison_analysis = self._analyze_comparison(zero_shot_results, few_shot_results)
            analysis_results['comparison'] = comparison_analysis
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        logger.info("Completed comprehensive result analysis")
        return analysis_results
    
    def _analyze_zero_shot_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze zero-shot results."""
        analysis = {}
        
        # Extract model results
        model_results = {}
        for model_name, model_data in results.items():
            if model_name in ['mbert', 'mt5'] and 'evaluation_results' in model_data:
                # Extract F1 scores by language
                f1_scores = []
                for lang, lang_results in model_data['evaluation_results'].items():
                    if lang != 'summary' and 'metrics' in lang_results:
                        f1_scores.append(lang_results['metrics'].get('f1', 0))
                model_results[model_name] = f1_scores
        
        # Perform statistical comparison
        if len(model_results) == 2:
            comparison_matrix = self.statistical_analyzer.compare_models(model_results)
            analysis['statistical_comparison'] = comparison_matrix.to_dict()
        
        # Generate visualizations
        for model_name, model_data in results.items():
            if model_name in ['mbert', 'mt5'] and 'evaluation_results' in model_data:
                # Plot performance by language
                plot_path = self.visualizer.plot_performance_by_language(
                    model_data['evaluation_results'],
                    title=f'{model_name.upper()} Performance by Language'
                )
                analysis[f'{model_name}_performance_plot'] = plot_path
        
        return analysis
    
    def _analyze_few_shot_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze few-shot results."""
        analysis = {}
        
        # Generate learning curve plot
        plot_path = self.visualizer.plot_few_shot_learning_curve(results)
        analysis['learning_curve_plot'] = plot_path
        
        return analysis
    
    def _analyze_comparison(self, 
                          zero_shot_results: Dict[str, Any],
                          few_shot_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comparison between zero-shot and few-shot results."""
        analysis = {}
        
        # Generate model comparison plot
        plot_path = self.visualizer.plot_model_comparison(
            zero_shot_results,
            title='Zero-Shot Model Comparison'
        )
        analysis['model_comparison_plot'] = plot_path
        
        return analysis
    
    def _save_analysis_results(self, analysis_results: Dict[str, Any]):
        """Save analysis results."""
        results_file = self.output_dir / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Saved analysis results to {results_file}")


def analyze_results(zero_shot_results: Dict[str, Any],
                   few_shot_results: Optional[Dict[str, Any]] = None,
                   output_dir: str = './results') -> Dict[str, Any]:
    """
    Convenience function to analyze results.
    
    Args:
        zero_shot_results: Zero-shot experiment results
        few_shot_results: Few-shot experiment results (optional)
        output_dir: Output directory for results
        
    Returns:
        Analysis results
    """
    analyzer = ResultAnalyzer(output_dir)
    return analyzer.analyze_results(zero_shot_results, few_shot_results)
