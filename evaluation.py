from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import numpy as np
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class ReportEvaluator:
    """Evaluator for ECG report generation with NLG metrics"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def compute_bleu(self, predictions, references, n=4):
        """
        Compute BLEU scores
        Args:
            predictions: List of predicted reports
            references: List of reference reports
            n: Compute BLEU-1 to BLEU-n
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        bleu_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            
            # Compute BLEU for different n-grams
            for i in range(1, n + 1):
                weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
                bleu = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores[f'bleu_{i}'].append(bleu)
        
        # Average scores
        avg_scores = {k: np.mean(v) for k, v in bleu_scores.items()}
        return avg_scores
    
    def compute_meteor(self, predictions, references):
        """Compute METEOR score"""
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            
            score = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(score)
        
        return {'meteor': np.mean(meteor_scores)}
    
    def compute_rouge(self, predictions, references):
        """Compute ROUGE-L score"""
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        return {'rouge': np.mean(rouge_scores)}
    
    def compute_bertscore(self, predictions, references, model_type='bert-base-uncased'):
        """
        Compute BERTScore
        Args:
            predictions: List of predicted reports
            references: List of reference reports
            model_type: BERT model to use for scoring
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=model_type,
            lang='en',
            verbose=False,
            device='cuda' if __name__ == '__main__' else 'cpu'
        )
        
        return {
            'bertscore': F1.mean().item(),
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item()
        }
    
    def evaluate_all(self, predictions, references):
        """
        Compute all metrics
        Args:
            predictions: List of predicted reports
            references: List of reference reports
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        print("Computing BLEU scores...")
        results.update(self.compute_bleu(predictions, references))
        
        print("Computing METEOR score...")
        results.update(self.compute_meteor(predictions, references))
        
        print("Computing ROUGE score...")
        results.update(self.compute_rouge(predictions, references))
        
        print("Computing BERTScore...")
        results.update(self.compute_bertscore(predictions, references))
        
        return results
    
    def print_results(self, results):
        """Pretty print evaluation results"""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        
        # Print in order matching the paper's table
        metrics_order = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bertscore', 'meteor', 'rouge']
        
        for metric in metrics_order:
            if metric in results:
                print(f"{metric.upper():15s}: {results[metric]:.3f}")
        
        print("=" * 50 + "\n")
    
    def compare_with_baseline(self, results, baseline_scores):
        """
        Compare results with baseline scores
        Args:
            results: Current results dictionary
            baseline_scores: Dictionary with baseline scores (e.g., from paper)
        """
        print("\n" + "=" * 70)
        print("Comparison with Baseline")
        print("=" * 70)
        print(f"{'Metric':<15} {'Current':<12} {'Baseline':<12} {'Difference':<12}")
        print("-" * 70)
        
        for metric in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bertscore', 'meteor', 'rouge']:
            if metric in results and metric in baseline_scores:
                current = results[metric]
                baseline = baseline_scores[metric]
                diff = current - baseline
                diff_str = f"{diff:+.3f}"
                print(f"{metric.upper():<15} {current:.3f:<12} {baseline:.3f:<12} {diff_str:<12}")
        
        print("=" * 70 + "\n")


def get_paper_baseline_scores():
    """Return the target scores from ECG-ReGen paper (Table I, PTB-XL)"""
    return {
        'bleu_1': 0.801,
        'bleu_2': 0.768,
        'bleu_3': 0.737,
        'bleu_4': 0.700,
        'bertscore': 0.920,
        'meteor': 0.836,
        'rouge': 0.836
    }
