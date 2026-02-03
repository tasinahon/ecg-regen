"""
Monitor training progress with actual report generation and metrics
Run this periodically to see if reports are improving beyond just loss
"""
import torch
import json
from model import ECGReGen
from dataset import PTBXLDataset
from retrieval import FAISSRetriever, generate_reports
from evaluation import ReportEvaluator
import argparse


def evaluate_model(checkpoint_path, test_data_path, train_data_path, num_samples=10):
    """Evaluate model with actual report generation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = ECGReGen()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded!\n")
    
    # Load test dataset
    test_dataset = PTBXLDataset(test_data_path, max_length=5000)
    
    # Build retrieval database
    print("Building retrieval database...")
    train_dataset = PTBXLDataset(train_data_path, max_length=5000)
    retriever = FAISSRetriever(model, train_dataset, device=device)
    print("Database ready!\n")
    
    # Generate reports for samples
    print("="*80)
    print("GENERATED REPORTS:")
    print("="*80)
    
    generated_reports = []
    reference_reports = []
    
    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[i]
        ecg = sample['ecg'].unsqueeze(0).to(device)
        true_report = sample['report']
        
        # Generate report
        with torch.no_grad():
            generated_report = generate_reports(model, retriever, [ecg])[0]
        
        generated_reports.append(generated_report)
        reference_reports.append(true_report)
        
        print(f"\n{'='*80}")
        print(f"Sample {i+1}:")
        print(f"{'='*80}")
        print(f"TRUE REPORT:\n{true_report}")
        print(f"\n{'-'*80}")
        print(f"GENERATED REPORT:\n{generated_report}")
        print(f"{'='*80}")
    
    # Compute metrics
    print("\n" + "="*80)
    print("EVALUATION METRICS:")
    print("="*80)
    
    evaluator = ReportEvaluator()
    
    # BLEU scores
    bleu_scores = evaluator.compute_bleu(generated_reports, reference_reports)
    print("\nBLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"  {metric.upper()}: {score:.4f}")
    
    # ROUGE score
    rouge_scores = evaluator.compute_rouge(generated_reports, reference_reports)
    print(f"\nROUGE-L: {rouge_scores['rouge']:.4f}")
    
    # METEOR score
    try:
        meteor_scores = evaluator.compute_meteor(generated_reports, reference_reports)
        print(f"METEOR: {meteor_scores['meteor']:.4f}")
    except Exception as e:
        print(f"METEOR: Error computing ({e})")
    
    print("\n" + "="*80)
    print("WHAT TO LOOK FOR:")
    print("="*80)
    print("""
1. BLEU-1 (0.30+): Are individual medical terms correct?
2. BLEU-4 (0.15+): Are phrases/sentence structure reasonable?
3. ROUGE-L (0.40+): Overall similarity to reference
4. CLINICAL ACCURACY: Do diagnoses/findings match?
5. COHERENCE: Are reports readable and logical?

IMPORTANT: Low scores with good clinical accuracy is OK!
Medical reports can say the same thing differently.
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/ptbxl_test_proper.jsonl',
                        help='Path to test data')
    parser.add_argument('--train_data', type=str, default='data/ptbxl_train_proper.jsonl',
                        help='Path to train data (for retrieval)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to show')
    
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.test_data, args.train_data, args.num_samples)
