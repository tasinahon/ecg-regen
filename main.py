import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm

from dataset import PTBXLDataset, collate_fn
from model import ECGReGen
from trainer import ECGReGenTrainer
from retrieval import FAISSRetriever, generate_reports
from evaluation import ReportEvaluator, get_paper_baseline_scores


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PTBXLDataset(
        jsonl_path=args.train_data,
        max_length=args.max_length
    )
    
    val_dataset = PTBXLDataset(
        jsonl_path=args.val_data,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers == 0),  # Only use pin_memory with single process
        persistent_workers=(args.num_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None  # Reduce memory overhead
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers == 0),  # Only use pin_memory with single process
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Create model
    print("Creating model...")
    model = ECGReGen(
        bert_model_name=args.bert_model,
        hidden_dim=args.hidden_dim
    )
    
    # Create trainer
    trainer = ECGReGenTrainer(
        model=model,
        tokenizer=model.tokenizer,
        device=device,
        mem_mask_ratio=args.mem_mask_ratio,
        mlm_mask_ratio=args.mlm_mask_ratio
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    # For progress-based sampling
    from retrieval import FAISSRetriever, generate_reports
    sample_ecgs = []  # Store some validation ECGs for quick testing
    sample_reports = []  # Store their true reports
    
    # Grab a few validation samples for monitoring
    for i, sample in enumerate(val_dataset):
        if i >= 3:  # Just 3 samples
            break
        sample_ecgs.append(sample['ecg'].unsqueeze(0))
        sample_reports.append(sample['report'])
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        model.train()
        train_losses = {'total_loss': 0, 'mem_loss': 0, 'mlm_loss': 0, 'etm_loss': 0}
        num_batches = 0
        total_batches = len(train_loader)
        last_report_percent = -5  # Track when we last showed reports
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            losses = trainer.train_step(batch, optimizer)
            
            for k, v in losses.items():
                train_losses[k] += v
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'],
                'mem': losses['mem_loss'],
                'mlm': losses['mlm_loss'],
                'etm': losses['etm_loss']
            })
            
            # Show sample reports every 5% progress
            current_percent = int((batch_idx / total_batches) * 100)
            if current_percent >= last_report_percent + 5 and current_percent % 5 == 0:
                last_report_percent = current_percent
                try:
                    print(f"\n{'='*80}")
                    print(f"Progress: {current_percent}% | Epoch {epoch+1} | Batch {batch_idx}/{total_batches}")
                    print(f"Losses: total={losses['total_loss']:.3f}, mem={losses['mem_loss']:.3f}, mlm={losses['mlm_loss']:.3f}, etm={losses['etm_loss']:.5f}")
                    print(f"{'='*80}")
                    
                    # Quick sample generation
                    model.eval()
                    with torch.no_grad():
                        for i, (ecg, true_report) in enumerate(zip(sample_ecgs, sample_reports)):
                            ecg_cuda = ecg.to(device)
                            embedding = model.get_ecg_embedding(ecg_cuda)
                            
                            print(f"\nSample {i+1}:")
                            print(f"True: {true_report[:80]}...")
                            print(f"Embedding norm: {embedding.norm().item():.3f}")
                    
                    print(f"{'='*80}\n")
                except Exception as e:
                    print(f"\n[Warning: Sample display failed: {e}]\n")
                finally:
                    model.train()
        
        # Average losses
        avg_train_losses = {k: v / num_batches for k, v in train_losses.items()}
        
        # Validation
        val_results = trainer.validate(val_loader, show_samples=False)
        
        print(f"Train Loss: {avg_train_losses['total_loss']:.4f} "
              f"(MEM: {avg_train_losses['mem_loss']:.4f}, "
              f"MLM: {avg_train_losses['mlm_loss']:.4f}, "
              f"ETM: {avg_train_losses['etm_loss']:.4f})")
        print(f"Val Loss: {val_results['val_loss']:.4f}")
        
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_results['val_loss'],
            }, checkpoint_path)
    
    print("\nTraining completed!")


def evaluate(args):
    """Evaluation function for report generation"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PTBXLDataset(
        jsonl_path=args.train_data,
        max_length=args.max_length
    )
    
    test_dataset = PTBXLDataset(
        jsonl_path=args.test_data,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers == 0)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.num_workers == 0)
    )
    
    # Create model
    print("Creating model...")
    model = ECGReGen(
        bert_model_name=args.bert_model,
        hidden_dim=args.hidden_dim
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Build FAISS index
    print("Building FAISS index...")
    retriever = FAISSRetriever(
        embedding_dim=args.hidden_dim,
        use_gpu=torch.cuda.is_available()
    )
    retriever.build_index(model, train_loader, device=device)
    
    # Save FAISS index
    index_path = os.path.join(args.output_dir, 'faiss_index.bin')
    metadata_path = os.path.join(args.output_dir, 'faiss_metadata.pkl')
    retriever.save(index_path, metadata_path)
    
    # Generate reports on test set
    print("Generating reports on test set...")
    predictions, ground_truths = generate_reports(
        model, retriever, test_loader, device=device, k=1
    )
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.json')
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump({
            'predictions': predictions,
            'ground_truths': ground_truths
        }, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Evaluate
    print("\nEvaluating report generation...")
    evaluator = ReportEvaluator()
    results = evaluator.evaluate_all(predictions, ground_truths)
    
    # Print results
    evaluator.print_results(results)
    
    # Compare with paper baseline
    baseline_scores = get_paper_baseline_scores()
    evaluator.compare_with_baseline(results, baseline_scores)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='ECG-ReGen: Report Generation via Retrieval')
    
    # Mode
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True,
                        help='Mode: train or evaluate')
    
    # Data paths
    parser.add_argument('--train_data', type=str, 
                        default='data/ptbxl_train_proper.jsonl',
                        help='Path to training JSONL file')
    parser.add_argument('--val_data', type=str,
                        default='data/ptbxl_val_proper.jsonl',
                        help='Path to validation JSONL file')
    parser.add_argument('--test_data', type=str,
                        default='data/ptbxl_test_proper.jsonl',
                        help='Path to test JSONL file')
    
    # Model hyperparameters
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT model name')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension size')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum ECG signal length')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (reduced default to prevent OOM)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--mem_mask_ratio', type=float, default=0.75,
                        help='Masking ratio for MEM task')
    parser.add_argument('--mlm_mask_ratio', type=float, default=0.15,
                        help='Masking ratio for MLM task')
    
    # Other settings
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (0=main process only, recommended to avoid OOM)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pt',
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)


if __name__ == '__main__':
    main()
