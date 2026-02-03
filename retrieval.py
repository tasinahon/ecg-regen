import torch
import numpy as np
import faiss
from tqdm import tqdm
import pickle


class FAISSRetriever:
    """FAISS-based retrieval system for ECG reports"""
    
    def __init__(self, embedding_dim=768, use_gpu=False):
        """
        Args:
            embedding_dim: Dimension of ECG embeddings
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.reports = []
        self.ids = []
        
    def build_index(self, model, dataloader, device='cuda'):
        """
        Build FAISS index from training data
        Args:
            model: Trained ECG-ReGen model
            dataloader: DataLoader with training data
            device: Device to run model on
        """
        model.eval()
        embeddings_list = []
        reports_list = []
        ids_list = []
        
        print("Extracting embeddings for FAISS index...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                ecg = batch['ecg'].to(device)
                
                # Get embeddings
                embedding = model.get_ecg_embedding(ecg)
                
                # L2 normalize
                embedding = F.normalize(embedding, p=2, dim=1)
                
                embeddings_list.append(embedding.cpu().numpy())
                reports_list.extend(batch['report'])
                ids_list.extend(batch['id'])
        
        # Concatenate all embeddings
        embeddings = np.vstack(embeddings_list).astype('float32')
        self.reports = reports_list
        self.ids = ids_list
        
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        # Create FAISS index
        # Use IndexFlatIP for cosine similarity (since we normalize embeddings)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Optionally use GPU
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, model, ecg, k=1, device='cuda'):
        """
        Retrieve k nearest neighbor reports for given ECG
        Args:
            model: Trained ECG-ReGen model
            ecg: ECG tensor (batch, 12, 5000)
            k: Number of neighbors to retrieve
            device: Device to run model on
        Returns:
            retrieved_reports: List of retrieved reports
            similarities: Similarity scores to neighbors
            indices: Indices of neighbors
        """
        model.eval()
        
        with torch.no_grad():
            # Get embedding
            embedding = model.get_ecg_embedding(ecg.to(device))
            
            # L2 normalize
            embedding = F.normalize(embedding, p=2, dim=1)
            embedding = embedding.cpu().numpy().astype('float32')
        
        # Search in FAISS index
        similarities, indices = self.index.search(embedding, k)
        
        # Retrieve reports with safety checks
        batch_reports = []
        for batch_idx in range(len(embedding)):
            reports_for_sample = []
            for neighbor_idx in indices[batch_idx]:
                if 0 <= neighbor_idx < len(self.reports):
                    reports_for_sample.append(self.reports[neighbor_idx])
                else:
                    reports_for_sample.append("")  # Fallback for invalid indices
            batch_reports.append(reports_for_sample)
        
        return batch_reports, similarities, indices
    
    def save(self, index_path, metadata_path):
        """Save FAISS index and metadata"""
        # Save FAISS index
        if self.use_gpu:
            # Convert to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'reports': self.reports,
            'ids': self.ids,
            'embedding_dim': self.embedding_dim
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"FAISS index saved to {index_path}")
        print(f"Metadata saved to {metadata_path}")
    
    def load(self, index_path, metadata_path):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.reports = metadata['reports']
        self.ids = metadata['ids']
        self.embedding_dim = metadata['embedding_dim']
        
        print(f"FAISS index loaded from {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")


import torch.nn.functional as F


def generate_reports(model, retriever, test_loader, device='cuda', k=1):
    """
    Retrieve reports for test set (retrieval-based, not generative)
    Args:
        model: Trained ECG-ReGen model
        retriever: FAISS retriever
        test_loader: Test data loader
        device: Device to run on
        k: Number of neighbors to retrieve (default 1 for top-1 report)
    Returns:
        predictions: List of retrieved reports
        ground_truths: List of ground truth reports
        retrieval_metrics: Dict with Recall@k and MRR
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    # Retrieval metrics
    recall_at_k = 0
    mrr_sum = 0.0
    total_samples = 0
    
    print("Retrieving reports via nearest neighbor search...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            ecg = batch['ecg']
            gt_reports = batch['report']
            
            # Retrieve top-k reports
            retrieved_reports, similarities, indices = retriever.retrieve(
                model, ecg, k=k, device=device
            )
            
            # Process each sample
            for i in range(len(ecg)):
                gt_report = gt_reports[i]
                retrieved = retrieved_reports[i] if retrieved_reports[i] else [""]
                
                # Use top-1 as prediction
                pred_report = retrieved[0] if retrieved else ""
                predictions.append(pred_report)
                ground_truths.append(gt_report)
                
                # Compute retrieval metrics
                if gt_report in retrieved:
                    recall_at_k += 1
                    rank = retrieved.index(gt_report) + 1
                    mrr_sum += 1.0 / rank
                
                total_samples += 1
    
    # Calculate final metrics
    retrieval_metrics = {
        f'recall@{k}': recall_at_k / total_samples if total_samples > 0 else 0.0,
        'mrr': mrr_sum / total_samples if total_samples > 0 else 0.0
    }
    
    return predictions, ground_truths, retrieval_metrics
