import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb
from pathlib import Path


class PTBXLDataset(Dataset):
    """PTB-XL dataset for ECG-ReGen with reports"""
    
    def __init__(self, jsonl_path, csv_path=None, max_length=5000, transform=None):
        """
        Args:
            jsonl_path: Path to JSONL file with ECG paths and reports
            csv_path: Path to PTB-XL database CSV (optional, for additional metadata)
            max_length: Maximum ECG signal length (default 5000 for 500Hz)
            transform: Optional data augmentation transforms
        """
        self.data = []
        self.max_length = max_length
        self.transform = transform
        
        # Load JSONL data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
        
        # Load CSV metadata if provided
        self.metadata = None
        if csv_path:
            self.metadata = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get ECG path and report
        ecg_path = item['ecg_path']
        report = item['messages'][1]['content']  # Assistant's response
        
        # Load ECG signal using wfdb
        try:
            # Remove the _hr suffix if present
            if ecg_path.endswith('_hr'):
                ecg_path = ecg_path[:-3]
            
            # Read the signal
            record = wfdb.rdrecord(ecg_path)
            ecg_signal = record.p_signal  # Shape: (samples, 12 leads)
            
            # Transpose to (12, samples)
            ecg_signal = ecg_signal.T
            
            # Pad or truncate to max_length
            if ecg_signal.shape[1] < self.max_length:
                # Pad with zeros
                padding = np.zeros((ecg_signal.shape[0], self.max_length - ecg_signal.shape[1]))
                ecg_signal = np.concatenate([ecg_signal, padding], axis=1)
            elif ecg_signal.shape[1] > self.max_length:
                # Truncate
                ecg_signal = ecg_signal[:, :self.max_length]
            
            # Convert to tensor
            ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)
            
            # Apply transforms if any
            if self.transform:
                ecg_signal = self.transform(ecg_signal)
            
            return {
                'ecg': ecg_signal,  # Shape: (12, max_length)
                'report': report,
                'id': item['id']
            }
        
        except Exception as e:
            print(f"Error loading ECG {ecg_path}: {e}")
            # Return a zero signal if loading fails
            return {
                'ecg': torch.zeros(12, self.max_length, dtype=torch.float32),
                'report': report,
                'id': item['id']
            }


def collate_fn(batch):
    """Custom collate function for batching"""
    ecgs = torch.stack([item['ecg'] for item in batch])
    reports = [item['report'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'ecg': ecgs,
        'report': reports,
        'id': ids
    }
