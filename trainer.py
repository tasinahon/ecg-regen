import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ECGReGenTrainer:
    """Trainer for ECG-ReGen with self-supervised pre-training (MEM + MLM + ETM)"""
    
    def __init__(self, model, tokenizer, device='cuda', mem_mask_ratio=0.75, mlm_mask_ratio=0.15):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mem_mask_ratio = mem_mask_ratio
        self.mlm_mask_ratio = mlm_mask_ratio
        
        self.model.to(device)
    
    def create_ecg_mask(self, batch_size, seq_len):
        """Create mask for Masked ECG Modeling (MEM)"""
        mask = torch.ones(batch_size, seq_len, device=self.device)
        
        # Randomly mask patches
        for i in range(batch_size):
            num_masked = int(seq_len * self.mem_mask_ratio)
            masked_indices = np.random.choice(seq_len, num_masked, replace=False)
            mask[i, masked_indices] = 0
        
        return mask
    
    def create_mlm_labels(self, text_ids, text_mask):
        """Create labels for Masked Language Modeling (MLM)"""
        labels = text_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_mask_ratio, device=self.device)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool, device=self.device), value=0.0
        )
        
        # Mask tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).bool() & masked_indices
        text_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=self.device)
        text_ids[indices_random] = random_words[indices_random]
        
        return text_ids, labels
    
    def create_negative_pairs(self, batch_size):
        """Create negative pairs for ECG-Text matching"""
        # Create proper negative pairs by ensuring no self-matches
        negative_indices = torch.arange(batch_size, device=self.device)
        # Shift by 1 to avoid self-matching (circular shift)
        negative_indices = (negative_indices + 1) % batch_size
        return negative_indices
    
    def train_step(self, batch, optimizer, update_weights=True):
        """Single training step with optional gradient accumulation
        
        Args:
            batch: Input batch
            optimizer: Optimizer
            update_weights: Whether to update weights (for gradient accumulation)
        """
        self.model.train()
        
        ecg = batch['ecg'].to(self.device)
        reports = batch['report']
        batch_size = ecg.size(0)
        
        # Tokenize text
        text_encoding = self.tokenizer(
            reports,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        text_ids = text_encoding['input_ids'].to(self.device)
        text_mask = text_encoding['attention_mask'].to(self.device)
        
        # Create masks for MEM and MLM
        ecg_features_shape = ecg.size(2) // 8  # After 3 pooling layers (2^3 = 8)
        ecg_mask = self.create_ecg_mask(batch_size, ecg_features_shape)
        text_ids_masked, mlm_labels = self.create_mlm_labels(text_ids.clone(), text_mask)
        
        # Create negative pairs for ETM
        negative_indices = self.create_negative_pairs(batch_size)
        text_ids_neg = text_ids[negative_indices]
        text_mask_neg = text_mask[negative_indices]
        
        # Forward pass with positive pairs
        outputs_pos = self.model(
            ecg=ecg,
            text_ids=text_ids_masked,
            text_mask=text_mask,
            ecg_mask=ecg_mask,
            text_mlm_labels=mlm_labels
        )
        
        # Forward pass with negative pairs
        outputs_neg = self.model(
            ecg=ecg,
            text_ids=text_ids_neg,
            text_mask=text_mask_neg
        )
        
        # Compute losses
        # 1. MEM Loss: Reconstruction loss
        if 'decoded_ecg' in outputs_pos:
            # The decoder reconstructs the ECG features in hidden_dim space
            # Compare decoded features with original encoded features
            decoded = outputs_pos['decoded_ecg']  # (batch, seq_len, hidden_dim)
            
            # Get original ECG features (without mask)
            with torch.no_grad():
                original_features = self.model.ecg_encoder(ecg, mask=None)  # (batch, seq_len, hidden_dim)
            
            # Only compute loss on masked positions
            mask_expanded = ecg_mask.unsqueeze(-1).expand_as(original_features)
            mem_loss = F.mse_loss(
                decoded * (1 - mask_expanded),
                original_features.detach() * (1 - mask_expanded)
            )
        else:
            mem_loss = torch.tensor(0.0, device=self.device)
        
        # 2. MLM Loss
        if 'mlm_logits' in outputs_pos:
            mlm_logits = outputs_pos['mlm_logits']
            mlm_loss = F.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_labels.view(-1),
                ignore_index=-100
            )
        else:
            mlm_loss = torch.tensor(0.0, device=self.device)
        
        # 3. ETM Loss: Binary classification (positive vs negative pairs)
        matching_logits = torch.cat([
            outputs_pos['matching_logits'],
            outputs_neg['matching_logits']
        ], dim=0)
        matching_labels_full = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device)
        ], dim=0)
        etm_loss = F.cross_entropy(matching_logits, matching_labels_full)
        
        # Total loss
        total_loss = mem_loss + mlm_loss + etm_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update weights if specified (for gradient accumulation)
        if update_weights:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()  # Zero gradients after update
        
        return {
            'total_loss': total_loss.item(),
            'mem_loss': mem_loss.item(),
            'mlm_loss': mlm_loss.item(),
            'etm_loss': etm_loss.item()
        }
    
    def validate(self, val_loader, show_samples=True, num_samples=3):
        """Validation step with optional sample generation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        samples = []
        
        with torch.no_grad():
            for batch in val_loader:
                ecg = batch['ecg'].to(self.device)
                reports = batch['report']
                batch_size = ecg.size(0)
                
                # Tokenize
                text_encoding = self.tokenizer(
                    reports,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                text_ids = text_encoding['input_ids'].to(self.device)
                text_mask = text_encoding['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    ecg=ecg,
                    text_ids=text_ids,
                    text_mask=text_mask
                )
                
                # Simple matching loss for validation
                matching_logits = outputs['matching_logits']
                matching_labels = torch.ones(batch_size, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(matching_logits, matching_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect samples for display
                if show_samples and len(samples) < num_samples:
                    for i in range(min(batch_size, num_samples - len(samples))):
                        samples.append({
                            'true_report': reports[i],
                            'ecg_embedding': outputs['ecg_embedding'][i].cpu()
                        })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {
            'val_loss': avg_loss,
            'samples': samples if show_samples else []
        }
