import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertConfig
import math


class ECGEncoder(nn.Module):
    """ECG Encoder with 1D Conv + Transformer"""
    
    def __init__(self, 
                 input_channels=12, 
                 hidden_dim=768,
                 num_layers=6,
                 num_heads=12,
                 mlp_dim=3072,
                 dropout=0.1,
                 patch_size=50):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        
        # 1D Convolutional layers for local feature extraction
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(256, hidden_dim, kernel_size=15, stride=1, padding=7)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, channels=12, length=5000)
            mask: optional mask for masked ECG modeling
        Returns:
            features: (batch, seq_len, hidden_dim)
        """
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # (batch, hidden_dim, seq_len)
        
        # Transpose for transformer: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Apply mask if provided (for MEM task)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Layer norm
        x = self.ln(x)
        
        return x


class ECGDecoder(nn.Module):
    """Decoder for Masked ECG Modeling (MEM)"""
    
    def __init__(self, hidden_dim=768, num_layers=2):
        super().__init__()
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection stays in hidden_dim space for feature reconstruction
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, memory):
        """
        Args:
            x: decoder input (batch, seq_len, hidden_dim)
            memory: encoder output (batch, seq_len, hidden_dim)
        Returns:
            reconstructed features (batch, seq_len, hidden_dim)
        """
        x = self.transformer_decoder(x, memory)
        x = self.output_proj(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module for ECG-Text matching"""
    
    def __init__(self, hidden_dim=768, num_layers=2):
        super().__init__()
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=12, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, ecg_features, text_features):
        """
        Cross-attention between ECG and text features
        Args:
            ecg_features: (batch, ecg_seq_len, hidden_dim)
            text_features: (batch, text_seq_len, hidden_dim)
        """
        # Cross attention: ECG queries text
        for cross_attn in self.cross_attn_layers:
            attn_out, _ = cross_attn(ecg_features, text_features, text_features)
            ecg_features = self.ln1(ecg_features + attn_out)
            
            # FFN
            ffn_out = self.ffn(ecg_features)
            ecg_features = self.ln2(ecg_features + ffn_out)
        
        return ecg_features


class ECGReGen(nn.Module):
    """Main ECG-ReGen model with multi-modal pre-training"""
    
    def __init__(self, 
                 ecg_encoder_config=None,
                 bert_model_name='bert-base-uncased',
                 hidden_dim=768):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ECG Encoder
        ecg_config = ecg_encoder_config or {}
        self.ecg_encoder = ECGEncoder(hidden_dim=hidden_dim, **ecg_config)
        
        # Text Encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # ECG Decoder for MEM
        self.ecg_decoder = ECGDecoder(hidden_dim=hidden_dim)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(hidden_dim=hidden_dim)
        
        # ECG-Text matching head
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
        
        # MLM head (use BERT's MLM head)
        self.mlm_head = nn.Linear(hidden_dim, self.text_encoder.config.vocab_size)
        
    def forward(self, ecg, text_ids, text_mask, ecg_mask=None, text_mlm_labels=None):
        """
        Forward pass for multi-modal pre-training
        Args:
            ecg: (batch, 12, 5000)
            text_ids: (batch, max_text_len)
            text_mask: (batch, max_text_len)
            ecg_mask: (batch, ecg_seq_len) - mask for MEM
            text_mlm_labels: (batch, max_text_len) - labels for MLM
        """
        # Encode ECG
        ecg_features = self.ecg_encoder(ecg, mask=ecg_mask)  # (batch, seq_len, hidden_dim)
        
        # Encode text
        text_outputs = self.text_encoder(input_ids=text_ids, attention_mask=text_mask)
        text_features = text_outputs.last_hidden_state  # (batch, text_len, hidden_dim)
        
        # Multi-modal fusion
        # Multi-modal fusion
        fused_features = self.fusion(ecg_features, text_features)
        
        # Global pooling for embeddings
        # Use ECG-only features for retrieval consistency (not fused)
        ecg_embedding = torch.max(ecg_features, dim=1)[0]  # (batch, hidden_dim)
        
        outputs = {
            'ecg_features': ecg_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'ecg_embedding': ecg_embedding
        }
        
        # MEM: Decode masked ECG
        if ecg_mask is not None:
            decoded_ecg = self.ecg_decoder(fused_features, ecg_features)
            outputs['decoded_ecg'] = decoded_ecg
        
        # MLM: Predict masked tokens
        if text_mlm_labels is not None:
            mlm_logits = self.mlm_head(text_features)
            outputs['mlm_logits'] = mlm_logits
        
        # ECG-Text matching
        matching_logits = self.matching_head(ecg_embedding)
        outputs['matching_logits'] = matching_logits
        
        return outputs
    
    def get_ecg_embedding(self, ecg):
        """Get ECG embedding for retrieval (consistent with training)"""
        with torch.no_grad():
            ecg_features = self.ecg_encoder(ecg)
            embedding = torch.max(ecg_features, dim=1)[0]  # Global max pooling
            return embedding
