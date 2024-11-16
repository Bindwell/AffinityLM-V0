import re
import torch
import ankh
import torch.nn as nn
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from Bio import SeqIO

@dataclass
class ModelConfig:
    def __init__(self,
                 input_dim=768,
                 embedding_dim=512,
                 linear_dim=256,
                 num_attention_layers=8,    # Increased base attention layers
                 num_heads=16,              # Increased base attention heads
                 cross_attention_layers=4,  # New: dedicated cross-attention
                 self_attention_layers=4,   # New: dedicated self-attention
                 dropout_rate=0.1,
                 feedforward_dim=2048):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.cross_attention_layers = cross_attention_layers
        self.self_attention_layers = self_attention_layers
        self.dropout_rate = dropout_rate
        self.feedforward_dim = feedforward_dim

class ProteinPairDataset(Dataset):
    """Dataset for protein pairs and their binding affinities"""
    def __init__(self, protein1_sequences: List[str], 
                 protein2_sequences: List[str], 
                 affinities: torch.Tensor,
                 mean: float = 6.51286529169358,
                 scale: float = 1.5614094578916633):
        assert len(protein1_sequences) == len(protein2_sequences) == len(affinities)
        # Convert sequences to strings explicitly
        self.protein1_sequences = [str(seq).strip() for seq in protein1_sequences]
        self.protein2_sequences = [str(seq).strip() for seq in protein2_sequences]
        # Normalize affinities
        self.affinities = (affinities - mean) / scale
        self.mean = mean
        self.scale = scale

    def __len__(self) -> int:
        return len(self.protein1_sequences)

    def __getitem__(self, idx: int) -> Tuple[str, str, float]:
        return (self.protein1_sequences[idx], 
                self.protein2_sequences[idx], 
                self.affinities[idx])

class CrossAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        
    def forward(self, query, key, value):
        attended = self.attention(
            query=self.norm1(query),
            key=self.norm1(key),
            value=self.norm1(value)
        )[0]
        return self.norm2(query + attended)

class ProteinProteinAffinityLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.mean = 6.51286529169358
        self.scale = 1.5614094578916633
        
        # Protein projection
        self.protein_projection = nn.Linear(
            config.input_dim,
            config.embedding_dim
        )
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 2, config.embedding_dim)
        )
        
        # Self-attention layers for each protein
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout_rate,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.self_attention_layers)
        ])
        
        # Cross-attention layers between proteins
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttention(config),
                CrossAttention(config)
            ]) for _ in range(config.cross_attention_layers)
        ])
        
        # Final transformer encoder layers
        self.final_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout_rate,
                batch_first=True,
                norm_first=True
            ) for _ in range(config.num_attention_layers)
        ])
        
        # Prediction head with attention
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.embedding_dim))
        
        self.affinity_head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.linear_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim, config.linear_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim // 2, 1)
        )

    def forward(self, protein1_embedding: torch.Tensor, 
                protein2_embedding: torch.Tensor,
                denormalize: bool = False) -> torch.Tensor:
        # Project proteins
        p1 = self.protein_projection(protein1_embedding)
        p2 = self.protein_projection(protein2_embedding)
        
        # Add positional embeddings
        p1 = p1 + self.pos_embedding[:, :1, :]
        p2 = p2 + self.pos_embedding[:, 1:, :]
        
        # Self-attention for each protein
        for layer in self.self_attention_layers:
            p1 = layer(p1)
            p2 = layer(p2)
        
        # Cross-attention between proteins
        for cross_layer1, cross_layer2 in self.cross_attention_layers:
            # P1 attending to P2
            p1_new = cross_layer1(p1, p2, p2)
            # P2 attending to P1
            p2_new = cross_layer2(p2, p1, p1)
            p1, p2 = p1_new, p2_new
        
        # Combine proteins for final attention
        combined = torch.cat([p1, p2], dim=1)
        
        # Final attention layers
        for layer in self.final_attention_layers:
            combined = layer(combined)
        
        # Attention pooling
        batch_size = combined.shape[0]
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled = self.attention_pooling(
            query=query,
            key=combined,
            value=combined
        )[0].squeeze(1)
        
        # Predict affinity
        prediction = self.affinity_head(pooled)
        
        if denormalize:
            return (prediction * self.scale) + self.mean
        return prediction

    def convert_to_affinity(self, normalized_prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert normalized prediction to actual affinity values"""
        neg_log10_affinity_M = (normalized_prediction * self.scale) + self.mean
        affinity_uM = (10**6) * (10**(-neg_log10_affinity_M))
        return {
            "neg_log10_affinity_M": neg_log10_affinity_M,
            "affinity_uM": affinity_uM
        }

class ProteinEmbeddingCache:
    """Cache for storing protein embeddings to avoid recomputation"""
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, protein_sequence: str) -> Optional[torch.Tensor]:
        return self.cache.get(protein_sequence)
    
    def set(self, protein_sequence: str, embedding: torch.Tensor):
        self.cache[protein_sequence] = embedding
    
    def save(self, filename: str):
        if self.cache_dir:
            torch.save(self.cache, self.cache_dir / filename)
    
    def load(self, filename: str) -> bool:
        if self.cache_dir and (self.cache_dir / filename).exists():
            self.cache = torch.load(self.cache_dir / filename)
            return True
        return False

class ProteinProteinAffinityTrainer:
    """Trainer class for the protein-protein affinity model"""
    def __init__(self, 
                 config: Optional[ModelConfig] = None,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.config = config or ModelConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.model = ProteinProteinAffinityLM(self.config).to(self.device)
        self.ankh_model, self.ankh_tokenizer = ankh.load_base_model()
        self.ankh_model.eval()
        self.ankh_model.to(self.device)
        
        # Initialize embedding cache
        self.protein_cache = ProteinEmbeddingCache(cache_dir)
        
    def encode_proteins(self, 
                       proteins: List[str], 
                       batch_size: int = 2) -> torch.Tensor:
        """Encode proteins using the Ankh model with caching"""
        embeddings = []
        
        for i in range(0, len(proteins), batch_size):
            batch = proteins[i:i+batch_size]
            batch_embeddings = []
            
            for protein in batch:
                protein = str(protein).strip()
                # Check cache first
                cached_embedding = self.protein_cache.get(protein)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                    continue
                
                # Compute embedding if not cached
                tokens = self.ankh_tokenizer([protein], 
                                          padding=True, 
                                          return_tensors="pt")
                with torch.no_grad():
                    output = self.ankh_model(
                        input_ids=tokens['input_ids'].to(self.device),
                        attention_mask=tokens['attention_mask'].to(self.device)
                    )
                    embedding = output.last_hidden_state.mean(dim=1)
                    self.protein_cache.set(protein, embedding.cpu())
                    batch_embeddings.append(embedding)
            
            embeddings.extend([emb.to(self.device) for emb in batch_embeddings])
        
        return torch.cat(embeddings)

    def prepare_data(self, 
                    protein1_sequences: List[str],
                    protein2_sequences: List[str],
                    affinities: List[float],
                    batch_size: int = 2,
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test data loaders"""
        # Convert affinities to tensor
        affinities_tensor = torch.tensor(affinities, dtype=torch.float32)
        
        # Split data
        train_p1, test_p1, train_p2, test_p2, train_aff, test_aff = train_test_split(
            protein1_sequences, protein2_sequences, affinities_tensor,
            test_size=test_size, random_state=42
        )
        
        train_p1, val_p1, train_p2, val_p2, train_aff, val_aff = train_test_split(
            train_p1, train_p2, train_aff,
            test_size=val_size, random_state=42
        )
        
        # Create datasets and dataloaders
        train_dataset = ProteinPairDataset(train_p1, train_p2, train_aff)
        val_dataset = ProteinPairDataset(val_p1, val_p2, val_aff)
        test_dataset = ProteinPairDataset(test_p1, test_p2, test_aff)
        
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size),
            DataLoader(test_dataset, batch_size=batch_size)
        )

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 100,
             learning_rate: float = 1e-4,
             save_dir: str = 'models',
             model_name: str = 'protein_protein_affinity.pt',
             patience: int = 10) -> Dict[str, List[float]]:
        """Train the model with early stopping and logging"""
        save_path = Path(save_dir) / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and model saving
# Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, save_path)
                print(f'Saved new best model with validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        return history

    def _train_epoch(self, 
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc='Training') as pbar:
            for p1_seqs, p2_seqs, affinities in pbar:
                # Encode proteins
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(p1_embeddings, p2_embeddings)
                loss = criterion(outputs.squeeze(), affinities)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)

    def _validate_epoch(self, 
                       val_loader: DataLoader,
                       criterion: nn.Module) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for p1_seqs, p2_seqs, affinities in val_loader:
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                outputs = self.model(p1_embeddings, p2_embeddings)
                loss = criterion(outputs.squeeze(), affinities)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data"""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for p1_seqs, p2_seqs, affinities in tqdm(test_loader, desc='Evaluating'):
                p1_embeddings = self.encode_proteins(p1_seqs)
                p2_embeddings = self.encode_proteins(p2_seqs)
                affinities = affinities.to(self.device)
                
                outputs = self.model(p1_embeddings, p2_embeddings)
                loss = criterion(outputs.squeeze(), affinities)
                total_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(affinities.cpu().numpy())
        
        mse = total_loss / len(test_loader)
        return {
            'mse': mse,
            'rmse': math.sqrt(mse),
            'predictions': predictions,
            'actuals': actuals
        }

    def predict(self, protein1_sequence: str, protein2_sequence: str) -> Dict[str, float]:
        """Make a prediction for a pair of proteins with denormalization"""
        self.model.eval()
        with torch.no_grad():
            # Encode proteins
            p1_embedding = self.encode_proteins([protein1_sequence])
            p2_embedding = self.encode_proteins([protein2_sequence])
            
            # Get normalized prediction
            normalized_pred = self.model(p1_embedding, p2_embedding)
            
            # Convert to actual affinity values
            return self.model.convert_to_affinity(normalized_pred)

# Configurations for experiments
class OverfitConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            embedding_dim=1024,
            linear_dim=512,
            num_attention_layers=12,        # Many final attention layers
            num_heads=32,                   # Many attention heads
            cross_attention_layers=8,       # Many cross-attention layers
            self_attention_layers=8,        # Many self-attention layers
            dropout_rate=0.0,              # No dropout for overfitting
            feedforward_dim=4096
        )

class UnderfitConfig(ModelConfig):
    def __init__(self):
        super().__init__(
            embedding_dim=512,
            linear_dim=256,
            num_attention_layers=4,
            num_heads=8,
            cross_attention_layers=2,
            self_attention_layers=2,
            dropout_rate=0.5,              # Heavy dropout
            feedforward_dim=1024
        )

def main():
    """Example usage with different configurations"""
    import logging
    from datetime import datetime
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('protein_affinity.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # Create directories for outputs
        output_dir = Path('new-output') 
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        # Load protein sequences and binding affinities from CSV data
        data_path = os.path.join(os.getcwd(), "data/Protein-Protein-Binding-Affinity-Data", "Data.csv")
        logger.info(f"Loading data from {data_path}")
        
        # Read the CSV file using pandas
        df = pd.read_csv(data_path)
            
        # Convert dataframe columns to lists and clean data
        df['protein1_sequence'] = df['protein1_sequence'].astype(str).str.strip()
        df['protein2_sequence'] = df['protein2_sequence'].astype(str).str.strip()
        protein1_sequences = df['protein1_sequence'].tolist()
        protein2_sequences = df['protein2_sequence'].tolist() 
        affinities = df['pkd'].astype(float).tolist()

        # Remove any invalid sequences
        valid_indices = [i for i in range(len(protein1_sequences)) 
                        if protein1_sequences[i] and protein2_sequences[i] 
                        and protein1_sequences[i].lower() != 'nan' 
                        and protein2_sequences[i].lower() != 'nan']
        
        protein1_sequences = [protein1_sequences[i] for i in valid_indices]
        protein2_sequences = [protein2_sequences[i] for i in valid_indices]
        affinities = [affinities[i] for i in valid_indices]

        logger.info(f"Loaded {len(protein1_sequences)} valid protein pairs")

        # Create different configurations for experiments
        configs = {
            'base': ModelConfig(),
            'overfit': OverfitConfig(),
            'underfit': UnderfitConfig()
        }

        results = {}
        
        for name, config in configs.items():
            logger.info(f"\nTraining with {name} configuration...")
            
            # Initialize trainer
            trainer = ProteinProteinAffinityTrainer(
                config=config,
                cache_dir=str(output_dir / f'embedding_cache_{name}')
            )

            # Prepare data
            train_loader, val_loader, test_loader = trainer.prepare_data(
                protein1_sequences=protein1_sequences,
                protein2_sequences=protein2_sequences,
                affinities=affinities,
                batch_size=2
            )

            # Train model
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                learning_rate=1e-4,
                save_dir=str(model_dir),
                model_name=f'protein_affinity_model_{name}.pt',
                patience=10
            )

            # Evaluate
            eval_results = trainer.evaluate(test_loader)
            results[name] = {
                'mse': float(eval_results['mse']),
                'rmse': float(eval_results['rmse']),
                'history': {
                    'train_loss': [float(x) for x in history['train_loss']],
                    'val_loss': [float(x) for x in history['val_loss']]
                }
            }

            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training History - {name.capitalize()} Configuration')
            plt.legend()
            plt.savefig(output_dir / f'training_history_{name}.png')
            plt.close()

        # Save comparative results
        with open(output_dir / 'comparative_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        logger.info("Experiment completed successfully")
        logger.info("\nResults summary:")
        for name, result in results.items():
            logger.info(f"\n{name.capitalize()} Configuration:")
            logger.info(f"MSE: {result['mse']:.4f}")
            logger.info(f"RMSE: {result['rmse']:.4f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()