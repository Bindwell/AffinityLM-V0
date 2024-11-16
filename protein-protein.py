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
                 input_dim=768,  # Add this line to match ESM embedding dimension
                 embedding_dim=256,
                 linear_dim=128,
                 num_attention_layers=3,
                 num_heads=4,
                 dropout_rate=0.1):
        self.input_dim = input_dim  # Add this line
        self.embedding_dim = embedding_dim
        self.linear_dim = linear_dim
        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
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

class ProteinProteinBindingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Update projection layers to use input_dim
        self.protein_projection = nn.Linear(config.input_dim, config.embedding_dim)
        
        # Rest of the initialization remains the same
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.embedding_dim,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
                batch_first=True
            ) for _ in range(config.num_attention_layers)
        ])
        
        self.fc1 = nn.Linear(config.embedding_dim * 2, config.linear_dim)
        self.fc2 = nn.Linear(config.linear_dim, 1)
        self.dropout = nn.Dropout(config.dropout_rate)

class ProteinProteinAffinityLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Project protein embeddings to model dimension
        self.protein_projection = nn.Linear(
            config.input_dim,  # Changed from config.protein_embedding_dim
            config.embedding_dim
        )
                
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True  # Improved training stability
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_attention_layers
        )
        
        # Prediction head
        self.affinity_head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),  # Added normalization
            nn.Linear(config.embedding_dim, config.linear_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim, config.linear_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.linear_dim // 2, 1)
        )

    def forward(self, protein1_embedding: torch.Tensor, 
                protein2_embedding: torch.Tensor) -> torch.Tensor:
        # Project proteins to common embedding space
        protein1_proj = self.protein_projection(protein1_embedding)
        protein2_proj = self.protein_projection(protein2_embedding)
        
        # Combine embeddings for transformer
        combined = torch.stack([protein1_proj, protein2_proj], dim=1)
        
        # Apply transformer and pool outputs
        transformed = self.transformer(combined)
        pooled = transformed.mean(dim=1)
        
        # Predict affinity
        return self.affinity_head(pooled)

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
                    batch_size: int = 32,
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
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
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




def main():
    """
    Example usage of the Protein-Protein Affinity Trainer
    Demonstrates training, evaluation, and inference workflows
    """
    import logging
    from datetime import datetime
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd  # Make sure pandas is imported

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
        output_dir = Path('output') 
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)

        # Load protein sequences and binding affinities from CSV data
        data_path = os.path.join(os.getcwd(), "data/Protein-Protein-Binding-Affinity-Data", "Data.csv")
        logger.info(f"Loading data from {data_path}")
        
        # Read the CSV file using pandas
        df = pd.read_csv(data_path)
            
        # Convert dataframe columns to lists
        protein1_sequences = df['protein1_sequence'].tolist()
        protein2_sequences = df['protein2_sequence'].tolist() 
        affinities = df['pkd'].tolist()

        # Remove any empty sequences
        valid_indices = [i for i in range(len(protein1_sequences)) 
                        if protein1_sequences[i] and protein2_sequences[i]]
        protein1_sequences = [protein1_sequences[i] for i in valid_indices]
        protein2_sequences = [protein2_sequences[i] for i in valid_indices]
        affinities = [affinities[i] for i in valid_indices]

        # Log data loading status
        logger.info(f"Loaded {len(protein1_sequences)} protein pairs")

        # Model configuration
        config = ModelConfig(
            input_dim=768,
            embedding_dim=256,
            linear_dim=128,
            num_attention_layers=3,
            num_heads=4,
            dropout_rate=0.1
        )

        # Save configuration
        with open(output_dir / 'config.json', 'w') as f:
            config_dict = {k: v for k, v in config.__dict__.items()}
            json.dump(config_dict, f, indent=4)

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ProteinProteinAffinityTrainer(
            config=config,
            cache_dir=str(output_dir / 'embedding_cache')
        )

        # Prepare data
        logger.info("Preparing data...")
        train_loader, val_loader, test_loader = trainer.prepare_data(
            protein1_sequences=protein1_sequences,
            protein2_sequences=protein2_sequences,
            affinities=affinities,
            batch_size=32,
            test_size=0.2,
            val_size=0.1
        )

        # Training
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            learning_rate=1e-4,
            save_dir=str(model_dir),
            model_name='protein_affinity_model.pt',
            patience=10
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(output_dir / 'training_history.png')
        plt.close()

        # Evaluation
        logger.info("Evaluating model...")
        results = trainer.evaluate(test_loader)

        # Save evaluation results
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            eval_results = {
                'mse': float(results['mse']),
                'rmse': float(results['rmse'])
            }
            json.dump(eval_results, f, indent=4)

        # Plot predictions vs actuals
        plt.figure(figsize=(8, 8))
        plt.scatter(results['actuals'], results['predictions'], alpha=0.5)
        plt.plot([min(results['actuals']), max(results['actuals'])],
                [min(results['actuals']), max(results['actuals'])],
                'r--', label='Perfect Prediction')
        plt.xlabel('Actual Affinity')
        plt.ylabel('Predicted Affinity')
        plt.title('Predictions vs Actuals')
        plt.legend()
        plt.savefig(output_dir / 'predictions_vs_actuals.png')
        plt.close()

        # Example inference with affinity conversion
        logger.info("Running example inference...")
        example_p1 = protein1_sequences[0]
        example_p2 = protein2_sequences[0]
        
        with torch.no_grad():
            p1_embedding = trainer.encode_proteins([example_p1])
            p2_embedding = trainer.encode_proteins([example_p2])
            prediction = trainer.model(p1_embedding, p2_embedding)

        
        logger.info(f"Example prediction for protein pair:")
        logger.info(f"Negative log10 affinity (M): {prediction['neg_log10_affinity_M'].item():.2f}")
        logger.info(f"Affinity (μM): {prediction['affinity_uM'].item():.2f}")

        # Save example to file
        with open(output_dir / 'example_inference.txt', 'w') as f:
            f.write(f"Protein 1: {example_p1}\n")
            f.write(f"Protein 2: {example_p2}\n")
            f.write(f"Negative log10 affinity (M): {prediction['neg_log10_affinity_M'].item():.2f}\n")
            f.write(f"Affinity (μM): {prediction['affinity_uM'].item():.2f}\n")

        logger.info(f"All outputs saved to {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()