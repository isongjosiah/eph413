"""
Polygenic Risk Score (PRS) Prediction with Deep Learning and Data Heterogeneity Analysis
======================================================================================
This codebase implements a comprehensive framework for PRS prediction using deep learning
while studying the effectiveness of handling various types of genomic data heterogeneity.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================= Data Structures =========================

@dataclass
class HeterogeneityConfig:
    """Configuration for heterogeneity simulation"""
    population_groups: int = 3
    batch_groups: int = 4
    tissue_types: int = 5
    allelic_variants: int = 3
    tumor_clones: int = 3
    cohort_studies: int = 3

@dataclass
class ModelConfig:
    """Configuration for deep learning model"""
    input_dim: int = None
    hidden_dims: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========================= Data Loading and Preprocessing =========================

class CINECADataLoader:
    """Handles loading and initial preprocessing of CINECA synthetic dataset"""
    
    def __init__(self, zip_path: str = None):
        self.zip_path = zip_path
        self.data = None
        self.genotype_data = None
        self.phenotype_data = None
        self.metadata = None
        
    def load_from_zip(self, zip_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from zipped CINECA dataset
        Returns: genotype_data, phenotype_data, metadata
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('temp_data/')
            
            # Assuming standard CINECA format
            genotype_path = 'temp_data/genotypes.csv'
            phenotype_path = 'temp_data/phenotypes.csv'
            metadata_path = 'temp_data/metadata.csv'
            
            # Load with error handling for different possible formats
            if os.path.exists(genotype_path):
                self.genotype_data = pd.read_csv(genotype_path)
            else:
                # Try alternative formats
                for file in os.listdir('temp_data/'):
                    if 'geno' in file.lower():
                        self.genotype_data = pd.read_csv(f'temp_data/{file}')
                        break
            
            if os.path.exists(phenotype_path):
                self.phenotype_data = pd.read_csv(phenotype_path)
            else:
                for file in os.listdir('temp_data/'):
                    if 'pheno' in file.lower():
                        self.phenotype_data = pd.read_csv(f'temp_data/{file}')
                        break
            
            if os.path.exists(metadata_path):
                self.metadata = pd.read_csv(metadata_path)
            else:
                for file in os.listdir('temp_data/'):
                    if 'meta' in file.lower():
                        self.metadata = pd.read_csv(f'temp_data/{file}')
                        break
                        
            logger.info(f"Data loaded successfully. Genotype shape: {self.genotype_data.shape if self.genotype_data is not None else 'None'}")
            return self.genotype_data, self.phenotype_data, self.metadata
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Generate synthetic data for demonstration
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic genomic data for demonstration purposes"""
        logger.info("Generating synthetic CINECA-like dataset...")
        
        n_samples = 1000
        n_snps = 500  # Number of SNPs
        
        # Generate genotype data (0, 1, 2 encoding for AA, Aa, aa)
        np.random.seed(42)
        genotype_data = np.random.choice([0, 1, 2], size=(n_samples, n_snps), 
                                        p=[0.25, 0.5, 0.25])
        
        # Generate phenotype (PRS score)
        # True PRS based on weighted sum of SNPs with noise
        true_weights = np.random.randn(n_snps) * 0.1
        prs_scores = np.dot(genotype_data, true_weights) + np.random.randn(n_samples) * 0.5
        
        # Generate metadata
        metadata = pd.DataFrame({
            'sample_id': [f'SAMPLE_{i:04d}' for i in range(n_samples)],
            'population': np.random.choice(['EUR', 'AFR', 'EAS', 'SAS', 'AMR'], n_samples),
            'batch': np.random.choice(['BATCH_1', 'BATCH_2', 'BATCH_3', 'BATCH_4'], n_samples),
            'tissue': np.random.choice(['Blood', 'Brain', 'Liver', 'Muscle', 'Kidney'], n_samples),
            'study': np.random.choice(['STUDY_A', 'STUDY_B', 'STUDY_C'], n_samples),
            'age': np.random.randint(20, 80, n_samples),
            'sex': np.random.choice(['M', 'F'], n_samples)
        })
        
        # Create DataFrames
        self.genotype_data = pd.DataFrame(
            genotype_data,
            columns=[f'SNP_{i:04d}' for i in range(n_snps)]
        )
        self.genotype_data['sample_id'] = metadata['sample_id']
        
        self.phenotype_data = pd.DataFrame({
            'sample_id': metadata['sample_id'],
            'prs_score': prs_scores
        })
        
        self.metadata = metadata
        
        return self.genotype_data, self.phenotype_data, self.metadata

# ========================= Heterogeneity Simulation =========================

class HeterogeneitySimulator:
    """Simulates various types of genomic data heterogeneity"""
    
    def __init__(self, config: HeterogeneityConfig):
        self.config = config
        
    def simulate_population_stratification(self, X: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate population stratification by introducing ancestry-specific allele frequency differences
        """
        X_stratified = X.copy()
        populations = metadata['population'].unique() if 'population' in metadata else ['POP1', 'POP2', 'POP3']
        
        for pop in populations:
            pop_mask = metadata['population'] == pop if 'population' in metadata else np.random.rand(len(X)) > 0.5
            if np.any(pop_mask):
                # Introduce population-specific allele frequency shifts
                shift = np.random.randn(X.shape[1]) * 0.2
                X_stratified[pop_mask] += shift
                
        logger.info(f"Applied population stratification for {len(populations)} populations")
        return X_stratified
    
    def simulate_batch_effects(self, X: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate batch effects by adding systematic technical variation
        """
        X_batched = X.copy()
        batches = metadata['batch'].unique() if 'batch' in metadata else ['B1', 'B2', 'B3', 'B4']
        
        for batch in batches:
            batch_mask = metadata['batch'] == batch if 'batch' in metadata else np.random.rand(len(X)) > 0.5
            if np.any(batch_mask):
                # Add batch-specific systematic bias
                batch_effect = np.random.randn() * 0.3
                noise = np.random.randn(*X[batch_mask].shape) * 0.1
                X_batched[batch_mask] = X_batched[batch_mask] * (1 + batch_effect) + noise
                
        logger.info(f"Applied batch effects for {len(batches)} batches")
        return X_batched
    
    def simulate_tissue_specific_expression(self, X: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate tissue-specific gene expression patterns
        """
        X_tissue = X.copy()
        tissues = metadata['tissue'].unique() if 'tissue' in metadata else ['T1', 'T2', 'T3', 'T4', 'T5']
        
        # Create tissue-specific expression profiles
        n_features = X.shape[1]
        for tissue in tissues:
            tissue_mask = metadata['tissue'] == tissue if 'tissue' in metadata else np.random.rand(len(X)) > 0.5
            if np.any(tissue_mask):
                # Randomly select genes to be tissue-specific
                tissue_specific_genes = np.random.choice(n_features, size=n_features//3, replace=False)
                expression_modifier = np.ones(n_features)
                expression_modifier[tissue_specific_genes] = np.random.uniform(0.2, 2.0, len(tissue_specific_genes))
                X_tissue[tissue_mask] *= expression_modifier
                
        logger.info(f"Applied tissue-specific expression for {len(tissues)} tissue types")
        return X_tissue
    
    def simulate_allelic_heterogeneity(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate allelic/locus heterogeneity where different variants lead to similar phenotypes
        """
        X_allelic = X.copy()
        y_allelic = y.copy()
        
        # Create multiple pathways that can lead to similar phenotypes
        n_pathways = self.config.allelic_variants
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        pathway_assignments = np.random.choice(n_pathways, n_samples)
        
        for pathway in range(n_pathways):
            pathway_mask = pathway_assignments == pathway
            if np.any(pathway_mask):
                # Each pathway uses different sets of variants
                pathway_features = np.random.choice(n_features, size=n_features//n_pathways, replace=False)
                mask = np.zeros(n_features, dtype=bool)
                mask[pathway_features] = True
                
                # Zero out non-pathway features for these samples
                X_allelic[np.ix_(pathway_mask, ~mask)] *= 0.1
                
        logger.info(f"Applied allelic heterogeneity with {n_pathways} pathways")
        return X_allelic, y_allelic
    
    def simulate_tumor_heterogeneity(self, X: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate intra-tumor heterogeneity with multiple clones
        """
        X_tumor = X.copy()
        n_samples = X.shape[0]
        
        # Identify tumor samples (for demonstration, randomly select 30% of samples)
        tumor_samples = np.random.rand(n_samples) < 0.3
        
        if np.any(tumor_samples):
            tumor_indices = np.where(tumor_samples)[0]
            
            for idx in tumor_indices:
                # Each tumor has multiple clones with different genetic profiles
                n_clones = np.random.randint(2, self.config.tumor_clones + 1)
                clone_weights = np.random.dirichlet(np.ones(n_clones))
                
                # Create clone-specific mutations
                for clone in range(n_clones):
                    clone_mutations = np.random.randn(X.shape[1]) * 0.3 * clone_weights[clone]
                    X_tumor[idx] += clone_mutations
                    
        logger.info(f"Applied tumor heterogeneity to {np.sum(tumor_samples)} samples")
        return X_tumor
    
    def simulate_cohort_heterogeneity(self, X: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """
        Simulate cohort and study heterogeneity
        """
        X_cohort = X.copy()
        studies = metadata['study'].unique() if 'study' in metadata else ['S1', 'S2', 'S3']
        
        for study in studies:
            study_mask = metadata['study'] == study if 'study' in metadata else np.random.rand(len(X)) > 0.5
            if np.any(study_mask):
                # Each study has different processing pipelines and demographics
                study_bias = np.random.randn(X.shape[1]) * 0.15
                demographic_effect = np.random.randn() * 0.2
                X_cohort[study_mask] = X_cohort[study_mask] * (1 + demographic_effect) + study_bias
                
        logger.info(f"Applied cohort heterogeneity for {len(studies)} studies")
        return X_cohort

# ========================= Data Partitioning =========================

class DataPartitioner:
    """Handles data partitioning for federated learning scenarios"""
    
    def __init__(self, heterogeneity_type: str = 'iid'):
        self.heterogeneity_type = heterogeneity_type
        
    def partition_data(self, X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, 
                       n_clients: int = 5) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data across multiple clients based on heterogeneity type
        """
        partitions = {}
        
        if self.heterogeneity_type == 'iid':
            # IID partitioning
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            splits = np.array_split(indices, n_clients)
            
            for i, split in enumerate(splits):
                partitions[i] = (X[split], y[split])
                
        elif self.heterogeneity_type == 'population':
            # Partition by population groups
            if 'population' in metadata:
                populations = metadata['population'].unique()
                for i, pop in enumerate(populations[:n_clients]):
                    mask = metadata['population'] == pop
                    partitions[i] = (X[mask], y[mask])
                    
        elif self.heterogeneity_type == 'batch':
            # Partition by batches
            if 'batch' in metadata:
                batches = metadata['batch'].unique()
                for i, batch in enumerate(batches[:n_clients]):
                    mask = metadata['batch'] == batch
                    partitions[i] = (X[mask], y[mask])
                    
        elif self.heterogeneity_type == 'tissue':
            # Partition by tissue types
            if 'tissue' in metadata:
                tissues = metadata['tissue'].unique()
                for i, tissue in enumerate(tissues[:n_clients]):
                    mask = metadata['tissue'] == tissue
                    partitions[i] = (X[mask], y[mask])
                    
        elif self.heterogeneity_type == 'study':
            # Partition by study/cohort
            if 'study' in metadata:
                studies = metadata['study'].unique()
                for i, study in enumerate(studies[:n_clients]):
                    mask = metadata['study'] == study
                    partitions[i] = (X[mask], y[mask])
                    
        else:
            # Default to random non-IID
            indices = np.arange(len(X))
            # Create imbalanced partitions
            proportions = np.random.dirichlet(np.ones(n_clients) * 0.5)
            cumsum = np.cumsum(proportions)
            splits = []
            prev = 0
            for prop in cumsum[:-1]:
                split_point = int(prop * len(indices))
                splits.append(indices[prev:split_point])
                prev = split_point
            splits.append(indices[prev:])
            
            for i, split in enumerate(splits):
                if len(split) > 0:
                    partitions[i] = (X[split], y[split])
                    
        logger.info(f"Created {len(partitions)} data partitions with {self.heterogeneity_type} heterogeneity")
        return partitions

# ========================= Deep Learning Model =========================

class PRSDataset(Dataset):
    """PyTorch Dataset for PRS prediction"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PRSDeepNet(nn.Module):
    """Deep Neural Network for PRS prediction"""
    
    def __init__(self, config: ModelConfig):
        super(PRSDeepNet, self).__init__()
        
        layers = []
        input_dim = config.input_dim
        
        # Build hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PRSModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.training_history = {'loss': [], 'val_loss': []}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> PRSDeepNet:
        """Train the PRS prediction model"""
        
        # Create datasets and dataloaders
        train_dataset = PRSDataset(X_train, y_train)
        val_dataset = PRSDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize model
        self.config.input_dim = X_train.shape[1]
        if self.config.hidden_dims is None:
            self.config.hidden_dims = [256, 128, 64]
            
        self.model = PRSDeepNet(self.config).to(self.config.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.training_history['loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return self.model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.config.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
            
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        return metrics

# ========================= Heterogeneity Analysis =========================

class HeterogeneityAnalyzer:
    """Analyzes the effectiveness of handling different types of heterogeneity"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_heterogeneity_impact(self, X_base: np.ndarray, y_base: np.ndarray, 
                                    metadata: pd.DataFrame, config: ModelConfig) -> Dict[str, Any]:
        """
        Comprehensive analysis of heterogeneity impact on model performance
        """
        heterogeneity_simulator = HeterogeneitySimulator(HeterogeneityConfig())
        results = {}
        
        # Split data for training and testing
        X_train_base, X_test_base, y_train_base, y_test_base, meta_train, meta_test = train_test_split(
            X_base, y_base, metadata, test_size=0.2, random_state=42
        )
        
        X_train_val, X_val, y_train_val, y_val = train_test_split(
            X_train_base, y_train_base, test_size=0.2, random_state=42
        )
        
        # Baseline model (no heterogeneity)
        logger.info("Training baseline model...")
        trainer_baseline = PRSModelTrainer(config)
        trainer_baseline.train(X_train_val, y_train_val, X_val, y_val)
        baseline_metrics = trainer_baseline.evaluate(X_test_base, y_test_base)
        results['baseline'] = baseline_metrics
        
        # Test each heterogeneity type
        heterogeneity_types = {
            'population_stratification': lambda X, y, m: (
                heterogeneity_simulator.simulate_population_stratification(X, m), y
            ),
            'batch_effects': lambda X, y, m: (
                heterogeneity_simulator.simulate_batch_effects(X, m), y
            ),
            'tissue_specific': lambda X, y, m: (
                heterogeneity_simulator.simulate_tissue_specific_expression(X, m), y
            ),
            'allelic_heterogeneity': lambda X, y, m: 
                heterogeneity_simulator.simulate_allelic_heterogeneity(X, y),
            'tumor_heterogeneity': lambda X, y, m: (
                heterogeneity_simulator.simulate_tumor_heterogeneity(X, m), y
            ),
            'cohort_heterogeneity': lambda X, y, m: (
                heterogeneity_simulator.simulate_cohort_heterogeneity(X, m), y
            )
        }
        
        for het_name, het_func in heterogeneity_types.items():
            logger.info(f"Analyzing {het_name}...")
            
            # Apply heterogeneity
            X_het_train, y_het_train = het_func(X_train_base, y_train_base, meta_train)
            X_het_test, y_het_test = het_func(X_test_base, y_test_base, meta_test)
            
            # Split for validation
            X_train_val_het, X_val_het, y_train_val_het, y_val_het = train_test_split(
                X_het_train, y_het_train, test_size=0.2, random_state=42
            )
            
            # Train model on heterogeneous data
            trainer_het = PRSModelTrainer(config)
            trainer_het.train(X_train_val_het, y_train_val_het, X_val_het, y_val_het)
            
            # Evaluate on heterogeneous test data
            het_metrics = trainer_het.evaluate(X_het_test, y_het_test)
            
            # Test baseline model on heterogeneous data (robustness check)
            baseline_on_het = trainer_baseline.evaluate(X_het_test, y_het_test)
            
            results[het_name] = {
                'trained_on_het': het_metrics,
                'baseline_on_het': baseline_on_het,
                'performance_drop': {
                    metric: baseline_on_het[metric] - baseline_metrics[metric]
                    for metric in baseline_metrics
                },
                'adaptation_gain': {
                    metric: het_metrics[metric] - baseline_on_het[metric]
                    for metric in het_metrics
                }
            }
        
        self.results = results
        return results
    
    def analyze_federated_scenarios(self, X: np.ndarray, y: np.ndarray, 
                                   metadata: pd.DataFrame, config: ModelConfig) -> Dict[str, Any]:
        """
        Analyze performance in federated learning scenarios with different partitioning strategies
        """
        federated_results = {}
        partitioning_strategies = ['iid', 'population', 'batch', 'tissue', 'study']
        
        for strategy in partitioning_strategies:
            logger.info(f"Testing federated scenario with {strategy} partitioning...")
            
            partitioner = DataPartitioner(heterogeneity_type=strategy)
            partitions = partitioner.partition_data(X, y, metadata, n_clients=5)
            
            # Train local models and evaluate
            local_performances = []
            for client_id, (X_client, y_client) in partitions.items():
                if len(X_client) > 50:  # Minimum samples required
                    # Split client data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_client, y_client, test_size=0.3, random_state=42
                    )
                    
                    if len(X_train) > 20:
                        X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42
                        )
                        
                        # Train local model
                        local_config = ModelConfig(
                            hidden_dims=[128, 64],
                            epochs=50,
                            batch_size=16
                        )
                        
                        trainer = PRSModelTrainer(local_config)
                        trainer.train(X_train_c, y_train_c, X_val_c, y_val_c)
                        
                        # Evaluate
                        metrics = trainer.evaluate(X_test, y_test)
                        local_performances.append(metrics)
            
            if local_performances:
                # Calculate average performance across clients
                avg_metrics = {}
                for metric in local_performances[0].keys():
                    avg_metrics[metric] = np.mean([p[metric] for p in local_performances])
                    avg_metrics[f'{metric}_std'] = np.std([p[metric] for p in local_performances])
                
                federated_results[strategy] = {
                    'avg_metrics': avg_metrics,
                    'n_clients': len(local_performances),
                    'client_performances': local_performances
                }
        
        return federated_results
    
    def visualize_results(self, save_path: str = 'heterogeneity_analysis.png'):
        """Create comprehensive visualization of heterogeneity analysis results"""
        if not self.results:
            logger.warning("No results to visualize. Run analysis first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Heterogeneity Impact Analysis on PRS Prediction', fontsize=16)
        
        # Extract metrics for visualization
        het_types = [k for k in self.results.keys() if k != 'baseline']
        metrics_to_plot = ['mse', 'r2']
        
        for idx, het_type in enumerate(het_types[:6]):
            ax = axes[idx // 3, idx % 3]
            
            if het_type in self.results:
                data = self.results[het_type]
                
                # Compare baseline vs trained on heterogeneous data
                categories = ['Baseline', 'Baseline\non Het', 'Trained\non Het']
                mse_values = [
                    self.results['baseline']['mse'],
                    data['baseline_on_het']['mse'],
                    data['trained_on_het']['mse']
                ]
                r2_values = [
                    self.results['baseline']['r2'],
                    data['baseline_on_het']['r2'],
                    data['trained_on_het']['r2']
                ]
                
                x = np.arange(len(categories))
                width = 0.35
                
                ax2 = ax.twinx()
                bars1 = ax.bar(x - width/2, mse_values, width, label='MSE', color='coral')
                bars2 = ax2.bar(x + width/2, r2_values, width, label='R²', color='skyblue')
                
                ax.set_xlabel('Model Type')
                ax.set_ylabel('MSE', color='coral')
                ax2.set_ylabel('R² Score', color='skyblue')
                ax.set_title(het_type.replace('_', ' ').title())
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.tick_params(axis='y', labelcolor='coral')
                ax2.tick_params(axis='y', labelcolor='skyblue')
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
        plt.show()

# ========================= Main Pipeline =========================

class PRSPipeline:
    """Main pipeline for PRS prediction with heterogeneity analysis"""
    
    def __init__(self):
        self.data_loader = CINECADataLoader()
        self.heterogeneity_analyzer = HeterogeneityAnalyzer()
        self.results = {}
        
    def run_complete_analysis(self, data_path: str = None) -> Dict[str, Any]:
        """
        Run complete PRS prediction pipeline with heterogeneity analysis
        """
        logger.info("="*80)
        logger.info("Starting PRS Prediction Pipeline with Heterogeneity Analysis")
        logger.info("="*80)
        
        # Step 1: Load and preprocess data
        logger.info("\n[Step 1] Loading and preprocessing data...")
        if data_path and os.path.exists(data_path):
            genotype_data, phenotype_data, metadata = self.data_loader.load_from_zip(data_path)
        else:
            genotype_data, phenotype_data, metadata = self.data_loader.generate_synthetic_data()
        
        # Prepare features and targets
        X = genotype_data.drop(['sample_id'], axis=1, errors='ignore').values
        y = phenotype_data['prs_score'].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Data shape: X={X_scaled.shape}, y={y.shape}")
        logger.info(f"Metadata columns: {metadata.columns.tolist()}")
        
        # Step 2: Configure model
        logger.info("\n[Step 2] Configuring deep learning model...")
        model_config = ModelConfig(
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.3,
            learning_rate=0.001,
            batch_size=32,
            epochs=100
        )
        
        # Step 3: Analyze heterogeneity impact
        logger.info("\n[Step 3] Analyzing heterogeneity impact...")
        heterogeneity_results = self.heterogeneity_analyzer.analyze_heterogeneity_impact(
            X_scaled, y, metadata, model_config
        )
        
        # Step 4: Analyze federated scenarios
        logger.info("\n[Step 4] Analyzing federated learning scenarios...")
        federated_results = self.heterogeneity_analyzer.analyze_federated_scenarios(
            X_scaled, y, metadata, model_config
        )
        
        # Step 5: Generate visualizations
        logger.info("\n[Step 5] Generating visualizations...")
        self.heterogeneity_analyzer.visualize_results()
        
        # Step 6: Compile comprehensive results
        logger.info("\n[Step 6] Compiling results...")
        self.results = {
            'data_summary': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'populations': metadata['population'].unique().tolist() if 'population' in metadata else [],
                'batches': metadata['batch'].unique().tolist() if 'batch' in metadata else [],
                'tissues': metadata['tissue'].unique().tolist() if 'tissue' in metadata else [],
                'studies': metadata['study'].unique().tolist() if 'study' in metadata else []
            },
            'heterogeneity_analysis': heterogeneity_results,
            'federated_analysis': federated_results
        }
        
        # Print summary report
        self._print_summary_report()
        
        # Save results to JSON
        self._save_results('prs_analysis_results.json')
        
        return self.results
    
    def _print_summary_report(self):
        """Print a comprehensive summary report"""
        print("\n" + "="*80)
        print("HETEROGENEITY ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        if 'heterogeneity_analysis' in self.results:
            het_results = self.results['heterogeneity_analysis']
            
            # Baseline performance
            print("\n1. BASELINE MODEL PERFORMANCE:")
            print("-" * 40)
            if 'baseline' in het_results:
                for metric, value in het_results['baseline'].items():
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # Heterogeneity impact
            print("\n2. HETEROGENEITY IMPACT ANALYSIS:")
            print("-" * 40)
            
            for het_type in ['population_stratification', 'batch_effects', 'tissue_specific',
                           'allelic_heterogeneity', 'tumor_heterogeneity', 'cohort_heterogeneity']:
                if het_type in het_results:
                    print(f"\n  {het_type.replace('_', ' ').upper()}:")
                    data = het_results[het_type]
                    
                    print(f"    Model trained on heterogeneous data:")
                    for metric, value in data['trained_on_het'].items():
                        print(f"      {metric.upper()}: {value:.4f}")
                    
                    print(f"    Performance drop (baseline on heterogeneous):")
                    for metric, value in data['performance_drop'].items():
                        print(f"      {metric.upper()}: {value:+.4f}")
                    
                    print(f"    Adaptation gain (trained vs baseline on het):")
                    for metric, value in data['adaptation_gain'].items():
                        print(f"      {metric.upper()}: {value:+.4f}")
        
        if 'federated_analysis' in self.results:
            print("\n3. FEDERATED LEARNING SCENARIOS:")
            print("-" * 40)
            
            fed_results = self.results['federated_analysis']
            for strategy, data in fed_results.items():
                print(f"\n  {strategy.upper()} Partitioning:")
                print(f"    Number of clients: {data['n_clients']}")
                print(f"    Average metrics across clients:")
                for metric, value in data['avg_metrics'].items():
                    if not metric.endswith('_std'):
                        std_val = data['avg_metrics'].get(f'{metric}_std', 0)
                        print(f"      {metric.upper()}: {value:.4f} ± {std_val:.4f}")
        
        print("\n" + "="*80)
    
    def _save_results(self, filename: str):
        """Save results to JSON file"""
        import json
        
        def convert_numpy(obj):
            """Convert numpy types for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Deep copy and convert results
        json_results = json.loads(json.dumps(self.results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

# ========================= Utility Functions =========================

def correct_batch_effects(X: np.ndarray, metadata: pd.DataFrame, 
                         method: str = 'combat') -> np.ndarray:
    """
    Apply batch effect correction methods
    """
    if 'batch' not in metadata:
        logger.warning("No batch information in metadata")
        return X
    
    if method == 'combat':
        # Simplified ComBat-like adjustment
        X_corrected = X.copy()
        batches = metadata['batch'].unique()
        
        # Calculate grand mean
        grand_mean = np.mean(X, axis=0)
        
        for batch in batches:
            batch_mask = metadata['batch'] == batch
            if np.any(batch_mask):
                # Calculate batch mean
                batch_mean = np.mean(X[batch_mask], axis=0)
                # Adjust batch samples
                X_corrected[batch_mask] = X[batch_mask] - batch_mean + grand_mean
        
        return X_corrected
    
    elif method == 'pca':
        # PCA-based batch correction
        pca = PCA(n_components=min(X.shape[0], X.shape[1], 100))
        X_pca = pca.fit_transform(X)
        
        # Remove first few PCs that might capture batch effects
        X_pca[:, :2] = 0
        X_corrected = pca.inverse_transform(X_pca)
        
        return X_corrected
    
    else:
        return X

def analyze_population_structure(X: np.ndarray, metadata: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze population structure using PCA
    """
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)
    
    results = {
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pc_scores': X_pca[:, :3],  # First 3 PCs
    }
    
    if 'population' in metadata:
        # Analyze population clustering
        from sklearn.metrics import silhouette_score
        
        populations = metadata['population']
        if len(np.unique(populations)) > 1:
            silhouette = silhouette_score(X_pca[:, :3], populations)
            results['population_silhouette'] = silhouette
    
    return results

# ========================= Example Usage =========================

def main():
    """
    Main function to demonstrate the complete pipeline
    """
    # Initialize pipeline
    pipeline = PRSPipeline()
    
    # Run complete analysis
    # Note: Replace 'cineca_data.zip' with actual path to CINECA dataset
    results = pipeline.run_complete_analysis(data_path='cineca_data.zip')
    
    # Additional analyses can be performed here
    logger.info("\nPipeline completed successfully!")
    
    return results

if __name__ == "__main__":
    print("running main")
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run the main pipeline
    results = main() 
