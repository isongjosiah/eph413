import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from scripts.data.synthetic.genomic import GeneticDataGenerator
from scripts.models.hprs_model import HierarchicalPRSModel
import os

class MembershipInferenceAttack:
    def __init__(self, n_shadow_models=5, n_rare_variants=500):
        self.n_shadow_models = n_shadow_models
        self.n_rare_variants = n_rare_variants
        self.attack_model = None
        self.shadow_models = []
        self.shadow_data = []
        self.data_dir = "scripts/models/mia_data"

    def train_shadow_models(self):
        print("\nTraining shadow models for MIA...")
        os.makedirs(self.data_dir, exist_ok=True)

        for i in range(self.n_shadow_models):
            print(f"  Training shadow model {i+1}/{self.n_shadow_models}")
            data_path = os.path.join(self.data_dir, f"shadow_data_{i}.pt")

            if os.path.exists(data_path):
                print(f"    Loading shadow data from {data_path}")
                shadow_data_tensors = torch.load(data_path)
                prs_tensor = shadow_data_tensors['prs']
                rare_tensor = shadow_data_tensors['rare']
                phenotype_tensor = shadow_data_tensors['phenotype']
                dataset = TensorDataset(prs_tensor, rare_tensor, phenotype_tensor)
            else:
                print("    Generating new shadow data...")
                data_generator = GeneticDataGenerator(n_rare_variants=self.n_rare_variants)
                client_datasets = data_generator.create_federated_datasets(n_clients=1)
                shadow_data = client_datasets[0]

                prs_tensor = torch.FloatTensor(shadow_data["prs_scores"].reshape(-1, 1))
                rare_tensor = torch.FloatTensor(shadow_data["rare_dosages"])
                phenotype_tensor = torch.FloatTensor(shadow_data["phenotype_binary"].reshape(-1, 1))
                
                print(f"    Saving shadow data to {data_path}")
                torch.save({
                    'prs': prs_tensor,
                    'rare': rare_tensor,
                    'phenotype': phenotype_tensor
                }, data_path)
                dataset = TensorDataset(prs_tensor, rare_tensor, phenotype_tensor)

            train_size = int(0.5 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            shadow_model = HierarchicalPRSModel(n_rare_variants=self.n_rare_variants)
            optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
            criterion = nn.BCELoss()

            for epoch in range(10):
                shadow_model.train()
                for prs, rare, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = shadow_model(prs, rare)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            self.shadow_models.append(shadow_model)
            self.shadow_data.append({'train': train_dataset, 'test': test_dataset})

    def generate_attack_data(self):
        print("Generating data for the attack model...")
        attack_X = []
        attack_y = []

        for i, shadow_model in enumerate(self.shadow_models):
            train_dataset = self.shadow_data[i]['train']
            test_dataset = self.shadow_data[i]['test']

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            shadow_model.eval()
            with torch.no_grad():
                for prs, rare, _ in train_loader:
                    member_preds = shadow_model(prs, rare).cpu().numpy()
                    attack_X.extend(member_preds)
                    attack_y.extend(np.ones(len(member_preds)))

            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            shadow_model.eval()
            with torch.no_grad():
                for prs, rare, _ in test_loader:
                    non_member_preds = shadow_model(prs, rare).cpu().numpy()
                    attack_X.extend(non_member_preds)
                    attack_y.extend(np.zeros(len(non_member_preds)))

        return np.array(attack_X), np.array(attack_y)

    def train_attack_model(self):
        attack_X, attack_y = self.generate_attack_data()
        self.attack_model = LogisticRegression(solver='liblinear')
        self.attack_model.fit(attack_X, attack_y)

    def run_attack(self, target_model, member_data, non_member_data):
        # Member predictions
        member_loader = DataLoader(member_data, batch_size=32, shuffle=False)
        target_model.eval()
        member_preds = []
        with torch.no_grad():
            for prs, rare, _ in member_loader:
                preds = target_model(prs, rare).cpu().numpy()
                member_preds.extend(preds)

        # Non-member predictions
        non_member_loader = DataLoader(non_member_data, batch_size=32, shuffle=False)
        target_model.eval()
        non_member_preds = []
        with torch.no_grad():
            for prs, rare, _ in non_member_loader:
                preds = target_model(prs, rare).cpu().numpy()
                non_member_preds.extend(preds)

        # Attack
        member_attack_preds = self.attack_model.predict(member_preds)
        non_member_attack_preds = self.attack_model.predict(non_member_preds)

        # Attack accuracy
        attack_accuracy = (np.sum(member_attack_preds) + (len(non_member_attack_preds) - np.sum(non_member_attack_preds))) / (len(member_attack_preds) + len(non_member_attack_preds))
        return attack_accuracy