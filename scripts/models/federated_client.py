"""
Flower client for federated learning.
"""

import flwr as fl
import torch
from scripts.models.central_model import PolygenicNeuralNetwork
from torch.utils.data import DataLoader, TensorDataset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train_model(
            self.train_dataset.tensors[0].numpy(),
            self.train_dataset.tensors[1].numpy(),
            self.val_dataset.tensors[0].numpy(),
            self.val_dataset.tensors[1].numpy(),
            epochs=1, # In FL, we typically train for a small number of epochs
        )
        return self.get_parameters(config={}), len(self.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.model.evaluate(
            self.val_dataset.tensors[0].numpy(), self.val_dataset.tensors[1].numpy()
        )
        return metrics["auroc"], len(self.val_dataset), {"auroc": metrics["auroc"], "auprc": metrics["auprc"]}
