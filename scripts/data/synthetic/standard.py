"""
This script generates synthetic data for federated learning with different levels of
data heterogeneity. It is designed to be a versatile tool for testing and comparing
federated learning algorithms under various data distribution scenarios.

The data generation process is guided by seminal papers in the field of federated
learning, ensuring that the generated datasets are realistic and suitable for
research purposes.

Usage:
    To generate synthetic data, use the `generate_synthetic_data` function.
    You can specify the number of clients, classes, samples, and the type of
    data distribution ('iid', 'non-iid-label', or 'non-iid-quantity').

    To visualize the data distribution, use the `visualize_data_distribution`
    function. This function can help you understand the level of heterogeneity
    in the generated dataset.
"""

import numpy as np
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

def generate_synthetic_data(num_clients, num_classes, num_samples, distribution, alpha=0.5, beta=0.5):
    """
    Generates synthetic data for federated learning with varying levels of heterogeneity.

    This function is guided by the principles outlined in seminal federated learning papers
    to simulate realistic data distributions among clients. The goal is to create datasets
    that can be used to evaluate the performance of federated learning algorithms under
    different data heterogeneity scenarios.

    References:
    - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
    - "Federated Optimization in Heterogeneous Networks" (Li et al., 2020)
    - "Federated Learning with Non-IID and Heterogeneous Data for Mobile and Edge Computing" (FedNH)

    Args:
        num_clients (int): The number of clients in the federated network.
        num_classes (int): The number of classes in the dataset.
        num_samples (int): The total number of samples in the dataset.
        distribution (str): The type of data distribution. Can be one of:
            - 'iid': Independent and identically distributed data.
            - 'non-iid-label': Non-IID data based on label distribution skew.
            - 'non-iid-quantity': Non-IID data based on quantity skew.
        alpha (float): Dirichlet distribution parameter for label skew. Lower alpha means higher heterogeneity.
        beta (float): Power law distribution parameter for quantity skew. Lower beta means higher heterogeneity.

    Returns:
        dict: A dictionary where keys are client IDs and values are their respective data and labels.
    """

    # Use MNIST as the base dataset for generating synthetic data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    data_by_client = {i: {'data': [], 'labels': []} for i in range(num_clients)}

    if distribution == 'iid':
        # In the IID setting, data is randomly shuffled and partitioned among clients.
        # This is the most basic assumption and serves as a baseline for comparison.
        # Each client receives an equal number of samples, and the class distribution
        # is expected to be uniform across clients.
        
        all_indices = np.arange(len(mnist_data))
        np.random.shuffle(all_indices)
        
        samples_per_client = num_samples // num_clients
        for i in range(num_clients):
            client_indices = all_indices[i * samples_per_client : (i + 1) * samples_per_client]
            for idx in client_indices:
                data, label = mnist_data[int(idx)]
                data_by_client[i]['data'].append(data)
                data_by_client[i]['labels'].append(label)

    elif distribution == 'non-iid-label':
        # This setting simulates label distribution skew, a common form of non-IID data.
        # We use a Dirichlet distribution to partition data among clients, as proposed in
        # "Federated Optimization in Heterogeneous Networks". A smaller alpha value leads
        # to a more skewed distribution, where each client may only have a subset of the
        # total classes. This is also known as label skew non-IID.
        
        labels = np.array(mnist_data.targets)
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            
        client_class_samples = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            proportions = label_distribution[c]
            proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)
            
            start = 0
            for i in range(num_clients):
                end = proportions[i]
                client_class_samples[i].extend(class_indices[c][start:end])
                start = end

        for i in range(num_clients):
            for idx in client_class_samples[i]:
                data, label = mnist_data[int(idx)]
                data_by_client[i]['data'].append(data)
                data_by_client[i]['labels'].append(label)

    elif distribution == 'non-iid-quantity':
        # This setting simulates quantity skew, where the number of samples per client
        # varies significantly. We use a power law distribution to assign a different
        # number of samples to each client. This is another common type of non-IID data
        # that can impact the performance of federated learning algorithms.
        
        all_indices = np.arange(len(mnist_data))
        np.random.shuffle(all_indices)
        
        proportions = np.random.power(beta, num_clients)
        proportions = proportions / proportions.sum()
        
        client_sample_counts = (proportions * num_samples).astype(int)
        
        start = 0
        for i in range(num_clients):
            end = start + client_sample_counts[i]
            client_indices = all_indices[start:end]
            for idx in client_indices:
                data, label = mnist_data[int(idx)]
                data_by_client[i]['data'].append(data)
                data_by_client[i]['labels'].append(label)
            start = end
            
    else:
        raise ValueError("Invalid distribution type. Choose from 'iid', 'non-iid-label', or 'non-iid-quantity'.")

    # Convert lists to tensors
    for i in range(num_clients):
        if len(data_by_client[i]['data']) > 0:
            data_by_client[i]['data'] = torch.stack(data_by_client[i]['data'])
            data_by_client[i]['labels'] = torch.tensor(data_by_client[i]['labels'])

    return data_by_client

def visualize_data_distribution(data_by_client, num_clients, num_classes, output_path=None):
    """
    Visualizes the data distribution among clients.

    This function creates a bar chart showing the number of samples per class for each client.
    This can be helpful for understanding the level of data heterogeneity.

    Args:
        data_by_client (dict): A dictionary where keys are client IDs and values are their respective data and labels.
        num_clients (int): The number of clients in the federated network.
        num_classes (int): The number of classes in the dataset.
        output_path (str, optional): If provided, the plot will be saved to this path.
    """
    fig, axes = plt.subplots(1, num_clients, figsize=(20, 5), sharey=True)
    fig.suptitle('Data Distribution per Client')

    for i in range(num_clients):
        if len(data_by_client[i]['labels']) > 0:
            client_labels = data_by_client[i]['labels'].numpy()
            class_counts = [np.sum(client_labels == j) for j in range(num_classes)]
            axes[i].bar(range(num_classes), class_counts)
        else:
            axes[i].bar(range(num_classes), [0] * num_classes)
        axes[i].set_title(f'Client {i}')
        axes[i].set_xlabel('Class')
        if i == 0:
            axes[i].set_ylabel('Number of Samples')

    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path)
    else:
        plt.show()

if __name__ == '__main__':
    # Example of how to use the data generator
    num_clients = 10
    num_classes = 10
    num_samples = 5000
    output_dir = "assets/data_generation"

    # --- IID Example ---
    print("Generating IID data...")
    iid_data = generate_synthetic_data(num_clients=num_clients, num_classes=num_classes, num_samples=num_samples, distribution='iid')
    print(f"Generated IID data for {len(iid_data)} clients.")
    print(f"Client 0 has {len(iid_data[0]['data'])} samples.")
    visualize_data_distribution(iid_data, num_clients, num_classes, os.path.join(output_dir, "iid_distribution.png"))

    # --- Non-IID Label Skew Example ---
    print("\nGenerating non-IID (label skew) data...")
    non_iid_label_data = generate_synthetic_data(num_clients=num_clients, num_classes=num_classes, num_samples=num_samples, distribution='non-iid-label', alpha=0.1)
    print(f"Generated non-IID (label skew) data for {len(non_iid_label_data)} clients.")
    for i in range(3):
        if len(non_iid_label_data[i]['labels']) > 0:
            print(f"Client {i} has labels: {np.unique(non_iid_label_data[i]['labels'])}")
        else:
            print(f"Client {i} has no data.")
    visualize_data_distribution(non_iid_label_data, num_clients, num_classes, os.path.join(output_dir, "non_iid_label_distribution.png"))

    # --- Non-IID Quantity Skew Example ---
    print("\nGenerating non-IID (quantity skew) data...")
    non_iid_quantity_data = generate_synthetic_data(num_clients=num_clients, num_classes=num_classes, num_samples=num_samples, distribution='non-iid-quantity', beta=0.5)
    print(f"Generated non-IID (quantity skew) data for {len(non_iid_quantity_data)} clients.")
    for i in range(3):
        print(f"Client {i} has {len(non_iid_quantity_data[i]['data'])} samples.")
    visualize_data_distribution(non_iid_quantity_data, num_clients, num_classes, os.path.join(output_dir, "non_iid_quantity_distribution.png"))
