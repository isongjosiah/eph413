
import json
import matplotlib.pyplot as plt

def plot_trust_evolution(data_file, output_file):
    with open(data_file, 'r') as f:
        results = json.load(f)

    trust_history = results.get("trust_history", {})
    if not trust_history:
        print("No trust history found in the results file.")
        return

    plt.figure(figsize=(10, 6))
    for client_id, history in trust_history.items():
        plt.plot(history, label=f"Client {client_id}")

    plt.xlabel("Round")
    plt.ylabel("Trust Score")
    plt.title("Trust Evolution of Byzantine Clients")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_trust_evolution("byzantine_simulation_results.json", "trust_evolution.png")
