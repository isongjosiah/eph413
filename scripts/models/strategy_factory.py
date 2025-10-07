from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad
from typing import List, Tuple, Union, Dict, Optional
from flwr.common import (
    EvaluateRes,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy

def get_strategy(strategy_name: str, initial_parameters):
    if strategy_name == "FedAvg":
        strategy = FedAvg(initial_parameters=initial_parameters)
    elif strategy_name == "FedProx":
        strategy = FedProx(proximal_mu=0.1, initial_parameters=initial_parameters)
    elif strategy_name == "FedAdam":
        strategy = FedAdam(initial_parameters=initial_parameters)
    elif strategy_name == "FedYogi":
        strategy = FedYogi(initial_parameters=initial_parameters)
    elif strategy_name == "FedAdagrad":
        strategy = FedAdagrad(initial_parameters=initial_parameters)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Wrap the strategy with a custom aggregation function
    base_aggregate_evaluate = strategy.aggregate_evaluate
    def custom_aggregate_evaluate(
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class to get aggregated loss
        aggregated_loss, aggregated_metrics = base_aggregate_evaluate(server_round, results, failures)

        # Aggregate custom metrics
        aurocs = [r.metrics["auroc"] for _, r in results]
        auprcs = [r.metrics["auprc"] for _, r in results]
        
        aggregated_metrics["auroc"] = sum(aurocs) / len(aurocs)
        aggregated_metrics["auprc"] = sum(auprcs) / len(auprcs)

        return aggregated_loss, aggregated_metrics

    strategy.aggregate_evaluate = custom_aggregate_evaluate
    return strategy
