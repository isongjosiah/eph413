from flwr.server.strategy import FedAvg, FedProx
from typing import List, Tuple, Union, Dict, Optional
from flwr.common import (
    EvaluateRes,
    FitRes,
    Metrics,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy

class CustomFedAvg(FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to get aggregated loss
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        # Aggregate custom metrics
        aurocs = [r.metrics["auroc"] for _, r in results]
        auprcs = [r.metrics["auprc"] for _, r in results]
        
        return aggregated_loss, {"auroc": sum(aurocs) / len(aurocs), "auprc": sum(auprcs) / len(auprcs)}

class CustomFedProx(FedProx, CustomFedAvg):
    pass
