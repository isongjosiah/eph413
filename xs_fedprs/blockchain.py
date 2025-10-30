import logging
import json
import io
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

import ipfs_http_client
import numpy as np
import yaml  # Using yaml for config loading

# Import purechainlib
try:
    from purechainlib import PureChain
except ImportError:
    print(
        "Error: purechainlib not found. Please install it using 'pip install purechainlib'"
    )
    PureChain = None
    Account = None

# Configure logging
logger = logging.getLogger(__name__)


# Helper to load config
def _load_blockchain_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            return config.get("blockchain", {})
    except Exception as e:
        logger.error(f"Error loading config file: {e}. Using defaults.")
        return {}


class BlockchainLogger:
    """
    Handles logging of training artifacts to IPFS (off-chain) and
    recording their provenance on PureChain using purechainlib (on-chain).

    This class handles serialization, hashing, IPFS upload, and the
    final on-chain transaction.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes clients for PureChain (purechainlib) and IPFS based on config.
        """
        config = _load_blockchain_config(config_path)

        self.chain = None
        self.contract = None
        self.server_account = None
        self.ipfs_client = None

        # --- Initialize IPFS Client ---
        try:
            ipfs_addr = config.get("ipfs_api_addr", "/ip4/127.0.0.1/tcp/5001")
            self.ipfs_client = ipfs_http_client.connect(ipfs_addr)
            logger.info(f"Connected to IPFS node at {ipfs_addr}")
        except Exception as e:
            logger.error(f"Failed to connect to IPFS at {ipfs_addr}: {e}")
            self.ipfs_client = None

        # --- Initialize PureChain Client (using purechainlib) ---
        if PureChain and Account:  # Check if imports succeeded
            try:
                node_url = config.get("purechain_node_url", "http://127.0.0.1:8545")
                self.chain = PureChain("testnet")
                self.chain.connect("0x795b0F05cA107dD038c01Ce54F04034Be3B87949")

                # --- Load Server Account ---
                private_key = config.get("server_private_key")
                if not private_key:
                    raise ValueError("server_private_key not found in config.")
                self.server_account = Account.from_key(private_key)
                logger.info(f"Loaded server account: {self.server_account.address}")

                # --- Load Smart Contract ---
                contract_address = config.get("contract_address")
                abi_path = config.get("contract_abi_path")
                if not contract_address or not abi_path:
                    raise ValueError("contract_address or contract_abi_path not found.")

                abi_file = Path(abi_path)
                if not abi_file.exists():
                    raise FileNotFoundError(
                        f"Contract ABI file not found at {abi_path}"
                    )

                with open(abi_file, "r") as f:
                    contract_abi = json.load(f)

                checksum_address = self.chain.to_checksum_address(contract_address)
                self.contract = self.chain.load_contract(
                    address=checksum_address, abi=contract_abi
                )
                logger.info(f"Loaded AuditTrail smart contract at {contract_address}")

            except Exception as e:
                logger.error(f"Failed to initialize PureChain/Contract: {e}")
                self.chain = None
                self.contract = None
        else:
            logger.error("purechainlib initialization failed. Library not found.")

    def _serialize_and_upload_model(
        self, model_ndarrays: List[np.ndarray]
    ) -> Optional[str]:
        """Serializes model weights and uploads them to IPFS."""
        if self.ipfs_client is None:
            logger.error("IPFS client not available. Cannot upload model.")
            return None

        try:
            # Serialize numpy arrays into a buffer
            with io.BytesIO() as buffer:
                np.save(
                    buffer, np.array(model_ndarrays, dtype=object), allow_pickle=True
                )
                buffer.seek(0)

                # Upload the buffer's content
                ipfs_result = self.ipfs_client.add(buffer)
                cid = ipfs_result["Hash"]
                logger.info(f"Global model uploaded to IPFS. CID: {cid}")
                return cid
        except Exception as e:
            logger.error(f"IPFS upload failed: {e}")
            return None

    def _hash_client_update(
        self, update_data: Dict[str, Any], round_number: int
    ) -> bytes:
        """
        Generates the client update hash (h_k^(t)) as per Equation 8.
        h_k^(t) = SHA-256(SHA-256(Δw_k^(t)) | τ_k^(t) | V_k^(t) | t)

        Args:
            update_data: A dictionary containing a client's results,
                         e.g., {'params_nd': ..., 'trust_score': ..., 'variant_set': ...}
            round_number: The current round number.

        Returns:
            32-byte SHA-256 hash.
        """
        # 1. Hash the model update (Δw_k^(t))
        update_bytes = b"".join([arr.tobytes() for arr in update_data["params_nd"]])
        update_hash = hashlib.sha256(update_bytes).hexdigest()

        # 2. Get other components
        trust_score_str = f"{update_data['trust_score']:.6f}"
        variant_set_str = ",".join(
            sorted([str(v) for v in update_data.get("variant_set", set())])
        )
        round_str = str(round_number)

        # 3. Combine components and hash
        combined_str = f"{update_hash}|{trust_score_str}|{variant_set_str}|{round_str}"

        return hashlib.sha256(combined_str.encode()).digest()

    def log_round_data(
        self,
        round_number: int,
        client_updates: List[Dict[str, Any]],  # List of dicts from server strategy
        global_model_ndarrays: List[np.ndarray],
        trusted_client_cids: List[str],  # Used for the bitmap
    ) -> Optional[str]:
        """
        Logs all data for a completed federated round.

        1. Uploads global model to IPFS.
        2. Gathers hashes and records on-chain.

        Args:
            round_number: The current round number.
            client_updates: List of client update dicts (containing params, trust, etc.).
            global_model_ndarrays: The aggregated global model weights.
            trusted_client_cids: List of CIDs that were included in aggregation.

        Returns:
            The on-chain transaction hash, or None if failed.
        """
        if self.contract is None or self.chain is None or self.server_account is None:
            logger.error(
                "BlockchainLogger not initialized properly. Cannot log round data."
            )
            return None

        # --- a. Upload global model to IPFS ---
        global_model_cid = self._serialize_and_upload_model(global_model_ndarrays)
        if global_model_cid is None:
            logger.error(
                f"Round {round_number}: Failed to upload model to IPFS. Aborting log."
            )
            return None

        # --- b. Construct on-chain payload ---

        # (Eq. 8) Calculate hashes for all client updates
        client_update_hashes_bytes = [
            self._hash_client_update(upd, round_number) for upd in client_updates
        ]

        # (Eq. 9) Create aggregation bitmap
        all_received_cids = [upd["cid"] for upd in client_updates]
        bitmap_list = [
            1 if cid in trusted_client_cids else 0 for cid in all_received_cids
        ]

        if not bitmap_list:
            aggregation_bitmap_bytes = b""
        else:
            aggregation_bitmap_bytes = int("".join(map(str, bitmap_list)), 2).to_bytes(
                (len(bitmap_list) + 7) // 8, byteorder="big"
            )

        # --- c. Call Smart Contract using purechainlib ---
        try:
            logger.info(f"Submitting on-chain record for round {round_number}...")

            tx_options = {
                "from": self.server_account.address,
                "nonce": self.chain.get_transaction_count(self.server_account.address),
                "gas": 2000000,
                "gasPrice": self.chain.gas_price,
            }

            tx_hash = self.contract.functions.addRoundRecord(
                round_number,
                global_model_cid,
                client_update_hashes_bytes,
                aggregation_bitmap_bytes,
            ).transact(tx_options, private_key=self.server_account.key)

            tx_hash_hex = tx_hash.hex()
            logger.info(f"Transaction submitted. Hash: {tx_hash_hex}")

            tx_receipt = self.chain.wait_for_transaction_receipt(tx_hash)

            if tx_receipt.status == 1:
                logger.info(
                    f"Round {round_number} successfully recorded on-chain. Block: {tx_receipt.blockNumber}"
                )
            else:
                logger.error(
                    f"Round {round_number} on-chain transaction FAILED. Receipt: {tx_receipt}"
                )

            return tx_hash_hex

        except Exception as e:
            logger.exception(
                f"Error submitting on-chain transaction via purechainlib: {e}"
            )
            return None


# --- Example Usage (Conceptual) ---
if __name__ == "__main__":
    logger.info("Conceptual test of BlockchainLogger with full logic...")

    # 1. Create a config.yaml file with:
    # blockchain:
    #   ipfs_api_addr: "/ip4/127.0.0.1/tcp/5001"
    #   purechain_node_url: "http://127.0.0.1:8545" # URL of your PureChain node
    #   server_private_key: "0x..." # Private key of your server's wallet
    #   contract_address: "0x..." # Address of deployed AuditTrail.sol
    #   contract_abi_path: "AuditTrail.json" # Path to the compiled contract ABI

    # 2. Compile and deploy AuditTrail.sol
    #    - Get the contract_address and ABI (save as AuditTrail.json)

    # 3. Fund the server_account address on your PureChain.

    # 4. Run this example:

    # logger_instance = BlockchainLogger(config_path="config.yaml")

    # if logger_instance.chain:
    #     # --- Create Dummy Data ---
    #     dummy_model = [np.random.rand(10, 5), np.random.rand(5)]
    #     dummy_updates = [
    #         {
    #             'cid': 'client_1',
    #             'params_nd': [np.random.rand(10, 5), np.random.rand(5)], # Raw parameters
    #             'trust_score': 1.0,
    #             'variant_set': {'rs123', 'rs456'}
    #         },
    #         {
    #             'cid': 'client_2',
    #             'params_nd': [np.random.rand(10, 5), np.random.rand(5)], # Raw parameters
    #             'trust_score': 0.8,
    #             'variant_set': {'rs123', 'rs789'}
    #         }
    #     ]
    #     dummy_trusted_cids = ['client_1', 'client_2']
    #     round_num = 1

    #     # --- Test Logging ---
    #     tx_hash = logger_instance.log_round_data(
    #         round_num,
    #         dummy_updates,
    #         dummy_model,
    #         dummy_trusted_cids
    #     )

    #     if tx_hash:
    #         print(f"Test log successful. Transaction Hash: {tx_hash}")
    #     else:
    #         print("Test log failed.")
    # else:
    #     print("BlockchainLogger initialization failed. Check config and connections.")
    pass
