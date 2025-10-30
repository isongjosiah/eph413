// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title AuditTrail
 * @notice A simple smart contract to log essential data from each federated
 * learning round, providing an immutable audit trail for XS-FedPRS.
 * It logs data described in Equations 8 and 9 of the manuscript.
 */
contract AuditTrail {
    
    // The authorized server (aggregator) that is allowed to log data
    address public authorizedServer;

    /**
     * @dev Defines the data structure for each round's record.
     * This corresponds to the aggregation record (A^(t)) [cite: 181] and
     * references the client update hashes (h_k^(t))[cite: 180].
     */
    struct RoundData {
        uint256 roundNumber;
        string globalModelCID;      // IPFS CID of the aggregated global model (off-chain weights)
        bytes32[] clientUpdateHashes; // Array of SHA-256 hashes (Eq. 8)
        bytes aggregationBitmap;    // Bitmap of clients included in aggregation (Eq. 9)
        uint256 timestamp;            // Timestamp of the block
    }

    // Mapping from the round number to the logged data
    mapping(uint256 => RoundData) public trainingRounds;

    // Event to emit when a new record is added
    event RoundRecorded(uint256 indexed roundNumber, string globalModelCID, uint256 numClientHashes);

    /**
     * @dev Restricts function access to the authorized server.
     */
    modifier onlyServer() {
        require(msg.sender == authorizedServer, "CALLER_NOT_AUTHORIZED_SERVER");
        _;
    }

    /**
     * @dev Sets the authorized server address upon deployment.
     */
    constructor(address _serverAddress) {
        authorizedServer = _serverAddress;
    }

    /**
     * @dev The server calls this function to log the data for a completed round.
     */
    function addRoundRecord(
        uint256 _roundNumber,
        string memory _globalModelCID,
        bytes32[] memory _clientUpdateHashes,
        bytes memory _aggregationBitmap
    ) public onlyServer {
        // Ensure this round hasn't been recorded already
        require(trainingRounds[_roundNumber].timestamp == 0, "ROUND_ALREADY_RECORDED");

        // Store the data
        trainingRounds[_roundNumber] = RoundData({
            roundNumber: _roundNumber,
            globalModelCID: _globalModelCID,
            clientUpdateHashes: _clientUpdateHashes,
            aggregationBitmap: _aggregationBitmap,
            timestamp: block.timestamp
        });

        emit RoundRecorded(_roundNumber, _globalModelCID, _clientUpdateHashes.length);
    }

    /**
     * @dev Public view function to retrieve the data for a specific round.
     */
    function getRoundData(uint256 _roundNumber) public view returns (RoundData memory) {
        return trainingRounds[_roundNumber];
    }

    /**
     * @dev Allows the owner to update the authorized server address.
     */
    function setAuthorizedServer(address _newServerAddress) public onlyServer {
        authorizedServer = _newServerAddress;
    }
}
