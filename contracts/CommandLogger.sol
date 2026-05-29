// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title CommandLogger
 * @dev Phase 4: Decentralized Security - Immutable Audit Trail
 * 
 * This smart contract stores a hash of every gesture-command and the
 * authorized User ID, creating an immutable audit trail on the blockchain.
 * Deployed on a local Ganache testnet for development/testing.
 * 
 * Purpose: "Command Provenance" - the ability to prove exactly who issued
 * a movement command if the robot malfunctions.
 */
contract CommandLogger {

    // ========================================================================
    // Data Structures
    // ========================================================================

    struct CommandEntry {
        bytes32 commandHash;      // Keccak256 hash of the gesture-command
        string userId;            // Authorized user identifier
        string gestureLabel;      // The recognized gesture (e.g., "forward")
        string commandJson;       // Full command JSON string
        uint256 timestamp;        // Block timestamp when logged
        uint256 blockNumber;      // Block number for ordering
        uint256 sequenceNumber;   // Sequential command number
    }

    // ========================================================================
    // State Variables
    // ========================================================================

    /// @dev Owner of the contract (deployer)
    address public owner;

    /// @dev Total number of commands logged
    uint256 public totalCommands;

    /// @dev Mapping from sequence number to command entry
    mapping(uint256 => CommandEntry) public commandLog;

    /// @dev Mapping from user ID to their command count
    mapping(string => uint256) public userCommandCount;

    /// @dev Mapping from command hash to whether it exists (for verification)
    mapping(bytes32 => bool) public commandExists;

    /// @dev List of authorized user IDs
    mapping(string => bool) public authorizedUsers;

    /// @dev Emergency stop flag
    bool public emergencyStop;

    // ========================================================================
    // Events
    // ========================================================================

    /// @dev Emitted when a new command is logged
    event CommandLogged(
        uint256 indexed sequenceNumber,
        bytes32 indexed commandHash,
        string userId,
        string gestureLabel,
        uint256 timestamp
    );

    /// @dev Emitted when a user is authorized or deauthorized
    event UserAuthorizationChanged(
        string userId,
        bool authorized,
        uint256 timestamp
    );

    /// @dev Emitted on emergency stop toggle
    event EmergencyStopToggled(
        bool stopped,
        uint256 timestamp
    );

    // ========================================================================
    // Modifiers
    // ========================================================================

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier notStopped() {
        require(!emergencyStop, "System is in emergency stop");
        _;
    }

    // ========================================================================
    // Constructor
    // ========================================================================

    constructor() {
        owner = msg.sender;
        totalCommands = 0;
        emergencyStop = false;

        // Authorize the default user
        authorizedUsers["SHUBHAM_SHUKLA"] = true;
        emit UserAuthorizationChanged("SHUBHAM_SHUKLA", true, block.timestamp);
    }

    // ========================================================================
    // Core Functions
    // ========================================================================

    /**
     * @dev Log a gesture command to the immutable audit trail.
     * @param _userId The authorized user who issued the command.
     * @param _gestureLabel The recognized gesture label (e.g., "forward").
     * @param _commandJson The full command JSON string.
     * @return sequenceNumber The sequence number of the logged command.
     * @return commandHash The keccak256 hash of the command.
     */
    function logCommand(
        string memory _userId,
        string memory _gestureLabel,
        string memory _commandJson
    ) public notStopped returns (uint256 sequenceNumber, bytes32 commandHash) {
        // Verify user is authorized
        require(authorizedUsers[_userId], "User not authorized");

        // Generate command hash from all parameters
        commandHash = keccak256(
            abi.encodePacked(
                _userId,
                _gestureLabel,
                _commandJson,
                block.timestamp,
                totalCommands
            )
        );

        // Create the command entry
        sequenceNumber = totalCommands;
        commandLog[sequenceNumber] = CommandEntry({
            commandHash: commandHash,
            userId: _userId,
            gestureLabel: _gestureLabel,
            commandJson: _commandJson,
            timestamp: block.timestamp,
            blockNumber: block.number,
            sequenceNumber: sequenceNumber
        });

        // Update state
        commandExists[commandHash] = true;
        userCommandCount[_userId]++;
        totalCommands++;

        // Emit event
        emit CommandLogged(
            sequenceNumber,
            commandHash,
            _userId,
            _gestureLabel,
            block.timestamp
        );

        return (sequenceNumber, commandHash);
    }

    /**
     * @dev Verify that a command exists in the audit trail.
     * @param _commandHash The hash of the command to verify.
     * @return exists Whether the command hash exists.
     */
    function verifyCommand(bytes32 _commandHash) public view returns (bool exists) {
        return commandExists[_commandHash];
    }

    /**
     * @dev Get a command entry by its sequence number.
     * @param _sequenceNumber The sequence number of the command.
     * @return entry The full command entry.
     */
    function getCommand(uint256 _sequenceNumber) public view returns (
        bytes32 commandHash,
        string memory userId,
        string memory gestureLabel,
        string memory commandJson,
        uint256 timestamp,
        uint256 blockNumber
    ) {
        require(_sequenceNumber < totalCommands, "Command does not exist");
        CommandEntry storage entry = commandLog[_sequenceNumber];
        return (
            entry.commandHash,
            entry.userId,
            entry.gestureLabel,
            entry.commandJson,
            entry.timestamp,
            entry.blockNumber
        );
    }

    /**
     * @dev Get the total number of commands issued by a specific user.
     * @param _userId The user ID to query.
     * @return count Number of commands by this user.
     */
    function getUserCommandCount(string memory _userId) public view returns (uint256 count) {
        return userCommandCount[_userId];
    }

    // ========================================================================
    // Admin Functions
    // ========================================================================

    /**
     * @dev Authorize a new user to issue commands.
     * @param _userId The user ID to authorize.
     */
    function authorizeUser(string memory _userId) public onlyOwner {
        authorizedUsers[_userId] = true;
        emit UserAuthorizationChanged(_userId, true, block.timestamp);
    }

    /**
     * @dev Revoke a user's authorization.
     * @param _userId The user ID to deauthorize.
     */
    function deauthorizeUser(string memory _userId) public onlyOwner {
        authorizedUsers[_userId] = false;
        emit UserAuthorizationChanged(_userId, false, block.timestamp);
    }

    /**
     * @dev Toggle emergency stop. When active, no commands can be logged.
     */
    function toggleEmergencyStop() public onlyOwner {
        emergencyStop = !emergencyStop;
        emit EmergencyStopToggled(emergencyStop, block.timestamp);
    }

    /**
     * @dev Check if a user is authorized.
     * @param _userId The user ID to check.
     * @return isAuthorized Whether the user is authorized.
     */
    function isUserAuthorized(string memory _userId) public view returns (bool isAuthorized) {
        return authorizedUsers[_userId];
    }
}
