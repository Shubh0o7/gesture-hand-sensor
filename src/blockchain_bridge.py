"""
Phase 4: Blockchain Bridge - Web3.py Integration with Ganache
Connects the Python gesture recognition system to a local Ganache
Ethereum blockchain for immutable command logging.
Ensures "Command Provenance" - proving who issued each movement command.
"""

import json
import hashlib
import time
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BlockchainConfig:
    """Configuration for the blockchain connection."""
    ganache_url: str = "http://127.0.0.1:7545"  # Default Ganache GUI port
    chain_id: int = 1337  # Default Ganache chain ID
    gas_limit: int = 3000000
    gas_price_gwei: int = 20
    contract_address: Optional[str] = None
    deployer_account_index: int = 0  # Use first Ganache account
    user_id: str = "SHUBHAM_SHUKLA"


# ============================================================================
# Smart Contract ABI (compiled from CommandLogger.sol)
# ============================================================================

# This ABI is derived from the CommandLogger.sol contract
# In production, you would compile with solc and load the ABI from the artifact
COMMAND_LOGGER_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sequenceNumber", "type": "uint256"},
            {"indexed": True, "name": "commandHash", "type": "bytes32"},
            {"indexed": False, "name": "userId", "type": "string"},
            {"indexed": False, "name": "gestureLabel", "type": "string"},
            {"indexed": False, "name": "timestamp", "type": "uint256"}
        ],
        "name": "CommandLogged",
        "type": "event"
    },
    {
        "inputs": [
            {"name": "_userId", "type": "string"},
            {"name": "_gestureLabel", "type": "string"},
            {"name": "_commandJson", "type": "string"}
        ],
        "name": "logCommand",
        "outputs": [
            {"name": "sequenceNumber", "type": "uint256"},
            {"name": "commandHash", "type": "bytes32"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "_commandHash", "type": "bytes32"}],
        "name": "verifyCommand",
        "outputs": [{"name": "exists", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_sequenceNumber", "type": "uint256"}],
        "name": "getCommand",
        "outputs": [
            {"name": "commandHash", "type": "bytes32"},
            {"name": "userId", "type": "string"},
            {"name": "gestureLabel", "type": "string"},
            {"name": "commandJson", "type": "string"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "blockNumber", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_userId", "type": "string"}],
        "name": "getUserCommandCount",
        "outputs": [{"name": "count", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalCommands",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "emergencyStop",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_userId", "type": "string"}],
        "name": "isUserAuthorized",
        "outputs": [{"name": "isAuthorized", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_userId", "type": "string"}],
        "name": "authorizeUser",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "toggleEmergencyStop",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Compiled bytecode placeholder - in real usage, compile CommandLogger.sol with solc
# For demo purposes, we provide a simplified deployment mechanism
COMMAND_LOGGER_BYTECODE = None  # Set after compilation


# ============================================================================
# Blockchain Bridge
# ============================================================================

class BlockchainBridge:
    """
    Bridge between the Python gesture recognition system and the
    Ethereum blockchain (Ganache testnet).
    
    Responsibilities:
    1. Connect to local Ganache instance
    2. Deploy or connect to the CommandLogger smart contract
    3. Log each recognized gesture-command as an immutable record
    4. Verify command provenance (who issued what command)
    5. Retrieve audit trail history
    """

    def __init__(self, config: Optional[BlockchainConfig] = None):
        """
        Initialize the blockchain bridge.

        Args:
            config: Blockchain configuration. Uses defaults if None.
        """
        self.config = config or BlockchainConfig()
        self.w3: Optional[Web3] = None
        self.contract = None
        self.account = None
        self.connected = False
        self.local_log: List[Dict] = []  # Fallback local log
        self.connect()

    def is_connected(self) -> bool:
        """Return True if connected to Ganache."""
        return bool(self.connected and self.w3 and self.w3.is_connected())

    def connect(self) -> bool:
        """
        Connect to the Ganache blockchain.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.config.ganache_url))
            
            # Add POA middleware for Ganache compatibility
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            if self.w3.is_connected():
                # Get the deployer account
                accounts = self.w3.eth.accounts
                if len(accounts) > self.config.deployer_account_index:
                    self.account = accounts[self.config.deployer_account_index]
                    balance = self.w3.eth.get_balance(self.account)
                    balance_eth = self.w3.from_wei(balance, 'ether')
                    
                    print(f"Connected to Ganache at {self.config.ganache_url}")
                    print(f"Account: {self.account}")
                    print(f"Balance: {balance_eth} ETH")
                    print(f"Chain ID: {self.w3.eth.chain_id}")
                    print(f"Block Number: {self.w3.eth.block_number}")
                    
                    self.connected = True
                    return True
                else:
                    print("Error: No accounts available in Ganache.")
                    return False
            else:
                print(f"Error: Cannot connect to Ganache at {self.config.ganache_url}")
                print("Make sure Ganache is running!")
                return False

        except Exception as e:
            print(f"Connection error: {e}")
            print("Falling back to local logging mode.")
            self.connected = False
            return False

    def deploy_contract(self) -> Optional[str]:
        """
        Deploy the CommandLogger smart contract to Ganache.
        
        Note: This requires the compiled bytecode. For demo purposes,
        if bytecode is not available, it creates a mock contract interface.

        Returns:
            Contract address if deployed, None otherwise.
        """
        if not self.connected:
            print("Not connected to blockchain. Call connect() first.")
            return None

        try:
            if COMMAND_LOGGER_BYTECODE:
                # Deploy with actual bytecode
                Contract = self.w3.eth.contract(
                    abi=COMMAND_LOGGER_ABI,
                    bytecode=COMMAND_LOGGER_BYTECODE
                )
                
                tx = Contract.constructor().build_transaction({
                    'from': self.account,
                    'nonce': self.w3.eth.get_transaction_count(self.account),
                    'gas': self.config.gas_limit,
                    'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei')
                })
                
                signed_tx = self.w3.eth.account.sign_transaction(tx, private_key=None)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                self.config.contract_address = tx_receipt.contractAddress
                self.contract = self.w3.eth.contract(
                    address=self.config.contract_address,
                    abi=COMMAND_LOGGER_ABI
                )
                
                print(f"Contract deployed at: {self.config.contract_address}")
                return self.config.contract_address
            else:
                print("Note: Contract bytecode not available.")
                print("To deploy, compile CommandLogger.sol with solc and provide bytecode.")
                print("Using local logging mode as fallback.")
                return None

        except Exception as e:
            print(f"Deployment error: {e}")
            return None

    def connect_to_contract(self, contract_address: str) -> bool:
        """
        Connect to an already-deployed CommandLogger contract.

        Args:
            contract_address: The deployed contract's address.

        Returns:
            True if connection successful.
        """
        if not self.connected:
            print("Not connected to blockchain.")
            return False

        try:
            self.config.contract_address = contract_address
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=COMMAND_LOGGER_ABI
            )
            print(f"Connected to contract at: {contract_address}")
            return True
        except Exception as e:
            print(f"Error connecting to contract: {e}")
            return False

    def log_command(self, gesture_label: str, command_json: str,
                    user_id: Optional[str] = None) -> Dict:
        """
        Log a gesture command to the blockchain (or local fallback).

        This is the main function called by the gesture recognition pipeline.
        It creates an immutable record of the command for audit purposes.

        Args:
            gesture_label: The recognized gesture (e.g., "forward").
            command_json: The full command JSON string.
            user_id: The user who issued the command.

        Returns:
            Dictionary with logging result (tx_hash, sequence_number, etc.)
        """
        if user_id is None:
            user_id = self.config.user_id

        # Generate local hash for the command
        command_data = f"{user_id}:{gesture_label}:{command_json}:{time.time()}"
        local_hash = hashlib.sha256(command_data.encode()).hexdigest()

        result = {
            'success': False,
            'gesture_label': gesture_label,
            'user_id': user_id,
            'local_hash': local_hash,
            'timestamp': time.time(),
            'mode': 'unknown'
        }

        # Try blockchain logging first
        if self.connected and self.contract:
            try:
                tx = self.contract.functions.logCommand(
                    user_id,
                    gesture_label,
                    command_json
                ).build_transaction({
                    'from': self.account,
                    'nonce': self.w3.eth.get_transaction_count(self.account),
                    'gas': self.config.gas_limit,
                    'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei')
                })

                # For Ganache, we can send transactions directly (no signing needed)
                tx_hash = self.w3.eth.send_transaction({
                    'from': self.account,
                    'to': self.config.contract_address,
                    'data': tx['data'],
                    'gas': self.config.gas_limit,
                    'gasPrice': self.w3.to_wei(self.config.gas_price_gwei, 'gwei')
                })

                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                result.update({
                    'success': True,
                    'mode': 'blockchain',
                    'tx_hash': tx_hash.hex(),
                    'block_number': tx_receipt.blockNumber,
                    'gas_used': tx_receipt.gasUsed
                })

                print(f"  [Blockchain] Command logged: {gesture_label} | "
                      f"TX: {tx_hash.hex()[:16]}... | Block: {tx_receipt.blockNumber}")

            except Exception as e:
                print(f"  [Blockchain] Error: {e}")
                # Fall through to local logging

        # Fallback: Local logging
        if not result['success']:
            local_entry = {
                'sequence_number': len(self.local_log),
                'user_id': user_id,
                'gesture_label': gesture_label,
                'command_json': command_json,
                'local_hash': local_hash,
                'timestamp': time.time()
            }
            self.local_log.append(local_entry)

            result.update({
                'success': True,
                'mode': 'local',
                'sequence_number': local_entry['sequence_number']
            })

            print(f"  [Local Log] Command logged: {gesture_label} | "
                  f"Seq: {local_entry['sequence_number']} | Hash: {local_hash[:16]}...")

        return result

    def verify_command(self, command_hash: str) -> bool:
        """
        Verify that a command exists in the audit trail.

        Args:
            command_hash: The hash of the command to verify.

        Returns:
            True if the command exists.
        """
        if self.connected and self.contract:
            try:
                hash_bytes = bytes.fromhex(command_hash.replace('0x', ''))
                exists = self.contract.functions.verifyCommand(hash_bytes).call()
                return exists
            except Exception as e:
                print(f"Verification error: {e}")

        # Check local log
        for entry in self.local_log:
            if entry['local_hash'] == command_hash:
                return True
        return False

    def get_command(self, sequence_number: int) -> Optional[Dict]:
        """
        Retrieve a command entry by sequence number.

        Args:
            sequence_number: The command's sequence number.

        Returns:
            Command entry dictionary or None.
        """
        if self.connected and self.contract:
            try:
                result = self.contract.functions.getCommand(sequence_number).call()
                return {
                    'command_hash': result[0].hex(),
                    'user_id': result[1],
                    'gesture_label': result[2],
                    'command_json': result[3],
                    'timestamp': result[4],
                    'block_number': result[5]
                }
            except Exception as e:
                print(f"Retrieval error: {e}")

        # Check local log
        if 0 <= sequence_number < len(self.local_log):
            return self.local_log[sequence_number]
        return None

    def get_total_commands(self) -> int:
        """
        Get the total number of logged commands.

        Returns:
            Total command count.
        """
        if self.connected and self.contract:
            try:
                return self.contract.functions.totalCommands().call()
            except Exception:
                pass
        return len(self.local_log)

    def get_audit_trail(self, last_n: int = 20) -> List[Dict]:
        """
        Get the recent audit trail.

        Args:
            last_n: Number of recent entries to retrieve.

        Returns:
            List of command entry dictionaries.
        """
        total = self.get_total_commands()
        start = max(0, total - last_n)
        trail = []

        for i in range(start, total):
            entry = self.get_command(i)
            if entry:
                trail.append(entry)

        return trail

    def save_local_log(self, filepath: str = '../data/command_log.json'):
        """
        Save the local command log to a JSON file.

        Args:
            filepath: Path to save the log file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'total_commands': len(self.local_log),
                'user_id': self.config.user_id,
                'exported_at': time.time(),
                'commands': self.local_log
            }, f, indent=2)
        print(f"Local log saved to {filepath} ({len(self.local_log)} entries)")

    def get_status(self) -> Dict:
        """
        Get the current status of the blockchain bridge.

        Returns:
            Status dictionary.
        """
        status = {
            'connected': self.connected,
            'mode': 'blockchain' if (self.connected and self.contract) else 'local',
            'ganache_url': self.config.ganache_url,
            'contract_address': self.config.contract_address,
            'account': self.account,
            'total_commands': self.get_total_commands(),
            'local_log_size': len(self.local_log)
        }

        if self.connected:
            try:
                status['block_number'] = self.w3.eth.block_number
                status['chain_id'] = self.w3.eth.chain_id
                if self.account:
                    balance = self.w3.eth.get_balance(self.account)
                    status['balance_eth'] = float(self.w3.from_wei(balance, 'ether'))
            except Exception:
                pass

        return status


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demonstrate the blockchain bridge functionality."""
    print("=" * 60)
    print("  Phase 4: Blockchain Bridge - Demo")
    print("=" * 60)

    bridge = BlockchainBridge()

    # Try to connect to Ganache
    print("\n--- Connecting to Ganache ---")
    connected = bridge.connect()

    if not connected:
        print("\nGanache not running. Using local logging mode.")
        print("To use blockchain mode:")
        print("  1. Install Ganache: https://trufflesuite.com/ganache/")
        print("  2. Start Ganache on port 7545")
        print("  3. Run this script again")

    # Log some test commands
    print("\n--- Logging Test Commands ---")
    test_commands = [
        ('forward', '{"action": "move", "speed": 0.5, "direction": "forward"}'),
        ('turn_left', '{"action": "turn", "speed": 0.3, "direction": "left", "angle": 45}'),
        ('fast', '{"action": "adjust_speed", "speed": 0.8}'),
        ('forward', '{"action": "move", "speed": 0.8, "direction": "forward"}'),
        ('stop', '{"action": "stop", "speed": 0.0}'),
    ]

    for gesture, command_json in test_commands:
        result = bridge.log_command(gesture, command_json)
        time.sleep(0.1)

    # Show audit trail
    print("\n--- Audit Trail ---")
    trail = bridge.get_audit_trail(last_n=10)
    for entry in trail:
        seq = entry.get('sequence_number', 'N/A')
        user = entry.get('user_id', 'N/A')
        gesture = entry.get('gesture_label', 'N/A')
        hash_val = entry.get('local_hash', entry.get('command_hash', 'N/A'))
        if isinstance(hash_val, str):
            hash_val = hash_val[:16]
        print(f"  Seq {seq}: [{user}] {gesture} | Hash: {hash_val}...")

    # Show status
    print("\n--- Bridge Status ---")
    status = bridge.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Save local log
    bridge.save_local_log('../data/command_log.json')

    print("\nPhase 4 demo complete!")


if __name__ == '__main__':
    demo()
