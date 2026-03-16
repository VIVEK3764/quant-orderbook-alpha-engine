import os
from dataclasses import dataclass, field
from typing import List

# Base directory for the project to ensure absolute paths resolve correctly if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class BinanceConfig:
    WS_ENDPOINT: str = "wss://stream.binance.com:9443/ws"
    SYMBOLS: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    
@dataclass
class StorageConfig:
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    RAW_ORDERBOOK_DIR: str = os.path.join(DATA_DIR, "raw", "orderbook")
    RAW_TRADES_DIR: str = os.path.join(DATA_DIR, "raw", "trades")
    PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
    
@dataclass
class IngestionConfig:
    ORDERBOOK_BUFFER_SIZE: int = 1000
    TRADES_BUFFER_SIZE: int = 1000
    FLUSH_INTERVAL_SEC: float = 5.0
    
@dataclass
class FeatureConfig:
    WINDOW_SIZES: List[int] = field(default_factory=lambda: [10, 50, 100])
    
@dataclass
class ModelConfig:
    PREDICT_STEPS: int = 10 # N steps forward
    TEST_SPLIT: float = 0.2

BINANCE = BinanceConfig()
STORAGE = StorageConfig()
INGESTION = IngestionConfig()
FEATURES = FeatureConfig()
MODEL = ModelConfig()

def get_stream_names() -> List[str]:
    """Generates the required Binance stream names based on configured symbols."""
    streams = []
    for symbol in BINANCE.SYMBOLS:
        symbol_lower = symbol.lower()
        # We can either subscribe to individual streams or a combined stream.
        # depth20@100ms provides top 20 levels every 100ms
        # trade provides real-time trades
        streams.append(f"{symbol_lower}@depth20@100ms") 
        streams.append(f"{symbol_lower}@trade")          
    return streams
