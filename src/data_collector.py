import asyncio
import json
import websockets
import logging
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BINANCE, INGESTION
from src.storage import OrderbookStorage, TradesStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataCollector")


class DataCollector:
    def __init__(self):
        self.ws_url = BINANCE.WS_ENDPOINT
        self.symbols = [s.lower() for s in BINANCE.SYMBOLS]
        
        self.orderbook_storage = OrderbookStorage()
        self.trades_storage = TradesStorage()
        
        self.orderbook_buffer: List[Dict[str, Any]] = []
        self.trades_buffer: List[Dict[str, Any]] = []
        
        self.is_running = False

    def get_stream_url(self) -> str:
        """Constructs the combined stream URL for Binance."""
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@depth10@100ms")  # 10 levels is usually enough for top-of-book
            streams.append(f"{symbol}@trade")
        streams_str = "/".join(streams)
        base_url = self.ws_url.replace("/ws", "/stream")
        return f"{base_url}?streams={streams_str}"

    async def flush_buffers_periodically(self):
        """Task to periodically write buffered data to parquet."""
        while self.is_running:
            await asyncio.sleep(INGESTION.FLUSH_INTERVAL_SEC)
            self._flush_orderbook()
            self._flush_trades()
            
    def _flush_orderbook(self):
        if not self.orderbook_buffer:
            return
        logger.info(f"Flushing {len(self.orderbook_buffer)} orderbook records to storage.")
        # Copy to avoid mutation during save
        records_to_save = self.orderbook_buffer.copy()
        self.orderbook_buffer.clear()
        self.orderbook_storage.save(records_to_save)

    def _flush_trades(self):
        if not self.trades_buffer:
            return
        logger.info(f"Flushing {len(self.trades_buffer)} trade records to storage.")
        records_to_save = self.trades_buffer.copy()
        self.trades_buffer.clear()
        self.trades_storage.save(records_to_save)

    def process_message(self, message: str):
        data = json.loads(message)
        
        # Combined stream has { "stream": "...", "data": {...} }
        if 'stream' not in data or 'data' not in data:
            return
            
        stream_name: str = data['stream']
        payload: dict = data['data']
        
        if '@depth' in stream_name:
            self._parse_orderbook(payload)
        elif '@trade' in stream_name:
            self._parse_trade(payload)

    def _parse_orderbook(self, payload: dict):
        """
        Extract top of book (level 0) for feature engineering.
        """
        try:
            # Binance partial book depth streams do not include 'E' (Event time) in the payload by default
            # unless we use the update streams. We can use local timestamp as fallback or track it.
            import time
            timestamp = payload.get('E', int(time.time() * 1000)) 
            bids = payload.get('bids', payload.get('b', []))
            asks = payload.get('asks', payload.get('a', []))
            
            if not bids or not asks:
                return
                
            best_bid_price, best_bid_size = float(bids[0][0]), float(bids[0][1])
            best_ask_price, best_ask_size = float(asks[0][0]), float(asks[0][1])
            
            record = {
                'timestamp': timestamp,
                'bid_price': best_bid_price,
                'bid_size': best_bid_size,
                'ask_price': best_ask_price,
                'ask_size': best_ask_size
            }
            self.orderbook_buffer.append(record)
            
            if len(self.orderbook_buffer) >= INGESTION.ORDERBOOK_BUFFER_SIZE:
                self._flush_orderbook()
                
        except Exception as e:
            logger.error(f"Error parsing orderbook: {e}")

    def _parse_trade(self, payload: dict):
        """Parse raw trade stream."""
        try:
            timestamp = payload.get('E')
            price = float(payload.get('p', 0))
            quantity = float(payload.get('q', 0))
            is_buyer_maker = payload.get('m', False) # If true, trade was a sell (buyer was maker)
            
            side = 'SELL' if is_buyer_maker else 'BUY'
            
            record = {
                'timestamp': timestamp,
                'price': price,
                'quantity': quantity,
                'side': side
            }
            self.trades_buffer.append(record)
            
            if len(self.trades_buffer) >= INGESTION.TRADES_BUFFER_SIZE:
                self._flush_trades()
                
        except Exception as e:
            logger.error(f"Error parsing trade: {e}")

    async def run(self):
        """Main connection loop with auto-reconnect."""
        self.is_running = True
        url = self.get_stream_url()
        
        asyncio.create_task(self.flush_buffers_periodically())
        
        while self.is_running:
            try:
                logger.info(f"Connecting to Binance WebSocket: {url}")
                async with websockets.connect(url) as ws:
                    logger.info("Connected.")
                    while self.is_running:
                        message = await ws.recv()
                        self.process_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
                
    def stop(self):
        self.is_running = False
        self._flush_orderbook()
        self._flush_trades()

if __name__ == "__main__":
    collector = DataCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        logger.info("Stopping data collection...")
        collector.stop()
