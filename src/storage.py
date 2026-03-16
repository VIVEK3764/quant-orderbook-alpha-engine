import os
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Dict, Any
from src.config import STORAGE

class ParquetStorage:
    """"""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _get_partition_path(self, timestamp_ms: int) -> str:
        """Determines the file path based on timestamp for partitioning by date/hour."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        date_str = dt.strftime("%Y-%m-%d")
        hour_str = dt.strftime("%H")
        
        partition_dir = os.path.join(self.base_dir, date_str)
        os.makedirs(partition_dir, exist_ok=True)
        
        return os.path.join(partition_dir, f"{hour_str}.parquet")

    def append_data(self, records: List[Dict[str, Any]], schema: pa.Schema):
        if not records:
            return
            
        # Group records by partition (hour)
        partitions: Dict[str, List[Dict[str, Any]]] = {}
        for row in records:
            # Assumes every record has a 'timestamp' in milliseconds
            ts = row.get('timestamp')
            if ts is None:
                continue
            path = self._get_partition_path(ts)
            partitions.setdefault(path, []).append(row)
            
        for path, part_records in partitions.items():
            df = pd.DataFrame(part_records)
            table = pa.Table.from_pandas(df, schema=schema)
            
            # If the file exists, append; otherwise, write a new file.
            # PyArrow's write_to_dataset handles partitioning, but since we define the exact file, 
            # we can use ParquetWriter to append if file exists in pandas/pyarrow ecosystem.
            # However, standard parquet appending is not trivially supported without rewriting or chunking.
            # A common workaround for "append" in parquet is to use a dataset approach or unique filenames.
            # For this simple implementation, we will uniquely name files by timestamp or write chunks if we want true appends in the file.
            # Standard practice: write a new file per chunk in the directory instead of appending to single file.
            
            # For the scope of this project, we write chunked files if we want to avoid reading/rewriting.
            # Let's adjust the path to include a unique timestamp for the chunk.
            first_ts = part_records[0]['timestamp']
            chunk_path = path.replace('.parquet', f'_{first_ts}.parquet')
            
            pq.write_table(table, chunk_path)


class OrderbookStorage(ParquetStorage):
    def __init__(self):
        super().__init__(STORAGE.RAW_ORDERBOOK_DIR)
        self.schema = pa.schema([
            ('timestamp', pa.int64()),
            ('bid_price', pa.float64()),
            ('bid_size', pa.float64()),
            ('ask_price', pa.float64()),
            ('ask_size', pa.float64())
        ])
        
    def save(self, records: List[Dict[str, Any]]):
        self.append_data(records, self.schema)


class TradesStorage(ParquetStorage):
    def __init__(self):
        super().__init__(STORAGE.RAW_TRADES_DIR)
        self.schema = pa.schema([
            ('timestamp', pa.int64()),
            ('price', pa.float64()),
            ('quantity', pa.float64()),
            ('side', pa.string())
        ])
        
    def save(self, records: List[Dict[str, Any]]):
        self.append_data(records, self.schema)
