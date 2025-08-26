"""
Enterprise Data Pipeline for Safe RL Production System.

This module provides comprehensive data management capabilities including:
- ETL pipeline for training data
- Real-time data streaming
- Data validation and quality checks
- Data versioning and lineage tracking
- GDPR compliance and data privacy
"""

import asyncio
import logging
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import redis
import boto3
from kafka import KafkaProducer, KafkaConsumer
import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite
from cryptography.fernet import Fernet
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

logger = logging.getLogger(__name__)

Base = declarative_base()


@dataclass
class DataRecord:
    """Represents a single data record with metadata."""
    id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    quality_score: Optional[float] = None
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            'id': self.id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'version': self.version,
            'metadata': self.metadata,
            'schema_version': self.schema_version,
            'quality_score': self.quality_score,
            'is_valid': self.is_valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataRecord':
        """Create record from dictionary."""
        return cls(
            id=data['id'],
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data.get('source', ''),
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {}),
            schema_version=data.get('schema_version', '1.0'),
            quality_score=data.get('quality_score'),
            is_valid=data.get('is_valid', True)
        )


@dataclass 
class DataPipelineConfig:
    """Configuration for data pipeline."""
    # Source configuration
    source_type: str  # 'database', 'kafka', 's3', 'file'
    source_config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing configuration
    batch_size: int = 1000
    processing_workers: int = 4
    quality_checks_enabled: bool = True
    encryption_enabled: bool = True
    
    # Storage configuration
    storage_type: str = 'database'  # 'database', 's3', 'local'
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming configuration
    streaming_enabled: bool = False
    streaming_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data retention
    retention_days: int = 365
    archival_enabled: bool = True
    
    # Privacy and compliance
    anonymization_enabled: bool = True
    gdpr_compliance: bool = True
    audit_logging: bool = True


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
    
    @abstractmethod
    async def connect(self):
        """Connect to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read a batch of data records."""
        pass
    
    @abstractmethod
    async def read_stream(self) -> AsyncIterator[DataRecord]:
        """Read data as a stream."""
        pass


class DatabaseDataSource(DataSource):
    """Database data source implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = None
        self.session_factory = None
        self.last_processed_id = 0
    
    async def connect(self):
        """Connect to database."""
        try:
            database_url = self.config['url']
            self.engine = sa.create_engine(database_url, pool_pre_ping=True)
            self.session_factory = sessionmaker(bind=self.engine)
            self.is_connected = True
            logger.info("Connected to database data source")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from database."""
        if self.engine:
            self.engine.dispose()
        self.is_connected = False
        logger.info("Disconnected from database")
    
    async def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read batch from database."""
        if not self.is_connected:
            await self.connect()
        
        session = self.session_factory()
        try:
            # Query for new records
            query = session.execute(
                sa.text("""
                SELECT id, data, created_at, source, metadata 
                FROM training_data 
                WHERE id > :last_id 
                ORDER BY id 
                LIMIT :batch_size
                """),
                {'last_id': self.last_processed_id, 'batch_size': batch_size}
            )
            
            records = []
            for row in query:
                record = DataRecord(
                    id=str(row.id),
                    data=json.loads(row.data) if isinstance(row.data, str) else row.data,
                    timestamp=row.created_at,
                    source=row.source or 'database',
                    metadata=json.loads(row.metadata) if isinstance(row.metadata, str) else (row.metadata or {})
                )
                records.append(record)
                self.last_processed_id = max(self.last_processed_id, row.id)
            
            return records
            
        except Exception as e:
            logger.error(f"Failed to read batch from database: {e}")
            raise
        finally:
            session.close()
    
    async def read_stream(self) -> AsyncIterator[DataRecord]:
        """Stream data from database."""
        while True:
            try:
                batch = await self.read_batch(100)
                if not batch:
                    await asyncio.sleep(1)  # Wait before checking again
                    continue
                
                for record in batch:
                    yield record
                    
            except Exception as e:
                logger.error(f"Error in database stream: {e}")
                await asyncio.sleep(5)


class KafkaDataSource(DataSource):
    """Kafka data source implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consumer = None
        self.producer = None
    
    async def connect(self):
        """Connect to Kafka."""
        try:
            kafka_config = {
                'bootstrap_servers': self.config['bootstrap_servers'],
                'auto_offset_reset': 'latest',
                'group_id': self.config.get('group_id', 'saferl-data-pipeline'),
                'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
                'security_protocol': self.config.get('security_protocol', 'PLAINTEXT')
            }
            
            self.consumer = KafkaConsumer(
                self.config['topic'],
                **kafka_config
            )
            
            self.is_connected = True
            logger.info("Connected to Kafka data source")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Kafka."""
        if self.consumer:
            self.consumer.close()
        self.is_connected = False
        logger.info("Disconnected from Kafka")
    
    async def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read batch from Kafka."""
        if not self.is_connected:
            await self.connect()
        
        records = []
        messages = self.consumer.poll(timeout_ms=5000, max_records=batch_size)
        
        for topic_partition, msgs in messages.items():
            for msg in msgs:
                record = DataRecord(
                    id=f"{msg.topic}_{msg.partition}_{msg.offset}",
                    data=msg.value,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                    source='kafka',
                    metadata={
                        'topic': msg.topic,
                        'partition': msg.partition,
                        'offset': msg.offset
                    }
                )
                records.append(record)
        
        return records
    
    async def read_stream(self) -> AsyncIterator[DataRecord]:
        """Stream data from Kafka."""
        if not self.is_connected:
            await self.connect()
        
        for msg in self.consumer:
            record = DataRecord(
                id=f"{msg.topic}_{msg.partition}_{msg.offset}",
                data=msg.value,
                timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                source='kafka',
                metadata={
                    'topic': msg.topic,
                    'partition': msg.partition,
                    'offset': msg.offset
                }
            )
            yield record


class S3DataSource(DataSource):
    """S3 data source implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.s3_client = None
        self.bucket = config['bucket']
        self.prefix = config.get('prefix', '')
        self.processed_files = set()
    
    async def connect(self):
        """Connect to S3."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('access_key_id'),
                aws_secret_access_key=self.config.get('secret_access_key'),
                region_name=self.config.get('region', 'us-west-2')
            )
            self.is_connected = True
            logger.info("Connected to S3 data source")
            
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from S3."""
        self.s3_client = None
        self.is_connected = False
        logger.info("Disconnected from S3")
    
    async def read_batch(self, batch_size: int) -> List[DataRecord]:
        """Read batch from S3."""
        if not self.is_connected:
            await self.connect()
        
        # List new files
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.prefix,
            MaxKeys=batch_size
        )
        
        records = []
        for obj in response.get('Contents', []):
            file_key = obj['Key']
            if file_key in self.processed_files:
                continue
            
            # Read file content
            try:
                obj_response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
                content = obj_response['Body'].read()
                
                # Parse based on file extension
                if file_key.endswith('.json'):
                    data = json.loads(content.decode('utf-8'))
                elif file_key.endswith('.parquet'):
                    # Handle parquet files
                    import io
                    data = pd.read_parquet(io.BytesIO(content)).to_dict('records')
                else:
                    data = {'raw_content': content.decode('utf-8')}
                
                record = DataRecord(
                    id=file_key,
                    data=data,
                    timestamp=obj['LastModified'],
                    source='s3',
                    metadata={
                        'bucket': self.bucket,
                        'key': file_key,
                        'size': obj['Size'],
                        'etag': obj['ETag']
                    }
                )
                records.append(record)
                self.processed_files.add(file_key)
                
            except Exception as e:
                logger.error(f"Failed to process S3 file {file_key}: {e}")
        
        return records
    
    async def read_stream(self) -> AsyncIterator[DataRecord]:
        """Stream data from S3."""
        while True:
            batch = await self.read_batch(10)
            if not batch:
                await asyncio.sleep(30)  # Check for new files every 30 seconds
                continue
            
            for record in batch:
                yield record


class DataValidator:
    """Data quality validation and checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.expectation_suites = {}
        self.quality_thresholds = config.get('quality_thresholds', {})
    
    def add_expectation_suite(self, name: str, suite: ExpectationSuite):
        """Add an expectation suite for data validation."""
        self.expectation_suites[name] = suite
        logger.info(f"Added expectation suite: {name}")
    
    async def validate_record(self, record: DataRecord) -> bool:
        """Validate a single data record."""
        try:
            # Basic validation
            if not record.data:
                record.is_valid = False
                return False
            
            # Schema validation
            if not self._validate_schema(record):
                record.is_valid = False
                return False
            
            # Data quality checks using Great Expectations
            if self.expectation_suites:
                quality_score = await self._run_expectations(record)
                record.quality_score = quality_score
                
                min_quality = self.quality_thresholds.get('min_quality_score', 0.8)
                if quality_score < min_quality:
                    record.is_valid = False
                    return False
            
            record.is_valid = True
            return True
            
        except Exception as e:
            logger.error(f"Validation error for record {record.id}: {e}")
            record.is_valid = False
            return False
    
    def _validate_schema(self, record: DataRecord) -> bool:
        """Validate record schema."""
        # Basic schema validation - can be extended
        required_fields = self.config.get('required_fields', [])
        
        for field in required_fields:
            if field not in record.data:
                logger.warning(f"Missing required field '{field}' in record {record.id}")
                return False
        
        return True
    
    async def _run_expectations(self, record: DataRecord) -> float:
        """Run Great Expectations validation."""
        try:
            # Convert record to pandas DataFrame for GE
            df = pd.DataFrame([record.data])
            
            total_expectations = 0
            passed_expectations = 0
            
            for suite_name, suite in self.expectation_suites.items():
                # Create GE DataFrame
                ge_df = ge.from_pandas(df)
                
                # Run expectations
                validation_result = ge_df.validate(expectation_suite=suite)
                
                for result in validation_result.results:
                    total_expectations += 1
                    if result.success:
                        passed_expectations += 1
            
            return passed_expectations / total_expectations if total_expectations > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Great Expectations validation failed: {e}")
            return 0.0


class DataProcessor:
    """Process and transform data records."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transformations = []
        self.encryption_key = None
        
        if config.get('encryption_enabled'):
            self.encryption_key = Fernet.generate_key()
            self.cipher = Fernet(self.encryption_key)
    
    def add_transformation(self, transform_func: Callable[[DataRecord], DataRecord]):
        """Add a data transformation function."""
        self.transformations.append(transform_func)
    
    async def process_record(self, record: DataRecord) -> DataRecord:
        """Process a single data record."""
        try:
            # Apply transformations
            for transform in self.transformations:
                record = await self._apply_transformation(transform, record)
            
            # Apply anonymization if enabled
            if self.config.get('anonymization_enabled'):
                record = await self._anonymize_record(record)
            
            # Apply encryption if enabled
            if self.config.get('encryption_enabled'):
                record = await self._encrypt_record(record)
            
            return record
            
        except Exception as e:
            logger.error(f"Processing error for record {record.id}: {e}")
            record.is_valid = False
            return record
    
    async def _apply_transformation(self, transform_func: Callable, record: DataRecord) -> DataRecord:
        """Apply a transformation function to record."""
        try:
            if asyncio.iscoroutinefunction(transform_func):
                return await transform_func(record)
            else:
                return transform_func(record)
        except Exception as e:
            logger.error(f"Transformation failed for record {record.id}: {e}")
            return record
    
    async def _anonymize_record(self, record: DataRecord) -> DataRecord:
        """Anonymize sensitive data in record."""
        sensitive_fields = self.config.get('sensitive_fields', [])
        
        for field in sensitive_fields:
            if field in record.data:
                # Hash sensitive data
                original_value = str(record.data[field])
                hashed_value = hashlib.sha256(original_value.encode()).hexdigest()
                record.data[field] = hashed_value
                
                # Track anonymization in metadata
                if 'anonymized_fields' not in record.metadata:
                    record.metadata['anonymized_fields'] = []
                record.metadata['anonymized_fields'].append(field)
        
        return record
    
    async def _encrypt_record(self, record: DataRecord) -> DataRecord:
        """Encrypt sensitive data in record."""
        if not self.cipher:
            return record
        
        encrypt_fields = self.config.get('encrypt_fields', [])
        
        for field in encrypt_fields:
            if field in record.data:
                original_value = json.dumps(record.data[field])
                encrypted_value = self.cipher.encrypt(original_value.encode()).decode()
                record.data[field] = encrypted_value
                
                # Track encryption in metadata
                if 'encrypted_fields' not in record.metadata:
                    record.metadata['encrypted_fields'] = []
                record.metadata['encrypted_fields'].append(field)
        
        return record


class DataStorage:
    """Data storage backend abstraction."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_type = config['type']
        self.backend = None
    
    async def initialize(self):
        """Initialize storage backend."""
        if self.storage_type == 'database':
            await self._init_database_backend()
        elif self.storage_type == 's3':
            await self._init_s3_backend()
        elif self.storage_type == 'local':
            await self._init_local_backend()
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")
    
    async def store_batch(self, records: List[DataRecord]) -> bool:
        """Store a batch of records."""
        try:
            if self.storage_type == 'database':
                return await self._store_to_database(records)
            elif self.storage_type == 's3':
                return await self._store_to_s3(records)
            elif self.storage_type == 'local':
                return await self._store_to_local(records)
            return False
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return False
    
    async def _init_database_backend(self):
        """Initialize database backend."""
        database_url = self.config['url']
        self.backend = sa.create_engine(database_url)
        
    async def _init_s3_backend(self):
        """Initialize S3 backend."""
        self.backend = boto3.client(
            's3',
            aws_access_key_id=self.config['access_key_id'],
            aws_secret_access_key=self.config['secret_access_key'],
            region_name=self.config.get('region', 'us-west-2')
        )
    
    async def _init_local_backend(self):
        """Initialize local file backend."""
        self.storage_path = Path(self.config['path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def _store_to_database(self, records: List[DataRecord]) -> bool:
        """Store records to database."""
        Session = sessionmaker(bind=self.backend)
        session = Session()
        
        try:
            for record in records:
                # Insert record
                session.execute(
                    sa.text("""
                    INSERT INTO processed_data 
                    (id, data, timestamp, source, version, metadata, quality_score, is_valid)
                    VALUES (:id, :data, :timestamp, :source, :version, :metadata, :quality_score, :is_valid)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        timestamp = EXCLUDED.timestamp,
                        metadata = EXCLUDED.metadata,
                        quality_score = EXCLUDED.quality_score,
                        is_valid = EXCLUDED.is_valid
                    """),
                    {
                        'id': record.id,
                        'data': json.dumps(record.data),
                        'timestamp': record.timestamp,
                        'source': record.source,
                        'version': record.version,
                        'metadata': json.dumps(record.metadata),
                        'quality_score': record.quality_score,
                        'is_valid': record.is_valid
                    }
                )
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Database storage failed: {e}")
            return False
        finally:
            session.close()
    
    async def _store_to_s3(self, records: List[DataRecord]) -> bool:
        """Store records to S3."""
        try:
            # Convert records to parquet for efficient storage
            data_list = [record.to_dict() for record in records]
            df = pd.DataFrame(data_list)
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_{timestamp}_{len(records)}.parquet"
            
            # Convert to parquet bytes
            table = pa.Table.from_pandas(df)
            parquet_buffer = pa.BufferOutputStream()
            pq.write_table(table, parquet_buffer)
            
            # Upload to S3
            self.backend.put_object(
                Bucket=self.config['bucket'],
                Key=f"{self.config.get('prefix', 'data')}/{filename}",
                Body=parquet_buffer.getvalue().to_pybytes()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"S3 storage failed: {e}")
            return False
    
    async def _store_to_local(self, records: List[DataRecord]) -> bool:
        """Store records to local filesystem."""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_{timestamp}_{len(records)}.json"
            filepath = self.storage_path / filename
            
            # Write records as JSON
            data_list = [record.to_dict() for record in records]
            with open(filepath, 'w') as f:
                json.dump(data_list, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            return False


class DataPipeline:
    """Main data pipeline orchestrator."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.data_source = None
        self.validator = None
        self.processor = None
        self.storage = None
        self.is_running = False
        self.stats = {
            'records_processed': 0,
            'records_valid': 0,
            'records_invalid': 0,
            'batches_processed': 0,
            'errors': 0,
            'start_time': None,
            'last_batch_time': None
        }
        
        # Threading for parallel processing
        self.processing_queue = queue.Queue()
        self.storage_queue = queue.Queue()
        self.worker_threads = []
    
    async def initialize(self):
        """Initialize pipeline components."""
        logger.info("Initializing data pipeline")
        
        # Initialize data source
        self.data_source = await self._create_data_source()
        await self.data_source.connect()
        
        # Initialize validator
        self.validator = DataValidator(self.config.source_config.get('validation', {}))
        
        # Initialize processor
        self.processor = DataProcessor(self.config.source_config.get('processing', {}))
        
        # Initialize storage
        storage_config = {
            'type': self.config.storage_type,
            **self.config.storage_config
        }
        self.storage = DataStorage(storage_config)
        await self.storage.initialize()
        
        logger.info("Data pipeline initialized successfully")
    
    async def _create_data_source(self) -> DataSource:
        """Create appropriate data source based on configuration."""
        source_type = self.config.source_type
        
        if source_type == 'database':
            return DatabaseDataSource(self.config.source_config)
        elif source_type == 'kafka':
            return KafkaDataSource(self.config.source_config)
        elif source_type == 's3':
            return S3DataSource(self.config.source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    async def run_batch_processing(self):
        """Run batch processing mode."""
        logger.info("Starting batch processing")
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        try:
            while self.is_running:
                # Read batch from source
                batch = await self.data_source.read_batch(self.config.batch_size)
                
                if not batch:
                    await asyncio.sleep(5)  # Wait before next batch
                    continue
                
                # Process batch
                processed_batch = await self._process_batch(batch)
                
                # Store batch
                success = await self.storage.store_batch(processed_batch)
                
                # Update statistics
                self._update_stats(processed_batch, success)
                
                logger.info(f"Processed batch: {len(batch)} records, {len([r for r in processed_batch if r.is_valid])} valid")
                
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.stats['errors'] += 1
        finally:
            self.is_running = False
    
    async def run_stream_processing(self):
        """Run stream processing mode."""
        logger.info("Starting stream processing")
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Start worker threads
        self._start_worker_threads()
        
        try:
            async for record in self.data_source.read_stream():
                if not self.is_running:
                    break
                
                # Add to processing queue
                self.processing_queue.put(record)
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            self.stats['errors'] += 1
        finally:
            self.is_running = False
            self._stop_worker_threads()
    
    async def _process_batch(self, batch: List[DataRecord]) -> List[DataRecord]:
        """Process a batch of records."""
        processed_records = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.processing_workers) as executor:
            tasks = [
                executor.submit(self._process_single_record, record) 
                for record in batch
            ]
            
            for task in tasks:
                try:
                    processed_record = task.result()
                    processed_records.append(processed_record)
                except Exception as e:
                    logger.error(f"Record processing failed: {e}")
                    self.stats['errors'] += 1
        
        return processed_records
    
    def _process_single_record(self, record: DataRecord) -> DataRecord:
        """Process a single record (synchronous for thread pool)."""
        # Run async processing in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._async_process_record(record))
        finally:
            loop.close()
    
    async def _async_process_record(self, record: DataRecord) -> DataRecord:
        """Async processing of a single record."""
        # Validate record
        is_valid = await self.validator.validate_record(record)
        
        if is_valid:
            # Process record
            record = await self.processor.process_record(record)
        
        return record
    
    def _start_worker_threads(self):
        """Start worker threads for stream processing."""
        # Processing workers
        for i in range(self.config.processing_workers):
            worker = threading.Thread(target=self._processing_worker, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
        
        # Storage worker
        storage_worker = threading.Thread(target=self._storage_worker, daemon=True)
        storage_worker.start()
        self.worker_threads.append(storage_worker)
    
    def _processing_worker(self):
        """Worker thread for processing records."""
        while self.is_running:
            try:
                record = self.processing_queue.get(timeout=1)
                processed_record = self._process_single_record(record)
                self.storage_queue.put(processed_record)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")
    
    def _storage_worker(self):
        """Worker thread for storing records."""
        batch = []
        last_store_time = time.time()
        
        while self.is_running:
            try:
                # Try to get record with timeout
                try:
                    record = self.storage_queue.get(timeout=1)
                    batch.append(record)
                    self.storage_queue.task_done()
                except queue.Empty:
                    pass
                
                # Store batch if it's full or enough time has passed
                current_time = time.time()
                should_store = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_store_time > 30)  # Store every 30 seconds
                )
                
                if should_store and batch:
                    # Store batch (sync call in thread)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        success = loop.run_until_complete(self.storage.store_batch(batch))
                        self._update_stats(batch, success)
                        batch = []
                        last_store_time = current_time
                    finally:
                        loop.close()
                
            except Exception as e:
                logger.error(f"Storage worker error: {e}")
    
    def _stop_worker_threads(self):
        """Stop all worker threads."""
        # Wait for queues to empty
        self.processing_queue.join()
        self.storage_queue.join()
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        self.worker_threads.clear()
    
    def _update_stats(self, batch: List[DataRecord], success: bool):
        """Update pipeline statistics."""
        self.stats['records_processed'] += len(batch)
        self.stats['records_valid'] += sum(1 for r in batch if r.is_valid)
        self.stats['records_invalid'] += sum(1 for r in batch if not r.is_valid)
        self.stats['batches_processed'] += 1
        self.stats['last_batch_time'] = datetime.now()
        
        if not success:
            self.stats['errors'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.stats.copy()
        
        if stats['start_time']:
            runtime = (datetime.now() - stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['records_per_second'] = stats['records_processed'] / runtime if runtime > 0 else 0
        
        return stats
    
    async def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping data pipeline")
        self.is_running = False
        
        if self.data_source:
            await self.data_source.disconnect()
        
        logger.info("Data pipeline stopped")


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    config = DataPipelineConfig(
        source_type='database',
        source_config={
            'url': 'postgresql://user:pass@localhost:5432/saferl',
            'validation': {
                'required_fields': ['state', 'action', 'reward'],
                'quality_thresholds': {'min_quality_score': 0.8}
            },
            'processing': {
                'anonymization_enabled': True,
                'sensitive_fields': ['user_id', 'session_id'],
                'encryption_enabled': True,
                'encrypt_fields': ['personal_data']
            }
        },
        batch_size=1000,
        processing_workers=4,
        storage_type='database',
        storage_config={
            'url': 'postgresql://user:pass@localhost:5432/saferl'
        }
    )
    
    # Create and run pipeline
    async def main():
        pipeline = DataPipeline(config)
        await pipeline.initialize()
        
        # Run in batch mode
        await pipeline.run_batch_processing()
        
        # Print statistics
        stats = pipeline.get_stats()
        print(f"Pipeline stats: {json.dumps(stats, indent=2, default=str)}")
        
        await pipeline.stop()
    
    asyncio.run(main())