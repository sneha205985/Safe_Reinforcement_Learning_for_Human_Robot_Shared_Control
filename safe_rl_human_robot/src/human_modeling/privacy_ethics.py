"""
Privacy and Ethics Compliance System.

This module implements comprehensive privacy protection and ethics compliance
for human subject research and data handling, including GDPR compliance,
data anonymization, encryption, audit trails, and ethical oversight.

Key Features:
- GDPR compliance (data minimization, consent management, right to erasure)
- Advanced data anonymization and pseudonymization
- End-to-end encryption for sensitive data
- Comprehensive audit logging
- Ethics review and approval workflows
- Data retention and deletion policies
- Privacy-preserving analytics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import secrets
import json
import datetime
from abc import ABC, abstractmethod
import logging
import sqlite3
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("Cryptography library not available - encryption features disabled")

# Privacy-preserving analytics
try:
    from scipy.stats import laplace
    DIFFERENTIAL_PRIVACY_AVAILABLE = True
except ImportError:
    DIFFERENTIAL_PRIVACY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsentType(Enum):
    """Types of consent for data processing."""
    RESEARCH_PARTICIPATION = "research_participation"
    DATA_COLLECTION = "data_collection"
    DATA_STORAGE = "data_storage"
    DATA_SHARING = "data_sharing"
    PUBLICATION = "publication"
    FUTURE_CONTACT = "future_contact"
    BIOMETRIC_DATA = "biometric_data"
    VIDEO_RECORDING = "video_recording"


class DataCategory(Enum):
    """Categories of data for GDPR compliance."""
    PERSONAL_IDENTIFIABLE = "personal_identifiable"  # Name, email, etc.
    BIOMETRIC = "biometric"  # EMG, eye-tracking, etc.
    BEHAVIORAL = "behavioral"  # Task performance, interactions
    PHYSIOLOGICAL = "physiological"  # Heart rate, etc.
    DEMOGRAPHIC = "demographic"  # Age, gender, etc.
    TECHNICAL = "technical"  # System logs, timestamps
    DERIVED = "derived"  # Computed features, predictions


class ProcessingPurpose(Enum):
    """Purposes for data processing under GDPR."""
    RESEARCH = "research"
    SYSTEM_IMPROVEMENT = "system_improvement"
    SAFETY_MONITORING = "safety_monitoring"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MODEL_TRAINING = "model_training"
    PUBLICATION = "publication"
    REGULATORY_COMPLIANCE = "regulatory_compliance"


class DataSubjectRight(Enum):
    """Data subject rights under GDPR."""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure (right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    DATA_PORTABILITY = "data_portability"  # Right to data portability
    OBJECT = "object"  # Right to object
    WITHDRAW_CONSENT = "withdraw_consent"  # Right to withdraw consent


@dataclass
class ConsentRecord:
    """Record of participant consent."""
    participant_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: datetime.datetime
    consent_version: str
    digital_signature: Optional[str] = None
    witness: Optional[str] = None
    expiry_date: Optional[datetime.datetime] = None
    withdrawal_date: Optional[datetime.datetime] = None
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        now = datetime.datetime.now()
        
        # Check if withdrawn
        if self.withdrawal_date and self.withdrawal_date <= now:
            return False
        
        # Check if expired
        if self.expiry_date and self.expiry_date <= now:
            return False
        
        return self.granted
    
    def withdraw(self, timestamp: Optional[datetime.datetime] = None):
        """Withdraw consent."""
        self.withdrawal_date = timestamp or datetime.datetime.now()
        self.granted = False


@dataclass 
class DataProcessingRecord:
    """Record of data processing activities."""
    processing_id: str
    participant_id: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    legal_basis: str  # GDPR legal basis
    processing_start: datetime.datetime
    processing_end: Optional[datetime.datetime] = None
    data_controller: str = "Research Team"
    data_processor: Optional[str] = None
    retention_period: Optional[datetime.timedelta] = None
    security_measures: List[str] = field(default_factory=list)
    
    def should_delete(self) -> bool:
        """Check if data should be deleted based on retention policy."""
        if not self.retention_period:
            return False
        
        if not self.processing_end:
            return False
        
        deletion_date = self.processing_end + self.retention_period
        return datetime.datetime.now() >= deletion_date


@dataclass
class AuditLogEntry:
    """Audit log entry for privacy compliance."""
    timestamp: datetime.datetime
    user_id: str
    action: str
    participant_id: Optional[str]
    data_categories: List[str]
    legal_basis: str
    purpose: str
    outcome: str
    ip_address: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action': self.action,
            'participant_id': self.participant_id,
            'data_categories': self.data_categories,
            'legal_basis': self.legal_basis,
            'purpose': self.purpose,
            'outcome': self.outcome,
            'ip_address': self.ip_address,
            'system_info': self.system_info
        }


class DataAnonymizer:
    """Advanced data anonymization and pseudonymization."""
    
    def __init__(self):
        self.pseudonym_mapping = {}  # Store mapping for re-identification if needed
        self.anonymization_log = []
        
    def anonymize_identifiers(self, data: Dict[str, Any], 
                            keep_mapping: bool = False) -> Dict[str, Any]:
        """Anonymize direct identifiers."""
        anonymized = data.copy()
        
        # Direct identifiers to remove/anonymize
        direct_identifiers = [
            'name', 'email', 'phone', 'address', 'ssn', 'id_number',
            'participant_name', 'contact_info'
        ]
        
        for identifier in direct_identifiers:
            if identifier in anonymized:
                if keep_mapping:
                    # Create pseudonym
                    pseudonym = self._generate_pseudonym(str(anonymized[identifier]))
                    self.pseudonym_mapping[pseudonym] = anonymized[identifier]
                    anonymized[identifier] = pseudonym
                else:
                    # Complete removal
                    del anonymized[identifier]
        
        return anonymized
    
    def k_anonymize(self, data: List[Dict[str, Any]], k: int = 5,
                   quasi_identifiers: List[str] = None) -> List[Dict[str, Any]]:
        """Apply k-anonymity to dataset."""
        if quasi_identifiers is None:
            quasi_identifiers = ['age', 'gender', 'location', 'occupation']
        
        # Group records by quasi-identifier combinations
        groups = {}
        for record in data:
            key = tuple(record.get(qi, 'unknown') for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # Apply generalization to groups with < k members
        anonymized_data = []
        
        for group_key, group_records in groups.items():
            if len(group_records) < k:
                # Generalize quasi-identifiers
                generalized_records = self._generalize_group(
                    group_records, quasi_identifiers)
                anonymized_data.extend(generalized_records)
            else:
                anonymized_data.extend(group_records)
        
        return anonymized_data
    
    def add_differential_privacy_noise(self, numeric_data: np.ndarray,
                                     epsilon: float = 1.0) -> np.ndarray:
        """Add differential privacy noise to numeric data."""
        if not DIFFERENTIAL_PRIVACY_AVAILABLE:
            logger.warning("Differential privacy not available - returning original data")
            return numeric_data
        
        # Add Laplace noise for differential privacy
        sensitivity = 1.0  # Assume unit sensitivity
        scale = sensitivity / epsilon
        
        noise = laplace.rvs(scale=scale, size=numeric_data.shape)
        return numeric_data + noise
    
    def _generate_pseudonym(self, original_value: str) -> str:
        """Generate pseudonym for original value."""
        # Use SHA-256 hash with salt for pseudonymization
        salt = secrets.token_bytes(32)
        hasher = hashlib.sha256()
        hasher.update(salt + original_value.encode())
        pseudonym = base64.urlsafe_b64encode(hasher.digest()[:16]).decode().rstrip('=')
        return f"PSEUDO_{pseudonym}"
    
    def _generalize_group(self, records: List[Dict[str, Any]], 
                         quasi_identifiers: List[str]) -> List[Dict[str, Any]]:
        """Generalize quasi-identifiers for small groups."""
        generalized = []
        
        for record in records:
            gen_record = record.copy()
            
            for qi in quasi_identifiers:
                if qi in gen_record:
                    if qi == 'age':
                        # Age range generalization
                        age = gen_record[qi]
                        if isinstance(age, (int, float)):
                            age_range = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                            gen_record[qi] = age_range
                    
                    elif qi == 'location':
                        # Location generalization (remove specific details)
                        location = str(gen_record[qi])
                        if ',' in location:
                            gen_record[qi] = location.split(',')[0]  # Keep only first part
                        
                    elif qi == 'occupation':
                        # Occupation category generalization
                        occupation = str(gen_record[qi]).lower()
                        if 'engineer' in occupation or 'developer' in occupation:
                            gen_record[qi] = 'Technology'
                        elif 'doctor' in occupation or 'nurse' in occupation:
                            gen_record[qi] = 'Healthcare'
                        else:
                            gen_record[qi] = 'Other'
            
            generalized.append(gen_record)
        
        return generalized


class DataEncryption:
    """Data encryption and secure storage."""
    
    def __init__(self, master_password: Optional[str] = None):
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography library required for encryption")
        
        self.master_password = master_password or secrets.token_urlsafe(32)
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption keys."""
        # Derive key from master password
        password_bytes = self.master_password.encode()
        salt = b'stable_salt_for_research'  # In production, use random salt per dataset
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.fernet = Fernet(key)
        
        # Generate RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_sensitive_data(self, data: Any) -> str:
        """Encrypt sensitive data using symmetric encryption."""
        # Convert data to JSON string
        json_data = json.dumps(data, default=str)
        data_bytes = json_data.encode()
        
        # Encrypt
        encrypted_data = self.fernet.encrypt(data_bytes)
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Any:
        """Decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            json_data = decrypted_bytes.decode()
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def encrypt_with_public_key(self, data: str) -> str:
        """Encrypt data with public key (for secure sharing)."""
        data_bytes = data.encode()
        
        encrypted = self.public_key.encrypt(
            data_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_with_private_key(self, encrypted_data: str) -> str:
        """Decrypt data with private key."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            
            decrypted = self.private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Private key decryption failed: {e}")
            return None
    
    def secure_delete(self, file_path: str, passes: int = 3):
        """Securely delete file by overwriting."""
        if not os.path.exists(file_path):
            return
        
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r+b') as file:
            for _ in range(passes):
                file.seek(0)
                file.write(secrets.token_bytes(file_size))
                file.flush()
                os.fsync(file.fileno())
        
        os.remove(file_path)
        logger.info(f"Securely deleted file: {file_path}")


class ConsentManager:
    """Manage participant consent and GDPR compliance."""
    
    def __init__(self, storage_path: str = "consent_records.db"):
        self.storage_path = storage_path
        self.consent_records: Dict[str, List[ConsentRecord]] = {}
        self._setup_database()
    
    def _setup_database(self):
        """Setup SQLite database for consent storage."""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consent_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id TEXT NOT NULL,
                consent_type TEXT NOT NULL,
                granted BOOLEAN NOT NULL,
                timestamp TEXT NOT NULL,
                consent_version TEXT NOT NULL,
                digital_signature TEXT,
                witness TEXT,
                expiry_date TEXT,
                withdrawal_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_consent(self, consent: ConsentRecord):
        """Record participant consent."""
        # Store in memory
        if consent.participant_id not in self.consent_records:
            self.consent_records[consent.participant_id] = []
        
        self.consent_records[consent.participant_id].append(consent)
        
        # Store in database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO consent_records 
            (participant_id, consent_type, granted, timestamp, consent_version,
             digital_signature, witness, expiry_date, withdrawal_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            consent.participant_id,
            consent.consent_type.value,
            consent.granted,
            consent.timestamp.isoformat(),
            consent.consent_version,
            consent.digital_signature,
            consent.witness,
            consent.expiry_date.isoformat() if consent.expiry_date else None,
            consent.withdrawal_date.isoformat() if consent.withdrawal_date else None
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded consent for {consent.participant_id}: {consent.consent_type.value}")
    
    def check_consent(self, participant_id: str, 
                     consent_type: ConsentType) -> bool:
        """Check if participant has valid consent for specific processing."""
        if participant_id not in self.consent_records:
            return False
        
        # Find most recent consent of this type
        relevant_consents = [c for c in self.consent_records[participant_id] 
                           if c.consent_type == consent_type]
        
        if not relevant_consents:
            return False
        
        # Get most recent consent
        most_recent = max(relevant_consents, key=lambda c: c.timestamp)
        return most_recent.is_valid()
    
    def withdraw_consent(self, participant_id: str, 
                        consent_type: ConsentType,
                        timestamp: Optional[datetime.datetime] = None):
        """Withdraw participant consent."""
        if participant_id not in self.consent_records:
            logger.warning(f"No consent records found for {participant_id}")
            return
        
        # Find and withdraw relevant consents
        for consent in self.consent_records[participant_id]:
            if consent.consent_type == consent_type and consent.granted:
                consent.withdraw(timestamp)
        
        # Update database
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        withdrawal_time = (timestamp or datetime.datetime.now()).isoformat()
        
        cursor.execute('''
            UPDATE consent_records 
            SET withdrawal_date = ?, granted = 0
            WHERE participant_id = ? AND consent_type = ? AND withdrawal_date IS NULL
        ''', (withdrawal_time, participant_id, consent_type.value))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Withdrew consent for {participant_id}: {consent_type.value}")
    
    def get_consent_status(self, participant_id: str) -> Dict[ConsentType, bool]:
        """Get comprehensive consent status for participant."""
        status = {}
        
        for consent_type in ConsentType:
            status[consent_type] = self.check_consent(participant_id, consent_type)
        
        return status
    
    def generate_consent_report(self) -> Dict[str, Any]:
        """Generate consent compliance report."""
        report = {
            'total_participants': len(self.consent_records),
            'consent_summary': {},
            'expired_consents': [],
            'withdrawn_consents': []
        }
        
        # Count consents by type
        for consent_type in ConsentType:
            valid_count = 0
            total_count = 0
            
            for participant_id in self.consent_records:
                if self.check_consent(participant_id, consent_type):
                    valid_count += 1
                
                # Count all consents of this type
                type_consents = [c for c in self.consent_records[participant_id]
                               if c.consent_type == consent_type]
                total_count += len(type_consents)
            
            report['consent_summary'][consent_type.value] = {
                'valid': valid_count,
                'total': total_count
            }
        
        return report


class AuditLogger:
    """Comprehensive audit logging for privacy compliance."""
    
    def __init__(self, log_file: str = "privacy_audit.log"):
        self.log_file = log_file
        self.audit_entries: List[AuditLogEntry] = []
        
        # Setup file logging
        audit_logger = logging.getLogger('privacy_audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        self.file_logger = audit_logger
    
    def log_data_access(self, user_id: str, participant_id: str,
                       data_categories: List[DataCategory],
                       purpose: ProcessingPurpose,
                       outcome: str = "success",
                       **kwargs):
        """Log data access event."""
        entry = AuditLogEntry(
            timestamp=datetime.datetime.now(),
            user_id=user_id,
            action="data_access",
            participant_id=participant_id,
            data_categories=[cat.value for cat in data_categories],
            legal_basis=kwargs.get('legal_basis', 'consent'),
            purpose=purpose.value,
            outcome=outcome,
            ip_address=kwargs.get('ip_address'),
            system_info=kwargs.get('system_info', {})
        )
        
        self.audit_entries.append(entry)
        self.file_logger.info(json.dumps(entry.to_dict()))
    
    def log_consent_change(self, user_id: str, participant_id: str,
                          consent_type: ConsentType, action: str,
                          outcome: str = "success", **kwargs):
        """Log consent-related changes."""
        entry = AuditLogEntry(
            timestamp=datetime.datetime.now(),
            user_id=user_id,
            action=f"consent_{action}",
            participant_id=participant_id,
            data_categories=[],
            legal_basis="consent_management",
            purpose="compliance",
            outcome=outcome,
            system_info={'consent_type': consent_type.value, **kwargs.get('system_info', {})}
        )
        
        self.audit_entries.append(entry)
        self.file_logger.info(json.dumps(entry.to_dict()))
    
    def log_data_deletion(self, user_id: str, participant_id: str,
                         data_categories: List[DataCategory],
                         reason: str, outcome: str = "success", **kwargs):
        """Log data deletion events."""
        entry = AuditLogEntry(
            timestamp=datetime.datetime.now(),
            user_id=user_id,
            action="data_deletion",
            participant_id=participant_id,
            data_categories=[cat.value for cat in data_categories],
            legal_basis="right_to_erasure",
            purpose="compliance",
            outcome=outcome,
            system_info={'deletion_reason': reason, **kwargs.get('system_info', {})}
        )
        
        self.audit_entries.append(entry)
        self.file_logger.info(json.dumps(entry.to_dict()))
    
    def generate_audit_report(self, start_date: Optional[datetime.datetime] = None,
                            end_date: Optional[datetime.datetime] = None) -> Dict[str, Any]:
        """Generate audit report for specified time period."""
        # Filter entries by date range
        filtered_entries = self.audit_entries
        
        if start_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_date]
        
        if end_date:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_date]
        
        # Generate summary statistics
        report = {
            'period_start': start_date.isoformat() if start_date else 'N/A',
            'period_end': end_date.isoformat() if end_date else 'N/A',
            'total_events': len(filtered_entries),
            'events_by_action': {},
            'events_by_user': {},
            'events_by_outcome': {},
            'data_subjects_affected': set(),
            'compliance_issues': []
        }
        
        # Analyze entries
        for entry in filtered_entries:
            # Count by action
            action = entry.action
            report['events_by_action'][action] = report['events_by_action'].get(action, 0) + 1
            
            # Count by user
            user = entry.user_id
            report['events_by_user'][user] = report['events_by_user'].get(user, 0) + 1
            
            # Count by outcome
            outcome = entry.outcome
            report['events_by_outcome'][outcome] = report['events_by_outcome'].get(outcome, 0) + 1
            
            # Track data subjects
            if entry.participant_id:
                report['data_subjects_affected'].add(entry.participant_id)
            
            # Identify potential compliance issues
            if entry.outcome == "failure" or entry.outcome == "error":
                report['compliance_issues'].append({
                    'timestamp': entry.timestamp.isoformat(),
                    'action': entry.action,
                    'participant_id': entry.participant_id,
                    'issue': entry.outcome
                })
        
        # Convert set to count
        report['unique_data_subjects'] = len(report['data_subjects_affected'])
        del report['data_subjects_affected']
        
        return report


class PrivacyComplianceManager:
    """Main privacy and ethics compliance management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.consent_manager = ConsentManager(
            config.get('consent_db_path', 'consent_records.db'))
        
        self.audit_logger = AuditLogger(
            config.get('audit_log_path', 'privacy_audit.log'))
        
        self.data_anonymizer = DataAnonymizer()
        
        # Initialize encryption if available
        try:
            self.encryption = DataEncryption(config.get('master_password'))
        except RuntimeError:
            logger.warning("Encryption not available - sensitive data will not be encrypted")
            self.encryption = None
        
        # Data retention policies
        self.retention_policies = config.get('retention_policies', {
            DataCategory.PERSONAL_IDENTIFIABLE: datetime.timedelta(days=1095),  # 3 years
            DataCategory.BIOMETRIC: datetime.timedelta(days=2555),  # 7 years
            DataCategory.BEHAVIORAL: datetime.timedelta(days=1825),  # 5 years
            DataCategory.DERIVED: datetime.timedelta(days=3650)  # 10 years
        })
        
        # Processing records
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        
        logger.info("Privacy Compliance Manager initialized")
    
    def register_data_processing(self, participant_id: str,
                                data_categories: List[DataCategory],
                                purposes: List[ProcessingPurpose],
                                legal_basis: str = "consent") -> str:
        """Register new data processing activity."""
        processing_id = f"proc_{participant_id}_{int(datetime.datetime.now().timestamp())}"
        
        # Check consent for each required category
        consent_valid = True
        required_consent_types = self._map_data_to_consent_types(data_categories)
        
        for consent_type in required_consent_types:
            if not self.consent_manager.check_consent(participant_id, consent_type):
                consent_valid = False
                logger.warning(f"Missing consent for {consent_type.value} - {participant_id}")
        
        if not consent_valid and legal_basis == "consent":
            raise ValueError(f"Insufficient consent for processing participant {participant_id}")
        
        # Create processing record
        record = DataProcessingRecord(
            processing_id=processing_id,
            participant_id=participant_id,
            data_categories=data_categories,
            processing_purposes=purposes,
            legal_basis=legal_basis,
            processing_start=datetime.datetime.now(),
            retention_period=self._calculate_retention_period(data_categories, purposes),
            security_measures=["encryption", "access_control", "audit_logging"]
        )
        
        self.processing_records[processing_id] = record
        
        # Log processing start
        self.audit_logger.log_data_access(
            user_id="system",
            participant_id=participant_id,
            data_categories=data_categories,
            purpose=purposes[0] if purposes else ProcessingPurpose.RESEARCH,
            legal_basis=legal_basis,
            system_info={'processing_id': processing_id}
        )
        
        logger.info(f"Registered data processing: {processing_id}")
        return processing_id
    
    def process_data_subject_request(self, participant_id: str,
                                   request_type: DataSubjectRight,
                                   user_id: str,
                                   additional_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process data subject rights requests."""
        response = {
            'request_type': request_type.value,
            'participant_id': participant_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'processing',
            'data': None,
            'actions_taken': []
        }
        
        try:
            if request_type == DataSubjectRight.ACCESS:
                # Right to access - provide all data
                data = self._collect_participant_data(participant_id)
                if self.encryption:
                    # Decrypt for access (normally would be done securely)
                    response['data'] = data
                else:
                    response['data'] = data
                
                response['actions_taken'].append("Data package prepared")
                response['status'] = 'completed'
            
            elif request_type == DataSubjectRight.ERASURE:
                # Right to be forgotten
                deleted_categories = self._delete_participant_data(participant_id, user_id)
                response['actions_taken'] = [f"Deleted {cat}" for cat in deleted_categories]
                response['status'] = 'completed'
            
            elif request_type == DataSubjectRight.RECTIFICATION:
                # Right to rectification
                if additional_info and 'corrections' in additional_info:
                    corrections = additional_info['corrections']
                    self._apply_data_corrections(participant_id, corrections)
                    response['actions_taken'].append("Data corrections applied")
                    response['status'] = 'completed'
                else:
                    response['status'] = 'requires_additional_info'
                    response['message'] = 'Please provide correction details'
            
            elif request_type == DataSubjectRight.WITHDRAW_CONSENT:
                # Withdraw consent for all categories
                for consent_type in ConsentType:
                    try:
                        self.consent_manager.withdraw_consent(participant_id, consent_type)
                        response['actions_taken'].append(f"Withdrew {consent_type.value}")
                    except:
                        pass  # May not have all consent types
                
                response['status'] = 'completed'
            
            elif request_type == DataSubjectRight.DATA_PORTABILITY:
                # Provide data in portable format
                data = self._collect_participant_data(participant_id, portable_format=True)
                response['data'] = data
                response['actions_taken'].append("Portable data package prepared")
                response['status'] = 'completed'
            
            else:
                response['status'] = 'not_implemented'
                response['message'] = f"Request type {request_type.value} not yet implemented"
        
        except Exception as e:
            response['status'] = 'error'
            response['message'] = str(e)
            logger.error(f"Error processing data subject request: {e}")
        
        # Log the request processing
        self.audit_logger.log_data_access(
            user_id=user_id,
            participant_id=participant_id,
            data_categories=[DataCategory.PERSONAL_IDENTIFIABLE],
            purpose=ProcessingPurpose.REGULATORY_COMPLIANCE,
            outcome=response['status'],
            system_info={'request_type': request_type.value}
        )
        
        return response
    
    def anonymize_dataset(self, data: List[Dict[str, Any]], 
                         anonymization_level: str = "standard") -> List[Dict[str, Any]]:
        """Anonymize dataset for sharing or publication."""
        logger.info(f"Anonymizing dataset with {len(data)} records")
        
        # Apply different levels of anonymization
        if anonymization_level == "basic":
            # Remove direct identifiers only
            anonymized = [self.data_anonymizer.anonymize_identifiers(record) 
                         for record in data]
        
        elif anonymization_level == "standard":
            # Remove identifiers + k-anonymity
            no_identifiers = [self.data_anonymizer.anonymize_identifiers(record) 
                            for record in data]
            anonymized = self.data_anonymizer.k_anonymize(no_identifiers, k=5)
        
        elif anonymization_level == "high":
            # Standard + differential privacy
            no_identifiers = [self.data_anonymizer.anonymize_identifiers(record) 
                            for record in data]
            k_anon = self.data_anonymizer.k_anonymize(no_identifiers, k=10)
            
            # Add differential privacy noise to numeric columns
            anonymized = []
            for record in k_anon:
                anon_record = record.copy()
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        noise_value = self.data_anonymizer.add_differential_privacy_noise(
                            np.array([value]), epsilon=1.0)[0]
                        anon_record[key] = float(noise_value)
                anonymized.append(anon_record)
        
        else:
            raise ValueError(f"Unknown anonymization level: {anonymization_level}")
        
        logger.info(f"Anonymization completed: {anonymization_level} level")
        return anonymized
    
    def check_retention_compliance(self) -> Dict[str, Any]:
        """Check compliance with data retention policies."""
        compliance_report = {
            'total_processing_records': len(self.processing_records),
            'records_due_for_deletion': [],
            'overdue_deletions': [],
            'compliant_records': 0
        }
        
        current_time = datetime.datetime.now()
        
        for processing_id, record in self.processing_records.items():
            if record.should_delete():
                if record.processing_end and current_time > (record.processing_end + record.retention_period):
                    # Overdue for deletion
                    compliance_report['overdue_deletions'].append({
                        'processing_id': processing_id,
                        'participant_id': record.participant_id,
                        'overdue_days': (current_time - (record.processing_end + record.retention_period)).days
                    })
                else:
                    # Due for deletion
                    compliance_report['records_due_for_deletion'].append({
                        'processing_id': processing_id,
                        'participant_id': record.participant_id,
                        'deletion_due_date': (record.processing_end + record.retention_period).isoformat()
                    })
            else:
                compliance_report['compliant_records'] += 1
        
        return compliance_report
    
    def _map_data_to_consent_types(self, data_categories: List[DataCategory]) -> List[ConsentType]:
        """Map data categories to required consent types."""
        consent_mapping = {
            DataCategory.PERSONAL_IDENTIFIABLE: [ConsentType.DATA_COLLECTION],
            DataCategory.BIOMETRIC: [ConsentType.BIOMETRIC_DATA],
            DataCategory.BEHAVIORAL: [ConsentType.DATA_COLLECTION],
            DataCategory.PHYSIOLOGICAL: [ConsentType.BIOMETRIC_DATA],
            DataCategory.DEMOGRAPHIC: [ConsentType.DATA_COLLECTION]
        }
        
        required_consents = set()
        for category in data_categories:
            if category in consent_mapping:
                required_consents.update(consent_mapping[category])
        
        return list(required_consents)
    
    def _calculate_retention_period(self, data_categories: List[DataCategory],
                                   purposes: List[ProcessingPurpose]) -> datetime.timedelta:
        """Calculate retention period based on data categories and purposes."""
        # Use the longest retention period among all categories
        max_retention = datetime.timedelta(days=365)  # Default 1 year
        
        for category in data_categories:
            if category in self.retention_policies:
                category_retention = self.retention_policies[category]
                if category_retention > max_retention:
                    max_retention = category_retention
        
        # Adjust based on purpose
        if ProcessingPurpose.PUBLICATION in purposes:
            # Publications may require longer retention
            max_retention = max(max_retention, datetime.timedelta(days=3650))  # 10 years
        
        return max_retention
    
    def _collect_participant_data(self, participant_id: str, 
                                 portable_format: bool = False) -> Dict[str, Any]:
        """Collect all data for a participant (for access requests)."""
        participant_data = {
            'participant_id': participant_id,
            'consent_records': [],
            'processing_records': [],
            'data_files': [],
            'export_timestamp': datetime.datetime.now().isoformat()
        }
        
        # Get consent records
        consent_status = self.consent_manager.get_consent_status(participant_id)
        for consent_type, status in consent_status.items():
            participant_data['consent_records'].append({
                'consent_type': consent_type.value,
                'current_status': status
            })
        
        # Get processing records
        for processing_id, record in self.processing_records.items():
            if record.participant_id == participant_id:
                participant_data['processing_records'].append({
                    'processing_id': processing_id,
                    'data_categories': [cat.value for cat in record.data_categories],
                    'purposes': [purpose.value for purpose in record.processing_purposes],
                    'start_date': record.processing_start.isoformat(),
                    'end_date': record.processing_end.isoformat() if record.processing_end else None
                })
        
        if portable_format:
            # Format for data portability (machine-readable JSON)
            participant_data['format'] = 'json'
            participant_data['schema_version'] = '1.0'
        
        return participant_data
    
    def _delete_participant_data(self, participant_id: str, user_id: str) -> List[str]:
        """Delete all data for a participant (right to be forgotten)."""
        deleted_categories = []
        
        # Find all processing records for this participant
        records_to_end = []
        for processing_id, record in self.processing_records.items():
            if record.participant_id == participant_id:
                record.processing_end = datetime.datetime.now()
                records_to_end.append(processing_id)
        
        # Log deletion for each record
        for processing_id in records_to_end:
            record = self.processing_records[processing_id]
            self.audit_logger.log_data_deletion(
                user_id=user_id,
                participant_id=participant_id,
                data_categories=record.data_categories,
                reason="right_to_be_forgotten",
                system_info={'processing_id': processing_id}
            )
            
            deleted_categories.extend([cat.value for cat in record.data_categories])
        
        # In a real implementation, would delete actual data files, database entries, etc.
        logger.info(f"Processed deletion request for participant {participant_id}")
        
        return list(set(deleted_categories))  # Remove duplicates
    
    def _apply_data_corrections(self, participant_id: str, corrections: Dict[str, Any]):
        """Apply data corrections (right to rectification)."""
        # In a real implementation, would update actual data storage
        logger.info(f"Applied data corrections for participant {participant_id}: {corrections}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Privacy and Ethics Compliance System...")
    
    # Initialize privacy compliance manager
    config = {
        'consent_db_path': ':memory:',  # In-memory database for testing
        'audit_log_path': 'test_audit.log',
        'retention_policies': {
            DataCategory.PERSONAL_IDENTIFIABLE: datetime.timedelta(days=1095),
            DataCategory.BIOMETRIC: datetime.timedelta(days=2555)
        }
    }
    
    privacy_manager = PrivacyComplianceManager(config)
    
    # Test consent management
    print("\n=== Testing Consent Management ===")
    
    participant_id = "TEST_P001"
    
    # Record initial consents
    consents = [
        ConsentRecord(
            participant_id=participant_id,
            consent_type=ConsentType.RESEARCH_PARTICIPATION,
            granted=True,
            timestamp=datetime.datetime.now(),
            consent_version="v1.0"
        ),
        ConsentRecord(
            participant_id=participant_id,
            consent_type=ConsentType.BIOMETRIC_DATA,
            granted=True,
            timestamp=datetime.datetime.now(),
            consent_version="v1.0"
        )
    ]
    
    for consent in consents:
        privacy_manager.consent_manager.record_consent(consent)
    
    # Check consent status
    consent_status = privacy_manager.consent_manager.get_consent_status(participant_id)
    print(f"Consent status for {participant_id}:")
    for consent_type, status in consent_status.items():
        print(f"  {consent_type.value}: {status}")
    
    # Test data processing registration
    print("\n=== Testing Data Processing Registration ===")
    
    processing_id = privacy_manager.register_data_processing(
        participant_id=participant_id,
        data_categories=[DataCategory.BIOMETRIC, DataCategory.BEHAVIORAL],
        purposes=[ProcessingPurpose.RESEARCH, ProcessingPurpose.MODEL_TRAINING],
        legal_basis="consent"
    )
    
    print(f"Registered data processing: {processing_id}")
    
    # Test data anonymization
    print("\n=== Testing Data Anonymization ===")
    
    sample_data = [
        {
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 25,
            'gender': 'M',
            'performance_score': 0.85,
            'reaction_time': 0.23
        },
        {
            'name': 'Jane Smith', 
            'email': 'jane@example.com',
            'age': 28,
            'gender': 'F',
            'performance_score': 0.92,
            'reaction_time': 0.19
        },
        {
            'name': 'Bob Johnson',
            'email': 'bob@example.com', 
            'age': 32,
            'gender': 'M',
            'performance_score': 0.78,
            'reaction_time': 0.28
        }
    ]
    
    # Test different anonymization levels
    for level in ['basic', 'standard', 'high']:
        anonymized = privacy_manager.anonymize_dataset(sample_data.copy(), level)
        print(f"\n{level.title()} anonymization:")
        for record in anonymized[:2]:  # Show first 2 records
            print(f"  {record}")
    
    # Test data subject rights
    print("\n=== Testing Data Subject Rights ===")
    
    # Test access request
    access_response = privacy_manager.process_data_subject_request(
        participant_id=participant_id,
        request_type=DataSubjectRight.ACCESS,
        user_id="admin_user"
    )
    
    print(f"Access request status: {access_response['status']}")
    print(f"Actions taken: {access_response['actions_taken']}")
    
    # Test consent withdrawal
    withdrawal_response = privacy_manager.process_data_subject_request(
        participant_id=participant_id,
        request_type=DataSubjectRight.WITHDRAW_CONSENT,
        user_id="admin_user"
    )
    
    print(f"Consent withdrawal status: {withdrawal_response['status']}")
    print(f"Actions taken: {withdrawal_response['actions_taken']}")
    
    # Test retention compliance
    print("\n=== Testing Retention Compliance ===")
    
    compliance_report = privacy_manager.check_retention_compliance()
    print(f"Compliance report:")
    print(f"  Total records: {compliance_report['total_processing_records']}")
    print(f"  Compliant records: {compliance_report['compliant_records']}")
    print(f"  Due for deletion: {len(compliance_report['records_due_for_deletion'])}")
    print(f"  Overdue deletions: {len(compliance_report['overdue_deletions'])}")
    
    # Generate audit report
    print("\n=== Testing Audit Report ===")
    
    audit_report = privacy_manager.audit_logger.generate_audit_report()
    print(f"Audit report:")
    print(f"  Total events: {audit_report['total_events']}")
    print(f"  Unique data subjects: {audit_report['unique_data_subjects']}")
    print(f"  Events by action: {audit_report['events_by_action']}")
    print(f"  Compliance issues: {len(audit_report['compliance_issues'])}")
    
    print("\nPrivacy and Ethics Compliance System test completed!")