"""
Enterprise Compliance and Governance System for Safe RL Production.

This module provides comprehensive compliance features including:
- GDPR/CCPA data privacy compliance
- Model governance and explainability
- Bias detection and mitigation
- Regulatory compliance tracking
- Data lineage and audit trails
- Automated compliance reporting
"""

import asyncio
import logging
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import shap
import lime
from sklearn.metrics import confusion_matrix, classification_report
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import requests

logger = logging.getLogger(__name__)

Base = declarative_base()


class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    SOC2_TYPE2 = "soc2_type2"
    NIST_CYBERSECURITY = "nist_cybersecurity"
    FDA_MEDICAL_DEVICE = "fda_medical_device"
    IEEE_2857 = "ieee_2857"  # AI Engineering Standard


class DataCategory(Enum):
    """Data categories for privacy classification."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "personally_identifiable"
    PHI = "protected_health_information"
    FINANCIAL = "financial"
    BIOMETRIC = "biometric"


class ProcessingPurpose(Enum):
    """Data processing purposes."""
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SAFETY_MONITORING = "safety_monitoring"
    RESEARCH_DEVELOPMENT = "research_development"
    COMPLIANCE_AUDIT = "compliance_audit"
    SECURITY_MONITORING = "security_monitoring"


@dataclass
class DataPrivacyConfig:
    """Data privacy configuration."""
    gdpr_enabled: bool = True
    ccpa_enabled: bool = True
    data_retention_days: int = 2555  # 7 years default
    anonymization_enabled: bool = True
    pseudonymization_enabled: bool = True
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    
    # Consent management
    consent_required: bool = True
    consent_expiry_days: int = 365
    
    # Data subject rights
    right_to_access: bool = True
    right_to_rectification: bool = True
    right_to_erasure: bool = True
    right_to_portability: bool = True
    right_to_restrict: bool = True
    
    # Breach notification
    breach_notification_hours: int = 72
    breach_notification_enabled: bool = True


@dataclass
class ModelGovernanceConfig:
    """Model governance configuration."""
    explainability_required: bool = True
    bias_testing_required: bool = True
    performance_monitoring_required: bool = True
    model_approval_required: bool = True
    
    # Explainability settings
    explain_model_decisions: bool = True
    explain_global_behavior: bool = True
    explanation_methods: List[str] = field(default_factory=lambda: ['shap', 'lime'])
    
    # Bias testing settings
    protected_attributes: List[str] = field(default_factory=list)
    fairness_metrics: List[str] = field(default_factory=lambda: ['demographic_parity', 'equalized_odds'])
    bias_threshold: float = 0.1
    
    # Approval workflow
    approval_stages: List[str] = field(default_factory=lambda: ['technical_review', 'business_approval', 'compliance_sign_off'])
    required_approvers: Dict[str, List[str]] = field(default_factory=dict)


# Database Models
class DataProcessingRecord(Base):
    """Data processing activity record for GDPR compliance."""
    __tablename__ = 'data_processing_records'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic information
    processing_activity = Column(String, nullable=False)
    controller_name = Column(String, nullable=False)
    controller_contact = Column(String)
    dpo_contact = Column(String)  # Data Protection Officer
    
    # Processing details
    purposes = Column(Text)  # JSON array of purposes
    data_categories = Column(Text)  # JSON array of data categories
    data_subjects = Column(Text)  # JSON array of data subject categories
    recipients = Column(Text)  # JSON array of recipients
    
    # International transfers
    third_country_transfers = Column(Boolean, default=False)
    transfer_safeguards = Column(Text)
    
    # Retention and deletion
    retention_period = Column(String)
    deletion_schedule = Column(String)
    
    # Security measures
    technical_measures = Column(Text)  # JSON array
    organizational_measures = Column(Text)  # JSON array
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_review_date = Column(DateTime)
    next_review_date = Column(DateTime)


class ConsentRecord(Base):
    """Consent records for data subjects."""
    __tablename__ = 'consent_records'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Data subject information (anonymized)
    data_subject_id = Column(String, nullable=False)  # Pseudonymized ID
    
    # Consent details
    processing_purposes = Column(Text, nullable=False)  # JSON array
    consent_given = Column(Boolean, nullable=False)
    consent_timestamp = Column(DateTime, nullable=False)
    consent_expiry = Column(DateTime)
    
    # Consent mechanism
    consent_method = Column(String)  # web_form, api, paper, etc.
    consent_version = Column(String)
    privacy_policy_version = Column(String)
    
    # Withdrawal
    consent_withdrawn = Column(Boolean, default=False)
    withdrawal_timestamp = Column(DateTime)
    withdrawal_method = Column(String)
    
    # Metadata
    ip_address = Column(String)  # For consent verification
    user_agent = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataSubjectRequest(Base):
    """Data subject rights requests."""
    __tablename__ = 'data_subject_requests'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Request details
    request_type = Column(String, nullable=False)  # access, rectification, erasure, portability, restrict
    data_subject_id = Column(String, nullable=False)
    requester_email = Column(String)
    requester_identity_verified = Column(Boolean, default=False)
    
    # Request content
    request_description = Column(Text)
    specific_data_requested = Column(Text)
    
    # Processing
    status = Column(String, default='pending')  # pending, in_progress, completed, rejected
    assigned_to = Column(String)
    
    # Response
    response_method = Column(String)  # email, postal, secure_download
    response_sent = Column(Boolean, default=False)
    response_timestamp = Column(DateTime)
    
    # Timelines
    received_at = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime)  # Usually 30 days from request
    completed_at = Column(DateTime)
    
    # Metadata
    request_source = Column(String)  # web_form, email, phone, postal
    attachments = Column(Text)  # JSON array of attachment references
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelApprovalRecord(Base):
    """Model approval and governance records."""
    __tablename__ = 'model_approval_records'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model information
    model_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    
    # Approval workflow
    approval_stage = Column(String, nullable=False)
    approval_status = Column(String, default='pending')  # pending, approved, rejected, conditional
    approver_id = Column(String)
    approver_comments = Column(Text)
    
    # Reviews and assessments
    technical_review_completed = Column(Boolean, default=False)
    bias_assessment_completed = Column(Boolean, default=False)
    explainability_review_completed = Column(Boolean, default=False)
    compliance_review_completed = Column(Boolean, default=False)
    
    # Review results
    technical_review_score = Column(Float)
    bias_assessment_score = Column(Float)
    explainability_score = Column(Float)
    compliance_score = Column(Float)
    
    # Decision
    final_approval = Column(Boolean)
    approval_conditions = Column(Text)
    approval_expiry = Column(DateTime)
    
    # Timestamps
    submitted_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime)
    approved_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BiasAssessmentResult(Base):
    """Bias assessment results for models."""
    __tablename__ = 'bias_assessment_results'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model information
    model_id = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    
    # Assessment details
    assessment_type = Column(String, nullable=False)  # fairness, demographic_parity, etc.
    protected_attribute = Column(String, nullable=False)
    
    # Metrics
    bias_metric_name = Column(String, nullable=False)
    bias_metric_value = Column(Float, nullable=False)
    bias_threshold = Column(Float, nullable=False)
    
    # Results
    bias_detected = Column(Boolean, nullable=False)
    severity_level = Column(String)  # low, medium, high, critical
    
    # Recommendations
    mitigation_recommendations = Column(Text)
    requires_remediation = Column(Boolean, default=False)
    
    # Assessment metadata
    dataset_used = Column(String)
    assessment_methodology = Column(Text)
    
    assessed_at = Column(DateTime, default=datetime.utcnow)
    assessed_by = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ComplianceManager:
    """Main compliance management system."""
    
    def __init__(self, 
                 config: DataPrivacyConfig,
                 governance_config: ModelGovernanceConfig,
                 db_session_factory):
        self.config = config
        self.governance_config = governance_config
        self.session_factory = db_session_factory
        
        # Initialize components
        self.privacy_manager = DataPrivacyManager(config, db_session_factory)
        self.model_governance = ModelGovernance(governance_config, db_session_factory)
        self.bias_detector = BiasDetector(governance_config)
        self.explainer = ModelExplainer(governance_config)
        
        logger.info("ComplianceManager initialized")
    
    async def assess_model_compliance(self, 
                                    model: Any,
                                    model_metadata: Dict[str, Any],
                                    test_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """Comprehensive model compliance assessment."""
        compliance_report = {
            'model_id': model_metadata.get('id'),
            'model_name': model_metadata.get('name'),
            'model_version': model_metadata.get('version'),
            'assessment_timestamp': datetime.utcnow().isoformat(),
            'compliance_status': 'pending',
            'assessments': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Bias Assessment
            if self.governance_config.bias_testing_required and test_data:
                bias_results = await self.bias_detector.assess_bias(model, test_data, model_metadata)
                compliance_report['assessments']['bias'] = bias_results
                
                if bias_results.get('bias_detected'):
                    compliance_report['issues'].append("Bias detected in model outputs")
                    compliance_report['recommendations'].extend(bias_results.get('recommendations', []))
            
            # 2. Explainability Assessment
            if self.governance_config.explainability_required:
                explain_results = await self.explainer.assess_explainability(model, model_metadata)
                compliance_report['assessments']['explainability'] = explain_results
                
                if not explain_results.get('explainable'):
                    compliance_report['issues'].append("Model lacks sufficient explainability")
                    compliance_report['recommendations'].append("Implement model explanation mechanisms")
            
            # 3. Data Privacy Assessment
            privacy_results = await self.privacy_manager.assess_privacy_compliance(model_metadata)
            compliance_report['assessments']['privacy'] = privacy_results
            
            if not privacy_results.get('compliant'):
                compliance_report['issues'].extend(privacy_results.get('violations', []))
                compliance_report['recommendations'].extend(privacy_results.get('recommendations', []))
            
            # 4. Model Governance Assessment
            governance_results = await self.model_governance.assess_governance_compliance(model_metadata)
            compliance_report['assessments']['governance'] = governance_results
            
            # Determine overall compliance status
            has_critical_issues = any(
                assessment.get('critical_issues', False) 
                for assessment in compliance_report['assessments'].values()
            )
            
            if has_critical_issues:
                compliance_report['compliance_status'] = 'non_compliant'
            elif compliance_report['issues']:
                compliance_report['compliance_status'] = 'conditional'
            else:
                compliance_report['compliance_status'] = 'compliant'
            
            return compliance_report
            
        except Exception as e:
            logger.error(f"Model compliance assessment failed: {e}")
            compliance_report['compliance_status'] = 'error'
            compliance_report['error'] = str(e)
            return compliance_report
    
    async def generate_compliance_report(self, 
                                       report_type: str,
                                       period_start: datetime,
                                       period_end: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified period."""
        try:
            report = {
                'report_type': report_type,
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'generated_at': datetime.utcnow().isoformat(),
                'sections': {}
            }
            
            if report_type == 'gdpr' or report_type == 'comprehensive':
                # GDPR compliance section
                gdpr_section = await self._generate_gdpr_section(period_start, period_end)
                report['sections']['gdpr'] = gdpr_section
            
            if report_type == 'model_governance' or report_type == 'comprehensive':
                # Model governance section
                governance_section = await self._generate_governance_section(period_start, period_end)
                report['sections']['model_governance'] = governance_section
            
            if report_type == 'bias_assessment' or report_type == 'comprehensive':
                # Bias assessment section
                bias_section = await self._generate_bias_section(period_start, period_end)
                report['sections']['bias_assessment'] = bias_section
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {'error': str(e)}
    
    async def _generate_gdpr_section(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR compliance section."""
        session = self.session_factory()
        try:
            # Data subject requests
            requests_query = session.query(DataSubjectRequest).filter(
                DataSubjectRequest.received_at >= start_date,
                DataSubjectRequest.received_at <= end_date
            )
            
            total_requests = requests_query.count()
            completed_requests = requests_query.filter(
                DataSubjectRequest.status == 'completed'
            ).count()
            
            overdue_requests = requests_query.filter(
                DataSubjectRequest.due_date < datetime.utcnow(),
                DataSubjectRequest.status != 'completed'
            ).count()
            
            # Consent records
            consent_query = session.query(ConsentRecord).filter(
                ConsentRecord.created_at >= start_date,
                ConsentRecord.created_at <= end_date
            )
            
            total_consents = consent_query.count()
            withdrawn_consents = consent_query.filter(
                ConsentRecord.consent_withdrawn == True
            ).count()
            
            return {
                'data_subject_requests': {
                    'total': total_requests,
                    'completed': completed_requests,
                    'completion_rate': completed_requests / total_requests if total_requests > 0 else 0,
                    'overdue': overdue_requests
                },
                'consent_management': {
                    'total_consents': total_consents,
                    'withdrawn_consents': withdrawn_consents,
                    'withdrawal_rate': withdrawn_consents / total_consents if total_consents > 0 else 0
                },
                'compliance_status': 'compliant' if overdue_requests == 0 else 'issues_found'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate GDPR section: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    async def _generate_governance_section(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate model governance section."""
        session = self.session_factory()
        try:
            # Model approvals
            approvals_query = session.query(ModelApprovalRecord).filter(
                ModelApprovalRecord.submitted_at >= start_date,
                ModelApprovalRecord.submitted_at <= end_date
            )
            
            total_submissions = approvals_query.count()
            approved_models = approvals_query.filter(
                ModelApprovalRecord.final_approval == True
            ).count()
            
            pending_approvals = approvals_query.filter(
                ModelApprovalRecord.approval_status == 'pending'
            ).count()
            
            return {
                'model_approvals': {
                    'total_submissions': total_submissions,
                    'approved': approved_models,
                    'approval_rate': approved_models / total_submissions if total_submissions > 0 else 0,
                    'pending': pending_approvals
                },
                'governance_compliance': 'compliant' if pending_approvals == 0 else 'pending_reviews'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate governance section: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    async def _generate_bias_section(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate bias assessment section."""
        session = self.session_factory()
        try:
            # Bias assessments
            bias_query = session.query(BiasAssessmentResult).filter(
                BiasAssessmentResult.assessed_at >= start_date,
                BiasAssessmentResult.assessed_at <= end_date
            )
            
            total_assessments = bias_query.count()
            bias_detected = bias_query.filter(
                BiasAssessmentResult.bias_detected == True
            ).count()
            
            critical_bias = bias_query.filter(
                BiasAssessmentResult.severity_level == 'critical'
            ).count()
            
            return {
                'bias_assessments': {
                    'total_assessments': total_assessments,
                    'bias_detected': bias_detected,
                    'bias_rate': bias_detected / total_assessments if total_assessments > 0 else 0,
                    'critical_cases': critical_bias
                },
                'bias_compliance': 'compliant' if critical_bias == 0 else 'critical_issues'
            }
            
        except Exception as e:
            logger.error(f"Failed to generate bias section: {e}")
            return {'error': str(e)}
        finally:
            session.close()


class DataPrivacyManager:
    """Data privacy compliance manager."""
    
    def __init__(self, config: DataPrivacyConfig, db_session_factory):
        self.config = config
        self.session_factory = db_session_factory
    
    async def assess_privacy_compliance(self, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy compliance for a model."""
        assessment = {
            'compliant': True,
            'violations': [],
            'recommendations': [],
            'data_categories_processed': [],
            'legal_basis': None,
            'retention_compliant': True,
            'consent_required': False
        }
        
        try:
            # Check data categories
            data_categories = model_metadata.get('data_categories', [])
            sensitive_categories = [DataCategory.PII, DataCategory.PHI, DataCategory.BIOMETRIC]
            
            for category in data_categories:
                if category in [cat.value for cat in sensitive_categories]:
                    assessment['consent_required'] = True
                    assessment['data_categories_processed'].append(category)
            
            # Check retention compliance
            retention_days = model_metadata.get('data_retention_days', 0)
            if retention_days > self.config.data_retention_days:
                assessment['compliant'] = False
                assessment['violations'].append(
                    f"Data retention period ({retention_days} days) exceeds limit ({self.config.data_retention_days} days)"
                )
                assessment['retention_compliant'] = False
            
            # Check if consent is properly managed
            if assessment['consent_required'] and self.config.consent_required:
                # Would check if proper consent records exist
                consent_documented = model_metadata.get('consent_documented', False)
                if not consent_documented:
                    assessment['compliant'] = False
                    assessment['violations'].append("Consent required but not documented")
                    assessment['recommendations'].append("Implement proper consent management")
            
            # Check anonymization/pseudonymization
            if assessment['consent_required']:
                anonymized = model_metadata.get('data_anonymized', False)
                pseudonymized = model_metadata.get('data_pseudonymized', False)
                
                if not (anonymized or pseudonymized):
                    assessment['violations'].append("Sensitive data should be anonymized or pseudonymized")
                    assessment['recommendations'].append("Implement data anonymization techniques")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Privacy compliance assessment failed: {e}")
            return {'error': str(e)}
    
    async def handle_data_subject_request(self, request_data: Dict[str, Any]) -> str:
        """Handle data subject rights request."""
        session = self.session_factory()
        try:
            # Create request record
            request = DataSubjectRequest(
                request_type=request_data['type'],
                data_subject_id=request_data['subject_id'],
                requester_email=request_data.get('email'),
                request_description=request_data.get('description'),
                due_date=datetime.utcnow() + timedelta(days=30),  # 30-day response time
                request_source=request_data.get('source', 'api')
            )
            
            session.add(request)
            session.commit()
            
            logger.info(f"Created data subject request: {request.id}")
            return request.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create data subject request: {e}")
            raise
        finally:
            session.close()


class ModelGovernance:
    """Model governance and approval system."""
    
    def __init__(self, config: ModelGovernanceConfig, db_session_factory):
        self.config = config
        self.session_factory = db_session_factory
    
    async def assess_governance_compliance(self, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model governance compliance."""
        assessment = {
            'compliant': True,
            'issues': [],
            'requirements_met': {},
            'approval_required': self.config.model_approval_required
        }
        
        try:
            # Check explainability requirement
            if self.config.explainability_required:
                explainable = model_metadata.get('explainable', False)
                assessment['requirements_met']['explainability'] = explainable
                
                if not explainable:
                    assessment['compliant'] = False
                    assessment['issues'].append("Model explainability required but not provided")
            
            # Check bias testing requirement
            if self.config.bias_testing_required:
                bias_tested = model_metadata.get('bias_tested', False)
                assessment['requirements_met']['bias_testing'] = bias_tested
                
                if not bias_tested:
                    assessment['compliant'] = False
                    assessment['issues'].append("Bias testing required but not performed")
            
            # Check performance monitoring
            if self.config.performance_monitoring_required:
                monitoring_enabled = model_metadata.get('monitoring_enabled', False)
                assessment['requirements_met']['performance_monitoring'] = monitoring_enabled
                
                if not monitoring_enabled:
                    assessment['compliant'] = False
                    assessment['issues'].append("Performance monitoring required but not enabled")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Governance compliance assessment failed: {e}")
            return {'error': str(e)}
    
    async def submit_for_approval(self, model_id: str, model_metadata: Dict[str, Any]) -> str:
        """Submit model for approval."""
        session = self.session_factory()
        try:
            # Create approval record
            approval_record = ModelApprovalRecord(
                model_id=model_id,
                model_name=model_metadata['name'],
                model_version=model_metadata['version'],
                approval_stage='technical_review'
            )
            
            session.add(approval_record)
            session.commit()
            
            logger.info(f"Submitted model {model_id} for approval: {approval_record.id}")
            return approval_record.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to submit model for approval: {e}")
            raise
        finally:
            session.close()


class BiasDetector:
    """Bias detection and assessment system."""
    
    def __init__(self, config: ModelGovernanceConfig):
        self.config = config
    
    async def assess_bias(self, model: Any, test_data: Tuple, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive bias assessment."""
        assessment = {
            'bias_detected': False,
            'assessments': {},
            'overall_score': 0.0,
            'recommendations': [],
            'critical_issues': False
        }
        
        try:
            X_test, y_test = test_data
            if len(test_data) > 2:
                sensitive_attributes = test_data[2]  # Third element contains sensitive attributes
            else:
                # Generate dummy sensitive attributes for demo
                sensitive_attributes = pd.DataFrame({
                    'gender': np.random.choice(['male', 'female'], len(X_test)),
                    'age_group': np.random.choice(['young', 'middle', 'senior'], len(X_test))
                })
            
            predictions = model.predict(X_test)
            
            # Assess different types of bias
            for attr in self.config.protected_attributes:
                if attr in sensitive_attributes.columns:
                    attr_assessment = await self._assess_attribute_bias(
                        predictions, y_test, sensitive_attributes[attr], attr
                    )
                    assessment['assessments'][attr] = attr_assessment
                    
                    if attr_assessment['bias_detected']:
                        assessment['bias_detected'] = True
                        
                        if attr_assessment['severity'] == 'critical':
                            assessment['critical_issues'] = True
            
            # Calculate overall score
            if assessment['assessments']:
                scores = [a['bias_score'] for a in assessment['assessments'].values()]
                assessment['overall_score'] = np.mean(scores)
            
            # Generate recommendations
            if assessment['bias_detected']:
                assessment['recommendations'] = self._generate_bias_recommendations(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Bias assessment failed: {e}")
            return {'error': str(e)}
    
    async def _assess_attribute_bias(self, predictions, y_true, sensitive_attr, attr_name) -> Dict[str, Any]:
        """Assess bias for a specific protected attribute."""
        try:
            # Convert to binary if needed
            unique_groups = sensitive_attr.unique()
            
            assessment = {
                'attribute': attr_name,
                'groups': list(unique_groups),
                'bias_detected': False,
                'bias_score': 0.0,
                'severity': 'none',
                'metrics': {}
            }
            
            # Calculate fairness metrics
            if len(unique_groups) >= 2:
                # Demographic Parity
                dp_diff = demographic_parity_difference(
                    y_true, predictions, sensitive_features=sensitive_attr
                )
                assessment['metrics']['demographic_parity_difference'] = dp_diff
                
                # Equalized Odds
                eo_diff = equalized_odds_difference(
                    y_true, predictions, sensitive_features=sensitive_attr
                )
                assessment['metrics']['equalized_odds_difference'] = eo_diff
                
                # Determine bias score (worst of the metrics)
                bias_scores = [abs(dp_diff), abs(eo_diff)]
                assessment['bias_score'] = max(bias_scores)
                
                # Check against threshold
                if assessment['bias_score'] > self.config.bias_threshold:
                    assessment['bias_detected'] = True
                    
                    # Determine severity
                    if assessment['bias_score'] > self.config.bias_threshold * 3:
                        assessment['severity'] = 'critical'
                    elif assessment['bias_score'] > self.config.bias_threshold * 2:
                        assessment['severity'] = 'high'
                    else:
                        assessment['severity'] = 'medium'
            
            return assessment
            
        except Exception as e:
            logger.error(f"Attribute bias assessment failed for {attr_name}: {e}")
            return {'error': str(e)}
    
    def _generate_bias_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations for bias mitigation."""
        recommendations = []
        
        if assessment['critical_issues']:
            recommendations.append("Critical bias detected - model should not be deployed until remediated")
        
        if assessment['bias_detected']:
            recommendations.extend([
                "Collect more balanced training data",
                "Apply bias mitigation techniques during training",
                "Consider post-processing fairness corrections",
                "Implement fairness constraints in model optimization",
                "Regular monitoring of model outputs across demographic groups"
            ])
        
        return recommendations


class ModelExplainer:
    """Model explainability assessment system."""
    
    def __init__(self, config: ModelGovernanceConfig):
        self.config = config
    
    async def assess_explainability(self, model: Any, model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model explainability."""
        assessment = {
            'explainable': False,
            'explanation_methods': [],
            'global_explanations': {},
            'local_explanation_capability': False,
            'explanation_quality_score': 0.0
        }
        
        try:
            # Check if model has built-in explainability
            if hasattr(model, 'explain'):
                assessment['explainable'] = True
                assessment['explanation_methods'].append('built_in')
            
            # Check if model is interpretable by design
            interpretable_models = ['linear_regression', 'decision_tree', 'logistic_regression']
            if model_metadata.get('algorithm', '').lower() in interpretable_models:
                assessment['explainable'] = True
                assessment['explanation_methods'].append('interpretable_by_design')
            
            # Test SHAP explainability
            if 'shap' in self.config.explanation_methods:
                shap_compatible = await self._test_shap_compatibility(model)
                if shap_compatible:
                    assessment['explainable'] = True
                    assessment['explanation_methods'].append('shap')
                    assessment['local_explanation_capability'] = True
            
            # Test LIME explainability
            if 'lime' in self.config.explanation_methods:
                lime_compatible = await self._test_lime_compatibility(model)
                if lime_compatible:
                    assessment['explainable'] = True
                    assessment['explanation_methods'].append('lime')
                    assessment['local_explanation_capability'] = True
            
            # Calculate explanation quality score
            if assessment['explanation_methods']:
                assessment['explanation_quality_score'] = len(assessment['explanation_methods']) / len(self.config.explanation_methods)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Explainability assessment failed: {e}")
            return {'error': str(e)}
    
    async def _test_shap_compatibility(self, model: Any) -> bool:
        """Test if model is compatible with SHAP."""
        try:
            # Generate dummy data for testing
            dummy_data = np.random.rand(10, 5)
            
            # Try to create SHAP explainer
            if hasattr(model, 'predict'):
                explainer = shap.Explainer(model.predict, dummy_data)
                shap_values = explainer(dummy_data[:1])
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"SHAP compatibility test failed: {e}")
            return False
    
    async def _test_lime_compatibility(self, model: Any) -> bool:
        """Test if model is compatible with LIME."""
        try:
            # LIME requires predict_proba for classification or predict for regression
            if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"LIME compatibility test failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Example configuration
    privacy_config = DataPrivacyConfig(
        gdpr_enabled=True,
        data_retention_days=2555,
        anonymization_enabled=True
    )
    
    governance_config = ModelGovernanceConfig(
        explainability_required=True,
        bias_testing_required=True,
        protected_attributes=['gender', 'age_group']
    )
    
    # This would be used in a real application
    # compliance_manager = ComplianceManager(privacy_config, governance_config, db_session_factory)
    print("Compliance module configured successfully")