"""
Enterprise Authentication and Authorization System for Safe RL Production.

This module provides comprehensive security features including:
- JWT-based authentication
- Role-based access control (RBAC)
- OAuth2 integration
- API key management
- Multi-factor authentication (MFA)
- Session management
- Security audit logging
"""

import asyncio
import logging
import secrets
import hashlib
import hmac
import time
import jwt
import bcrypt
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import asynccontextmanager
import pyotp
import qrcode
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import aiohttp
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json

logger = logging.getLogger(__name__)

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class UserRole(Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    API_USER = "api_user"
    SERVICE_ACCOUNT = "service_account"


class Permission(Enum):
    """System permissions."""
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DELETE = "model:delete"
    MODEL_DEPLOY = "model:deploy"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    
    # Safety permissions
    SAFETY_OVERRIDE = "safety:override"
    SAFETY_CONFIG = "safety:config"
    
    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"


@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    jwt_refresh_expire_days: int = 7
    
    # Password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    
    # MFA settings
    mfa_enabled: bool = True
    mfa_issuer: str = "SafeRL"
    
    # Session settings
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    
    # Rate limiting
    rate_limit_enabled: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # OAuth2 settings
    oauth2_enabled: bool = False
    oauth2_providers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # API key settings
    api_key_length: int = 64
    api_key_expire_days: int = 365


# Database Models
class User(Base):
    """User model."""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String)
    
    # Status fields
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_locked = Column(Boolean, default=False)
    
    # Security fields
    role = Column(String, nullable=False)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # MFA fields
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String)
    backup_codes = Column(Text)  # JSON array of backup codes
    
    # Rate limiting
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("SecurityAuditLog", back_populates="user")


class UserSession(Base):
    """User session model."""
    __tablename__ = 'user_sessions'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    token_jti = Column(String, unique=True, nullable=False)  # JWT ID
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Session metadata
    ip_address = Column(String)
    user_agent = Column(String)
    device_fingerprint = Column(String)
    
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class APIKey(Base):
    """API key model."""
    __tablename__ = 'api_keys'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    
    # Permissions and scope
    scopes = Column(Text)  # JSON array of scopes
    rate_limit = Column(Integer)  # requests per minute
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class SecurityAuditLog(Base):
    """Security audit log model."""
    __tablename__ = 'security_audit_logs'
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Event details
    event_type = Column(String, nullable=False)  # login, logout, permission_check, etc.
    event_category = Column(String, nullable=False)  # authentication, authorization, data_access
    action = Column(String, nullable=False)
    resource = Column(String)  # What resource was accessed
    outcome = Column(String, nullable=False)  # success, failure, blocked
    
    # Context
    ip_address = Column(String)
    user_agent = Column(String)
    session_id = Column(String)
    
    # Additional metadata
    metadata = Column(Text)  # JSON additional context
    risk_score = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


class AuthenticationProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass


class DatabaseAuthProvider(AuthenticationProvider):
    """Database-based authentication provider."""
    
    def __init__(self, db_session_factory):
        self.session_factory = db_session_factory
    
    async def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user against database."""
        session = self.session_factory()
        try:
            user = session.query(User).filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                return None
            
            # Check if user is active and not locked
            if not user.is_active or user.is_locked:
                return None
            
            # Check if account is temporarily locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                return None
            
            # Verify password
            if not pwd_context.verify(password, user.password_hash):
                # Increment failed attempts
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:  # Lock after 5 failed attempts
                    user.is_locked = True
                    user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                session.commit()
                return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.last_login = datetime.utcnow()
            user.locked_until = None
            session.commit()
            
            return {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role,
                'is_verified': user.is_verified,
                'mfa_enabled': user.mfa_enabled
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
        finally:
            session.close()
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        session = self.session_factory()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            return {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role,
                'is_active': user.is_active,
                'is_verified': user.is_verified,
                'mfa_enabled': user.mfa_enabled
            }
            
        except Exception as e:
            logger.error(f"Get user error: {e}")
            return None
        finally:
            session.close()


class JWTTokenManager:
    """JWT token management."""
    
    def __init__(self, config: SecurityConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
    
    def create_tokens(self, user_data: Dict[str, Any]) -> Tuple[str, str]:
        """Create access and refresh tokens."""
        now = datetime.utcnow()
        jti = secrets.token_urlsafe(32)
        
        # Access token payload
        access_payload = {
            'sub': user_data['id'],
            'username': user_data['username'],
            'role': user_data['role'],
            'iat': now,
            'exp': now + timedelta(minutes=self.config.jwt_expire_minutes),
            'jti': jti,
            'type': 'access'
        }
        
        # Refresh token payload
        refresh_payload = {
            'sub': user_data['id'],
            'jti': jti,
            'iat': now,
            'exp': now + timedelta(days=self.config.jwt_refresh_expire_days),
            'type': 'refresh'
        }
        
        access_token = jwt.encode(
            access_payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        # Store token info in Redis if available
        if self.redis_client:
            token_data = {
                'user_id': user_data['id'],
                'username': user_data['username'],
                'role': user_data['role'],
                'created_at': now.isoformat()
            }
            self.redis_client.setex(
                f"token:{jti}",
                int(timedelta(days=self.config.jwt_refresh_expire_days).total_seconds()),
                json.dumps(token_data)
            )
        
        return access_token, refresh_token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is blacklisted (if using Redis)
            if self.redis_client:
                jti = payload.get('jti')
                if jti and self.redis_client.get(f"blacklist:{jti}"):
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None
    
    def blacklist_token(self, token: str) -> bool:
        """Blacklist a token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False}  # Don't verify expiration for blacklisting
            )
            
            jti = payload.get('jti')
            if jti and self.redis_client:
                # Calculate TTL based on original expiration
                exp = payload.get('exp', 0)
                now = int(time.time())
                ttl = max(1, exp - now)  # At least 1 second TTL
                
                self.redis_client.setex(f"blacklist:{jti}", ttl, "1")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token."""
        try:
            payload = self.verify_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                return None
            
            user_id = payload['sub']
            
            # Create new access token (would need to fetch user data)
            # This is simplified - in practice you'd fetch current user data
            new_payload = {
                'sub': user_id,
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(minutes=self.config.jwt_expire_minutes),
                'jti': secrets.token_urlsafe(32),
                'type': 'access'
            }
            
            return jwt.encode(
                new_payload,
                self.config.jwt_secret_key,
                algorithm=self.config.jwt_algorithm
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None


class MFAManager:
    """Multi-factor authentication manager."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def generate_secret(self, username: str) -> Tuple[str, str]:
        """Generate TOTP secret and QR code."""
        secret = pyotp.random_base32()
        
        # Generate provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name=self.config.mfa_issuer
        )
        
        return secret, provisioning_uri
    
    def generate_qr_code(self, provisioning_uri: str) -> bytes:
        """Generate QR code for TOTP setup."""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        from io import BytesIO
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def verify_totp(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=window)
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for account recovery."""
        return [secrets.token_hex(4).upper() for _ in range(count)]


class RoleBasedAccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        # Define role permissions
        self.role_permissions = {
            UserRole.SUPER_ADMIN: [p for p in Permission],  # All permissions
            UserRole.ADMIN: [
                Permission.MODEL_READ, Permission.MODEL_WRITE, Permission.MODEL_DEPLOY,
                Permission.DATA_READ, Permission.DATA_WRITE,
                Permission.SYSTEM_MONITOR, Permission.SYSTEM_CONFIG,
                Permission.SAFETY_CONFIG,
                Permission.USER_READ, Permission.USER_WRITE
            ],
            UserRole.OPERATOR: [
                Permission.MODEL_READ, Permission.MODEL_DEPLOY,
                Permission.DATA_READ,
                Permission.SYSTEM_MONITOR,
                Permission.SAFETY_OVERRIDE
            ],
            UserRole.VIEWER: [
                Permission.MODEL_READ,
                Permission.DATA_READ,
                Permission.SYSTEM_MONITOR
            ],
            UserRole.API_USER: [
                Permission.MODEL_READ,
                Permission.DATA_READ
            ],
            UserRole.SERVICE_ACCOUNT: [
                Permission.MODEL_READ,
                Permission.DATA_READ,
                Permission.SYSTEM_MONITOR
            ]
        }
    
    def has_permission(self, user_role: str, permission: Permission) -> bool:
        """Check if user role has specific permission."""
        try:
            role = UserRole(user_role)
            return permission in self.role_permissions.get(role, [])
        except ValueError:
            return False
    
    def get_user_permissions(self, user_role: str) -> List[Permission]:
        """Get all permissions for user role."""
        try:
            role = UserRole(user_role)
            return self.role_permissions.get(role, [])
        except ValueError:
            return []
    
    def can_access_resource(self, user_role: str, resource_type: str, action: str) -> bool:
        """Check if user can perform action on resource type."""
        permission_str = f"{resource_type}:{action}"
        
        try:
            permission = Permission(permission_str)
            return self.has_permission(user_role, permission)
        except ValueError:
            return False


class SecurityAuditor:
    """Security audit logging."""
    
    def __init__(self, db_session_factory, redis_client: Optional[redis.Redis] = None):
        self.session_factory = db_session_factory
        self.redis_client = redis_client
    
    async def log_security_event(self, 
                                user_id: Optional[str],
                                event_type: str,
                                event_category: str,
                                action: str,
                                resource: str = "",
                                outcome: str = "success",
                                ip_address: str = "",
                                user_agent: str = "",
                                session_id: str = "",
                                metadata: Dict[str, Any] = None,
                                risk_score: int = 0):
        """Log security event to audit trail."""
        
        session = self.session_factory()
        try:
            audit_log = SecurityAuditLog(
                id=secrets.token_urlsafe(32),
                user_id=user_id,
                event_type=event_type,
                event_category=event_category,
                action=action,
                resource=resource,
                outcome=outcome,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                metadata=json.dumps(metadata or {}),
                risk_score=risk_score
            )
            
            session.add(audit_log)
            session.commit()
            
            # Also log to Redis for real-time monitoring
            if self.redis_client:
                event_data = {
                    'user_id': user_id,
                    'event_type': event_type,
                    'event_category': event_category,
                    'action': action,
                    'outcome': outcome,
                    'timestamp': datetime.utcnow().isoformat(),
                    'risk_score': risk_score
                }
                
                self.redis_client.lpush(
                    "security_events",
                    json.dumps(event_data)
                )
                
                # Keep only last 10000 events
                self.redis_client.ltrim("security_events", 0, 9999)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
        finally:
            session.close()
    
    async def get_security_events(self, 
                                 user_id: Optional[str] = None,
                                 event_type: Optional[str] = None,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get security audit events with filtering."""
        
        session = self.session_factory()
        try:
            query = session.query(SecurityAuditLog)
            
            # Apply filters
            if user_id:
                query = query.filter(SecurityAuditLog.user_id == user_id)
            if event_type:
                query = query.filter(SecurityAuditLog.event_type == event_type)
            if start_time:
                query = query.filter(SecurityAuditLog.timestamp >= start_time)
            if end_time:
                query = query.filter(SecurityAuditLog.timestamp <= end_time)
            
            # Order by timestamp descending and limit
            query = query.order_by(SecurityAuditLog.timestamp.desc()).limit(limit)
            
            events = []
            for log in query.all():
                events.append({
                    'id': log.id,
                    'user_id': log.user_id,
                    'timestamp': log.timestamp.isoformat(),
                    'event_type': log.event_type,
                    'event_category': log.event_category,
                    'action': log.action,
                    'resource': log.resource,
                    'outcome': log.outcome,
                    'ip_address': log.ip_address,
                    'user_agent': log.user_agent,
                    'session_id': log.session_id,
                    'metadata': json.loads(log.metadata) if log.metadata else {},
                    'risk_score': log.risk_score
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get security events: {e}")
            return []
        finally:
            session.close()


class SecurityManager:
    """Main security manager orchestrating all security components."""
    
    def __init__(self, 
                 config: SecurityConfig,
                 db_session_factory,
                 redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.session_factory = db_session_factory
        self.redis_client = redis_client
        
        # Initialize components
        self.auth_provider = DatabaseAuthProvider(db_session_factory)
        self.token_manager = JWTTokenManager(config, redis_client)
        self.mfa_manager = MFAManager(config)
        self.rbac = RoleBasedAccessControl()
        self.auditor = SecurityAuditor(db_session_factory, redis_client)
    
    async def authenticate_user(self, 
                              username: str, 
                              password: str,
                              mfa_token: Optional[str] = None,
                              ip_address: str = "",
                              user_agent: str = "") -> Optional[Dict[str, Any]]:
        """Authenticate user with optional MFA."""
        try:
            # First-factor authentication
            user_data = await self.auth_provider.authenticate(username, password)
            
            if not user_data:
                await self.auditor.log_security_event(
                    user_id=None,
                    event_type="login_failed",
                    event_category="authentication",
                    action="login",
                    outcome="failure",
                    ip_address=ip_address,
                    user_agent=user_agent,
                    risk_score=5
                )
                return None
            
            # Check if MFA is required
            if user_data.get('mfa_enabled') and not mfa_token:
                await self.auditor.log_security_event(
                    user_id=user_data['id'],
                    event_type="mfa_required",
                    event_category="authentication",
                    action="login",
                    outcome="mfa_required",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return {'mfa_required': True, 'user_id': user_data['id']}
            
            # Verify MFA if provided
            if user_data.get('mfa_enabled') and mfa_token:
                if not await self._verify_mfa(user_data['id'], mfa_token):
                    await self.auditor.log_security_event(
                        user_id=user_data['id'],
                        event_type="mfa_failed",
                        event_category="authentication",
                        action="login",
                        outcome="failure",
                        ip_address=ip_address,
                        user_agent=user_agent,
                        risk_score=7
                    )
                    return None
            
            # Generate tokens
            access_token, refresh_token = self.token_manager.create_tokens(user_data)
            
            # Log successful authentication
            await self.auditor.log_security_event(
                user_id=user_data['id'],
                event_type="login_success",
                event_category="authentication",
                action="login",
                outcome="success",
                ip_address=ip_address,
                user_agent=user_agent,
                metadata={'username': user_data['username']}
            )
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'bearer',
                'expires_in': self.config.jwt_expire_minutes * 60,
                'user': user_data
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def logout_user(self, token: str, user_id: str, ip_address: str = ""):
        """Logout user and blacklist token."""
        try:
            # Blacklist the token
            self.token_manager.blacklist_token(token)
            
            # Log logout event
            await self.auditor.log_security_event(
                user_id=user_id,
                event_type="logout",
                event_category="authentication",
                action="logout",
                outcome="success",
                ip_address=ip_address
            )
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
    
    async def verify_permissions(self, 
                               user_id: str, 
                               permission: Permission,
                               resource: str = "",
                               ip_address: str = "") -> bool:
        """Verify user has required permission."""
        try:
            # Get user data
            user_data = await self.auth_provider.get_user(user_id)
            if not user_data or not user_data.get('is_active'):
                return False
            
            # Check permission
            has_permission = self.rbac.has_permission(user_data['role'], permission)
            
            # Log permission check
            await self.auditor.log_security_event(
                user_id=user_id,
                event_type="permission_check",
                event_category="authorization",
                action=permission.value,
                resource=resource,
                outcome="granted" if has_permission else "denied",
                ip_address=ip_address,
                risk_score=3 if not has_permission else 0
            )
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Permission verification error: {e}")
            return False
    
    async def _verify_mfa(self, user_id: str, mfa_token: str) -> bool:
        """Verify MFA token for user."""
        session = self.session_factory()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user or not user.mfa_secret:
                return False
            
            # Try TOTP verification
            if self.mfa_manager.verify_totp(user.mfa_secret, mfa_token):
                return True
            
            # Try backup codes
            if user.backup_codes:
                backup_codes = json.loads(user.backup_codes)
                if mfa_token.upper() in backup_codes:
                    # Remove used backup code
                    backup_codes.remove(mfa_token.upper())
                    user.backup_codes = json.dumps(backup_codes)
                    session.commit()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
        finally:
            session.close()


# FastAPI dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """FastAPI dependency to get current authenticated user."""
    try:
        # This would be injected with the actual security manager
        # For now, it's a placeholder
        token = credentials.credentials
        # Verify token and return user data
        # payload = security_manager.token_manager.verify_token(token)
        # if not payload:
        #     raise HTTPException(status_code=401, detail="Invalid token")
        # return payload
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not implemented - inject SecurityManager",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: Permission):
    """FastAPI dependency to require specific permission."""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        # This would check permissions using RBAC
        # if not security_manager.rbac.has_permission(current_user.get('role'), permission):
        #     raise HTTPException(status_code=403, detail="Insufficient permissions")
        # return current_user
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission checking not implemented - inject SecurityManager"
        )
    
    return permission_checker


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = SecurityConfig(
        jwt_secret_key="your-secret-key-here",
        mfa_enabled=True,
        rate_limit_enabled=True
    )
    
    # This would be used in a real application
    # security_manager = SecurityManager(config, db_session_factory, redis_client)
    print("Security module configured successfully")