"""
Advanced Alerting System for Safe RL Production Deployment.

This module provides comprehensive alerting capabilities including:
- Multi-channel alert delivery (Slack, email, PagerDuty, webhooks)
- Alert severity levels and escalation
- Alert suppression and deduplication
- Custom alert rules and conditions
"""

import asyncio
import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import requests
import redis
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertContext:
    """Context information for an alert."""
    metric_name: str
    current_value: float
    threshold: float
    condition: str
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Represents an alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    context: AlertContext
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None
    escalation_level: int = 0
    notification_count: int = 0
    
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.FIRING and not self.is_suppressed()
    
    def is_suppressed(self) -> bool:
        """Check if alert is currently suppressed."""
        if self.suppressed_until is None:
            return False
        return datetime.now() < self.suppressed_until
    
    def suppress(self, duration_minutes: int):
        """Suppress alert for specified duration."""
        self.suppressed_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()
    
    def resolve(self):
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
    
    def escalate(self):
        """Escalate alert to next level."""
        self.escalation_level += 1
        self.updated_at = datetime.now()
    
    def increment_notifications(self):
        """Increment notification counter."""
        self.notification_count += 1
        self.updated_at = datetime.now()


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit', 10)  # messages per minute
        self.last_sent = {}  # Rate limiting tracking
    
    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        pass
    
    def can_send(self, alert: Alert) -> bool:
        """Check if notification can be sent (rate limiting)."""
        if not self.enabled:
            return False
        
        now = time.time()
        minute_window = now // 60
        
        if minute_window not in self.last_sent:
            self.last_sent[minute_window] = 0
        
        # Clean old entries
        for window in list(self.last_sent.keys()):
            if window < minute_window - 1:
                del self.last_sent[window]
        
        if self.last_sent[minute_window] >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for {self.name}")
            return False
        
        return True
    
    def record_sent(self):
        """Record that a notification was sent."""
        minute_window = time.time() // 60
        if minute_window not in self.last_sent:
            self.last_sent[minute_window] = 0
        self.last_sent[minute_window] += 1


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#alerts')
        self.username = config.get('username', 'SafeRL Bot')
        self.icon_emoji = config.get('icon_emoji', ':robot_face:')
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        if not self.can_send(alert) or not self.webhook_url:
            return False
        
        try:
            color = self._get_color_for_severity(alert.severity)
            
            payload = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "attachments": [
                    {
                        "color": color,
                        "title": f"🚨 {alert.rule_name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": alert.context.metric_name,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": f"{alert.context.current_value:.4f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.context.threshold:.4f}",
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                                "short": False
                            }
                        ],
                        "footer": "Safe RL Monitoring",
                        "ts": int(alert.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.record_sent()
            logger.info(f"Slack notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get color code for severity level."""
        colors = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.INFO: "good"
        }
        return colors.get(severity, "warning")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        if not self.can_send(alert) or not self.smtp_server or not self.to_emails:
            return False
        
        try:
            subject = f"[{alert.severity.value.upper()}] SafeRL Alert: {alert.rule_name}"
            body = self._create_email_body(alert)
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            self.record_sent()
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body."""
        severity_color = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.INFO: "#17a2b8"
        }.get(alert.severity, "#6c757d")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <div style="border: 2px solid {severity_color}; border-radius: 8px; padding: 20px;">
                <h2 style="color: {severity_color}; margin-top: 0;">
                    🚨 SafeRL Alert: {alert.rule_name}
                </h2>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
                    <strong>Message:</strong> {alert.message}
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Severity:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: {severity_color};">
                            <strong>{alert.severity.value.upper()}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Metric:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{alert.context.metric_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Current Value:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{alert.context.current_value:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Threshold:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{alert.context.threshold:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;"><strong>Time:</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}</td>
                    </tr>
                </table>
                
                <div style="margin-top: 20px; font-size: 12px; color: #6c757d;">
                    Alert ID: {alert.id}<br>
                    Generated by Safe RL Monitoring System
                </div>
            </div>
        </body>
        </html>
        """


class PagerDutyNotificationChannel(NotificationChannel):
    """PagerDuty notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.integration_key = config.get('integration_key')
        self.api_url = config.get('api_url', 'https://events.pagerduty.com/v2/enqueue')
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        if not self.can_send(alert) or not self.integration_key:
            return False
        
        try:
            event_action = "trigger" if alert.status == AlertStatus.FIRING else "resolve"
            
            payload = {
                "routing_key": self.integration_key,
                "event_action": event_action,
                "dedup_key": f"saferl_{alert.rule_name}",
                "payload": {
                    "summary": f"SafeRL Alert: {alert.message}",
                    "severity": self._map_severity(alert.severity),
                    "source": "Safe RL Monitoring",
                    "component": alert.context.metric_name,
                    "group": "Safe RL",
                    "class": alert.rule_name,
                    "custom_details": {
                        "metric_name": alert.context.metric_name,
                        "current_value": alert.context.current_value,
                        "threshold": alert.context.threshold,
                        "condition": alert.context.condition,
                        "labels": alert.context.labels,
                        "alert_id": alert.id
                    }
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.record_sent()
            logger.info(f"PagerDuty notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False
    
    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map alert severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.INFO: "info"
        }
        return mapping.get(severity, "info")


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.url = config.get('url')
        self.method = config.get('method', 'POST')
        self.headers = config.get('headers', {})
        self.auth = config.get('auth')
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        if not self.can_send(alert) or not self.url:
            return False
        
        try:
            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "context": {
                    "metric_name": alert.context.metric_name,
                    "current_value": alert.context.current_value,
                    "threshold": alert.context.threshold,
                    "condition": alert.context.condition,
                    "labels": alert.context.labels,
                    "timestamp": alert.created_at.isoformat()
                },
                "escalation_level": alert.escalation_level,
                "notification_count": alert.notification_count
            }
            
            kwargs = {
                'json': payload,
                'headers': self.headers,
                'timeout': 10
            }
            
            if self.auth:
                kwargs['auth'] = (self.auth.get('username'), self.auth.get('password'))
            
            response = requests.request(self.method, self.url, **kwargs)
            response.raise_for_status()
            
            self.record_sent()
            logger.info(f"Webhook notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


@dataclass
class EscalationPolicy:
    """Defines escalation policy for alerts."""
    name: str
    levels: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_channels_for_level(self, level: int) -> List[str]:
        """Get notification channels for escalation level."""
        if level >= len(self.levels):
            return []
        return self.levels[level].get('channels', [])
    
    def get_wait_time_for_level(self, level: int) -> int:
        """Get wait time in minutes before escalating to next level."""
        if level >= len(self.levels):
            return 0
        return self.levels[level].get('wait_minutes', 15)


class AlertManager:
    """Central alert management system."""
    
    def __init__(self, config_file: Optional[str] = None, redis_client: Optional[redis.Redis] = None):
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.redis_client = redis_client
        self.suppression_rules: List[Dict[str, Any]] = []
        self.running = False
        self.escalation_task = None
        
        if config_file:
            self.load_config(config_file)
        
        logger.info("AlertManager initialized")
    
    def load_config(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load notification channels
            for name, channel_config in config.get('notification_channels', {}).items():
                channel_type = channel_config.get('type')
                
                if channel_type == 'slack':
                    channel = SlackNotificationChannel(name, channel_config)
                elif channel_type == 'email':
                    channel = EmailNotificationChannel(name, channel_config)
                elif channel_type == 'pagerduty':
                    channel = PagerDutyNotificationChannel(name, channel_config)
                elif channel_type == 'webhook':
                    channel = WebhookNotificationChannel(name, channel_config)
                else:
                    logger.warning(f"Unknown channel type: {channel_type}")
                    continue
                
                self.notification_channels[name] = channel
                logger.info(f"Loaded notification channel: {name}")
            
            # Load escalation policies
            for name, policy_config in config.get('escalation_policies', {}).items():
                policy = EscalationPolicy(name=name, levels=policy_config.get('levels', []))
                self.escalation_policies[name] = policy
                logger.info(f"Loaded escalation policy: {name}")
            
            # Load suppression rules
            self.suppression_rules = config.get('suppression_rules', [])
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
    
    def add_notification_channel(self, name: str, channel: NotificationChannel):
        """Add a notification channel."""
        self.notification_channels[name] = channel
        logger.info(f"Added notification channel: {name}")
    
    def add_escalation_policy(self, name: str, policy: EscalationPolicy):
        """Add an escalation policy."""
        self.escalation_policies[name] = policy
        logger.info(f"Added escalation policy: {name}")
    
    async def create_alert(self, rule_name: str, message: str, severity: AlertSeverity,
                          context: AlertContext, escalation_policy: str = None) -> Alert:
        """Create a new alert."""
        alert_id = self._generate_alert_id(rule_name, context)
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            existing_alert = self.active_alerts[alert_id]
            existing_alert.updated_at = datetime.now()
            existing_alert.context = context  # Update with latest context
            return existing_alert
        
        # Check suppression rules
        if self._is_suppressed(rule_name, context):
            logger.info(f"Alert {alert_id} is suppressed by rules")
            return None
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            rule_name=rule_name,
            severity=severity,
            status=AlertStatus.FIRING,
            message=message,
            context=context
        )
        
        self.active_alerts[alert_id] = alert
        
        # Store in Redis
        if self.redis_client:
            self._store_alert_in_redis(alert)
        
        # Send initial notifications
        await self._send_notifications(alert, escalation_policy)
        
        logger.info(f"Created alert: {alert_id}")
        return alert
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return
        
        alert = self.active_alerts[alert_id]
        alert.resolve()
        
        # Update in Redis
        if self.redis_client:
            self._store_alert_in_redis(alert)
        
        # Send resolution notifications
        await self._send_resolution_notifications(alert)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Resolved alert: {alert_id}")
    
    def suppress_alert(self, alert_id: str, duration_minutes: int):
        """Suppress an alert for specified duration."""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return
        
        alert = self.active_alerts[alert_id]
        alert.suppress(duration_minutes)
        
        # Update in Redis
        if self.redis_client:
            self._store_alert_in_redis(alert)
        
        logger.info(f"Suppressed alert {alert_id} for {duration_minutes} minutes")
    
    def add_suppression_rule(self, rule: Dict[str, Any]):
        """Add a suppression rule."""
        self.suppression_rules.append(rule)
        logger.info(f"Added suppression rule: {rule}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return [alert for alert in self.active_alerts.values() if alert.is_active()]
    
    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        return self.active_alerts.get(alert_id)
    
    def start_escalation_manager(self):
        """Start the escalation manager."""
        if self.running:
            return
        
        self.running = True
        
        async def escalation_loop():
            while self.running:
                try:
                    await self._process_escalations()
                except Exception as e:
                    logger.error(f"Error in escalation loop: {e}")
                
                await asyncio.sleep(60)  # Check every minute
        
        self.escalation_task = asyncio.create_task(escalation_loop())
        logger.info("Started escalation manager")
    
    def stop_escalation_manager(self):
        """Stop the escalation manager."""
        self.running = False
        if self.escalation_task:
            self.escalation_task.cancel()
        logger.info("Stopped escalation manager")
    
    def _generate_alert_id(self, rule_name: str, context: AlertContext) -> str:
        """Generate unique alert ID."""
        # Create ID based on rule name and key context labels
        key_labels = sorted(context.labels.items())
        labels_str = ','.join([f"{k}={v}" for k, v in key_labels])
        return f"{rule_name}_{context.metric_name}_{labels_str}"
    
    def _is_suppressed(self, rule_name: str, context: AlertContext) -> bool:
        """Check if alert should be suppressed."""
        for rule in self.suppression_rules:
            if self._matches_suppression_rule(rule, rule_name, context):
                return True
        return False
    
    def _matches_suppression_rule(self, rule: Dict[str, Any], rule_name: str, context: AlertContext) -> bool:
        """Check if alert matches suppression rule."""
        # Match by rule name pattern
        if 'rule_pattern' in rule:
            import re
            if not re.match(rule['rule_pattern'], rule_name):
                return False
        
        # Match by labels
        if 'labels' in rule:
            for key, value in rule['labels'].items():
                if context.labels.get(key) != value:
                    return False
        
        # Match by time window
        if 'time_windows' in rule:
            current_time = datetime.now().time()
            for window in rule['time_windows']:
                start = datetime.strptime(window['start'], '%H:%M').time()
                end = datetime.strptime(window['end'], '%H:%M').time()
                if start <= current_time <= end:
                    return True
        
        return True
    
    async def _send_notifications(self, alert: Alert, escalation_policy: str = None):
        """Send notifications for alert."""
        channels_to_notify = []
        
        if escalation_policy and escalation_policy in self.escalation_policies:
            policy = self.escalation_policies[escalation_policy]
            channels_to_notify = policy.get_channels_for_level(alert.escalation_level)
        else:
            # Default: notify all channels based on severity
            channels_to_notify = self._get_default_channels_for_severity(alert.severity)
        
        success_count = 0
        for channel_name in channels_to_notify:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]
                try:
                    success = await channel.send_notification(alert)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel_name}: {e}")
        
        alert.increment_notifications()
        logger.info(f"Sent notifications for alert {alert.id} via {success_count} channels")
    
    async def _send_resolution_notifications(self, alert: Alert):
        """Send resolution notifications."""
        # Only send resolution notifications for critical alerts
        if alert.severity != AlertSeverity.CRITICAL:
            return
        
        channels_to_notify = self._get_default_channels_for_severity(alert.severity)
        
        for channel_name in channels_to_notify:
            if channel_name in self.notification_channels:
                channel = self.notification_channels[channel_name]
                try:
                    await channel.send_notification(alert)
                except Exception as e:
                    logger.error(f"Failed to send resolution notification via {channel_name}: {e}")
    
    def _get_default_channels_for_severity(self, severity: AlertSeverity) -> List[str]:
        """Get default notification channels for severity level."""
        if severity == AlertSeverity.CRITICAL:
            return list(self.notification_channels.keys())  # All channels
        elif severity == AlertSeverity.WARNING:
            return [name for name, channel in self.notification_channels.items() 
                   if not isinstance(channel, PagerDutyNotificationChannel)]
        else:  # INFO
            return [name for name, channel in self.notification_channels.items() 
                   if isinstance(channel, SlackNotificationChannel)]
    
    async def _process_escalations(self):
        """Process alert escalations."""
        current_time = datetime.now()
        
        for alert in self.active_alerts.values():
            if not alert.is_active():
                continue
            
            # Find escalation policy for this alert
            policy = None
            for policy_name, escalation_policy in self.escalation_policies.items():
                if policy_name in alert.context.labels.get('escalation_policy', policy_name):
                    policy = escalation_policy
                    break
            
            if not policy:
                continue
            
            # Check if enough time has passed for escalation
            wait_time = policy.get_wait_time_for_level(alert.escalation_level)
            if wait_time <= 0:
                continue
            
            time_since_last_update = (current_time - alert.updated_at).total_seconds() / 60
            
            if time_since_last_update >= wait_time:
                # Escalate alert
                alert.escalate()
                await self._send_notifications(alert, policy.name)
                
                logger.info(f"Escalated alert {alert.id} to level {alert.escalation_level}")
    
    def _store_alert_in_redis(self, alert: Alert):
        """Store alert in Redis."""
        if not self.redis_client:
            return
        
        try:
            alert_data = {
                'id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'updated_at': alert.updated_at.isoformat(),
                'escalation_level': alert.escalation_level,
                'notification_count': alert.notification_count,
                'context': {
                    'metric_name': alert.context.metric_name,
                    'current_value': alert.context.current_value,
                    'threshold': alert.context.threshold,
                    'condition': alert.context.condition,
                    'labels': alert.context.labels,
                    'timestamp': alert.context.timestamp.isoformat()
                }
            }
            
            # Store individual alert
            key = f"alert:{alert.id}"
            self.redis_client.setex(key, 86400, json.dumps(alert_data, default=str))
            
            # Add to alert index
            self.redis_client.zadd("alerts:active", {alert.id: time.time()})
            
        except Exception as e:
            logger.error(f"Failed to store alert in Redis: {e}")


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None

def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager

def initialize_alerting(config_file: Optional[str] = None, 
                       redis_client: Optional[redis.Redis] = None) -> AlertManager:
    """Initialize the global alert manager."""
    global _global_alert_manager
    _global_alert_manager = AlertManager(config_file, redis_client)
    return _global_alert_manager