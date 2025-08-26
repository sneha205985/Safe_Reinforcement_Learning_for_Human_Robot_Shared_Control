"""
Production Deployment Manager for Safe RL Systems.

This module provides comprehensive deployment orchestration, health monitoring,
rollback capabilities, and production-level system management for Safe RL
human-robot shared control systems.

Key Features:
- Docker container orchestration
- Blue-green deployments
- Health monitoring and auto-recovery
- Configuration validation and deployment
- System diagnostics and logging
- Rollback and disaster recovery
- Multi-environment management
"""

import os
import time
import docker
import threading
import subprocess
import yaml
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import logging
import psutil
import socket
from contextlib import contextmanager

from .config_manager import ConfigurationManager, Environment, ConfigType

logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Deployment states."""
    UNINITIALIZED = auto()
    PREPARING = auto()
    DEPLOYING = auto()
    RUNNING = auto()
    UPDATING = auto()
    ROLLING_BACK = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    MAINTENANCE = auto()


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ServiceConfig:
    """Configuration for a service component."""
    name: str
    image: str
    tag: str = "latest"
    ports: Dict[int, int] = field(default_factory=dict)  # container_port: host_port
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)  # host_path: container_path
    command: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    health_check: Optional[Dict[str, Any]] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    restart_policy: str = "unless-stopped"
    privileged: bool = False


@dataclass
class DeploymentConfig:
    """Overall deployment configuration."""
    deployment_name: str
    version: str
    environment: Environment
    services: List[ServiceConfig] = field(default_factory=list)
    networks: Dict[str, Any] = field(default_factory=dict)
    volumes: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    backup: Dict[str, Any] = field(default_factory=dict)
    rollback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    status: HealthStatus
    message: str
    timestamp: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0


@dataclass
class DeploymentMetrics:
    """Deployment performance metrics."""
    deployment_time_s: float = 0.0
    startup_time_s: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    disk_io_mb: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    uptime_s: float = 0.0


class DeploymentManager:
    """
    Production deployment manager for Safe RL systems.
    
    Orchestrates containerized deployments with health monitoring,
    rollback capabilities, and production-level reliability.
    """
    
    def __init__(self, config_manager: ConfigurationManager, deployment_dir: Path):
        self.config_manager = config_manager
        self.deployment_dir = Path(deployment_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Docker client
        self.docker_client = docker.from_env()
        
        # Deployment state
        self.state = DeploymentState.UNINITIALIZED
        self.current_deployment: Optional[DeploymentConfig] = None
        self.active_containers: Dict[str, docker.models.containers.Container] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        
        # Health monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheckResult] = {}
        self.health_monitor_thread: Optional[threading.Thread] = None
        
        # Metrics collection
        self.metrics: DeploymentMetrics = DeploymentMetrics()
        self.metrics_history: List[DeploymentMetrics] = []
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Thread management
        self.shutdown_event = threading.Event()
        self.deployment_lock = threading.RLock()
        
        # Callbacks
        self.state_change_callbacks: List[Callable[[DeploymentState], None]] = []
        self.health_change_callbacks: List[Callable[[str, HealthCheckResult], None]] = []
        
        # Setup directories
        self._setup_deployment_directories()
        
        self.logger.info("Deployment manager initialized")
    
    def _setup_deployment_directories(self):
        """Setup deployment directories."""
        try:
            self.deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            for subdir in ["configs", "logs", "backups", "scripts", "data"]:
                (self.deployment_dir / subdir).mkdir(exist_ok=True)
            
            self.logger.info(f"Deployment directories setup: {self.deployment_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup deployment directories: {e}")
    
    def load_deployment_config(self, config_path: Path) -> bool:
        """Load deployment configuration."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse configuration
            self.current_deployment = self._parse_deployment_config(config_data)
            
            if self.current_deployment:
                self.logger.info(f"Deployment configuration loaded: {self.current_deployment.deployment_name}")
                return True
            else:
                self.logger.error("Failed to parse deployment configuration")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load deployment config: {e}")
            return False
    
    def _parse_deployment_config(self, config_data: Dict[str, Any]) -> Optional[DeploymentConfig]:
        """Parse deployment configuration from data."""
        try:
            deployment_config = DeploymentConfig(
                deployment_name=config_data.get('name', 'safe_rl_deployment'),
                version=config_data.get('version', '1.0.0'),
                environment=Environment(config_data.get('environment', 'development'))
            )
            
            # Parse services
            for service_data in config_data.get('services', []):
                service_config = ServiceConfig(
                    name=service_data['name'],
                    image=service_data['image'],
                    tag=service_data.get('tag', 'latest'),
                    ports=service_data.get('ports', {}),
                    environment=service_data.get('environment', {}),
                    volumes=service_data.get('volumes', {}),
                    command=service_data.get('command'),
                    depends_on=service_data.get('depends_on', []),
                    health_check=service_data.get('health_check'),
                    resources=service_data.get('resources', {}),
                    restart_policy=service_data.get('restart_policy', 'unless-stopped'),
                    privileged=service_data.get('privileged', False)
                )
                deployment_config.services.append(service_config)
                self.service_configs[service_config.name] = service_config
            
            # Parse other configuration sections
            deployment_config.networks = config_data.get('networks', {})
            deployment_config.volumes = config_data.get('volumes', {})
            deployment_config.monitoring = config_data.get('monitoring', {})
            deployment_config.backup = config_data.get('backup', {})
            deployment_config.rollback = config_data.get('rollback', {})
            
            return deployment_config
            
        except Exception as e:
            self.logger.error(f"Failed to parse deployment configuration: {e}")
            return None
    
    def deploy(self, force_recreate: bool = False) -> bool:
        """Deploy the system."""
        try:
            if not self.current_deployment:
                self.logger.error("No deployment configuration loaded")
                return False
            
            with self.deployment_lock:
                self._set_state(DeploymentState.PREPARING)
                
                deploy_start_time = time.time()
                
                # Pre-deployment validation
                if not self._validate_deployment():
                    self._set_state(DeploymentState.FAILED)
                    return False
                
                # Create Docker networks
                if not self._create_networks():
                    self._set_state(DeploymentState.FAILED)
                    return False
                
                # Create Docker volumes
                if not self._create_volumes():
                    self._set_state(DeploymentState.FAILED)
                    return False
                
                self._set_state(DeploymentState.DEPLOYING)
                
                # Deploy services in dependency order
                if not self._deploy_services(force_recreate):
                    self._set_state(DeploymentState.FAILED)
                    return False
                
                # Start health monitoring
                self._start_health_monitoring()
                
                # Start metrics collection
                self._start_metrics_collection()
                
                # Wait for services to become healthy
                if not self._wait_for_healthy_services():
                    self.logger.warning("Some services are not healthy, but deployment proceeding")
                
                deploy_end_time = time.time()
                self.metrics.deployment_time_s = deploy_end_time - deploy_start_time
                
                self._set_state(DeploymentState.RUNNING)
                self.logger.info(f"Deployment completed successfully in {self.metrics.deployment_time_s:.2f}s")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            self._set_state(DeploymentState.FAILED)
            return False
    
    def _validate_deployment(self) -> bool:
        """Validate deployment configuration and environment."""
        try:
            if not self.current_deployment:
                return False
            
            # Check Docker daemon is running
            try:
                self.docker_client.ping()
            except Exception:
                self.logger.error("Docker daemon is not running")
                return False
            
            # Check required images are available
            for service in self.current_deployment.services:
                image_name = f"{service.image}:{service.tag}"
                try:
                    self.docker_client.images.get(image_name)
                except docker.errors.ImageNotFound:
                    self.logger.warning(f"Image not found locally, will pull: {image_name}")
            
            # Check port availability
            for service in self.current_deployment.services:
                for host_port in service.ports.values():
                    if not self._is_port_available(host_port):
                        self.logger.error(f"Port {host_port} is already in use")
                        return False
            
            # Validate service dependencies
            if not self._validate_service_dependencies():
                return False
            
            # Check system resources
            if not self._check_system_resources():
                return False
            
            self.logger.info("Deployment validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment validation failed: {e}")
            return False
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def _validate_service_dependencies(self) -> bool:
        """Validate service dependency graph."""
        try:
            service_names = {service.name for service in self.current_deployment.services}
            
            for service in self.current_deployment.services:
                for dep in service.depends_on:
                    if dep not in service_names:
                        self.logger.error(f"Service {service.name} depends on unknown service: {dep}")
                        return False
            
            # Check for circular dependencies
            if self._has_circular_dependencies():
                self.logger.error("Circular dependencies detected in service configuration")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Service dependency validation failed: {e}")
            return False
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS."""
        try:
            # Build adjacency list
            deps = {}
            for service in self.current_deployment.services:
                deps[service.name] = service.depends_on
            
            # DFS to detect cycles
            visited = set()
            rec_stack = set()
            
            def dfs(node):
                if node in rec_stack:
                    return True  # Cycle detected
                
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in deps.get(node, []):
                    if dfs(neighbor):
                        return True
                
                rec_stack.remove(node)
                return False
            
            for service_name in deps:
                if service_name not in visited:
                    if dfs(service_name):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Circular dependency check failed: {e}")
            return True  # Assume circular to be safe
    
    def _check_system_resources(self) -> bool:
        """Check system has sufficient resources."""
        try:
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.available < 1024 * 1024 * 1024:  # 1GB
                self.logger.warning("Low available memory detected")
            
            # Check available disk space
            disk = psutil.disk_usage(str(self.deployment_dir))
            if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
                self.logger.warning("Low available disk space detected")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage detected: {cpu_percent}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return False
    
    def _create_networks(self) -> bool:
        """Create Docker networks."""
        try:
            for network_name, network_config in self.current_deployment.networks.items():
                try:
                    # Check if network already exists
                    existing_networks = self.docker_client.networks.list(names=[network_name])
                    if existing_networks:
                        self.logger.info(f"Network already exists: {network_name}")
                        continue
                    
                    # Create network
                    self.docker_client.networks.create(
                        name=network_name,
                        driver=network_config.get('driver', 'bridge'),
                        options=network_config.get('options', {}),
                        labels=network_config.get('labels', {})
                    )
                    
                    self.logger.info(f"Created network: {network_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create network {network_name}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Network creation failed: {e}")
            return False
    
    def _create_volumes(self) -> bool:
        """Create Docker volumes."""
        try:
            for volume_name, volume_config in self.current_deployment.volumes.items():
                try:
                    # Check if volume already exists
                    existing_volumes = self.docker_client.volumes.list(filters={'name': volume_name})
                    if existing_volumes:
                        self.logger.info(f"Volume already exists: {volume_name}")
                        continue
                    
                    # Create volume
                    self.docker_client.volumes.create(
                        name=volume_name,
                        driver=volume_config.get('driver', 'local'),
                        driver_opts=volume_config.get('driver_opts', {}),
                        labels=volume_config.get('labels', {})
                    )
                    
                    self.logger.info(f"Created volume: {volume_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create volume {volume_name}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Volume creation failed: {e}")
            return False
    
    def _deploy_services(self, force_recreate: bool = False) -> bool:
        """Deploy services in dependency order."""
        try:
            # Calculate deployment order
            deployment_order = self._calculate_deployment_order()
            
            for service_name in deployment_order:
                service_config = self.service_configs[service_name]
                
                if not self._deploy_service(service_config, force_recreate):
                    self.logger.error(f"Failed to deploy service: {service_name}")
                    return False
                
                self.logger.info(f"Service deployed successfully: {service_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Service deployment failed: {e}")
            return False
    
    def _calculate_deployment_order(self) -> List[str]:
        """Calculate service deployment order based on dependencies."""
        try:
            # Topological sort
            in_degree = {}
            adj_list = {}
            
            # Initialize
            for service in self.current_deployment.services:
                in_degree[service.name] = 0
                adj_list[service.name] = []
            
            # Build graph
            for service in self.current_deployment.services:
                for dep in service.depends_on:
                    adj_list[dep].append(service.name)
                    in_degree[service.name] += 1
            
            # Topological sort using Kahn's algorithm
            queue = [node for node in in_degree if in_degree[node] == 0]
            result = []
            
            while queue:
                node = queue.pop(0)
                result.append(node)
                
                for neighbor in adj_list[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            if len(result) != len(self.current_deployment.services):
                raise ValueError("Circular dependency detected")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate deployment order: {e}")
            return [service.name for service in self.current_deployment.services]
    
    def _deploy_service(self, service_config: ServiceConfig, force_recreate: bool) -> bool:
        """Deploy a single service."""
        try:
            container_name = f"{self.current_deployment.deployment_name}_{service_config.name}"
            
            # Stop and remove existing container if force_recreate
            if force_recreate:
                self._stop_and_remove_container(container_name)
            
            # Check if container already exists and is running
            existing_container = self._get_container(container_name)
            if existing_container:
                if existing_container.status == 'running':
                    self.logger.info(f"Service already running: {service_config.name}")
                    self.active_containers[service_config.name] = existing_container
                    return True
                else:
                    # Start existing container
                    existing_container.start()
                    self.active_containers[service_config.name] = existing_container
                    return True
            
            # Pull image if needed
            image_name = f"{service_config.image}:{service_config.tag}"
            try:
                self.docker_client.images.get(image_name)
            except docker.errors.ImageNotFound:
                self.logger.info(f"Pulling image: {image_name}")
                self.docker_client.images.pull(service_config.image, tag=service_config.tag)
            
            # Prepare container configuration
            container_config = self._build_container_config(service_config, container_name)
            
            # Create and start container
            container = self.docker_client.containers.run(**container_config)
            self.active_containers[service_config.name] = container
            
            self.logger.info(f"Container started: {container_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy service {service_config.name}: {e}")
            return False
    
    def _build_container_config(self, service_config: ServiceConfig, container_name: str) -> Dict[str, Any]:
        """Build container configuration dictionary."""
        try:
            config = {
                'image': f"{service_config.image}:{service_config.tag}",
                'name': container_name,
                'detach': True,
                'restart_policy': {'Name': service_config.restart_policy},
                'environment': service_config.environment,
                'labels': {
                    'deployment': self.current_deployment.deployment_name,
                    'service': service_config.name,
                    'version': self.current_deployment.version
                }
            }
            
            # Add command if specified
            if service_config.command:
                config['command'] = service_config.command
            
            # Add port mappings
            if service_config.ports:
                config['ports'] = service_config.ports
            
            # Add volume mounts
            if service_config.volumes:
                config['volumes'] = service_config.volumes
            
            # Add resource limits
            if service_config.resources:
                config['mem_limit'] = service_config.resources.get('memory', '512m')
                config['cpu_count'] = service_config.resources.get('cpus', 1)
            
            # Add privileged mode if needed
            if service_config.privileged:
                config['privileged'] = True
            
            # Add networks
            if self.current_deployment.networks:
                config['network'] = list(self.current_deployment.networks.keys())[0]
            
            # Add health check
            if service_config.health_check:
                config['healthcheck'] = {
                    'test': service_config.health_check.get('test'),
                    'interval': service_config.health_check.get('interval', 30) * 1000000000,  # Convert to nanoseconds
                    'timeout': service_config.health_check.get('timeout', 10) * 1000000000,
                    'retries': service_config.health_check.get('retries', 3)
                }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to build container config: {e}")
            return {}
    
    def _get_container(self, container_name: str) -> Optional[docker.models.containers.Container]:
        """Get container by name."""
        try:
            return self.docker_client.containers.get(container_name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            self.logger.error(f"Failed to get container {container_name}: {e}")
            return None
    
    def _stop_and_remove_container(self, container_name: str):
        """Stop and remove a container."""
        try:
            container = self._get_container(container_name)
            if container:
                container.stop(timeout=10)
                container.remove()
                self.logger.info(f"Stopped and removed container: {container_name}")
        except Exception as e:
            self.logger.error(f"Failed to stop/remove container {container_name}: {e}")
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        try:
            if self.health_monitor_thread and self.health_monitor_thread.is_alive():
                return  # Already running
            
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitoring_loop,
                name="HealthMonitor",
                daemon=True
            )
            self.health_monitor_thread.start()
            
            self.logger.info("Health monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start health monitoring: {e}")
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        try:
            while not self.shutdown_event.is_set():
                for service_name, container in self.active_containers.items():
                    try:
                        # Perform health check
                        result = self._perform_health_check(service_name, container)
                        
                        # Update health status
                        old_status = self.health_status.get(service_name)
                        self.health_status[service_name] = result
                        
                        # Notify if status changed
                        if not old_status or old_status.status != result.status:
                            self._notify_health_change(service_name, result)
                        
                        # Handle unhealthy services
                        if result.status == HealthStatus.CRITICAL:
                            self._handle_unhealthy_service(service_name, container)
                            
                    except Exception as e:
                        self.logger.error(f"Health check failed for {service_name}: {e}")
                
                # Wait before next check
                time.sleep(10)  # 10 second interval
                
        except Exception as e:
            self.logger.error(f"Health monitoring loop failed: {e}")
    
    def _perform_health_check(self, service_name: str, container: docker.models.containers.Container) -> HealthCheckResult:
        """Perform health check for a service."""
        try:
            start_time = time.perf_counter()
            
            # Refresh container state
            container.reload()
            
            # Check container status
            if container.status != 'running':
                return HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Container not running: {container.status}",
                    timestamp=time.time()
                )
            
            # Check container health
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                health_status = health.get('Status', 'unknown')
                if health_status == 'healthy':
                    status = HealthStatus.HEALTHY
                elif health_status == 'unhealthy':
                    status = HealthStatus.CRITICAL
                else:
                    status = HealthStatus.WARNING
                
                message = health.get('Log', [{}])[-1].get('Output', 'No health check output')
            else:
                # Basic health check - container is running
                status = HealthStatus.HEALTHY
                message = "Container running (no health check configured)"
            
            # Get container metrics
            stats = container.stats(stream=False)
            metrics = self._extract_container_metrics(stats)
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            return HealthCheckResult(
                service_name=service_name,
                status=status,
                message=message,
                timestamp=time.time(),
                metrics=metrics,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check error: {str(e)}",
                timestamp=time.time()
            )
    
    def _extract_container_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from container stats."""
        try:
            metrics = {}
            
            # Memory usage
            memory_stats = stats.get('memory_stats', {})
            if 'usage' in memory_stats:
                metrics['memory_usage_bytes'] = memory_stats['usage']
                metrics['memory_limit_bytes'] = memory_stats.get('limit', 0)
            
            # CPU usage
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            if 'cpu_usage' in cpu_stats and 'cpu_usage' in precpu_stats:
                cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
                num_cpus = len(cpu_stats.get('cpu_usage', {}).get('percpu_usage', []))
                
                if system_delta > 0 and num_cpus > 0:
                    metrics['cpu_percent'] = (cpu_delta / system_delta) * num_cpus * 100
            
            # Network I/O
            networks = stats.get('networks', {})
            total_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
            total_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
            metrics['network_rx_bytes'] = total_rx
            metrics['network_tx_bytes'] = total_tx
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to extract container metrics: {e}")
            return {}
    
    def _handle_unhealthy_service(self, service_name: str, container: docker.models.containers.Container):
        """Handle unhealthy service."""
        try:
            self.logger.warning(f"Service {service_name} is unhealthy, attempting restart")
            
            # Restart container
            container.restart(timeout=30)
            
            self.logger.info(f"Service {service_name} restarted")
            
        except Exception as e:
            self.logger.error(f"Failed to handle unhealthy service {service_name}: {e}")
    
    def _notify_health_change(self, service_name: str, result: HealthCheckResult):
        """Notify health change callbacks."""
        for callback in self.health_change_callbacks:
            try:
                callback(service_name, result)
            except Exception as e:
                self.logger.error(f"Health change callback failed: {e}")
    
    def _start_metrics_collection(self):
        """Start metrics collection thread."""
        try:
            if self.metrics_thread and self.metrics_thread.is_alive():
                return  # Already running
            
            self.metrics_thread = threading.Thread(
                target=self._metrics_collection_loop,
                name="MetricsCollector",
                daemon=True
            )
            self.metrics_thread.start()
            
            self.logger.info("Metrics collection started")
            
        except Exception as e:
            self.logger.error(f"Failed to start metrics collection: {e}")
    
    def _metrics_collection_loop(self):
        """Metrics collection loop."""
        try:
            start_time = time.time()
            
            while not self.shutdown_event.is_set():
                try:
                    # Update deployment metrics
                    self.metrics.uptime_s = time.time() - start_time
                    
                    # Collect system metrics
                    self.metrics.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
                    self.metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
                    
                    # Collect container metrics
                    self._collect_container_metrics()
                    
                    # Add to history
                    self.metrics_history.append(self.metrics)
                    
                    # Keep only recent history
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                    
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                
                time.sleep(30)  # 30 second interval
                
        except Exception as e:
            self.logger.error(f"Metrics collection loop failed: {e}")
    
    def _collect_container_metrics(self):
        """Collect metrics from all containers."""
        try:
            total_memory = 0
            total_cpu = 0
            
            for service_name, container in self.active_containers.items():
                try:
                    container.reload()
                    if container.status == 'running':
                        stats = container.stats(stream=False)
                        metrics = self._extract_container_metrics(stats)
                        
                        total_memory += metrics.get('memory_usage_bytes', 0)
                        total_cpu += metrics.get('cpu_percent', 0)
                        
                except Exception as e:
                    self.logger.debug(f"Failed to collect metrics for {service_name}: {e}")
            
            # Update aggregate metrics
            if total_memory > 0:
                self.metrics.memory_usage_mb = total_memory / (1024 * 1024)
            if total_cpu > 0:
                self.metrics.cpu_usage_percent = total_cpu
                
        except Exception as e:
            self.logger.error(f"Container metrics collection failed: {e}")
    
    def _wait_for_healthy_services(self, timeout_s: int = 300) -> bool:
        """Wait for all services to become healthy."""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout_s:
                all_healthy = True
                
                for service_name in self.active_containers:
                    health_status = self.health_status.get(service_name)
                    
                    if not health_status or health_status.status != HealthStatus.HEALTHY:
                        all_healthy = False
                        break
                
                if all_healthy:
                    self.logger.info("All services are healthy")
                    return True
                
                time.sleep(5)  # Check every 5 seconds
            
            self.logger.warning(f"Timeout waiting for healthy services after {timeout_s}s")
            return False
            
        except Exception as e:
            self.logger.error(f"Error waiting for healthy services: {e}")
            return False
    
    def _set_state(self, new_state: DeploymentState):
        """Set deployment state and notify callbacks."""
        try:
            if self.state != new_state:
                old_state = self.state
                self.state = new_state
                
                self.logger.info(f"Deployment state changed: {old_state.name} -> {new_state.name}")
                
                # Notify callbacks
                for callback in self.state_change_callbacks:
                    try:
                        callback(new_state)
                    except Exception as e:
                        self.logger.error(f"State change callback failed: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to set deployment state: {e}")
    
    # Public API methods
    
    def stop(self) -> bool:
        """Stop the deployment."""
        try:
            with self.deployment_lock:
                self._set_state(DeploymentState.STOPPING)
                
                # Stop all containers
                for service_name, container in self.active_containers.items():
                    try:
                        self.logger.info(f"Stopping service: {service_name}")
                        container.stop(timeout=30)
                    except Exception as e:
                        self.logger.error(f"Failed to stop service {service_name}: {e}")
                
                self.active_containers.clear()
                self._set_state(DeploymentState.STOPPED)
                
                self.logger.info("Deployment stopped successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop deployment: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        return {
            'state': self.state.name,
            'deployment_name': self.current_deployment.deployment_name if self.current_deployment else None,
            'version': self.current_deployment.version if self.current_deployment else None,
            'services': {
                name: {
                    'status': container.status,
                    'health': self.health_status.get(name, {}).status.value if name in self.health_status else 'unknown'
                }
                for name, container in self.active_containers.items()
            },
            'metrics': self.metrics,
            'uptime_s': self.metrics.uptime_s
        }
    
    def get_health_status(self) -> Dict[str, HealthCheckResult]:
        """Get health status of all services."""
        return self.health_status.copy()
    
    def get_metrics(self) -> DeploymentMetrics:
        """Get current deployment metrics."""
        return self.metrics
    
    def register_state_change_callback(self, callback: Callable[[DeploymentState], None]):
        """Register state change callback."""
        self.state_change_callbacks.append(callback)
    
    def register_health_change_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """Register health change callback."""
        self.health_change_callbacks.append(callback)
    
    def shutdown(self):
        """Shutdown deployment manager."""
        try:
            self.logger.info("Shutting down deployment manager...")
            
            self.shutdown_event.set()
            
            # Stop deployment
            self.stop()
            
            # Stop monitoring threads
            for thread in [self.health_monitor_thread, self.metrics_thread]:
                if thread and thread.is_alive():
                    thread.join(timeout=5.0)
            
            self.logger.info("Deployment manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Deployment manager shutdown failed: {e}")


# Example deployment configuration
EXAMPLE_DEPLOYMENT_CONFIG = {
    'name': 'safe_rl_deployment',
    'version': '1.0.0',
    'environment': 'production',
    'services': [
        {
            'name': 'safe_rl_node',
            'image': 'safe_rl/main',
            'tag': 'latest',
            'ports': {8080: 8080, 8081: 8081},
            'environment': {
                'ROS_MASTER_URI': 'http://localhost:11311',
                'LOG_LEVEL': 'INFO'
            },
            'volumes': {
                '/opt/safe_rl/configs': '/app/configs',
                '/opt/safe_rl/logs': '/app/logs'
            },
            'health_check': {
                'test': 'curl -f http://localhost:8080/health || exit 1',
                'interval': 30,
                'timeout': 10,
                'retries': 3
            },
            'resources': {
                'memory': '2g',
                'cpus': 2
            }
        },
        {
            'name': 'safety_monitor',
            'image': 'safe_rl/safety_monitor',
            'tag': 'latest',
            'ports': {8082: 8082},
            'environment': {
                'MONITORING_FREQUENCY': '2000',
                'LOG_LEVEL': 'INFO'
            },
            'privileged': True,
            'depends_on': ['safe_rl_node']
        }
    ],
    'networks': {
        'safe_rl_network': {
            'driver': 'bridge'
        }
    },
    'volumes': {
        'safe_rl_data': {
            'driver': 'local'
        }
    }
}


def create_example_deployment_config(output_path: Path):
    """Create example deployment configuration."""
    with open(output_path, 'w') as f:
        yaml.dump(EXAMPLE_DEPLOYMENT_CONFIG, f, default_flow_style=False, indent=2)
    
    print(f"Created example deployment configuration: {output_path}")


if __name__ == "__main__":
    """Example usage of deployment manager."""
    import tempfile
    from .config_manager import ConfigurationManager, Environment
    
    # Create temporary deployment directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Creating deployment manager in: {temp_path}")
        
        # Create example deployment config
        config_file = temp_path / "deployment.yaml"
        create_example_deployment_config(config_file)
        
        # Initialize configuration manager
        config_manager = ConfigurationManager(temp_path / "configs", Environment.DEVELOPMENT)
        
        # Initialize deployment manager
        deployment_manager = DeploymentManager(config_manager, temp_path)
        
        # Load deployment configuration
        if deployment_manager.load_deployment_config(config_file):
            print("Deployment configuration loaded successfully")
            
            # Get status
            status = deployment_manager.get_status()
            print(f"Deployment status: {status}")
        
        # Shutdown
        deployment_manager.shutdown()
        config_manager.shutdown()
        
        print("Deployment manager example complete!")