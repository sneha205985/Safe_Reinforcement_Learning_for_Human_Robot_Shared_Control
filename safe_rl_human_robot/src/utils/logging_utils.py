"""
Logging utilities for Safe RL Human-Robot Shared Control system.

This module provides comprehensive logging setup including file logging,
console output, and integration with experiment tracking systems.
"""

import logging
import logging.handlers
from typing import Optional, Dict, Any, Union
from pathlib import Path
import sys
import json
import time
from datetime import datetime
import traceback
import torch
import numpy as np
from contextlib import contextmanager

try:
    import tensorboardX
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SafeRLFormatter(logging.Formatter):
    """Custom formatter for Safe RL logging with color support."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_thread: bool = False):
        """
        Initialize formatter.
        
        Args:
            use_colors: Whether to use ANSI color codes
            include_thread: Whether to include thread information
        """
        self.use_colors = use_colors
        self.include_thread = include_thread
        
        # Format string
        if include_thread:
            format_str = '[%(asctime)s] [%(name)s] [%(threadName)s] [%(levelname)s] %(message)s'
        else:
            format_str = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
        
        super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record with optional colors."""
        # Apply color if enabled and outputting to terminal
        if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            log_color = self.COLORS.get(record.levelname, '')
            reset_color = self.COLORS['RESET']
            
            # Temporarily modify the record
            original_levelname = record.levelname
            record.levelname = f"{log_color}{record.levelname}{reset_color}"
            
            formatted = super().format(record)
            
            # Restore original levelname
            record.levelname = original_levelname
            
            return formatted
        else:
            return super().format(record)


class MetricsLogger:
    """Logger for training metrics and experiment tracking."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 tensorboard_enabled: bool = True,
                 wandb_enabled: bool = False,
                 wandb_project: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            tensorboard_enabled: Enable TensorBoard logging
            wandb_enabled: Enable Weights & Biases logging
            wandb_project: W&B project name
            experiment_name: Name of current experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"safe_rl_{int(time.time())}"
        
        # Initialize TensorBoard
        self.tb_writer = None
        if tensorboard_enabled and TENSORBOARD_AVAILABLE:
            tb_log_dir = self.log_dir / "tensorboard" / self.experiment_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = tensorboardX.SummaryWriter(str(tb_log_dir))
            logging.info(f"TensorBoard logging enabled: {tb_log_dir}")
        elif tensorboard_enabled:
            logging.warning("TensorBoard requested but tensorboardX not available")
        
        # Initialize W&B
        self.wandb_enabled = False
        if wandb_enabled and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=wandb_project or "safe-rl-human-robot",
                    name=self.experiment_name,
                    dir=str(self.log_dir)
                )
                self.wandb_enabled = True
                logging.info("Weights & Biases logging enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize W&B: {e}")
        elif wandb_enabled:
            logging.warning("W&B requested but wandb not available")
        
        # Metrics storage
        self.step = 0
        self.episode = 0
        self.metrics_history = []
    
    def log_scalar(self, name: str, value: Union[float, int, torch.Tensor], step: Optional[int] = None):
        """
        Log scalar metric.
        
        Args:
            name: Metric name
            value: Scalar value
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
        
        # Convert tensors to scalars
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        
        # W&B logging
        if self.wandb_enabled:
            wandb.log({name: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, Union[float, int, torch.Tensor]], 
                   step: Optional[int] = None):
        """
        Log multiple scalar metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number
        """
        if step is None:
            step = self.step
        
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], 
                     step: Optional[int] = None):
        """
        Log histogram of values.
        
        Args:
            name: Histogram name
            values: Array of values
            step: Step number
        """
        if step is None:
            step = self.step
        
        # Convert to numpy if needed
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
        
        # W&B logging
        if self.wandb_enabled:
            wandb.log({f"{name}_hist": wandb.Histogram(values)}, step=step)
    
    def log_parameters(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        Log model parameters as histograms.
        
        Args:
            model: PyTorch model
            step: Step number
        """
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f"params/{name}", param, step)
                self.log_histogram(f"gradients/{name}", param.grad, step)
                self.log_scalar(f"grad_norm/{name}", torch.norm(param.grad), step)
    
    def log_constraint_violations(self, violations: Dict[str, Any], step: Optional[int] = None):
        """
        Log constraint violation statistics.
        
        Args:
            violations: Dictionary of constraint violation data
            step: Step number
        """
        if step is None:
            step = self.step
        
        for constraint_name, violation_data in violations.items():
            if isinstance(violation_data, dict):
                for key, value in violation_data.items():
                    self.log_scalar(f"constraints/{constraint_name}/{key}", value, step)
            else:
                self.log_scalar(f"constraints/{constraint_name}", violation_data, step)
    
    def log_episode_stats(self, stats: Dict[str, Any], episode: Optional[int] = None):
        """
        Log episode-level statistics.
        
        Args:
            stats: Dictionary of episode statistics
            episode: Episode number
        """
        if episode is None:
            episode = self.episode
        
        for key, value in stats.items():
            self.log_scalar(f"episode/{key}", value, episode)
        
        # Store in history
        stats_with_episode = {"episode": episode, **stats}
        self.metrics_history.append(stats_with_episode)
    
    def increment_step(self):
        """Increment internal step counter."""
        self.step += 1
    
    def increment_episode(self):
        """Increment internal episode counter."""
        self.episode += 1
    
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics history to JSON file."""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        logging.info(f"Metrics saved to {filepath}")
    
    def close(self):
        """Close logging resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_enabled:
            wandb.finish()


class SafetyMonitor:
    """Monitor for safety constraint violations and system health."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize safety monitor.
        
        Args:
            log_file: File to log safety events
        """
        self.safety_events = []
        self.violation_counts = {}
        self.log_file = log_file
        
        if log_file:
            # Set up dedicated safety logger
            self.safety_logger = logging.getLogger("safety_monitor")
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '[%(asctime)s] [SAFETY] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.safety_logger.addHandler(handler)
            self.safety_logger.setLevel(logging.INFO)
        else:
            self.safety_logger = logging.getLogger(__name__)
    
    def log_violation(self, constraint_name: str, violation_value: float, 
                     context: Optional[Dict[str, Any]] = None):
        """
        Log constraint violation.
        
        Args:
            constraint_name: Name of violated constraint
            violation_value: Severity of violation
            context: Additional context information
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "constraint": constraint_name,
            "violation_value": violation_value,
            "context": context or {}
        }
        
        self.safety_events.append(event)
        self.violation_counts[constraint_name] = self.violation_counts.get(constraint_name, 0) + 1
        
        # Log to safety logger
        self.safety_logger.warning(
            f"Constraint violation: {constraint_name} = {violation_value:.6f}"
        )
        
        if context:
            self.safety_logger.info(f"Context: {context}")
    
    def log_critical_event(self, event_type: str, description: str, 
                          context: Optional[Dict[str, Any]] = None):
        """
        Log critical safety event.
        
        Args:
            event_type: Type of critical event
            description: Event description
            context: Additional context
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": "critical",
            "event_type": event_type,
            "description": description,
            "context": context or {}
        }
        
        self.safety_events.append(event)
        
        self.safety_logger.critical(f"CRITICAL EVENT: {event_type} - {description}")
        if context:
            self.safety_logger.critical(f"Context: {context}")
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        total_violations = sum(self.violation_counts.values())
        return {
            "total_violations": total_violations,
            "violation_counts": self.violation_counts.copy(),
            "total_events": len(self.safety_events),
            "violation_rate": total_violations / max(len(self.safety_events), 1)
        }
    
    def save_events(self, filepath: str):
        """Save safety events to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.safety_events, f, indent=2, default=str)


def setup_logger(name: str = "safe_rl",
                level: Union[str, int] = logging.INFO,
                log_dir: str = "logs",
                log_filename: Optional[str] = None,
                console_output: bool = True,
                file_output: bool = True,
                max_log_files: int = 10,
                max_file_size_mb: int = 100) -> logging.Logger:
    """
    Set up comprehensive logging for the Safe RL system.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        log_filename: Log filename (auto-generated if None)
        console_output: Enable console output
        file_output: Enable file output
        max_log_files: Maximum number of log files to keep
        max_file_size_mb: Maximum size per log file in MB
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = SafeRLFormatter(use_colors=True, include_thread=False)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"safe_rl_{timestamp}.log"
        
        log_filepath = log_path / log_filename
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_filepath,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_log_files
        )
        file_handler.setLevel(logging.DEBUG)  # More detailed for file logging
        file_formatter = SafeRLFormatter(use_colors=False, include_thread=True)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add exception hook for uncaught exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = exception_handler
    
    logger.info(f"Logger '{name}' initialized")
    logger.info(f"Log level: {logging.getLevelName(logger.level)}")
    logger.info(f"Log directory: {log_path.absolute()}")
    
    return logger


@contextmanager
def log_execution_time(logger: logging.Logger, operation_name: str, 
                      log_level: int = logging.INFO):
    """
    Context manager to log execution time of operations.
    
    Args:
        logger: Logger instance
        operation_name: Name of operation being timed
        log_level: Log level for timing message
    """
    start_time = time.time()
    logger.log(log_level, f"Starting {operation_name}")
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.log(log_level, f"Completed {operation_name} in {execution_time:.3f}s")


def log_system_info(logger: logging.Logger):
    """Log system information for debugging."""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    logger.info("=" * 30)


def configure_torch_logging(logger: logging.Logger):
    """Configure PyTorch-specific logging."""
    # Set PyTorch logging levels
    torch_loggers = [
        "torch",
        "torch.nn",
        "torch.optim",
        "torch.distributed"
    ]
    
    for torch_logger_name in torch_loggers:
        torch_logger = logging.getLogger(torch_logger_name)
        torch_logger.setLevel(logging.WARNING)  # Reduce verbosity
    
    # Log torch settings
    logger.info(f"PyTorch settings:")
    logger.info(f"  Default tensor type: {torch.get_default_dtype()}")
    logger.info(f"  Number of threads: {torch.get_num_threads()}")
    logger.info(f"  Gradient check: {torch.is_grad_enabled()}")
    
    # Set up anomaly detection in debug mode
    if logger.level <= logging.DEBUG:
        torch.autograd.set_detect_anomaly(True)
        logger.debug("PyTorch anomaly detection enabled")