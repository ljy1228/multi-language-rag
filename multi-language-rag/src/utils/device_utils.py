"""
Device utilities for Mac MPS/CPU detection and memory management.
Optimized for Apple Silicon Macs with MPS support.
"""

import torch
import psutil
import logging
from typing import Optional, Dict, Any, Union
import warnings

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and memory optimization for Mac systems."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], str]] = None):
        """
        Initialize device manager with configuration.
        
        Args:
            config: Configuration dictionary with device settings or device string
        """
        if isinstance(config, str):
            # Handle string input (e.g., "auto", "mps", "cpu")
            self.config = {"device": config}
        else:
            self.config = config or {}
        self.device = self._select_device()
        self.memory_info = self._get_memory_info()
        
        logger.info(f"Device selected: {self.device}")
        logger.info(f"Memory info: {self.memory_info}")
    
    def _select_device(self) -> torch.device:
        """Select the best available device for computation."""
        use_mps = self.config.get('use_mps', True)
        use_cpu = self.config.get('use_cpu', False)
        
        # Check MPS availability (Apple Silicon GPU)
        if use_mps and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("Using MPS (Apple Silicon GPU)")
                return torch.device("mps")
            else:
                logger.warning("MPS not built, falling back to CPU")
        
        # Fallback to CPU
        if use_cpu or not torch.cuda.is_available():
            logger.info("Using CPU")
            return torch.device("cpu")
        
        # CUDA fallback (if available)
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            return torch.device("cuda")
        
        # Default to CPU
        logger.info("Defaulting to CPU")
        return torch.device("cpu")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get memory information for the system."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    def get_device(self) -> torch.device:
        """Get the selected device."""
        return self.device
    
    def is_mps(self) -> bool:
        """Check if using MPS device."""
        return self.device.type == "mps"
    
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self.device.type == "cpu"
    
    def get_optimal_batch_size(self, model_size_mb: float, sequence_length: int = 384) -> int:
        """
        Estimate optimal batch size based on available memory.
        
        Args:
            model_size_mb: Model size in MB
            sequence_length: Maximum sequence length
            
        Returns:
            Recommended batch size
        """
        available_memory_gb = self.memory_info['available_gb']
        
        # Rough estimation: each sample needs ~2x model size in memory
        # Plus overhead for gradients, optimizer states, etc.
        memory_per_sample_mb = (model_size_mb * 2.5) + (sequence_length * 0.1)
        memory_per_sample_gb = memory_per_sample_mb / 1024
        
        # Use 70% of available memory to be safe
        usable_memory_gb = available_memory_gb * 0.7
        optimal_batch_size = int(usable_memory_gb / memory_per_sample_gb)
        
        # Clamp to reasonable bounds
        optimal_batch_size = max(1, min(optimal_batch_size, 32))
        
        logger.info(f"Estimated optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def optimize_for_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to the model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Move model to device
        model = model.to(self.device)
        
        return model
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if self.is_mps():
            # MPS doesn't have detailed memory stats like CUDA
            return {
                'allocated_gb': 0.0,
                'cached_gb': 0.0,
                'max_allocated_gb': 0.0
            }
        elif torch.cuda.is_available() and self.device.type == "cuda":
            return {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
                'cached_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3)
            }
        else:
            # CPU memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'allocated_gb': memory_info.rss / (1024**3),
                'cached_gb': 0.0,
                'max_allocated_gb': 0.0
            }
    
    def clear_cache(self):
        """Clear device cache to free memory."""
        if self.is_mps():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            # For CPU, we can't clear cache but can suggest garbage collection
            import gc
            gc.collect()
        
        logger.info("Cleared device cache")


def get_device_manager(config: Optional[Dict[str, Any]] = None) -> DeviceManager:
    """
    Get a device manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DeviceManager instance
    """
    return DeviceManager(config)


def setup_mixed_precision(device_manager: DeviceManager) -> bool:
    """
    Setup mixed precision training if supported.
    
    Args:
        device_manager: Device manager instance
        
    Returns:
        True if mixed precision is enabled, False otherwise
    """
    # Mixed precision is supported on MPS, CUDA, and some CPU operations
    if device_manager.is_mps() or (torch.cuda.is_available() and device_manager.device.type == "cuda"):
        logger.info("Mixed precision training enabled")
        return True
    else:
        logger.warning("Mixed precision not supported on CPU, using FP32")
        return False


def get_optimal_dataloader_workers(device_manager: DeviceManager) -> int:
    """
    Get optimal number of dataloader workers based on device.
    
    Args:
        device_manager: Device manager instance
        
    Returns:
        Number of workers
    """
    if device_manager.is_cpu():
        # For CPU, use more workers
        return min(4, psutil.cpu_count())
    else:
        # For GPU/MPS, use fewer workers to avoid overhead
        return min(2, psutil.cpu_count())


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_global_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def set_global_device_manager(device_manager: DeviceManager):
    """Set the global device manager instance."""
    global _device_manager
    _device_manager = device_manager
