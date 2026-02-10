"""Enhanced progress tracking utilities for event scanning"""

import time
import psutil
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, UTC


@dataclass
class StageMetrics:
    """Metrics for a processing stage"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    items_processed: int = 0
    memory_usage_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get stage duration in milliseconds"""
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    @property
    def processing_rate(self) -> float:
        """Get items processed per second"""
        duration_seconds = self.duration_ms / 1000
        if duration_seconds <= 0:
            return 0.0
        return self.items_processed / duration_seconds


@dataclass
class DetailedProgressTracker:
    """Enhanced progress tracker with detailed metrics and bottleneck detection"""
    total_items: int
    current_item: int = 0
    current_operation: str = ""
    current_stage: str = ""
    
    # UI components
    streamlit_progress_bar: Optional[Any] = None
    streamlit_status: Optional[Any] = None
    streamlit_metrics_container: Optional[Any] = None
    
    # Detailed tracking
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    stage_order: List[str] = field(default_factory=list)
    processing_rates: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    bottlenecks: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    start_time: float = field(default_factory=time.perf_counter)
    last_update_time: float = field(default_factory=time.perf_counter)
    
    def start_stage(self, stage_name: str, details: Dict[str, Any] = None) -> None:
        """Start timing a processing stage"""
        
        # End previous stage if exists
        if self.current_stage and self.current_stage in self.stages:
            self.end_stage(self.current_stage)
        
        self.current_stage = stage_name
        
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(
                name=stage_name,
                start_time=time.perf_counter()
            )
            self.stage_order.append(stage_name)
        else:
            # Restart existing stage
            self.stages[stage_name].start_time = time.perf_counter()
            self.stages[stage_name].end_time = None
        
        self._update_ui()
    
    def end_stage(self, stage_name: str) -> None:
        """End timing and calculate stage performance"""
        
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.end_time = time.perf_counter()
            
            # Sample memory usage
            try:
                process = psutil.Process()
                stage.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(stage.memory_usage_mb)
            except:
                stage.memory_usage_mb = 0.0
            
            # Calculate processing rate
            if stage.processing_rate > 0:
                self.processing_rates.append(stage.processing_rate)
        
        if self.current_stage == stage_name:
            self.current_stage = ""
        
        self._detect_bottlenecks()
        self._update_ui()
    
    def update(self, increment: int = 1, operation: str = "", stage_items: int = 0) -> None:
        """Update progress by increment"""
        
        self.current_item = min(self.current_item + increment, self.total_items)
        
        if operation:
            self.current_operation = operation
        
        # Update current stage items
        if self.current_stage and self.current_stage in self.stages:
            self.stages[self.current_stage].items_processed += stage_items or increment
        
        # Sample processing rate
        current_time = time.perf_counter()
        time_delta = current_time - self.last_update_time
        
        if time_delta > 0:
            rate = increment / time_delta
            self.processing_rates.append(rate)
            
            # Keep only recent rates (last 100 samples)
            if len(self.processing_rates) > 100:
                self.processing_rates = self.processing_rates[-100:]
        
        self.last_update_time = current_time
        self._update_ui()
    
    def add_stage_error(self, stage_name: str, error: str) -> None:
        """Add an error to a specific stage"""
        
        if stage_name in self.stages:
            self.stages[stage_name].errors.append(error)
    
    def _detect_bottlenecks(self) -> None:
        """Identify performance bottlenecks"""
        
        self.bottlenecks.clear()
        
        if len(self.stages) < 2:
            return
        
        # Find slowest stage by duration
        completed_stages = [s for s in self.stages.values() if s.end_time is not None]
        
        if completed_stages:
            slowest_stage = max(completed_stages, key=lambda s: s.duration_ms)
            avg_duration = sum(s.duration_ms for s in completed_stages) / len(completed_stages)
            
            if slowest_stage.duration_ms > avg_duration * 2:
                self.bottlenecks["slow_stage"] = f"{slowest_stage.name} is taking {slowest_stage.duration_ms:.0f}ms (2x average)"
        
        # Check memory usage
        if self.memory_samples:
            current_memory = self.memory_samples[-1]
            if current_memory > 1000:  # > 1GB
                self.bottlenecks["high_memory"] = f"High memory usage: {current_memory:.1f}MB"
        
        # Check processing rate
        if self.processing_rates:
            recent_rates = self.processing_rates[-10:]  # Last 10 samples
            avg_rate = sum(recent_rates) / len(recent_rates)
            
            if avg_rate < 1.0:  # Less than 1 item per second
                self.bottlenecks["slow_processing"] = f"Slow processing rate: {avg_rate:.2f} items/sec"
    
    def _update_ui(self) -> None:
        """Update Streamlit UI components"""
        
        # Update progress bar
        if self.streamlit_progress_bar is not None:
            progress = self.current_item / max(self.total_items, 1)
            self.streamlit_progress_bar.progress(progress)
        
        # Update status text
        if self.streamlit_status is not None:
            status_parts = []
            
            if self.current_stage:
                status_parts.append(f"Stage: {self.current_stage}")
            
            if self.current_operation:
                status_parts.append(self.current_operation)
            
            status_parts.append(f"({self.current_item}/{self.total_items})")
            
            status_text = " | ".join(status_parts)
            self.streamlit_status.text(status_text)
        
        # Update metrics
        if self.streamlit_metrics_container is not None:
            with self.streamlit_metrics_container:
                self._render_metrics()
    
    def _render_metrics(self) -> None:
        """Render detailed metrics in Streamlit"""
        
        import streamlit as st
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            progress_pct = self.get_progress_percent()
            st.metric("Progress", f"{progress_pct:.1f}%")
        
        with col2:
            current_rate = self.get_current_processing_rate()
            st.metric("Rate", f"{current_rate:.1f}/sec")
        
        with col3:
            memory_usage = self.get_current_memory_usage()
            st.metric("Memory", f"{memory_usage:.1f}MB")
        
        with col4:
            eta = self.get_estimated_time_remaining()
            st.metric("ETA", eta)
        
        # Show bottlenecks if any
        if self.bottlenecks:
            st.warning("⚠️ **Bottlenecks Detected:**")
            for bottleneck_type, description in self.bottlenecks.items():
                st.write(f"• {description}")
    
    def get_progress_percent(self) -> float:
        """Get progress as percentage"""
        return (self.current_item / max(self.total_items, 1)) * 100
    
    def get_current_processing_rate(self) -> float:
        """Get current processing rate (items/sec)"""
        if not self.processing_rates:
            return 0.0
        
        # Average of last 10 samples
        recent_rates = self.processing_rates[-10:]
        return sum(recent_rates) / len(recent_rates)
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if not self.memory_samples:
            return 0.0
        return self.memory_samples[-1]
    
    def get_estimated_time_remaining(self) -> str:
        """Get estimated time remaining"""
        
        remaining_items = self.total_items - self.current_item
        current_rate = self.get_current_processing_rate()
        
        if current_rate <= 0 or remaining_items <= 0:
            return "N/A"
        
        seconds_remaining = remaining_items / current_rate
        
        if seconds_remaining < 60:
            return f"{seconds_remaining:.0f}s"
        elif seconds_remaining < 3600:
            return f"{seconds_remaining/60:.1f}m"
        else:
            return f"{seconds_remaining/3600:.1f}h"
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of all stages"""
        
        summary = {}
        
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            summary[stage_name] = {
                "duration_ms": stage.duration_ms,
                "items_processed": stage.items_processed,
                "processing_rate": stage.processing_rate,
                "memory_mb": stage.memory_usage_mb,
                "errors": len(stage.errors),
                "status": "completed" if stage.end_time else "running"
            }
        
        return summary
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        
        total_duration = time.perf_counter() - self.start_time
        
        return {
            "total_duration_ms": total_duration * 1000,
            "items_processed": self.current_item,
            "overall_rate": self.current_item / max(total_duration, 0.001),
            "stages_completed": len([s for s in self.stages.values() if s.end_time]),
            "total_stages": len(self.stages),
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "bottlenecks": self.bottlenecks,
            "stage_breakdown": self.get_stage_summary()
        }
    
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.current_item >= self.total_items


# Legacy compatibility
@dataclass
class ProgressTracker:
    """Legacy progress tracker for backward compatibility"""
    total_items: int
    current_item: int = 0
    current_operation: str = ""
    streamlit_progress_bar: Optional[Any] = None
    streamlit_status: Optional[Any] = None
    
    def update(self, increment: int = 1, operation: str = "") -> None:
        """Update progress by increment"""
        self.current_item = min(self.current_item + increment, self.total_items)
        if operation:
            self.current_operation = operation
        
        # Update Streamlit UI if available
        if self.streamlit_progress_bar is not None:
            progress = self.current_item / max(self.total_items, 1)
            self.streamlit_progress_bar.progress(progress)
        
        if self.streamlit_status is not None:
            status_text = f"{self.current_operation} ({self.current_item}/{self.total_items})"
            self.streamlit_status.text(status_text)
    
    def set_operation(self, operation: str) -> None:
        """Set current operation without incrementing"""
        self.current_operation = operation
        if self.streamlit_status is not None:
            status_text = f"{self.current_operation} ({self.current_item}/{self.total_items})"
            self.streamlit_status.text(status_text)
    
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.current_item >= self.total_items
    
    def get_progress_percent(self) -> float:
        """Get progress as percentage"""
        return (self.current_item / max(self.total_items, 1)) * 100


def create_streamlit_progress(total_items: int, operation: str = "Processing") -> ProgressTracker:
    """Create a progress tracker with Streamlit UI components"""
    try:
        import streamlit as st
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tracker = ProgressTracker(
            total_items=total_items,
            current_operation=operation,
            streamlit_progress_bar=progress_bar,
            streamlit_status=status_text
        )
        
        return tracker
        
    except ImportError:
        # Return basic tracker if Streamlit not available
        return ProgressTracker(total_items=total_items, current_operation=operation)


def create_detailed_streamlit_progress(total_items: int, operation: str = "Processing") -> DetailedProgressTracker:
    """Create a detailed progress tracker with enhanced Streamlit UI"""
    try:
        import streamlit as st
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        tracker = DetailedProgressTracker(
            total_items=total_items,
            current_operation=operation,
            streamlit_progress_bar=progress_bar,
            streamlit_status=status_text,
            streamlit_metrics_container=metrics_container
        )
        
        return tracker
        
    except ImportError:
        # Return basic detailed tracker if Streamlit not available
        return DetailedProgressTracker(total_items=total_items, current_operation=operation)
