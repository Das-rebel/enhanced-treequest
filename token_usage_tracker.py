#!/usr/bin/env python3
"""
Token Usage Tracker for Claude Code Sessions
Monitors token consumption and triggers orchestrator switching at 70% threshold
Provides real-time usage tracking with predictive analytics
"""

import os
import json
import time
import asyncio
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

class SessionStatus(Enum):
    """Claude Code session status levels"""
    ACTIVE = "active"
    APPROACHING_LIMIT = "approaching_limit"  # 70%+ usage
    CRITICAL = "critical"                    # 90%+ usage
    EXCEEDED = "exceeded"                    # 100%+ usage
    SWITCHED = "switched"                    # Switched to Gemini CLI

@dataclass
class TokenUsage:
    """Token usage metrics for a session"""
    session_id: str
    start_time: float
    current_tokens: int = 0
    estimated_limit: int = 100000  # Conservative Claude Code limit
    input_tokens: int = 0
    output_tokens: int = 0
    api_call_tokens: int = 0
    context_tokens: int = 0
    peak_usage: int = 0
    usage_history: List[Dict] = None
    
    def __post_init__(self):
        if self.usage_history is None:
            self.usage_history = []
    
    @property
    def usage_percentage(self) -> float:
        """Current usage as percentage of limit"""
        return (self.current_tokens / self.estimated_limit) * 100
    
    @property
    def status(self) -> SessionStatus:
        """Current session status based on usage"""
        percentage = self.usage_percentage
        if percentage >= 100:
            return SessionStatus.EXCEEDED
        elif percentage >= 90:
            return SessionStatus.CRITICAL
        elif percentage >= 70:
            return SessionStatus.APPROACHING_LIMIT
        else:
            return SessionStatus.ACTIVE
    
    @property
    def remaining_tokens(self) -> int:
        """Estimated remaining tokens before limit"""
        return max(0, self.estimated_limit - self.current_tokens)
    
    @property
    def time_to_limit(self) -> Optional[float]:
        """Estimated minutes until token limit based on current rate"""
        if len(self.usage_history) < 2:
            return None
        
        # Calculate token rate from recent history
        recent_history = self.usage_history[-10:]  # Last 10 data points
        if len(recent_history) < 2:
            return None
        
        time_diff = recent_history[-1]['timestamp'] - recent_history[0]['timestamp']
        token_diff = recent_history[-1]['tokens'] - recent_history[0]['tokens']
        
        if time_diff <= 0 or token_diff <= 0:
            return None
        
        tokens_per_second = token_diff / time_diff
        remaining_time = self.remaining_tokens / tokens_per_second
        return remaining_time / 60  # Convert to minutes

class TokenEstimator:
    """Advanced token estimation for different types of content"""
    
    def __init__(self):
        if TIKTOKEN_AVAILABLE:
            try:
                # Use GPT-4 encoding as baseline (similar to Claude)
                self.encoder = tiktoken.encoding_for_model("gpt-4")
            except:
                # Fallback to simple estimation
                self.encoder = None
        else:
            # tiktoken not available, use simple estimation
            self.encoder = None
    
    def estimate_tokens(self, text: str, content_type: str = "general") -> int:
        """Estimate token count for text with type-specific adjustments"""
        if not text:
            return 0
        
        if self.encoder:
            base_tokens = len(self.encoder.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            base_tokens = len(text) // 4
        
        # Adjust based on content type
        multipliers = {
            "code": 1.2,        # Code tends to use more tokens
            "markdown": 1.1,    # Markdown formatting adds tokens
            "json": 1.15,       # Structured data
            "general": 1.0,     # Regular text
            "context": 1.05,    # Context/system prompts
            "tools": 1.3        # Tool calls and responses
        }
        
        multiplier = multipliers.get(content_type, 1.0)
        return int(base_tokens * multiplier)
    
    def estimate_conversation_tokens(self, messages: List[Dict]) -> int:
        """Estimate tokens for a conversation history"""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            
            # Role overhead
            total += 4  # Role designation
            
            # Content tokens
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if item.get('type') == 'text':
                        total += self.estimate_tokens(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        total += self.estimate_tokens(json.dumps(item), "tools")
        
        return total

class TokenUsageTracker:
    """Comprehensive token usage tracking for Claude Code sessions"""
    
    def __init__(self, 
                 session_limit: int = 100000,
                 warning_threshold: float = 0.7,
                 critical_threshold: float = 0.9,
                 tracking_file: Optional[str] = None):
        
        self.session_limit = session_limit
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.tracking_file = tracking_file or os.path.expanduser("~/.treequest/session_usage.json")
        
        # Current session tracking
        self.current_session: Optional[TokenUsage] = None
        self.session_start_time = time.time()
        self.session_id = f"session_{int(self.session_start_time)}"
        
        # Token estimation
        self.estimator = TokenEstimator()
        
        # Callbacks for threshold events
        self.threshold_callbacks: Dict[str, List[Callable]] = {
            'warning': [],
            'critical': [],
            'exceeded': []
        }
        
        # Create tracking directory
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize session
        self._initialize_session()
        
        # Start background monitoring
        self._start_monitoring()
    
    def _initialize_session(self):
        """Initialize a new tracking session"""
        self.current_session = TokenUsage(
            session_id=self.session_id,
            start_time=self.session_start_time,
            estimated_limit=self.session_limit
        )
        
        self.logger.info(f"🚀 Token tracking initialized for session {self.session_id}")
        self.logger.info(f"📊 Session limit: {self.session_limit:,} tokens")
        self.logger.info(f"⚠️ Warning threshold: {self.warning_threshold*100:.0f}%")
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while True:
                try:
                    self._update_usage_estimate()
                    self._check_thresholds()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    self.logger.error(f"❌ Monitoring error: {e}")
                    time.sleep(10)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_usage_estimate(self):
        """Update token usage estimates based on system activity"""
        if not self.current_session:
            return
        
        # Try to estimate based on Claude Code process activity
        try:
            # Look for Claude Code processes
            claude_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'claude' in proc.info['name'].lower():
                    claude_processes.append(proc)
            
            if claude_processes:
                # Estimate based on memory usage (rough correlation)
                total_memory = sum(proc.info['memory_info'].rss for proc in claude_processes)
                # Very rough estimate: 1MB memory ≈ 1000 tokens of context
                estimated_tokens = total_memory // (1024 * 1024) * 1000
                
                # Cap the estimate reasonably
                estimated_tokens = min(estimated_tokens, self.session_limit)
                
                # Update if this is higher than current estimate
                if estimated_tokens > self.current_session.current_tokens:
                    self.current_session.current_tokens = estimated_tokens
                    self.current_session.peak_usage = max(
                        self.current_session.peak_usage,
                        estimated_tokens
                    )
                    
                    # Record in history
                    self.current_session.usage_history.append({
                        'timestamp': time.time(),
                        'tokens': estimated_tokens,
                        'memory_mb': total_memory // (1024 * 1024),
                        'processes': len(claude_processes)
                    })
                    
                    # Keep history reasonable size
                    if len(self.current_session.usage_history) > 100:
                        self.current_session.usage_history = self.current_session.usage_history[-50:]
        
        except Exception as e:
            self.logger.debug(f"Could not update usage estimate: {e}")
    
    def track_input(self, text: str, content_type: str = "general"):
        """Track input tokens manually"""
        if not self.current_session:
            return
        
        tokens = self.estimator.estimate_tokens(text, content_type)
        self.current_session.input_tokens += tokens
        self.current_session.current_tokens += tokens
        
        self.logger.debug(f"📝 Input tracked: {tokens} tokens ({content_type})")
        self._update_peak_and_history()
    
    def track_output(self, text: str, content_type: str = "general"):
        """Track output tokens manually"""
        if not self.current_session:
            return
        
        tokens = self.estimator.estimate_tokens(text, content_type)
        self.current_session.output_tokens += tokens
        self.current_session.current_tokens += tokens
        
        self.logger.debug(f"📤 Output tracked: {tokens} tokens ({content_type})")
        self._update_peak_and_history()
    
    def track_api_call(self, request_tokens: int, response_tokens: int):
        """Track API call tokens"""
        if not self.current_session:
            return
        
        total_tokens = request_tokens + response_tokens
        self.current_session.api_call_tokens += total_tokens
        self.current_session.current_tokens += total_tokens
        
        self.logger.debug(f"🔌 API call tracked: {total_tokens} tokens")
        self._update_peak_and_history()
    
    def track_context(self, context_text: str):
        """Track context/system prompt tokens"""
        if not self.current_session:
            return
        
        tokens = self.estimator.estimate_tokens(context_text, "context")
        self.current_session.context_tokens += tokens
        self.current_session.current_tokens += tokens
        
        self.logger.debug(f"🧠 Context tracked: {tokens} tokens")
        self._update_peak_and_history()
    
    def _update_peak_and_history(self):
        """Update peak usage and history after token tracking"""
        if not self.current_session:
            return
        
        self.current_session.peak_usage = max(
            self.current_session.peak_usage,
            self.current_session.current_tokens
        )
        
        self.current_session.usage_history.append({
            'timestamp': time.time(),
            'tokens': self.current_session.current_tokens,
            'input': self.current_session.input_tokens,
            'output': self.current_session.output_tokens,
            'api': self.current_session.api_call_tokens,
            'context': self.current_session.context_tokens
        })
        
        # Keep history manageable
        if len(self.current_session.usage_history) > 100:
            self.current_session.usage_history = self.current_session.usage_history[-50:]
    
    def _check_thresholds(self):
        """Check if any thresholds have been crossed"""
        if not self.current_session:
            return
        
        percentage = self.current_session.usage_percentage
        
        # Check for threshold crossings
        if percentage >= 100 and self.current_session.status != SessionStatus.EXCEEDED:
            self.logger.critical(f"🚨 TOKEN LIMIT EXCEEDED: {percentage:.1f}%")
            self._trigger_callbacks('exceeded')
        elif percentage >= self.critical_threshold * 100 and percentage < 100:
            if self.current_session.status != SessionStatus.CRITICAL:
                self.logger.warning(f"⚠️ CRITICAL TOKEN USAGE: {percentage:.1f}%")
                self._trigger_callbacks('critical')
        elif percentage >= self.warning_threshold * 100 and percentage < self.critical_threshold * 100:
            if self.current_session.status != SessionStatus.APPROACHING_LIMIT:
                self.logger.warning(f"🔔 TOKEN WARNING: {percentage:.1f}% - Consider switching to Gemini CLI")
                self._trigger_callbacks('warning')
    
    def _trigger_callbacks(self, threshold_type: str):
        """Trigger registered callbacks for threshold events"""
        callbacks = self.threshold_callbacks.get(threshold_type, [])
        for callback in callbacks:
            try:
                callback(self.current_session)
            except Exception as e:
                self.logger.error(f"❌ Callback error for {threshold_type}: {e}")
    
    def register_threshold_callback(self, threshold_type: str, callback: Callable):
        """Register callback for threshold events"""
        if threshold_type in self.threshold_callbacks:
            self.threshold_callbacks[threshold_type].append(callback)
            self.logger.info(f"✅ Registered {threshold_type} threshold callback")
    
    def should_switch_orchestrator(self) -> bool:
        """Check if orchestrator should switch to Gemini CLI"""
        if not self.current_session:
            return False
        
        return self.current_session.usage_percentage >= self.warning_threshold * 100
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        if not self.current_session:
            return {}
        
        return {
            'session_id': self.current_session.session_id,
            'current_tokens': self.current_session.current_tokens,
            'limit': self.current_session.estimated_limit,
            'percentage': self.current_session.usage_percentage,
            'status': self.current_session.status.value,
            'remaining': self.current_session.remaining_tokens,
            'time_to_limit_minutes': self.current_session.time_to_limit,
            'should_switch': self.should_switch_orchestrator(),
            'breakdown': {
                'input': self.current_session.input_tokens,
                'output': self.current_session.output_tokens,
                'api_calls': self.current_session.api_call_tokens,
                'context': self.current_session.context_tokens
            },
            'session_duration_minutes': (time.time() - self.current_session.start_time) / 60
        }
    
    def save_session_data(self):
        """Save current session data to file"""
        if not self.current_session:
            return
        
        try:
            data = asdict(self.current_session)
            data['last_updated'] = time.time()
            
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.debug(f"💾 Session data saved to {self.tracking_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save session data: {e}")
    
    def get_usage_report(self) -> str:
        """Generate a formatted usage report"""
        if not self.current_session:
            return "No active session"
        
        summary = self.get_usage_summary()
        
        status_emoji = {
            'active': '✅',
            'approaching_limit': '🔔',
            'critical': '⚠️',
            'exceeded': '🚨',
            'switched': '🔄'
        }
        
        emoji = status_emoji.get(summary['status'], '❓')
        
        report = [
            f"📊 Claude Code Token Usage Report {emoji}",
            "=" * 50,
            f"Session ID: {summary['session_id']}",
            f"Current Usage: {summary['current_tokens']:,} / {summary['limit']:,} tokens",
            f"Usage Percentage: {summary['percentage']:.1f}%",
            f"Status: {summary['status'].replace('_', ' ').title()}",
            f"Remaining: {summary['remaining']:,} tokens",
            "",
            "📋 Token Breakdown:",
            f"   Input Tokens: {summary['breakdown']['input']:,}",
            f"   Output Tokens: {summary['breakdown']['output']:,}",
            f"   API Call Tokens: {summary['breakdown']['api_calls']:,}",
            f"   Context Tokens: {summary['breakdown']['context']:,}",
            "",
            f"⏱️ Session Duration: {summary['session_duration_minutes']:.1f} minutes"
        ]
        
        if summary['time_to_limit_minutes']:
            report.append(f"⏳ Est. Time to Limit: {summary['time_to_limit_minutes']:.1f} minutes")
        
        if summary['should_switch']:
            report.append("")
            report.append("🔄 RECOMMENDATION: Switch to Gemini CLI orchestrator")
        
        return "\n".join(report)

# Global tracker instance
token_tracker = TokenUsageTracker()

def setup_token_tracking(session_limit: int = 100000, 
                        warning_threshold: float = 0.7) -> TokenUsageTracker:
    """Setup global token tracking with custom parameters"""
    global token_tracker
    token_tracker = TokenUsageTracker(
        session_limit=session_limit,
        warning_threshold=warning_threshold
    )
    return token_tracker

# Context managers for automatic tracking
class TrackTokenUsage:
    """Context manager for automatic token tracking"""
    
    def __init__(self, operation_name: str, content_type: str = "general"):
        self.operation_name = operation_name
        self.content_type = content_type
        self.start_tokens = 0
    
    def __enter__(self):
        if token_tracker.current_session:
            self.start_tokens = token_tracker.current_session.current_tokens
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if token_tracker.current_session:
            end_tokens = token_tracker.current_session.current_tokens
            used_tokens = end_tokens - self.start_tokens
            if used_tokens > 0:
                token_tracker.logger.info(
                    f"🔢 {self.operation_name}: {used_tokens} tokens used"
                )

# CLI interface for testing
async def main():
    """Main CLI interface for testing token tracking"""
    print("📊 Token Usage Tracker Test")
    print("=" * 40)
    
    # Setup tracking
    tracker = setup_token_tracking()
    
    # Simulate some usage
    tracker.track_input("Hello, this is a test prompt for token tracking", "general")
    tracker.track_context("System prompt with context information")
    tracker.track_output("This is a simulated AI response with some content", "general")
    tracker.track_api_call(100, 150)
    
    # Print current status
    print(tracker.get_usage_report())
    
    # Test threshold callback
    def warning_callback(session):
        print(f"🔔 WARNING: Session {session.session_id} at {session.usage_percentage:.1f}%")
    
    tracker.register_threshold_callback('warning', warning_callback)
    
    # Simulate approaching threshold
    for i in range(10):
        tracker.track_input("Large prompt " * 1000, "general")
        await asyncio.sleep(0.1)
        
        if tracker.should_switch_orchestrator():
            print("🔄 Orchestrator switch recommended!")
            break
    
    # Save session data
    tracker.save_session_data()

if __name__ == "__main__":
    asyncio.run(main())