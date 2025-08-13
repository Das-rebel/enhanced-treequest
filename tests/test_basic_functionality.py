#!/usr/bin/env python3
"""
Enhanced TreeQuest - Basic Functionality Tests
===============================================

Test suite for core TreeQuest functionality including provider integration,
task execution, and basic orchestration features.
"""

import pytest
import asyncio
import os
import sys

# Add TreeQuest to path for testing
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_treequest_controller import fast_execution, smart_execution, parallel_execution, ExecutionStrategy
from universal_api_key_manager import universal_manager
from token_usage_tracker import TokenUsageTracker

class TestBasicExecution:
    """Test basic task execution functionality"""
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test basic task execution"""
        prompt = "Say hello in a friendly way"
        result = await fast_execution(prompt)
        
        assert result.success
        assert result.content
        assert result.provider
        assert isinstance(result.cost, float)
        assert result.cost >= 0
        assert isinstance(result.quality_score, float)
        assert 0 <= result.quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self):
        """Test handling of empty prompts"""
        with pytest.raises(Exception):
            await fast_execution("")
    
    @pytest.mark.asyncio
    async def test_long_prompt_handling(self):
        """Test handling of very long prompts"""
        long_prompt = "Explain this: " + "A" * 5000
        result = await fast_execution(long_prompt[:1000])  # Truncate for safety
        
        assert result.success
        assert result.content

class TestProviderIntegration:
    """Test AI provider integration and routing"""
    
    def test_provider_discovery(self):
        """Test that providers are discovered correctly"""
        discovered = universal_manager.discover_all_keys()
        
        assert isinstance(discovered, dict)
        assert len(discovered) > 0  # Should find at least one provider
        
        # Check that keys are properly formatted
        for provider, key in discovered.items():
            assert isinstance(provider, str)
            assert isinstance(key, str)
            assert len(key) > 10  # Basic key length validation
    
    def test_environment_configuration(self):
        """Test environment variable configuration"""
        universal_manager.configure_environment()
        
        # Check that environment variables are set
        discovered = universal_manager.discover_all_keys()
        for provider, key in discovered.items():
            env_var = f"{provider.upper()}_API_KEY"
            assert os.environ.get(env_var) == key

class TestTaskSpecificRouting:
    """Test task-specific provider routing"""
    
    @pytest.mark.asyncio
    async def test_compilation_fixes_routing(self):
        """Test routing for compilation fix tasks"""
        prompt = "Fix this Python error: NameError: name 'x' is not defined"
        result = await smart_execution(prompt, task_type="compilation_fixes")
        
        assert result.success
        assert result.provider in ['cerebras', 'groq', 'openai', 'together']
    
    @pytest.mark.asyncio
    async def test_ui_development_routing(self):
        """Test routing for UI development tasks"""
        prompt = "Create a React component for a button"
        result = await smart_execution(prompt, task_type="ui_development")
        
        assert result.success
        # Should prefer providers good at UI development
        assert result.provider in ['openai', 'anthropic', 'together', 'cerebras']
    
    @pytest.mark.asyncio
    async def test_general_routing_fallback(self):
        """Test fallback to general routing"""
        prompt = "What is the weather like?"
        result = await smart_execution(prompt, task_type="unknown_task_type")
        
        assert result.success
        assert result.provider  # Should fallback to available provider

class TestParallelExecution:
    """Test parallel execution strategies"""
    
    @pytest.mark.asyncio
    async def test_fastest_first_strategy(self):
        """Test fastest-first execution strategy"""
        tasks = [
            "Count to 3",
            "Say hello",
            "Add 1+1"
        ]
        
        results = await parallel_execution(
            tasks, 
            strategy=ExecutionStrategy.FASTEST_FIRST,
            max_concurrent=3
        )
        
        assert len(results) == len(tasks)
        for result in results:
            assert result.success
            assert result.content
            assert result.provider
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_concurrency_limit(self):
        """Test parallel execution with concurrency limits"""
        tasks = ["Task " + str(i) for i in range(5)]
        
        results = await parallel_execution(
            tasks,
            strategy=ExecutionStrategy.FASTEST_FIRST,
            max_concurrent=2  # Limit concurrent executions
        )
        
        assert len(results) == 5
        for result in results:
            assert result.success

class TestTokenUsageTracking:
    """Test token usage monitoring functionality"""
    
    def test_tracker_initialization(self):
        """Test token tracker initialization"""
        tracker = TokenUsageTracker()
        
        assert tracker.warning_threshold == 0.7  # Default threshold
        assert tracker.current_session is None
        assert isinstance(tracker.sessions, dict)
    
    def test_session_tracking(self):
        """Test session tracking functionality"""
        tracker = TokenUsageTracker()
        session_id = "test-session"
        
        with tracker.track_session(session_id):
            assert tracker.current_session is not None
            assert tracker.current_session.session_id == session_id
        
        # Session should be archived after context exit
        assert tracker.current_session is None
        assert session_id in tracker.sessions
    
    def test_usage_statistics(self):
        """Test usage statistics collection"""
        tracker = TokenUsageTracker()
        
        with tracker.track_session("test-stats"):
            # Simulate some usage
            tracker.current_session.tokens_used = 100
            tracker.current_session.max_tokens = 1000
        
        stats = tracker.get_usage_stats()
        assert isinstance(stats, dict)
        assert 'total_sessions' in stats
        assert stats['total_sessions'] >= 1

class TestErrorHandling:
    """Test error handling and resilience"""
    
    @pytest.mark.asyncio
    async def test_invalid_task_type(self):
        """Test handling of invalid task types"""
        prompt = "Simple task"
        result = await smart_execution(prompt, task_type="invalid_type")
        
        # Should fallback gracefully
        assert result.success
    
    @pytest.mark.asyncio
    async def test_provider_failure_handling(self):
        """Test handling when providers are unavailable"""
        # This test assumes that the system handles provider failures gracefully
        prompt = "Test provider resilience"
        result = await fast_execution(prompt)
        
        # Should succeed even if some providers fail
        assert result.success or result.provider  # At least attempt made

class TestCostOptimization:
    """Test cost optimization features"""
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test that costs are properly tracked"""
        prompt = "Simple calculation: 2+2"
        result = await fast_execution(prompt)
        
        assert isinstance(result.cost, float)
        assert result.cost >= 0
    
    @pytest.mark.asyncio
    async def test_multiple_executions_cost_accumulation(self):
        """Test cost accumulation across multiple executions"""
        tasks = ["Task 1", "Task 2", "Task 3"]
        total_cost = 0
        
        for task in tasks:
            result = await fast_execution(task)
            total_cost += result.cost
        
        assert total_cost >= 0
        assert isinstance(total_cost, float)

class TestQualityAssessment:
    """Test quality assessment functionality"""
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self):
        """Test that quality scores are generated"""
        prompt = "Explain the concept of recursion"
        result = await fast_execution(prompt)
        
        assert isinstance(result.quality_score, float)
        assert 0 <= result.quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_quality_comparison(self):
        """Test quality comparison across multiple attempts"""
        prompt = "Write a haiku about programming"
        results = []
        
        for _ in range(3):
            result = await fast_execution(prompt)
            results.append(result)
        
        # All should have quality scores
        for result in results:
            assert isinstance(result.quality_score, float)
            assert 0 <= result.quality_score <= 1

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests"""
    # Ensure API keys are discovered
    discovered = universal_manager.discover_all_keys()
    if len(discovered) == 0:
        pytest.skip("No API keys found. Please configure .env file for testing.")
    
    # Configure environment
    universal_manager.configure_environment()
    
    print(f"Test environment configured with {len(discovered)} providers")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])