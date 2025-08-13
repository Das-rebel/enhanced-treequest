#!/usr/bin/env python3
"""
Enhanced TreeQuest - Advanced Usage Examples
=============================================

This file demonstrates advanced features and patterns for power users.
Includes parallel execution, token monitoring, and custom orchestration.
"""

import asyncio
import sys
import os
import time
from typing import List, Dict

# Add TreeQuest to path
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')

from enhanced_treequest_controller import (
    parallel_execution, 
    ExecutionStrategy,
    fast_execution
)
from token_usage_tracker import TokenUsageTracker
from universal_api_key_manager import universal_manager

async def example_1_parallel_strategies():
    """Example 1: Different parallel execution strategies"""
    print("=== Example 1: Parallel Execution Strategies ===")
    
    tasks = [
        "Create a Python class for managing user sessions",
        "Write a JavaScript function for data validation",
        "Design a REST API endpoint for file uploads",
        "Implement a caching mechanism using Redis"
    ]
    
    strategies = [
        (ExecutionStrategy.FASTEST_FIRST, "Fastest Response"),
        (ExecutionStrategy.QUALITY_CONSENSUS, "Best Quality"),
        (ExecutionStrategy.COST_OPTIMIZED, "Lowest Cost")
    ]
    
    for strategy, description in strategies:
        print(f"\n🚀 Strategy: {description}")
        print("-" * 30)
        
        start_time = time.time()
        
        # Execute tasks in parallel
        results = await parallel_execution(
            tasks, 
            strategy=strategy,
            max_concurrent=4
        )
        
        execution_time = time.time() - start_time
        total_cost = sum(r.cost for r in results)
        avg_quality = sum(r.quality_score for r in results) / len(results)
        
        print(f"⏱️  Total time: {execution_time:.2f}s")
        print(f"💰 Total cost: ${total_cost:.6f}")
        print(f"📊 Average quality: {avg_quality:.2f}")
        
        print("Providers used:")
        for i, result in enumerate(results):
            print(f"  Task {i+1}: {result.provider}")

async def example_2_token_monitoring():
    """Example 2: Advanced token usage monitoring"""
    print("\n=== Example 2: Token Usage Monitoring ===")
    
    tracker = TokenUsageTracker(warning_threshold=0.5)  # 50% threshold for demo
    
    # Simulate a session with multiple tasks
    session_id = "advanced-demo-session"
    
    print(f"🔄 Starting monitored session: {session_id}")
    
    tasks = [
        "Explain machine learning concepts",
        "Create a comprehensive Python tutorial",
        "Design a microservices architecture",
        "Write documentation for an API",
        "Analyze code performance bottlenecks"
    ]
    
    with tracker.track_session(session_id):
        for i, task in enumerate(tasks, 1):
            print(f"\n📝 Task {i}: {task[:40]}...")
            
            result = await fast_execution(task)
            
            # Check token usage after each task
            current_session = tracker.current_session
            if current_session:
                usage_pct = current_session.usage_percentage
                print(f"📊 Token usage: {usage_pct:.1f}%")
                
                if tracker.should_switch_orchestrator():
                    print("⚠️  Warning: Approaching token limit!")
                    print("🔄 Recommendation: Switch to external orchestrator")
                    break
            
            print(f"✅ Completed with {result.provider}")
    
    # Show final usage statistics
    stats = tracker.get_usage_stats()
    print(f"\n📈 Session Summary:")
    print(f"  Total sessions: {stats.get('total_sessions', 0)}")
    print(f"  Active sessions: {stats.get('active_sessions', 0)}")

async def example_3_custom_provider_selection():
    """Example 3: Custom provider selection logic"""
    print("\n=== Example 3: Custom Provider Selection ===")
    
    # Get available providers
    discovered = universal_manager.discover_all_keys()
    print(f"🔍 Available providers: {list(discovered.keys())}")
    
    # Define custom selection criteria
    def select_provider_for_task(task: str, available_providers: List[str]) -> str:
        """Custom logic for provider selection"""
        if "fast" in task.lower() or "quick" in task.lower():
            # Prefer speed providers
            speed_providers = ['groq', 'cerebras', 'together']
            for provider in speed_providers:
                if provider in available_providers:
                    return provider
        
        elif "creative" in task.lower() or "design" in task.lower():
            # Prefer creative providers
            creative_providers = ['openai', 'anthropic', 'together']
            for provider in creative_providers:
                if provider in available_providers:
                    return provider
        
        elif "research" in task.lower() or "analysis" in task.lower():
            # Prefer research providers
            research_providers = ['perplexity', 'google', 'openai']
            for provider in research_providers:
                if provider in available_providers:
                    return provider
        
        # Default fallback
        return available_providers[0] if available_providers else 'openai'
    
    # Test custom selection
    test_tasks = [
        "Quick debugging help for Python error",
        "Creative marketing copy for a product launch",
        "Research analysis on quantum computing trends"
    ]
    
    available = list(discovered.keys())
    
    for task in test_tasks:
        selected = select_provider_for_task(task, available)
        print(f"📝 Task: {task}")
        print(f"🎯 Selected provider: {selected}")
        
        # Execute with selected provider logic (simplified)
        result = await fast_execution(task)
        print(f"✅ Actually used: {result.provider}")
        print("---")

async def example_4_batch_processing():
    """Example 4: Efficient batch processing with optimization"""
    print("\n=== Example 4: Batch Processing ===")
    
    # Simulate a large batch of similar tasks
    code_review_tasks = [
        "Review this function for potential bugs: def calculate_total(items): return sum(item.price for item in items)",
        "Check this SQL query for security issues: SELECT * FROM users WHERE id = ?",
        "Analyze this React component for performance: const UserList = ({users}) => users.map(user => <div>{user.name}</div>)",
        "Review error handling in: try { fetchData() } catch(e) { console.log(e) }",
        "Check this API endpoint: @app.route('/api/users/<int:user_id>')"
    ]
    
    print(f"📋 Processing {len(code_review_tasks)} code review tasks...")
    
    # Batch processing with optimal concurrency
    start_time = time.time()
    
    results = await parallel_execution(
        code_review_tasks,
        strategy=ExecutionStrategy.FASTEST_FIRST,
        max_concurrent=3  # Limit concurrent requests
    )
    
    processing_time = time.time() - start_time
    
    print(f"⏱️  Batch completed in {processing_time:.2f} seconds")
    print(f"💰 Total cost: ${sum(r.cost for r in results):.6f}")
    print(f"📊 Average quality: {sum(r.quality_score for r in results) / len(results):.2f}")
    
    # Show provider distribution
    provider_usage = {}
    for result in results:
        provider_usage[result.provider] = provider_usage.get(result.provider, 0) + 1
    
    print("🔄 Provider distribution:")
    for provider, count in provider_usage.items():
        print(f"  {provider}: {count} tasks")

async def example_5_adaptive_retry_logic():
    """Example 5: Adaptive retry logic with intelligent fallbacks"""
    print("\n=== Example 5: Adaptive Retry Logic ===")
    
    async def execute_with_retry(prompt: str, max_retries: int = 3) -> Dict:
        """Execute task with intelligent retry logic"""
        attempts = []
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                result = await fast_execution(prompt)
                
                # Check if result meets quality threshold
                if result.quality_score >= 0.7:  # Quality threshold
                    print(f"✅ Success on attempt {attempt + 1}")
                    return {
                        'success': True,
                        'result': result,
                        'attempts': attempt + 1
                    }
                else:
                    print(f"⚠️  Low quality ({result.quality_score:.2f}), retrying...")
                    attempts.append(result)
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                attempts.append(None)
        
        # If all retries failed, return best attempt
        valid_attempts = [a for a in attempts if a is not None]
        if valid_attempts:
            best_attempt = max(valid_attempts, key=lambda x: x.quality_score)
            return {
                'success': False,
                'result': best_attempt,
                'attempts': max_retries,
                'note': 'Returned best of failed attempts'
            }
        
        return {
            'success': False,
            'result': None,
            'attempts': max_retries,
            'note': 'All attempts failed'
        }
    
    # Test adaptive retry
    test_prompt = "Explain the concept of blockchain in simple terms"
    
    result = await execute_with_retry(test_prompt)
    
    print(f"📊 Final result:")
    print(f"  Success: {result['success']}")
    print(f"  Attempts used: {result['attempts']}")
    if result['result']:
        print(f"  Provider: {result['result'].provider}")
        print(f"  Quality: {result['result'].quality_score:.2f}")
    if 'note' in result:
        print(f"  Note: {result['note']}")

async def example_6_performance_benchmarking():
    """Example 6: Performance benchmarking across providers"""
    print("\n=== Example 6: Performance Benchmarking ===")
    
    benchmark_task = "Write a Python function to implement binary search"
    
    # Get available providers
    discovered = universal_manager.discover_all_keys()
    available_providers = list(discovered.keys())[:5]  # Test first 5 providers
    
    print(f"🏁 Benchmarking task across {len(available_providers)} providers...")
    print(f"Task: {benchmark_task}")
    
    benchmark_results = []
    
    for provider in available_providers:
        print(f"\n🔄 Testing {provider}...")
        
        start_time = time.time()
        try:
            # Note: This is a simplified benchmark
            # In real implementation, you'd force specific provider usage
            result = await fast_execution(benchmark_task)
            execution_time = time.time() - start_time
            
            benchmark_results.append({
                'provider': provider,
                'success': True,
                'time': execution_time,
                'cost': result.cost,
                'quality': result.quality_score,
                'actual_provider': result.provider  # May differ due to routing
            })
            
            print(f"✅ Success in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            benchmark_results.append({
                'provider': provider,
                'success': False,
                'time': execution_time,
                'error': str(e)
            })
            print(f"❌ Failed in {execution_time:.2f}s: {e}")
    
    # Show benchmark summary
    print(f"\n📊 Benchmark Results Summary:")
    print("=" * 50)
    
    successful_results = [r for r in benchmark_results if r['success']]
    
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['time'])
        cheapest = min(successful_results, key=lambda x: x['cost'])
        highest_quality = max(successful_results, key=lambda x: x['quality'])
        
        print(f"🏃 Fastest: {fastest['provider']} ({fastest['time']:.2f}s)")
        print(f"💰 Cheapest: {cheapest['provider']} (${cheapest['cost']:.6f})")
        print(f"🏆 Highest Quality: {highest_quality['provider']} ({highest_quality['quality']:.2f})")

async def main():
    """Run all advanced usage examples"""
    print("🌳 Enhanced TreeQuest - Advanced Usage Examples")
    print("=" * 60)
    
    try:
        await example_1_parallel_strategies()
        await example_2_token_monitoring()
        await example_3_custom_provider_selection()
        await example_4_batch_processing()
        await example_5_adaptive_retry_logic()
        await example_6_performance_benchmarking()
        
        print("\n🎉 All advanced examples completed!")
        print("💡 You're now ready to build sophisticated AI orchestration workflows")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure your API keys are configured and providers are available")

if __name__ == "__main__":
    asyncio.run(main())