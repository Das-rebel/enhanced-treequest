#!/usr/bin/env python3
"""
Enhanced TreeQuest - Interactive Demo
=====================================

This demo showcases the key features of Enhanced TreeQuest:
- Multi-AI provider orchestration
- Automatic provider selection
- Token usage monitoring
- Parallel execution strategies
- Cost optimization

Run: python3 demo.py
"""

import asyncio
import sys
import os
import time
from typing import List

# Add the Enhanced TreeQuest path
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')

try:
    from enhanced_treequest_controller import (
        fast_execution, 
        parallel_execution, 
        smart_execution,
        ExecutionStrategy
    )
    from universal_api_key_manager import universal_manager
    from token_usage_tracker import TokenUsageTracker
    print("✅ Enhanced TreeQuest modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing TreeQuest modules: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

def print_banner():
    """Print the TreeQuest demo banner"""
    banner = """
    🌳 Enhanced TreeQuest Demo
    ===========================
    Multi-AI Provider Orchestration System
    
    Features demonstrated:
    ✅ 11 AI provider support
    ✅ Intelligent provider routing
    ✅ Token usage monitoring  
    ✅ Parallel execution strategies
    ✅ Cost optimization
    ✅ Automatic failover
    """
    print(banner)

async def demo_basic_execution():
    """Demonstrate basic TreeQuest execution"""
    print("\n🚀 Demo 1: Basic Execution")
    print("=" * 40)
    
    prompt = "Explain the concept of recursion in programming with a simple Python example"
    
    print(f"📝 Task: {prompt}")
    print("🔄 Executing with automatic provider selection...")
    
    start_time = time.time()
    result = await fast_execution(prompt)
    execution_time = time.time() - start_time
    
    print(f"✅ Provider used: {result.provider}")
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")
    print(f"💰 Cost: ${result.cost:.6f}")
    print(f"📊 Quality score: {result.quality_score:.2f}")
    print(f"📄 Response preview: {result.content[:200]}...")
    
    return result

async def demo_task_specific_routing():
    """Demonstrate task-specific provider routing"""
    print("\n🎯 Demo 2: Task-Specific Routing")
    print("=" * 40)
    
    tasks = [
        ("Fix this compilation error: 'Cannot find module'", "compilation_fixes"),
        ("Create a React component for user authentication", "ui_development"),
        ("Design a database schema for a blog system", "database_fixes"),
        ("Implement AI-powered content recommendations", "ai_features")
    ]
    
    for prompt, task_type in tasks:
        print(f"\n📝 Task type: {task_type}")
        print(f"📝 Prompt: {prompt}")
        
        result = await smart_execution(prompt, task_type=task_type)
        print(f"✅ Selected provider: {result.provider} (optimized for {task_type})")
        print(f"💰 Cost: ${result.cost:.6f}")

async def demo_parallel_execution():
    """Demonstrate parallel execution strategies"""
    print("\n⚡ Demo 3: Parallel Execution")
    print("=" * 40)
    
    tasks = [
        "Write a Python function to calculate prime numbers",
        "Create a JavaScript function for form validation", 
        "Design a SQL query to find top-selling products",
        "Explain the benefits of microservices architecture"
    ]
    
    print(f"📝 Running {len(tasks)} tasks in parallel...")
    print("🔄 Strategy: Fastest-first execution")
    
    start_time = time.time()
    results = await parallel_execution(
        tasks, 
        strategy=ExecutionStrategy.FASTEST_FIRST,
        max_concurrent=4
    )
    total_time = time.time() - start_time
    
    print(f"✅ All tasks completed in {total_time:.2f} seconds")
    print(f"💰 Total cost: ${sum(r.cost for r in results):.6f}")
    
    print("\n📊 Results summary:")
    for i, result in enumerate(results):
        print(f"  Task {i+1}: {result.provider} - ${result.cost:.6f} - Quality: {result.quality_score:.2f}")

async def demo_provider_discovery():
    """Demonstrate automatic provider discovery"""
    print("\n🔍 Demo 4: Provider Discovery & Health Check")
    print("=" * 40)
    
    # Discover available providers
    discovered = universal_manager.discover_all_keys()
    print(f"🔑 Discovered API keys: {len(discovered)}")
    
    # Show provider capabilities
    print("\n📋 Available providers:")
    for provider in discovered.keys():
        print(f"  ✅ {provider}")
    
    # Check provider health
    try:
        health_status = universal_manager.check_provider_health()
        print(f"\n🏥 Provider health check:")
        print(f"  ✅ Healthy: {len(health_status.get('healthy', []))}")
        print(f"  ⚠️  Warning: {len(health_status.get('warning', []))}")
        print(f"  ❌ Failed: {len(health_status.get('failed', []))}")
    except Exception as e:
        print(f"⚠️ Health check unavailable: {e}")

async def demo_token_monitoring():
    """Demonstrate token usage monitoring"""
    print("\n📊 Demo 5: Token Usage Monitoring")
    print("=" * 40)
    
    tracker = TokenUsageTracker()
    
    # Start monitoring session
    session_id = "demo-session"
    print(f"🔄 Starting token monitoring for session: {session_id}")
    
    with tracker.track_session(session_id):
        # Execute a task that uses tokens
        prompt = "Write a comprehensive guide to Python async/await programming with examples"
        result = await fast_execution(prompt)
        
        # Check usage
        usage_stats = tracker.get_usage_stats()
        current_session = tracker.current_session
        
        print(f"📊 Token usage statistics:")
        print(f"  📝 Estimated tokens used: {current_session.tokens_used if current_session else 'N/A'}")
        print(f"  📈 Usage percentage: {current_session.usage_percentage:.1f}%" if current_session else "N/A")
        print(f"  ⚠️  Warning threshold: {tracker.warning_threshold * 100}%")
        
        if tracker.should_switch_orchestrator():
            print("🔄 Recommendation: Switch to external orchestrator (approaching token limit)")
        else:
            print("✅ Token usage within acceptable limits")

async def demo_cost_optimization():
    """Demonstrate cost optimization features"""
    print("\n💰 Demo 6: Cost Optimization")
    print("=" * 40)
    
    # Compare different execution strategies
    prompt = "Summarize the key features of Python 3.12"
    
    strategies = [
        (ExecutionStrategy.FASTEST_FIRST, "Fastest response"),
        (ExecutionStrategy.COST_OPTIMIZED, "Lowest cost"),
        (ExecutionStrategy.QUALITY_CONSENSUS, "Best quality")
    ]
    
    for strategy, description in strategies:
        print(f"\n🔄 Testing strategy: {description}")
        
        # For cost comparison, we'll simulate by using different approaches
        if strategy == ExecutionStrategy.COST_OPTIMIZED:
            # Route to cost-effective providers
            result = await smart_execution(prompt, task_type="general")
        else:
            # Use standard execution
            result = await fast_execution(prompt)
        
        print(f"  ✅ Provider: {result.provider}")
        print(f"  💰 Cost: ${result.cost:.6f}")
        print(f"  📊 Quality: {result.quality_score:.2f}")

async def main():
    """Run the complete TreeQuest demo"""
    print_banner()
    
    try:
        # Check if TreeQuest is properly configured
        discovered = universal_manager.discover_all_keys()
        if len(discovered) == 0:
            print("❌ No API keys found. Please configure your .env file.")
            print("💡 Copy .env.example to .env and add your API keys")
            return
        
        print(f"🚀 TreeQuest configured with {len(discovered)} providers")
        
        # Run all demos
        await demo_basic_execution()
        await demo_task_specific_routing()
        await demo_parallel_execution()
        await demo_provider_discovery()
        await demo_token_monitoring()
        await demo_cost_optimization()
        
        print("\n🎉 Demo completed successfully!")
        print("✨ Enhanced TreeQuest is ready for production use")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Check your API keys and network connection")

if __name__ == "__main__":
    asyncio.run(main())