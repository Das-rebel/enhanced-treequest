#!/usr/bin/env python3
"""
Enhanced TreeQuest - Basic Usage Examples
==========================================

This file demonstrates the fundamental usage patterns of Enhanced TreeQuest.
Perfect for getting started and understanding core concepts.
"""

import asyncio
import sys
import os

# Add TreeQuest to path
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')

from enhanced_treequest_controller import fast_execution, smart_execution

async def example_1_simple_execution():
    """Example 1: Simple task execution with automatic provider selection"""
    print("=== Example 1: Simple Execution ===")
    
    prompt = "Write a Python function to check if a number is prime"
    
    print(f"Task: {prompt}")
    result = await fast_execution(prompt)
    
    print(f"✅ Provider used: {result.provider}")
    print(f"💰 Cost: ${result.cost:.6f}")
    print(f"📄 Response:\n{result.content}")
    print()

async def example_2_task_specific_routing():
    """Example 2: Task-specific provider routing for optimal results"""
    print("=== Example 2: Task-Specific Routing ===")
    
    # Different types of tasks that benefit from specific providers
    tasks = [
        ("Fix this TypeScript error: Property 'x' does not exist on type", "compilation_fixes"),
        ("Create a React component for a user profile card", "ui_development"),
        ("Design a database schema for an e-commerce system", "database_fixes"),
        ("Implement natural language search functionality", "ai_features")
    ]
    
    for prompt, task_type in tasks:
        print(f"Task type: {task_type}")
        print(f"Prompt: {prompt}")
        
        result = await smart_execution(prompt, task_type=task_type)
        
        print(f"✅ Selected provider: {result.provider}")
        print(f"💰 Cost: ${result.cost:.6f}")
        print(f"📊 Quality score: {result.quality_score:.2f}")
        print("---")

async def example_3_cost_aware_execution():
    """Example 3: Cost-aware execution with budget considerations"""
    print("=== Example 3: Cost-Aware Execution ===")
    
    # For cost-sensitive tasks, route to efficient providers
    prompt = "Summarize the main benefits of cloud computing"
    
    # Use cost-optimized routing
    result = await smart_execution(prompt, task_type="cost_sensitive")
    
    print(f"Task: {prompt}")
    print(f"✅ Cost-optimized provider: {result.provider}")
    print(f"💰 Cost: ${result.cost:.6f}")
    print(f"📄 Summary: {result.content[:200]}...")
    print()

async def example_4_multiple_attempts():
    """Example 4: Multiple execution attempts for reliability"""
    print("=== Example 4: Multiple Attempts for Reliability ===")
    
    prompt = "Generate a creative marketing slogan for a tech startup"
    
    print(f"Task: {prompt}")
    print("Running 3 attempts for comparison...")
    
    results = []
    for i in range(3):
        result = await fast_execution(prompt)
        results.append(result)
        print(f"Attempt {i+1}: {result.provider} - Quality: {result.quality_score:.2f}")
    
    # Find best result
    best_result = max(results, key=lambda r: r.quality_score)
    print(f"\n🏆 Best result (Quality: {best_result.quality_score:.2f}):")
    print(f"Provider: {best_result.provider}")
    print(f"Slogan: {best_result.content}")
    print()

async def example_5_error_handling():
    """Example 5: Proper error handling and fallback strategies"""
    print("=== Example 5: Error Handling ===")
    
    prompts = [
        "Write a simple hello world program",  # Should succeed
        "",  # Empty prompt - may cause issues
        "A" * 10000,  # Very long prompt - may hit limits
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Test {i}: {'Valid prompt' if len(prompt) < 100 else 'Edge case prompt'}")
        
        try:
            result = await fast_execution(prompt[:100] + "..." if len(prompt) > 100 else prompt)
            print(f"✅ Success: {result.provider}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("💡 TreeQuest automatically tries fallback providers")
        
        print("---")

async def main():
    """Run all basic usage examples"""
    print("🌳 Enhanced TreeQuest - Basic Usage Examples")
    print("=" * 50)
    
    try:
        await example_1_simple_execution()
        await example_2_task_specific_routing()
        await example_3_cost_aware_execution()
        await example_4_multiple_attempts()
        await example_5_error_handling()
        
        print("🎉 All examples completed successfully!")
        print("💡 Ready to use Enhanced TreeQuest in your projects")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Make sure your API keys are configured in .env")

if __name__ == "__main__":
    asyncio.run(main())