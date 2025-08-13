#!/usr/bin/env python3
"""
Test Model-wise Usage Tracking Integration
Tests the updated system with all discovered API keys and model-wise tracking
"""

import asyncio
import sys
import os
import time
from typing import Dict, List

# Add paths
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')
sys.path.append('/Users/Subho/CascadeProjects/multi-ai-treequest')

from universal_api_key_manager import UniversalAPIKeyManager
from token_usage_tracker import TokenUsageTracker, setup_token_tracking

async def test_model_wise_usage():
    """Test model-wise usage tracking with discovered API keys"""
    print("🧪 Testing Model-wise Usage Tracking Integration")
    print("=" * 60)
    
    # 1. Test API Key Discovery (with fix)
    print("\n1️⃣ API Key Discovery Test...")
    api_manager = UniversalAPIKeyManager()
    discovered = api_manager.discover_all_keys()
    
    print(f"   ✅ Discovered {len(discovered)} API keys")
    for provider, key in discovered.items():
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        print(f"   🔑 {provider}: {masked_key}")
    
    # Configure environment variables for all discovered keys
    api_manager.configure_environment()
    print(f"   🔧 Environment configured with {len(discovered)} providers")
    
    # 2. Test Token Usage Tracking Setup
    print("\n2️⃣ Token Usage Tracking Setup...")
    tracker = setup_token_tracking(
        session_limit=100000,
        warning_threshold=0.7
    )
    
    print(f"   📊 Session ID: {tracker.current_session.session_id}")
    print(f"   🚨 Warning threshold: {tracker.warning_threshold*100:.0f}%")
    
    # Setup model-wise tracking callbacks
    def on_warning(session):
        print(f"   🔔 TOKEN WARNING: {session.usage_percentage:.1f}% used")
        active_providers = list(discovered.keys())
        print(f"   🤖 Available providers: {', '.join(active_providers[:5])}...")
    
    def on_critical(session):
        print(f"   ⚠️ CRITICAL: {session.usage_percentage:.1f}% - Switch to alternate model!")
    
    tracker.register_threshold_callback('warning', on_warning)
    tracker.register_threshold_callback('critical', on_critical)
    print("   ✅ Model-wise callbacks registered")
    
    # 3. Test Model Usage Simulation
    print("\n3️⃣ Model Usage Simulation...")
    
    # Simulate usage with different models
    model_usage = {
        'openai': 5000,
        'cerebras': 2000,
        'google': 3000,
        'groq': 1000,
        'together': 2500
    }
    
    print("   📈 Simulating model usage:")
    for model, tokens in model_usage.items():
        if model in discovered:
            tracker.track_api_call(tokens // 2, tokens // 2)
            print(f"   🤖 {model}: {tokens} tokens")
    
    # Get current usage
    usage_summary = tracker.get_usage_summary()
    print(f"\n   📊 Total Usage: {usage_summary['current_tokens']:,} tokens")
    print(f"   📊 Usage Percentage: {usage_summary['percentage']:.1f}%")
    
    # 4. Test Provider Routing Based on Usage
    print("\n4️⃣ Provider Routing Test...")
    
    # Get optimal providers for different scenarios
    cost_providers = api_manager.get_optimal_providers('cost')
    speed_providers = api_manager.get_optimal_providers('urgent')
    quality_providers = api_manager.get_optimal_providers('quality')
    
    print(f"   💰 Cost-optimized: {' → '.join(cost_providers[:3])}")
    print(f"   🏃 Speed-optimized: {' → '.join(speed_providers[:3])}")
    print(f"   🎯 Quality-optimized: {' → '.join(quality_providers[:3])}")
    
    # 5. Test Threshold Detection
    print("\n5️⃣ Threshold Detection Test...")
    
    # Simulate high usage to trigger thresholds
    print("   📈 Simulating high token usage...")
    for i in range(5):
        tracker.track_input("Large prompt simulation " * 2000, "simulation")
        current_usage = tracker.get_usage_summary()
        print(f"   Step {i+1}: {current_usage['percentage']:.1f}%")
        
        if current_usage['should_switch']:
            print("   🔄 ORCHESTRATOR SWITCH RECOMMENDED!")
            break
        
        await asyncio.sleep(0.1)
    
    # 6. Generate Usage Report
    print("\n6️⃣ Usage Report...")
    report = tracker.get_usage_report()
    print(report)
    
    # 7. Test Configuration Persistence
    print("\n7️⃣ Configuration Persistence...")
    api_manager.save_configuration()
    tracker.save_session_data()
    print("   💾 Configuration and session data saved")
    
    print("\n✅ Model-wise Usage Tracking Integration Test Complete!")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   🔑 API Keys Discovered: {len(discovered)}")
    print(f"   📊 Token Usage: {usage_summary['percentage']:.1f}%")
    print(f"   🤖 Model Providers: {', '.join(discovered.keys())}")
    print(f"   ⚠️ Switch Recommended: {'Yes' if usage_summary['should_switch'] else 'No'}")
    
    return True

async def test_provider_selection():
    """Test intelligent provider selection based on usage and cost"""
    print("\n🎯 Testing Intelligent Provider Selection")
    print("=" * 50)
    
    api_manager = UniversalAPIKeyManager()
    discovered = api_manager.discover_all_keys()
    
    # Test different task scenarios
    scenarios = [
        ("urgent", "Quick debugging task"),
        ("quality", "Complex architecture design"),
        ("cost", "Simple content generation"),
        ("research", "Technical research and analysis")
    ]
    
    for task_type, description in scenarios:
        optimal_providers = api_manager.get_optimal_providers(task_type)
        print(f"   📝 {description}")
        print(f"   🎯 Optimal providers ({task_type}): {' → '.join(optimal_providers[:3])}")
        print()
    
    print("✅ Provider selection test complete!")

if __name__ == "__main__":
    asyncio.run(test_model_wise_usage())
    asyncio.run(test_provider_selection())