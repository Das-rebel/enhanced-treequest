#!/usr/bin/env python3
"""
Universal API Key Manager for Enhanced TreeQuest
Auto-discovers, validates, and manages API keys across all AI providers
Supports 15+ providers with health checking and rotation capabilities
"""

import os
import asyncio
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import logging
from pathlib import Path

class ProviderStatus(Enum):
    """API provider status levels"""
    ACTIVE = "active"           # Working with valid key
    INVALID_KEY = "invalid_key" # Key exists but invalid
    RATE_LIMITED = "rate_limited" # Temporarily rate limited
    NO_KEY = "no_key"          # No API key found
    OFFLINE = "offline"        # Provider service unavailable
    QUOTA_EXCEEDED = "quota_exceeded" # API quota exceeded

@dataclass
class ProviderHealth:
    """Health status for an AI provider"""
    provider: str
    status: ProviderStatus
    api_key_present: bool
    last_check: float
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    quota_remaining: Optional[int] = None
    rate_limit_reset: Optional[float] = None
    monthly_usage: float = 0.0
    cost_per_token: float = 0.0

class UniversalAPIKeyManager:
    """Comprehensive API key management for all AI providers"""
    
    # Comprehensive provider configuration with 2025 updates
    PROVIDER_CONFIG = {
        # Commercial Providers
        'openai': {
            'env_keys': ['OPENAI_API_KEY', 'OPENAI_TOKEN'],
            'health_endpoint': 'https://api.openai.com/v1/models',
            'cost_per_1k_tokens': 0.03,  # GPT-4o pricing
            'header_format': 'Bearer {key}',
            'test_model': 'gpt-4o',
            'priority': 1
        },
        'anthropic': {
            'env_keys': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
            'health_endpoint': 'https://api.anthropic.com/v1/messages',
            'cost_per_1k_tokens': 0.015,  # Claude 3.5 Sonnet pricing
            'header_format': 'Bearer {key}',
            'test_model': 'claude-3-5-sonnet-20241022',
            'priority': 2
        },
        'google': {
            'env_keys': ['GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_AI_KEY'],
            'health_endpoint': 'https://generativelanguage.googleapis.com/v1/models',
            'cost_per_1k_tokens': 0.001,  # Gemini Pro pricing
            'header_format': 'Bearer {key}',
            'test_model': 'gemini-pro',
            'priority': 3
        },
        'cerebras': {
            'env_keys': ['CEREBRAS_API_KEY'],
            'health_endpoint': 'https://api.cerebras.ai/v1/models',
            'cost_per_1k_tokens': 0.0006,  # Ultra-fast inference
            'header_format': 'Bearer {key}',
            'test_model': 'llama3.1-70b',
            'priority': 4
        },
        'groq': {
            'env_keys': ['GROQ_API_KEY'],
            'health_endpoint': 'https://api.groq.com/openai/v1/models',
            'cost_per_1k_tokens': 0.0002,  # Very fast and cheap
            'header_format': 'Bearer {key}',
            'test_model': 'llama-3.1-70b-versatile',
            'priority': 5
        },
        'together': {
            'env_keys': ['TOGETHER_API_KEY', 'TOGETHER_AI_KEY'],
            'health_endpoint': 'https://api.together.xyz/v1/models',
            'cost_per_1k_tokens': 0.0008,  # Good balance
            'header_format': 'Bearer {key}',
            'test_model': 'meta-llama/Llama-3-70b-chat-hf',
            'priority': 6
        },
        'mistral': {
            'env_keys': ['MISTRAL_API_KEY'],
            'health_endpoint': 'https://api.mistral.ai/v1/models',
            'cost_per_1k_tokens': 0.002,  # European AI
            'header_format': 'Bearer {key}',
            'test_model': 'mistral-large-latest',
            'priority': 7
        },
        'openrouter': {
            'env_keys': ['OPENROUTER_API_KEY'],
            'health_endpoint': 'https://openrouter.ai/api/v1/models',
            'cost_per_1k_tokens': 0.001,  # Access to many models
            'header_format': 'Bearer {key}',
            'test_model': 'anthropic/claude-3.5-sonnet',
            'priority': 8
        },
        'xai': {
            'env_keys': ['XAI_API_KEY', 'GROK_API_KEY'],
            'health_endpoint': 'https://api.x.ai/v1/models',
            'cost_per_1k_tokens': 0.005,  # Grok models
            'header_format': 'Bearer {key}',
            'test_model': 'grok-beta',
            'priority': 9
        },
        'deepseek': {
            'env_keys': ['DEEPSEEK_API_KEY'],
            'health_endpoint': 'https://api.deepseek.com/v1/models',
            'cost_per_1k_tokens': 0.0003,  # Competitive Chinese models
            'header_format': 'Bearer {key}',
            'test_model': 'deepseek-chat',
            'priority': 10
        },
        'perplexity': {
            'env_keys': ['PERPLEXITY_API_KEY'],
            'health_endpoint': 'https://api.perplexity.ai/models',
            'cost_per_1k_tokens': 0.002,  # Research-focused
            'header_format': 'Bearer {key}',
            'test_model': 'llama-3.1-sonar-large-128k-online',
            'priority': 11
        },
        
        # Free/Local Providers
        'ollama': {
            'env_keys': ['OLLAMA_API_KEY', 'OLLAMA_HOST'],
            'health_endpoint': 'http://localhost:11434/api/tags',
            'cost_per_1k_tokens': 0.0,  # Free local
            'header_format': None,  # No auth needed
            'test_model': 'llama3.1',
            'priority': 12
        },
        'huggingface': {
            'env_keys': ['HUGGINGFACE_API_KEY', 'HF_TOKEN'],
            'health_endpoint': 'https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium',
            'cost_per_1k_tokens': 0.0,  # Free tier available
            'header_format': 'Bearer {key}',
            'test_model': 'microsoft/DialoGPT-medium',
            'priority': 13
        },
        'replicate': {
            'env_keys': ['REPLICATE_API_TOKEN'],
            'health_endpoint': 'https://api.replicate.com/v1/models',
            'cost_per_1k_tokens': 0.001,  # Per-second billing
            'header_format': 'Token {key}',
            'test_model': 'meta/llama-2-70b-chat',
            'priority': 14
        },
        'cohere': {
            'env_keys': ['COHERE_API_KEY'],
            'health_endpoint': 'https://api.cohere.ai/v1/models',
            'cost_per_1k_tokens': 0.0015,  # Command models
            'header_format': 'Bearer {key}',
            'test_model': 'command',
            'priority': 15
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.path.expanduser("~/.treequest/api_keys.json")
        self.health_status: Dict[str, ProviderHealth] = {}
        self.discovered_keys: Dict[str, str] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def discover_all_keys(self) -> Dict[str, str]:
        """Discover all available API keys from multiple sources"""
        discovered = {}
        
        self.logger.info("🔍 Discovering API keys across all providers...")
        
        for provider, config in self.PROVIDER_CONFIG.items():
            key = self._find_provider_key(provider, config['env_keys'])
            if key:
                discovered[provider] = key
                self.logger.info(f"✅ Found {provider} API key")
        
        # Additional discovery from .env files
        discovered.update(self._discover_from_env_files())
        
        # Discovery from configuration files
        discovered.update(self._discover_from_config_files())
        
        self.discovered_keys = discovered
        self.logger.info(f"🚀 Discovered {len(discovered)} API keys: {', '.join(discovered.keys())}")
        
        return discovered
    
    def _find_provider_key(self, provider: str, env_keys: List[str]) -> Optional[str]:
        """Find API key for a specific provider"""
        for env_key in env_keys:
            value = os.environ.get(env_key)
            if value and len(value.strip()) > 10:  # Valid key length check
                return value.strip()
        return None
    
    def _discover_from_env_files(self) -> Dict[str, str]:
        """Discover keys from .env files in common locations"""
        discovered = {}
        env_locations = [
            '.env',
            os.path.expanduser('~/.env'),
            '/Users/Subho/CascadeProjects/brain-spark-platform/.env',
            '/Users/Subho/CascadeProjects/multi-ai-treequest/.env',  # Main TreeQuest .env file
            '/Users/Subho/.env'
        ]
        
        for env_file in env_locations:
            if os.path.exists(env_file):
                try:
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '=' in line and not line.startswith('#'):
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                
                                # Check if this looks like an API key
                                if 'API_KEY' in key and len(value) > 10:
                                    provider = key.replace('_API_KEY', '').lower()
                                    if provider in self.PROVIDER_CONFIG:
                                        discovered[provider] = value
                                        self.logger.info(f"✅ Found {provider} key in {env_file}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read {env_file}: {e}")
        
        return discovered
    
    def _discover_from_config_files(self) -> Dict[str, str]:
        """Discover keys from configuration files"""
        discovered = {}
        config_locations = [
            os.path.expanduser('~/.config/treequest/config.json'),
            os.path.expanduser('~/.claude/config.json'),
            self.config_file
        ]
        
        for config_file in config_locations:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        api_keys = config.get('api_keys', {})
                        for provider, key in api_keys.items():
                            if provider in self.PROVIDER_CONFIG and len(key) > 10:
                                discovered[provider] = key
                                self.logger.info(f"✅ Found {provider} key in {config_file}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not read {config_file}: {e}")
        
        return discovered
    
    async def validate_all_keys(self) -> Dict[str, ProviderHealth]:
        """Validate all discovered API keys by testing health endpoints"""
        if not self.session:
            async with self:
                return await self.validate_all_keys()
        
        self.logger.info("🔍 Validating all API keys...")
        
        # Validate all discovered keys in parallel
        validation_tasks = []
        for provider, key in self.discovered_keys.items():
            task = self._validate_provider_key(provider, key)
            validation_tasks.append(task)
        
        if validation_tasks:
            health_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(health_results):
                if isinstance(result, Exception):
                    provider = list(self.discovered_keys.keys())[i]
                    self.health_status[provider] = ProviderHealth(
                        provider=provider,
                        status=ProviderStatus.OFFLINE,
                        api_key_present=True,
                        last_check=time.time(),
                        error_message=str(result)
                    )
                else:
                    self.health_status[result.provider] = result
        
        # Add providers without keys
        for provider in self.PROVIDER_CONFIG:
            if provider not in self.health_status:
                self.health_status[provider] = ProviderHealth(
                    provider=provider,
                    status=ProviderStatus.NO_KEY,
                    api_key_present=False,
                    last_check=time.time()
                )
        
        # Log summary
        active_count = sum(1 for h in self.health_status.values() if h.status == ProviderStatus.ACTIVE)
        self.logger.info(f"✅ Validation complete: {active_count}/{len(self.PROVIDER_CONFIG)} providers active")
        
        return self.health_status
    
    async def _validate_provider_key(self, provider: str, api_key: str) -> ProviderHealth:
        """Validate a specific provider's API key"""
        config = self.PROVIDER_CONFIG[provider]
        start_time = time.time()
        
        try:
            headers = {}
            if config['header_format']:
                auth_header = config['header_format'].format(key=api_key)
                headers['Authorization'] = auth_header
            
            # Special handling for different providers
            if provider == 'anthropic':
                headers['anthropic-version'] = '2023-06-01'
                headers['content-type'] = 'application/json'
            elif provider == 'google':
                # Use API key in URL for Google
                url = f"{config['health_endpoint']}?key={api_key}"
            else:
                url = config['health_endpoint']
            
            async with self.session.get(url, headers=headers) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    return ProviderHealth(
                        provider=provider,
                        status=ProviderStatus.ACTIVE,
                        api_key_present=True,
                        last_check=time.time(),
                        response_time=response_time,
                        cost_per_token=config['cost_per_1k_tokens'] / 1000
                    )
                elif response.status == 401:
                    return ProviderHealth(
                        provider=provider,
                        status=ProviderStatus.INVALID_KEY,
                        api_key_present=True,
                        last_check=time.time(),
                        error_message="Invalid API key"
                    )
                elif response.status == 429:
                    retry_after = response.headers.get('retry-after')
                    return ProviderHealth(
                        provider=provider,
                        status=ProviderStatus.RATE_LIMITED,
                        api_key_present=True,
                        last_check=time.time(),
                        rate_limit_reset=time.time() + (int(retry_after) if retry_after else 60),
                        error_message="Rate limited"
                    )
                else:
                    return ProviderHealth(
                        provider=provider,
                        status=ProviderStatus.OFFLINE,
                        api_key_present=True,
                        last_check=time.time(),
                        error_message=f"HTTP {response.status}"
                    )
        
        except Exception as e:
            return ProviderHealth(
                provider=provider,
                status=ProviderStatus.OFFLINE,
                api_key_present=True,
                last_check=time.time(),
                error_message=str(e)
            )
    
    def get_active_providers(self) -> List[str]:
        """Get list of all active providers with working API keys"""
        return [
            provider for provider, health in self.health_status.items()
            if health.status == ProviderStatus.ACTIVE
        ]
    
    def get_optimal_providers(self, task_type: str = 'general', max_cost: Optional[float] = None) -> List[str]:
        """Get optimal provider order for a task with cost constraints"""
        active = self.get_active_providers()
        
        # Filter by cost if specified
        if max_cost:
            active = [
                p for p in active 
                if self.health_status[p].cost_per_token <= max_cost
            ]
        
        # Sort by priority (lower number = higher priority)
        active.sort(key=lambda p: self.PROVIDER_CONFIG[p]['priority'])
        
        # Task-specific reordering
        task_preferences = {
            'urgent': ['cerebras', 'groq', 'together'],  # Speed priority
            'quality': ['anthropic', 'openai', 'google'],  # Quality priority
            'cost': ['ollama', 'huggingface', 'deepseek'],  # Cost priority
            'research': ['perplexity', 'google', 'openai']  # Research priority
        }
        
        if task_type in task_preferences:
            preferred = [p for p in task_preferences[task_type] if p in active]
            remaining = [p for p in active if p not in preferred]
            return preferred + remaining
        
        return active
    
    def save_configuration(self):
        """Save current configuration to file"""
        config = {
            'discovered_keys': {k: v for k, v in self.discovered_keys.items()},
            'health_status': {k: asdict(v) for k, v in self.health_status.items()},
            'last_updated': time.time()
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            self.logger.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
    
    def load_configuration(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.discovered_keys = config.get('discovered_keys', {})
                
                # Reconstruct health status
                health_data = config.get('health_status', {})
                for provider, data in health_data.items():
                    # Convert status string back to enum
                    data['status'] = ProviderStatus(data['status'])
                    self.health_status[provider] = ProviderHealth(**data)
                
                self.logger.info(f"📂 Configuration loaded from {self.config_file}")
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
        
        return False
    
    def export_environment_vars(self) -> Dict[str, str]:
        """Export all active keys as environment variables"""
        env_vars = {}
        for provider, key in self.discovered_keys.items():
            # Check if provider has valid health status and is active
            health = self.health_status.get(provider)
            if health and hasattr(health, 'status') and health.status == ProviderStatus.ACTIVE:
                env_name = f"{provider.upper()}_API_KEY"
                env_vars[env_name] = key
            elif provider in self.discovered_keys and len(key) > 10:
                # Include provider if we have a valid key (fallback)
                env_name = f"{provider.upper()}_API_KEY"
                env_vars[env_name] = key
        return env_vars
    
    def configure_environment(self):
        """Configure environment variables for all active providers"""
        env_vars = self.export_environment_vars()
        for env_name, value in env_vars.items():
            os.environ[env_name] = value
        
        self.logger.info(f"🔧 Configured {len(env_vars)} environment variables")
    
    def get_health_report(self) -> str:
        """Generate a comprehensive health report"""
        report = ["🏥 Universal API Key Manager Health Report", "=" * 50]
        
        status_counts = {}
        for health in self.health_status.values():
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        report.append(f"📊 Provider Status Summary:")
        for status, count in status_counts.items():
            report.append(f"   {status.replace('_', ' ').title()}: {count}")
        
        report.append("\n📋 Individual Provider Status:")
        for provider, health in sorted(self.health_status.items()):
            status_emoji = {
                ProviderStatus.ACTIVE: "✅",
                ProviderStatus.INVALID_KEY: "🔑",
                ProviderStatus.RATE_LIMITED: "⏱️",
                ProviderStatus.NO_KEY: "❌",
                ProviderStatus.OFFLINE: "🔴",
                ProviderStatus.QUOTA_EXCEEDED: "💳"
            }
            
            emoji = status_emoji.get(health.status, "❓")
            report.append(f"   {emoji} {provider.ljust(15)} - {health.status.value}")
            
            if health.response_time:
                report.append(f"      Response time: {health.response_time:.3f}s")
            if health.error_message:
                report.append(f"      Error: {health.error_message}")
        
        active_providers = self.get_active_providers()
        total_cost = sum(
            self.health_status[p].cost_per_token 
            for p in active_providers 
            if self.health_status[p].cost_per_token
        )
        
        report.append(f"\n💰 Cost Analysis:")
        report.append(f"   Active Providers: {len(active_providers)}")
        report.append(f"   Average Cost/Token: ${total_cost/len(active_providers):.6f}")
        
        return "\n".join(report)

# Global instance for easy access
universal_manager = UniversalAPIKeyManager()

async def setup_enhanced_treequest():
    """Setup enhanced TreeQuest with universal API key management"""
    async with universal_manager:
        # Discover and validate all keys
        discovered = universal_manager.discover_all_keys()
        health = await universal_manager.validate_all_keys()
        
        # Configure environment
        universal_manager.configure_environment()
        
        # Save configuration
        universal_manager.save_configuration()
        
        # Return active providers
        active = universal_manager.get_active_providers()
        print(f"🚀 Enhanced TreeQuest configured with {len(active)} active providers")
        return active

# CLI interface for testing
async def main():
    """Main CLI interface for testing"""
    print("🔧 Universal API Key Manager Test")
    print("=" * 40)
    
    async with universal_manager:
        # Discover keys
        discovered = universal_manager.discover_all_keys()
        print(f"📊 Discovered {len(discovered)} API keys")
        
        # Validate all keys
        health = await universal_manager.validate_all_keys()
        
        # Print health report
        print("\n" + universal_manager.get_health_report())
        
        # Show optimal providers for different tasks
        print("\n🎯 Optimal Provider Recommendations:")
        for task_type in ['urgent', 'quality', 'cost', 'research']:
            optimal = universal_manager.get_optimal_providers(task_type)
            print(f"   {task_type.capitalize()}: {' → '.join(optimal[:3])}")

if __name__ == "__main__":
    asyncio.run(main())