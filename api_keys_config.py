#!/usr/bin/env python3
"""
Centralized API Key Configuration for TreeQuest
Auto-discovers and configures all available API keys for optimal performance
Updated to use Universal API Key Manager for complete discovery
"""

import os
import sys
from typing import Dict, Optional, List

# Import the enhanced Universal API Key Manager
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')
try:
    from universal_api_key_manager import UniversalAPIKeyManager, universal_manager
    ENHANCED_DISCOVERY = True
except ImportError:
    ENHANCED_DISCOVERY = False

class APIKeyManager:
    """Manages API keys across all AI providers - Enhanced with Universal Discovery"""
    
    def __init__(self):
        if ENHANCED_DISCOVERY:
            # Use the enhanced Universal API Key Manager
            self.universal_manager = universal_manager
            self.discovered_keys = self.universal_manager.discover_all_keys()
            self.universal_manager.configure_environment()
            print(f"🔧 Enhanced API discovery: {len(self.discovered_keys)} providers found")
        else:
            # Fallback to basic discovery
            self.discovered_keys = self._discover_all_keys_basic()
            print(f"⚠️ Basic API discovery: {len(self.discovered_keys)} providers found")
        
        self.provider_priorities = self._get_provider_priorities()
    
    def _discover_all_keys_basic(self) -> Dict[str, str]:
        """Basic discovery as fallback when Universal Manager not available"""
        keys = {}
        
        # Primary API keys from environment
        key_mappings = {
            'OPENAI_API_KEY': 'openai',
            'ANTHROPIC_API_KEY': 'anthropic', 
            'GOOGLE_API_KEY': 'google',
            'CEREBRAS_API_KEY': 'cerebras',
            'GROQ_API_KEY': 'groq',
            'TOGETHER_API_KEY': 'together',
            'MISTRAL_API_KEY': 'mistral',
            'OPENROUTER_API_KEY': 'openrouter',
            'XAI_API_KEY': 'xai',
            'DEEPSEEK_API_KEY': 'deepseek',
            'HUGGINGFACE_API_KEY': 'huggingface',
            'PERPLEXITY_API_KEY': 'perplexity'
        }
        
        for env_key, provider in key_mappings.items():
            value = os.environ.get(env_key)
            if value and len(value) > 10:  # Valid key length check
                keys[provider] = value
                print(f"✅ Found {provider} API key")
        
        # Check for additional keys in known formats
        for key in os.environ:
            if 'API_KEY' in key and key not in key_mappings:
                value = os.environ[key]
                if value and len(value) > 10:
                    provider_name = key.replace('_API_KEY', '').lower()
                    keys[provider_name] = value
                    print(f"✅ Found additional {provider_name} API key")
        
        return keys
    
    def _get_provider_priorities(self) -> Dict[str, List[str]]:
        """Define provider priorities by task type for optimal routing"""
        return {
            'compilation_fixes': ['cerebras', 'groq', 'openai', 'together'],
            'ui_development': ['openai', 'anthropic', 'together', 'cerebras'],
            'database_fixes': ['cerebras', 'groq', 'openai'],
            'ai_features': ['openai', 'anthropic', 'together'],
            'general': ['cerebras', 'openai', 'groq', 'together', 'anthropic']
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of all available providers with valid API keys"""
        return list(self.discovered_keys.keys())
    
    def get_optimal_providers(self, task_type: str = 'general') -> List[str]:
        """Get optimal provider order for a specific task type"""
        priorities = self.provider_priorities.get(task_type, self.provider_priorities['general'])
        return [p for p in priorities if p in self.discovered_keys]
    
    def configure_environment(self):
        """Configure environment variables for all discovered keys"""
        for provider, key in self.discovered_keys.items():
            env_name = f"{provider.upper()}_API_KEY"
            os.environ[env_name] = key
            
    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        return self.discovered_keys.get(provider)
    
    def export_for_treequest(self) -> Dict[str, str]:
        """Export all keys in format expected by TreeQuest"""
        env_vars = {}
        for provider, key in self.discovered_keys.items():
            env_name = f"{provider.upper()}_API_KEY"
            env_vars[env_name] = key
        return env_vars

# Global instance
api_manager = APIKeyManager()

def setup_treequest_environment():
    """Setup environment for TreeQuest with all available API keys"""
    api_manager.configure_environment()
    available = api_manager.get_available_providers()
    print(f"🚀 TreeQuest configured with {len(available)} providers: {', '.join(available)}")
    return available

if __name__ == "__main__":
    setup_treequest_environment()