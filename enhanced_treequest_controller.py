#!/usr/bin/env python3
"""
Enhanced TreeQuest Controller with OpenAI, Cerebras, and Claude Code Integration
Optimizes task routing for accuracy and speed with intelligent provider selection
"""

import asyncio
import json
import time
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import sys

# Add the multi-ai-treequest path
sys.path.append('/Users/Subho/CascadeProjects/multi-ai-treequest')
sys.path.append('/Users/Subho/CascadeProjects/brain-spark-platform')

from ai_wrappers import BaseAIWrapper, ModelProvider, TaskResult, AIWrapperFactory
from cost_tracker import CostTracker
from api_keys_config import setup_treequest_environment, api_manager

class TaskComplexity(Enum):
    """Task complexity levels for intelligent routing"""
    SIMPLE = "simple"      # Quick queries, basic calculations
    MEDIUM = "medium"      # Code generation, analysis, implementation
    COMPLEX = "complex"    # Multi-step reasoning, architecture design
    EXPERT = "expert"      # Deep analysis, research, specialized knowledge

class TaskType(Enum):
    """Task-specific categories based on benchmark performance"""
    # Development Tasks
    FRONTEND = "frontend"           # React, Vue, HTML/CSS, UI/UX
    BACKEND = "backend"            # APIs, databases, server logic
    TESTING = "testing"            # Unit tests, integration tests, test strategies
    DEBUGGING = "debugging"        # Bug fixing, error analysis
    CODE_REVIEW = "code_review"    # Code quality, best practices
    DEVOPS = "devops"             # Docker, CI/CD, deployment
    
    # Analysis Tasks  
    DATA_ANALYSIS = "data_analysis"    # Data processing, statistics
    ARCHITECTURE = "architecture"     # System design, patterns
    SECURITY = "security"             # Security analysis, vulnerabilities
    PERFORMANCE = "performance"       # Optimization, profiling
    
    # Content Tasks
    DOCUMENTATION = "documentation"   # Technical writing, API docs
    EXPLANATION = "explanation"       # Educational, tutorials
    RESEARCH = "research"            # Information gathering, analysis
    
    # Specialized Tasks
    MATH = "math"                    # Mathematical problems, calculations
    CREATIVE = "creative"            # Content creation, brainstorming
    GENERAL = "general"              # General questions, conversations

class ProviderStrength(Enum):
    """Provider strength categories"""
    SPEED = "speed"           # Cerebras - ultra-fast inference
    ACCURACY = "accuracy"     # Claude Code - high-quality reasoning
    BALANCE = "balance"       # OpenAI - good speed/accuracy balance

@dataclass
class TaskRequest:
    """Enhanced task request with routing preferences"""
    prompt: str
    context: Optional[str] = None
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    priority: ProviderStrength = ProviderStrength.BALANCE
    task_type: TaskType = TaskType.GENERAL
    max_cost: Optional[float] = None
    quality_threshold: float = 0.7
    speed_requirement: str = "normal"  # "urgent", "normal", "quality"
    description: str = ""

@dataclass  
class EnhancedTaskResult:
    """Enhanced result with provider performance metrics"""
    content: str
    provider: str
    cost: float
    execution_time: float
    quality_score: float
    success: bool
    tokens_used: int = 0
    error_message: Optional[str] = None
    confidence_score: float = 0.85

class ClaudeCodeWrapper:
    """Wrapper for Claude Code CLI integration"""
    
    def __init__(self):
        self.provider = "claude_code"
        self.cost_per_token = 0.000015  # Approximate Claude cost
        
    async def execute_task(self, prompt: str, context: Optional[str] = None) -> TaskResult:
        """Execute task using Claude Code CLI"""
        start_time = time.time()
        
        try:
            # Create a temporary file with the task
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                f.write(full_prompt)
                temp_file = f.name
            
            # Execute Claude Code in headless mode
            cmd = [
                'claude', 
                '--headless',
                '--file', temp_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                content = result.stdout.strip()
                tokens_used = len(content.split())  # Rough token estimate
                cost = tokens_used * self.cost_per_token
                
                return TaskResult(
                    content=content,
                    tokens_used=tokens_used,
                    cost=cost,
                    execution_time=execution_time,
                    success=True,
                    provider=ModelProvider.ANTHROPIC,  # Claude provider
                    model_name="claude-sonnet-4"
                )
            else:
                return TaskResult(
                    content="",
                    tokens_used=0,
                    cost=0.0,
                    execution_time=execution_time,
                    success=False,
                    error_message=result.stderr,
                    provider=ModelProvider.ANTHROPIC
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskResult(
                content="",
                tokens_used=0,
                cost=0.0,
                execution_time=execution_time,
                success=False,
                error_message=str(e),
                provider=ModelProvider.ANTHROPIC
            )
        finally:
            # Clean up temp file
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass

class EnhancedTreeQuestController:
    """Enhanced multi-AI controller with intelligent routing"""
    
    def __init__(self):
        # Auto-configure all available API keys
        available_providers = setup_treequest_environment()
        
        # Initialize standard wrappers with auto-configured keys
        self.standard_wrappers = AIWrapperFactory.create_all_available_wrappers()
        
        # Claude Code wrapper for orchestration only (not as AI provider)
        self.claude_wrapper = ClaudeCodeWrapper()
        
        # AI providers only (excluding Claude Code)
        self.all_wrappers = self.standard_wrappers
        
        self.cost_tracker = CostTracker()
        
        # Track providers with key authentication issues to avoid multi-attempts
        self.failed_auth_providers = set()
        
        print(f"🚀 Enhanced TreeQuest Controller initialized with {len(self.all_wrappers)} AI providers")
        print(f"📊 Available providers: {list(self.all_wrappers.keys())}")
        print(f"🔑 Auto-configured API keys for: {', '.join(available_providers)}")
        
        # Enhanced routing matrix with latest 2025 models (August 2025): [Complexity][Priority] -> Provider Order
        self.routing_matrix = {
            TaskComplexity.SIMPLE: {
                ProviderStrength.SPEED: ['groq', 'ollama', 'deepseek', 'cerebras', 'huggingface', 'together', 'xai', 'openai'],
                ProviderStrength.BALANCE: ['ollama', 'deepseek', 'groq', 'together', 'cerebras', 'openai', 'gemini', 'mistral'],
                ProviderStrength.ACCURACY: ['openai', 'anthropic', 'gemini', 'together', 'mistral', 'openrouter', 'groq', 'cerebras']  # GPT-5 and Claude 4 prioritized
            },
            TaskComplexity.MEDIUM: {
                ProviderStrength.SPEED: ['groq', 'deepseek', 'together', 'cerebras', 'ollama', 'xai', 'openai', 'gemini'],
                ProviderStrength.BALANCE: ['together', 'deepseek', 'groq', 'openai', 'anthropic', 'gemini', 'mistral', 'ollama', 'cerebras'],
                ProviderStrength.ACCURACY: ['openai', 'anthropic', 'gemini', 'together', 'mistral', 'openrouter', 'groq', 'cerebras']  # Latest models first
            },
            TaskComplexity.COMPLEX: {
                ProviderStrength.SPEED: ['groq', 'together', 'deepseek', 'openai', 'anthropic', 'cerebras', 'gemini', 'xai'],
                ProviderStrength.BALANCE: ['anthropic', 'openai', 'together', 'gemini', 'mistral', 'openrouter', 'groq', 'deepseek', 'cerebras'],
                ProviderStrength.ACCURACY: ['anthropic', 'openai', 'gemini', 'together', 'mistral', 'openrouter', 'groq', 'cerebras']  # Claude 4 Opus prioritized for complex tasks
            },
            TaskComplexity.EXPERT: {
                ProviderStrength.SPEED: ['anthropic', 'openai', 'gemini', 'together', 'mistral', 'groq', 'deepseek', 'cerebras'],
                ProviderStrength.BALANCE: ['anthropic', 'openai', 'gemini', 'together', 'mistral', 'openrouter', 'groq', 'deepseek', 'cerebras'],
                ProviderStrength.ACCURACY: ['anthropic', 'openai', 'gemini', 'together', 'mistral', 'openrouter', 'groq', 'cerebras']  # Claude 4 Opus & GPT-5 for expert tasks
            }
        }
        
        # Speed-based routing for urgent tasks (prioritizing free/cheap options, Claude Code excluded)
        self.speed_routing = {
            'urgent': ['groq', 'ollama', 'deepseek', 'cerebras', 'together', 'xai', 'openai', 'gemini'],
            'normal': ['ollama', 'deepseek', 'together', 'groq', 'openai', 'cerebras', 'gemini', 'mistral'],
            'quality': ['gemini', 'together', 'openai', 'mistral', 'openrouter', 'groq', 'cerebras', 'ollama']
        }
        
        # Task-specific model preferences based on benchmarks and specializations
        self.task_specific_routing = {
            # Development Tasks - Based on coding benchmarks (HumanEval, MBPP)
            TaskType.FRONTEND: {
                'primary': ['anthropic', 'openai'],      # Claude 4 & GPT-4o excel at React/Vue/CSS
                'secondary': ['together', 'groq'],       # Llama models good at web dev
                'reason': 'Claude 4 leads in UI/UX generation, GPT-4o strong at modern frameworks'
            },
            TaskType.BACKEND: {
                'primary': ['openai', 'anthropic'],      # GPT-4o & Claude 4 best for APIs/databases
                'secondary': ['mistral', 'together'],    # Mistral strong at system architecture
                'reason': 'GPT-4o leads backend architecture, Claude 4 excellent at database design'
            },
            TaskType.TESTING: {
                'primary': ['anthropic', 'openai'],      # Claude 4 exceptional at test strategies
                'secondary': ['groq', 'together'],       # Fast models for simple test generation
                'reason': 'Claude 4 leads in comprehensive testing strategies and edge cases'
            },
            TaskType.DEBUGGING: {
                'primary': ['anthropic', 'openai'],      # Claude 4 best at error analysis
                'secondary': ['groq', 'cerebras'],       # Fast debugging for simple issues
                'reason': 'Claude 4 superior at complex debugging and root cause analysis'
            },
            TaskType.CODE_REVIEW: {
                'primary': ['anthropic', 'openai'],      # Claude 4 excels at code quality analysis
                'secondary': ['mistral', 'together'],    # Good at best practices
                'reason': 'Claude 4 leads in identifying code smells and security issues'
            },
            TaskType.DEVOPS: {
                'primary': ['openai', 'together'],       # GPT-4o strong at Docker/K8s, Llama at configs
                'secondary': ['anthropic', 'mistral'],   # Good at infrastructure design
                'reason': 'GPT-4o leads DevOps automation, Together AI strong at config files'
            },
            
            # Analysis Tasks - Based on reasoning benchmarks (GPQA, DROP)
            TaskType.DATA_ANALYSIS: {
                'primary': ['openai', 'anthropic'],      # GPT-4o leads data analysis, Claude 4 at insights
                'secondary': ['mistral', 'together'],    # Good statistical reasoning
                'reason': 'GPT-4o excels at data visualization, Claude 4 superior at statistical insights'
            },
            TaskType.ARCHITECTURE: {
                'primary': ['anthropic', 'openai'],      # Claude 4 leads system design thinking
                'secondary': ['mistral', 'together'],    # Good architectural patterns
                'reason': 'Claude 4 superior at complex system architecture and scalability'
            },
            TaskType.SECURITY: {
                'primary': ['anthropic', 'openai'],      # Claude 4 leads security analysis
                'secondary': ['mistral', 'together'],    # Good at vulnerability detection
                'reason': 'Claude 4 excels at security threat modeling and vulnerability assessment'
            },
            TaskType.PERFORMANCE: {
                'primary': ['openai', 'anthropic'],      # GPT-4o strong at optimization strategies
                'secondary': ['cerebras', 'groq'],       # Fast models for simple optimizations
                'reason': 'GPT-4o leads performance optimization, Claude 4 excellent at profiling analysis'
            },
            
            # Content Tasks - Based on writing benchmarks (HELM, MT-Bench)
            TaskType.DOCUMENTATION: {
                'primary': ['anthropic', 'openai'],      # Claude 4 leads technical writing
                'secondary': ['mistral', 'together'],    # Good at structured documentation
                'reason': 'Claude 4 superior at clear, comprehensive technical documentation'
            },
            TaskType.EXPLANATION: {
                'primary': ['anthropic', 'openai'],      # Claude 4 best at educational content
                'secondary': ['gemini', 'mistral'],      # Good at explanations
                'reason': 'Claude 4 excels at breaking down complex concepts clearly'
            },
            TaskType.RESEARCH: {
                'primary': ['openai', 'gemini'],         # GPT-4o with browsing, Gemini with real-time data
                'secondary': ['anthropic', 'together'],  # Good analytical reasoning
                'reason': 'GPT-4o + browsing leads research, Gemini strong at current information'
            },
            
            # Specialized Tasks - Based on domain-specific benchmarks
            TaskType.MATH: {
                'primary': ['openai', 'anthropic'],      # GPT-4o leads math reasoning (GSM8K, MATH)
                'secondary': ['gemini', 'mistral'],      # Strong mathematical reasoning
                'reason': 'GPT-4o leads complex mathematics, Claude 4 excellent at proof verification'
            },
            TaskType.CREATIVE: {
                'primary': ['anthropic', 'openai'],      # Claude 4 leads creative writing
                'secondary': ['gemini', 'mistral'],      # Good creative capabilities
                'reason': 'Claude 4 superior at creative content generation and storytelling'
            },
            TaskType.GENERAL: {
                'primary': ['openai', 'anthropic'],      # Balanced general capability
                'secondary': ['gemini', 'together'],     # Good all-around performance
                'reason': 'GPT-4o and Claude 4 lead general capability benchmarks'
            }
        }
        
        # Speed-optimized routing for urgent tasks (override specializations)
        self.speed_override_routing = {
            'fastest': ['cerebras', 'groq'],             # <1s response time
            'fast': ['together', 'mistral'],             # <2s response time
            'balanced': ['openai', 'anthropic'],         # 2-4s response time
        }
        
        print(f"🚀 Enhanced TreeQuest initialized with {len(self.all_wrappers)} AI providers:")
        for provider in self.all_wrappers.keys():
            print(f"   ✅ {provider}")
        print(f"📊 Task-specific routing enabled for {len(self.task_specific_routing)} specialized task types")
    
    def get_optimal_provider_order(self, task_request: TaskRequest) -> List[str]:
        """Get optimal provider order based on task requirements and specializations"""
        
        # Speed override for urgent tasks (bypass specializations)
        if task_request.speed_requirement == 'urgent':
            return self.speed_routing['urgent']
        
        # Task-specific routing takes priority for specialized tasks
        if task_request.task_type != TaskType.GENERAL and task_request.task_type in self.task_specific_routing:
            task_routing = self.task_specific_routing[task_request.task_type]
            
            # Combine primary and secondary providers for fallback
            specialized_order = task_routing['primary'] + task_routing['secondary']
            
            # For quality tasks, stick to task specialization
            if task_request.speed_requirement == 'quality':
                return specialized_order
            
            # For normal tasks, add general fallbacks
            fallback_providers = ['groq', 'cerebras', 'ollama', 'huggingface']
            final_order = specialized_order + [p for p in fallback_providers if p not in specialized_order]
            
            return final_order
        
        # Quality override for general tasks
        if task_request.speed_requirement == 'quality':
            return self.speed_routing['quality']
        
        # Use complexity-based routing matrix for general tasks
        complexity = task_request.complexity
        priority = task_request.priority
        
        return self.routing_matrix.get(complexity, {}).get(
            priority, 
            ['ollama', 'deepseek', 'together', 'groq', 'openai', 'cerebras', 'gemini', 'mistral']
        )
    
    async def execute_with_fallback(self, task_request: TaskRequest) -> EnhancedTaskResult:
        """Execute task with intelligent fallback chain"""
        
        provider_order = self.get_optimal_provider_order(task_request)
        
        # Filter out providers with authentication failures to avoid multi-attempts
        available_providers = [
            p for p in provider_order 
            if not self._should_skip_provider(p) and p in self.all_wrappers
        ]
        
        if not available_providers:
            return EnhancedTaskResult(
                content="", provider="none", cost=0.0, execution_time=0.0,
                quality_score=0.0, success=False, 
                error_message="No providers available - all have authentication issues"
            )
        
        last_error = None
        
        # Show task-specific reasoning if applicable
        if task_request.task_type != TaskType.GENERAL and task_request.task_type in self.task_specific_routing:
            task_routing = self.task_specific_routing[task_request.task_type]
            print(f"🎯 Task Type: {task_request.task_type.value.upper()}")
            print(f"📊 Routing Reason: {task_routing['reason']}")
        
        print(f"🔄 Executing with available providers: {' → '.join(available_providers)}")
        
        for provider_name in available_providers:
                
            print(f"   🔄 Trying {provider_name}...")
            
            try:
                wrapper = self.all_wrappers[provider_name]
                start_time = time.time()
                
                # Execute task with AI provider (Claude Code not included as provider)
                result = await wrapper.execute_task(task_request.prompt, task_request.context)
                
                if result.success:
                    # Calculate quality score
                    quality_score = self.calculate_quality_score(result, task_request)
                    execution_time = time.time() - start_time
                    
                    # Provider-specific confidence scores (Updated August 2025)
                    confidence_scores = {
                        'claude_code': 0.95,
                        'anthropic': 0.92,  # Claude 4 models - excellent reasoning and coding
                        'openai': 0.94,     # GPT-5 models - smartest, fastest
                        'together': 0.87,   # Strong open-source models including Llama 3.3
                        'gemini': 0.89,     # Gemini 2.0 with improved capabilities
                        'mistral': 0.86,    # Mistral Large latest
                        'openrouter': 0.87,
                        'groq': 0.82,       # Ultra-fast inference with Llama 3.3
                        'xai': 0.80,        # Grok beta improving
                        'deepseek': 0.78,   # Competitive Chinese models
                        'ollama': 0.75,
                        'huggingface': 0.72,
                        'cerebras': 0.84    # Fast inference with Llama 3.3 70B
                    }
                    
                    enhanced_result = EnhancedTaskResult(
                        content=result.content,
                        provider=provider_name,
                        cost=result.cost,
                        execution_time=execution_time,
                        quality_score=quality_score,
                        success=True,
                        tokens_used=result.tokens_used,
                        confidence_score=confidence_scores.get(provider_name, 0.75)
                    )
                    
                    print(f"   ✅ Success with {provider_name} (Quality: {quality_score:.2f})")
                    return enhanced_result
                    
            except Exception as e:
                last_error = str(e)
                print(f"   ❌ {provider_name} failed: {e}")
                continue
        
        # All providers failed
        return EnhancedTaskResult(
            content="",
            provider="none",
            cost=0.0,
            execution_time=0.0,
            quality_score=0.0,
            success=False,
            error_message=f"All providers failed. Last error: {last_error}"
        )
    
    def calculate_quality_score(self, result: TaskResult, task_request: TaskRequest) -> float:
        """Calculate quality score based on result characteristics"""
        score = 0.5  # Base score
        
        # Content length score
        content_length = len(result.content)
        if content_length > 1000:
            score += 0.2
        elif content_length > 500:
            score += 0.1
        
        # Code detection bonus for code tasks
        if 'code' in task_request.prompt.lower() or 'implement' in task_request.prompt.lower():
            if '```' in result.content or 'def ' in result.content:
                score += 0.2
        
        # Structure bonus
        if '\n' in result.content and len(result.content.split('\n')) > 3:
            score += 0.1
        
        # Provider-specific quality adjustments
        provider_quality_bonus = {
            'claude_code': 0.1,    # High quality reasoning
            'together': 0.05,      # Good open-source models
            'openai': 0.08,        # Reliable commercial
            'gemini': 0.07,        # Strong Google models
            'mistral': 0.06,       # High-quality European AI
            'openrouter': 0.06,    # Access to premium models
            'groq': 0.03,          # Fast but variable quality
            'xai': 0.04,           # Grok models
            'deepseek': 0.05,      # Strong Chinese models
            'ollama': 0.02,        # Local, depends on model
            'huggingface': 0.01,   # Variable quality
            'cerebras': 0.04       # Fast, decent quality
        }
        
        provider_name = getattr(result, 'provider', None)
        if isinstance(provider_name, str):
            score += provider_quality_bonus.get(provider_name, 0)
        elif hasattr(provider_name, 'value'):
            score += provider_quality_bonus.get(provider_name.value, 0)
        
        return min(1.0, score)
    
    def _is_auth_error(self, error_message: str) -> bool:
        """Check if an error is related to API key authentication"""
        if not error_message:
            return False
            
        auth_keywords = [
            'api key', 'authentication', 'unauthorized', 'invalid key', 
            'forbidden', '401', '403', 'api_key', 'auth', 'token'
        ]
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in auth_keywords)
    
    def _mark_provider_auth_failed(self, provider_name: str, error_message: str):
        """Mark a provider as having authentication issues"""
        if self._is_auth_error(error_message):
            self.failed_auth_providers.add(provider_name)
            print(f"🔑 {provider_name} marked as auth failed - will skip in future attempts")
    
    def _should_skip_provider(self, provider_name: str) -> bool:
        """Check if provider should be skipped due to previous auth failures"""
        return provider_name in self.failed_auth_providers
    
    async def execute_parallel_validation(self, task_request: TaskRequest) -> Dict[str, EnhancedTaskResult]:
        """Execute task across multiple providers for validation"""
        
        provider_order = self.get_optimal_provider_order(task_request)[:2]  # Top 2 providers
        tasks = []
        
        for provider_name in provider_order:
            if provider_name in self.all_wrappers:
                task = self.execute_single_provider(task_request, provider_name)
                tasks.append((provider_name, task))
        
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for i, (provider_name, _) in enumerate(tasks):
            result = completed_tasks[i]
            if not isinstance(result, Exception):
                results[provider_name] = result
        
        return results
    
    async def execute_single_provider(self, task_request: TaskRequest, provider_name: str) -> EnhancedTaskResult:
        """Execute task with a single provider with auth failure tracking"""
        if provider_name not in self.all_wrappers:
            return EnhancedTaskResult(
                content="", provider=provider_name, cost=0.0, execution_time=0.0,
                quality_score=0.0, success=False, error_message="Provider not available"
            )
        
        # Skip providers with previous auth failures to avoid multi-attempts
        if self._should_skip_provider(provider_name):
            return EnhancedTaskResult(
                content="", provider=provider_name, cost=0.0, execution_time=0.0,
                quality_score=0.0, success=False, 
                error_message="Provider skipped due to previous authentication failure"
            )
        
        try:
            wrapper = self.all_wrappers[provider_name]
            start_time = time.time()
            
            result = await wrapper.execute_task(task_request.prompt, task_request.context)
            execution_time = time.time() - start_time
            quality_score = self.calculate_quality_score(result, task_request)
            
            # If task failed, check if it's an auth error
            if not result.success:
                self._mark_provider_auth_failed(provider_name, result.error_message)
            
            return EnhancedTaskResult(
                content=result.content,
                provider=provider_name,
                cost=result.cost,
                execution_time=execution_time,
                quality_score=quality_score,
                success=result.success,
                tokens_used=result.tokens_used,
                error_message=result.error_message
            )
            
        except Exception as e:
            error_message = str(e)
            self._mark_provider_auth_failed(provider_name, error_message)
            
            return EnhancedTaskResult(
                content="", provider=provider_name, cost=0.0, execution_time=0.0,
                quality_score=0.0, success=False, error_message=error_message
            )

# Convenience functions for different use cases
async def fast_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task prioritizing speed (Cerebras → OpenAI → Claude)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.SPEED,
        speed_requirement='urgent'
    )
    return await controller.execute_with_fallback(task)

async def accurate_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task prioritizing accuracy (Claude → OpenAI → Cerebras)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

async def balanced_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task with balanced speed/accuracy (Together → Groq → OpenAI)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.BALANCE,
        speed_requirement='normal'
    )
    return await controller.execute_with_fallback(task)

async def cost_optimized_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task prioritizing cost efficiency (Ollama → HuggingFace → Groq)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        complexity=TaskComplexity.SIMPLE,
        priority=ProviderStrength.SPEED,
        speed_requirement='normal',
        max_cost=0.001  # Very low cost threshold
    )
    return await controller.execute_with_fallback(task)

async def local_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task using only local/free providers (Ollama → HuggingFace)"""
    controller = EnhancedTreeQuestController()
    
    # Override routing to only use free providers
    free_providers = ['ollama', 'huggingface']
    
    for provider_name in free_providers:
        if provider_name in controller.all_wrappers:
            try:
                result = await controller.execute_single_provider(
                    TaskRequest(prompt=prompt, context=context), 
                    provider_name
                )
                if result.success:
                    return result
            except Exception as e:
                continue
    
    # If all free providers fail, return error
    return EnhancedTaskResult(
        content="",
        provider="none",
        cost=0.0,
        execution_time=0.0,
        quality_score=0.0,
        success=False,
        error_message="All free providers failed"
    )

# New convenience functions for latest 2025 models
async def gpt5_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task using GPT-5 (latest OpenAI model, August 2025)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.ACCURACY,
        complexity=TaskComplexity.EXPERT,
        speed_requirement='quality',
        description="GPT-5 execution for highest quality results"
    )
    
    # Force use of OpenAI first
    if 'openai' in controller.all_wrappers:
        try:
            result = await controller.execute_single_provider(task, 'openai')
            if result.success:
                return result
        except:
            pass
    
    # Fallback to standard execution
    return await controller.execute_with_fallback(task)

async def claude4_execution(prompt: str, context: str = None, model_variant: str = "opus") -> EnhancedTaskResult:
    """Execute task using Claude 4 models (latest Anthropic models, 2025)
    
    Args:
        model_variant: 'opus' for Claude Opus 4.1 (best coding), 'sonnet' for Claude Sonnet 4
    """
    controller = EnhancedTreeQuestController()
    
    # Set complexity based on variant
    complexity = TaskComplexity.EXPERT if model_variant == "opus" else TaskComplexity.COMPLEX
    
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.ACCURACY,
        complexity=complexity,
        speed_requirement='quality',
        description=f"Claude 4 {model_variant.title()} execution for advanced reasoning"
    )
    
    # Force use of Anthropic first
    if 'anthropic' in controller.all_wrappers:
        try:
            result = await controller.execute_single_provider(task, 'anthropic')
            if result.success:
                return result
        except:
            pass
    
    # Fallback to standard execution
    return await controller.execute_with_fallback(task)

async def llama33_execution(prompt: str, context: str = None, provider: str = "cerebras") -> EnhancedTaskResult:
    """Execute task using Llama 3.3 70B models (latest open-source, 2025)
    
    Args:
        provider: 'cerebras' (ultra-fast), 'groq' (fast), or 'together' (balanced)
    """
    controller = EnhancedTreeQuestController()
    
    # Map provider preferences
    provider_map = {
        'cerebras': ProviderStrength.SPEED,
        'groq': ProviderStrength.SPEED, 
        'together': ProviderStrength.BALANCE
    }
    
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=provider_map.get(provider, ProviderStrength.BALANCE),
        complexity=TaskComplexity.MEDIUM,
        speed_requirement='urgent' if provider in ['cerebras', 'groq'] else 'normal',
        description=f"Llama 3.3 70B execution via {provider}"
    )
    
    # Force use of specified provider first
    if provider in controller.all_wrappers:
        try:
            result = await controller.execute_single_provider(task, provider)
            if result.success:
                return result
        except:
            pass
    
    # Fallback to standard execution
    return await controller.execute_with_fallback(task)

async def latest_models_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute task using only the latest 2025 models (GPT-5, Claude 4, Gemini 2.0)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.ACCURACY,
        complexity=TaskComplexity.EXPERT,
        speed_requirement='quality',
        description="Latest 2025 models execution"
    )
    
    # Try latest models in order of capability
    latest_providers = ['anthropic', 'openai', 'gemini']
    
    for provider_name in latest_providers:
        if provider_name in controller.all_wrappers:
            try:
                result = await controller.execute_single_provider(task, provider_name)
                if result.success:
                    return result
            except:
                continue
    
    # Fallback to standard execution if latest models fail
    return await controller.execute_with_fallback(task)

async def benchmark_latest_models(prompt: str, context: str = None) -> Dict[str, EnhancedTaskResult]:
    """Benchmark latest 2025 models against each other"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        priority=ProviderStrength.ACCURACY,
        complexity=TaskComplexity.EXPERT
    )
    
    # Test all latest models
    latest_providers = ['openai', 'anthropic', 'gemini', 'together', 'cerebras', 'groq']
    results = {}
    
    for provider_name in latest_providers:
        if provider_name in controller.all_wrappers:
            try:
                result = await controller.execute_single_provider(task, provider_name)
                results[provider_name] = result
            except Exception as e:
                results[provider_name] = EnhancedTaskResult(
                    content="", provider=provider_name, cost=0.0, execution_time=0.0,
                    quality_score=0.0, success=False, error_message=str(e)
                )
    
    return results

# Task-specific execution functions with automatic detection
async def smart_execution(prompt: str, context: str = None, task_type: TaskType = None) -> EnhancedTaskResult:
    """Smart execution with automatic task type detection and specialized routing"""
    controller = EnhancedTreeQuestController()
    
    # Auto-detect task type if not provided
    if task_type is None:
        task_type = controller.detect_task_type(prompt)
    
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=task_type,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='normal'
    )
    return await controller.execute_with_fallback(task)

async def frontend_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute frontend development tasks (React, Vue, HTML/CSS, UI/UX)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.FRONTEND,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

async def backend_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute backend development tasks (APIs, databases, server logic)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.BACKEND,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

async def testing_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute testing tasks (unit tests, integration tests, test strategies)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.TESTING,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

async def debugging_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute debugging tasks (bug fixing, error analysis)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.DEBUGGING,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='normal'
    )
    return await controller.execute_with_fallback(task)

async def security_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute security analysis tasks (vulnerability assessment, threat modeling)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.SECURITY,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

async def documentation_execution(prompt: str, context: str = None) -> EnhancedTaskResult:
    """Execute documentation tasks (technical writing, API docs)"""
    controller = EnhancedTreeQuestController()
    task = TaskRequest(
        prompt=prompt,
        context=context,
        task_type=TaskType.DOCUMENTATION,
        priority=ProviderStrength.ACCURACY,
        speed_requirement='quality'
    )
    return await controller.execute_with_fallback(task)

# Add the detect_task_type method to the class
def add_task_detection_to_controller():
    """Add task detection method to EnhancedTreeQuestController"""
    
    def detect_task_type(self, prompt: str) -> TaskType:
        """Automatically detect task type from prompt content"""
        prompt_lower = prompt.lower()
        
        # Development task keywords
        frontend_keywords = ['react', 'vue', 'html', 'css', 'javascript', 'frontend', 'ui', 'ux', 'component', 'bootstrap', 'tailwind']
        backend_keywords = ['api', 'database', 'server', 'backend', 'express', 'django', 'flask', 'node.js', 'sql', 'mongodb']
        testing_keywords = ['test', 'testing', 'unit test', 'integration test', 'jest', 'pytest', 'mocha', 'cypress']
        debugging_keywords = ['debug', 'error', 'fix bug', 'troubleshoot', 'issue', 'problem', 'exception']
        code_review_keywords = ['code review', 'refactor', 'optimize', 'best practice', 'clean code', 'review']
        devops_keywords = ['docker', 'kubernetes', 'ci/cd', 'deployment', 'aws', 'azure', 'terraform', 'devops']
        
        # Analysis task keywords
        data_keywords = ['data analysis', 'statistics', 'pandas', 'numpy', 'visualization', 'chart', 'graph']
        architecture_keywords = ['architecture', 'system design', 'scalability', 'microservices', 'design pattern']
        security_keywords = ['security', 'vulnerability', 'authentication', 'authorization', 'encryption', 'penetration']
        performance_keywords = ['performance', 'optimization', 'speed', 'memory', 'profiling', 'bottleneck']
        
        # Content task keywords
        documentation_keywords = ['documentation', 'readme', 'api doc', 'technical writing', 'manual']
        explanation_keywords = ['explain', 'how to', 'tutorial', 'guide', 'teach', 'learn']
        research_keywords = ['research', 'analyze', 'compare', 'study', 'investigate']
        
        # Specialized task keywords
        math_keywords = ['math', 'calculate', 'equation', 'formula', 'statistics', 'probability']
        creative_keywords = ['creative', 'story', 'content', 'marketing', 'brainstorm', 'idea']
        
        # Check for keywords in order of specificity
        if any(keyword in prompt_lower for keyword in frontend_keywords):
            return TaskType.FRONTEND
        elif any(keyword in prompt_lower for keyword in backend_keywords):
            return TaskType.BACKEND
        elif any(keyword in prompt_lower for keyword in testing_keywords):
            return TaskType.TESTING
        elif any(keyword in prompt_lower for keyword in debugging_keywords):
            return TaskType.DEBUGGING
        elif any(keyword in prompt_lower for keyword in code_review_keywords):
            return TaskType.CODE_REVIEW
        elif any(keyword in prompt_lower for keyword in devops_keywords):
            return TaskType.DEVOPS
        elif any(keyword in prompt_lower for keyword in data_keywords):
            return TaskType.DATA_ANALYSIS
        elif any(keyword in prompt_lower for keyword in architecture_keywords):
            return TaskType.ARCHITECTURE
        elif any(keyword in prompt_lower for keyword in security_keywords):
            return TaskType.SECURITY
        elif any(keyword in prompt_lower for keyword in performance_keywords):
            return TaskType.PERFORMANCE
        elif any(keyword in prompt_lower for keyword in documentation_keywords):
            return TaskType.DOCUMENTATION
        elif any(keyword in prompt_lower for keyword in explanation_keywords):
            return TaskType.EXPLANATION
        elif any(keyword in prompt_lower for keyword in research_keywords):
            return TaskType.RESEARCH
        elif any(keyword in prompt_lower for keyword in math_keywords):
            return TaskType.MATH
        elif any(keyword in prompt_lower for keyword in creative_keywords):
            return TaskType.CREATIVE
        else:
            return TaskType.GENERAL
    
    # Add method to class
    EnhancedTreeQuestController.detect_task_type = detect_task_type

# Execute the addition
add_task_detection_to_controller()