#!/usr/bin/env python3
"""
Gemini CLI Wrapper for TreeQuest Orchestrator Switching
Provides programmatic interface to Gemini CLI for seamless orchestration
Supports state preservation and command translation from Claude Code
"""

import os
import json
import asyncio
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

class GeminiExecutionMode(Enum):
    """Gemini CLI execution modes"""
    INTERACTIVE = "interactive"     # Interactive session
    SINGLE_SHOT = "single_shot"    # One command execution
    STREAMING = "streaming"        # Streaming response
    BATCH = "batch"               # Batch processing

@dataclass
class GeminiCommand:
    """Gemini CLI command structure"""
    prompt: str
    context: Optional[str] = None
    mode: GeminiExecutionMode = GeminiExecutionMode.SINGLE_SHOT
    model: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    tools: List[str] = None
    files: List[str] = None
    timeout: int = 60
    
    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        if self.files is None:
            self.files = []

@dataclass
class GeminiResponse:
    """Gemini CLI response structure"""
    content: str
    success: bool
    execution_time: float
    tokens_used: Optional[int] = None
    model_used: str = "gemini-pro"
    error_message: Optional[str] = None
    exit_code: int = 0
    raw_output: Optional[str] = None

class GeminiCLIWrapper:
    """Wrapper for Gemini CLI with TreeQuest integration"""
    
    def __init__(self, 
                 gemini_path: str = "/Users/Subho/gemini-cli",
                 config_file: Optional[str] = None):
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.gemini_path = Path(gemini_path)
        self.config_file = config_file or os.path.expanduser("~/.treequest/gemini_config.json")
        
        # Verify Gemini CLI installation
        self.gemini_binary = self._find_gemini_binary()
        
        # Session management
        self.active_sessions: Dict[str, Any] = {}
        self.session_counter = 0
        
        # Configuration
        self.default_config = {
            'model': 'gemini-pro',
            'temperature': 0.7,
            'max_tokens': 4096,
            'timeout': 60,
            'api_key': os.environ.get('GOOGLE_API_KEY'),
            'tools_enabled': True
        }
        
        # Load configuration
        self._load_config()
        
        # Initialize Gemini CLI if needed
        self._ensure_gemini_initialized()
    
    def _find_gemini_binary(self) -> Optional[Path]:
        """Find Gemini CLI binary"""
        possible_paths = [
            self.gemini_path / "bin" / "gemini",
            self.gemini_path / "gemini",
            Path("/usr/local/bin/gemini"),
            Path("/opt/homebrew/bin/gemini")
        ]
        
        # Also check if it's in PATH
        try:
            result = subprocess.run(['which', 'gemini'], capture_output=True, text=True)
            if result.returncode == 0:
                possible_paths.append(Path(result.stdout.strip()))
        except:
            pass
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                # Check if it's executable
                if os.access(path, os.X_OK):
                    self.logger.info(f"✅ Found Gemini CLI at: {path}")
                    return path
        
        # Try npm global installation
        try:
            result = subprocess.run(['npm', 'list', '-g', 'gemini-cli'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Try to run it directly
                test_result = subprocess.run(['gemini', '--version'], 
                                           capture_output=True, text=True)
                if test_result.returncode == 0:
                    self.logger.info("✅ Found Gemini CLI in npm global packages")
                    return Path('gemini')  # Use PATH
        except:
            pass
        
        self.logger.warning("⚠️ Gemini CLI not found. Will attempt to install.")
        return None
    
    def _ensure_gemini_initialized(self):
        """Ensure Gemini CLI is properly initialized"""
        if not self.gemini_binary:
            self._install_gemini_cli()
        
        # Test basic functionality
        if self.gemini_binary and not self._test_gemini_connection():
            self._configure_gemini_cli()
    
    def _install_gemini_cli(self):
        """Install Gemini CLI if not found"""
        self.logger.info("📦 Installing Gemini CLI...")
        
        try:
            # Try npm installation first
            result = subprocess.run([
                'npm', 'install', '-g', 'gemini-cli'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.gemini_binary = Path('gemini')
                self.logger.info("✅ Gemini CLI installed via npm")
                return
        except Exception as e:
            self.logger.warning(f"⚠️ npm install failed: {e}")
        
        # Try building from source if the directory exists
        if self.gemini_path.exists():
            try:
                os.chdir(self.gemini_path)
                
                # Install dependencies
                subprocess.run(['npm', 'install'], check=True, timeout=120)
                
                # Build the project
                subprocess.run(['npm', 'run', 'build'], check=True, timeout=120)
                
                # Create symlink or copy binary
                built_binary = self.gemini_path / "dist" / "gemini"
                if built_binary.exists():
                    self.gemini_binary = built_binary
                    self.logger.info("✅ Gemini CLI built from source")
                    return
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to build from source: {e}")
        
        self.logger.error("❌ Could not install Gemini CLI")
        raise RuntimeError("Gemini CLI installation failed")
    
    def _test_gemini_connection(self) -> bool:
        """Test if Gemini CLI is working properly"""
        try:
            result = subprocess.run([
                str(self.gemini_binary), '--version'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.logger.info("✅ Gemini CLI connection test passed")
                return True
            else:
                self.logger.warning(f"⚠️ Gemini CLI test failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️ Gemini CLI test error: {e}")
            return False
    
    def _configure_gemini_cli(self):
        """Configure Gemini CLI with API keys and settings"""
        self.logger.info("🔧 Configuring Gemini CLI...")
        
        # Set up API key if available
        api_key = self.default_config.get('api_key')
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
            self.logger.info("✅ Google API key configured")
        else:
            self.logger.warning("⚠️ No Google API key found")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.default_config.update(config)
                self.logger.info(f"📂 Configuration loaded from {self.config_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load config: {e}")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.default_config, f, indent=2)
            self.logger.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"❌ Could not save config: {e}")
    
    async def execute_command(self, command: GeminiCommand) -> GeminiResponse:
        """Execute a command through Gemini CLI"""
        start_time = time.time()
        
        if not self.gemini_binary:
            return GeminiResponse(
                content="",
                success=False,
                execution_time=0.0,
                error_message="Gemini CLI not available"
            )
        
        try:
            # Build command arguments
            cmd_args = [str(self.gemini_binary)]
            
            # Add model specification
            if command.model != "gemini-pro":
                cmd_args.extend(['--model', command.model])
            
            # Add temperature
            if command.temperature != 0.7:
                cmd_args.extend(['--temperature', str(command.temperature)])
            
            # Add max tokens
            if command.max_tokens:
                cmd_args.extend(['--max-tokens', str(command.max_tokens)])
            
            # Handle different execution modes
            if command.mode == GeminiExecutionMode.STREAMING:
                cmd_args.append('--stream')
            elif command.mode == GeminiExecutionMode.BATCH:
                cmd_args.append('--batch')
            
            # Add tools if specified
            for tool in command.tools:
                cmd_args.extend(['--tool', tool])
            
            # Handle context and prompt
            if command.context:
                full_prompt = f"{command.context}\n\n{command.prompt}"
            else:
                full_prompt = command.prompt
            
            # For file input or complex prompts, use stdin
            if len(full_prompt) > 1000 or '\n' in full_prompt:
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=full_prompt.encode()),
                    timeout=command.timeout
                )
            else:
                # For simple prompts, use command line argument
                cmd_args.extend(['--prompt', full_prompt])
                
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=command.timeout
                )
            
            execution_time = time.time() - start_time
            
            # Process results
            if process.returncode == 0:
                content = stdout.decode('utf-8').strip()
                return GeminiResponse(
                    content=content,
                    success=True,
                    execution_time=execution_time,
                    model_used=command.model,
                    exit_code=process.returncode,
                    raw_output=content
                )
            else:
                error_msg = stderr.decode('utf-8').strip()
                return GeminiResponse(
                    content="",
                    success=False,
                    execution_time=execution_time,
                    error_message=error_msg,
                    exit_code=process.returncode,
                    raw_output=stdout.decode('utf-8')
                )
        
        except asyncio.TimeoutError:
            return GeminiResponse(
                content="",
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Timeout after {command.timeout} seconds"
            )
        except Exception as e:
            return GeminiResponse(
                content="",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def execute_prompt(self, 
                           prompt: str, 
                           context: Optional[str] = None,
                           model: str = "gemini-pro",
                           **kwargs) -> GeminiResponse:
        """Simple prompt execution with sensible defaults"""
        command = GeminiCommand(
            prompt=prompt,
            context=context,
            model=model,
            **kwargs
        )
        return await self.execute_command(command)
    
    async def execute_streaming(self, 
                              prompt: str, 
                              context: Optional[str] = None,
                              callback: Optional[callable] = None) -> GeminiResponse:
        """Execute with streaming response"""
        command = GeminiCommand(
            prompt=prompt,
            context=context,
            mode=GeminiExecutionMode.STREAMING
        )
        
        # For streaming, we need to handle real-time output
        if not callback:
            return await self.execute_command(command)
        
        # Implement streaming with callback
        start_time = time.time()
        
        try:
            cmd_args = [str(self.gemini_binary), '--stream']
            
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send input
            process.stdin.write(full_prompt.encode())
            process.stdin.close()
            
            # Read streaming output
            content_parts = []
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                    if not line:
                        break
                    
                    text = line.decode('utf-8').rstrip()
                    if text:
                        content_parts.append(text)
                        callback(text)  # Send chunk to callback
                        
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
            
            await process.wait()
            
            full_content = '\n'.join(content_parts)
            execution_time = time.time() - start_time
            
            return GeminiResponse(
                content=full_content,
                success=process.returncode == 0,
                execution_time=execution_time,
                exit_code=process.returncode
            )
            
        except Exception as e:
            return GeminiResponse(
                content="",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new interactive session"""
        if not session_id:
            self.session_counter += 1
            session_id = f"gemini_session_{self.session_counter}"
        
        self.active_sessions[session_id] = {
            'created_at': time.time(),
            'command_history': [],
            'context': []
        }
        
        self.logger.info(f"🚀 Created Gemini session: {session_id}")
        return session_id
    
    async def execute_in_session(self, 
                               session_id: str, 
                               prompt: str,
                               preserve_context: bool = True) -> GeminiResponse:
        """Execute command within a session context"""
        if session_id not in self.active_sessions:
            return GeminiResponse(
                content="",
                success=False,
                execution_time=0.0,
                error_message=f"Session {session_id} not found"
            )
        
        session = self.active_sessions[session_id]
        
        # Build context from session history if requested
        context = None
        if preserve_context and session['context']:
            context = '\n'.join(session['context'][-5:])  # Last 5 interactions
        
        # Execute the command
        response = await self.execute_prompt(prompt, context)
        
        # Update session history
        session['command_history'].append({
            'timestamp': time.time(),
            'prompt': prompt,
            'response': response.content,
            'success': response.success
        })
        
        # Update context for future commands
        if preserve_context and response.success:
            session['context'].append(f"User: {prompt}")
            session['context'].append(f"Assistant: {response.content}")
            
            # Keep context manageable
            if len(session['context']) > 20:
                session['context'] = session['context'][-10:]
        
        return response
    
    def close_session(self, session_id: str):
        """Close and clean up a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"🔚 Closed Gemini session: {session_id}")
    
    def translate_claude_command(self, claude_command: str) -> GeminiCommand:
        """Translate Claude Code command to Gemini CLI format"""
        # Basic command translation patterns
        translations = {
            'claude': 'gemini',
            '--headless': '',  # Gemini CLI doesn't have headless mode
            '--file': '--input-file',
            '--model claude-': '--model gemini-',
        }
        
        # Apply translations
        translated = claude_command
        for old, new in translations.items():
            translated = translated.replace(old, new)
        
        # Extract prompt if it's a simple command
        if '--prompt' in translated:
            parts = translated.split('--prompt', 1)
            if len(parts) > 1:
                prompt = parts[1].strip().strip('"\'')
                return GeminiCommand(prompt=prompt)
        
        # Default simple execution
        return GeminiCommand(prompt=translated)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of Gemini CLI wrapper"""
        return {
            'gemini_binary_available': self.gemini_binary is not None,
            'gemini_path': str(self.gemini_binary) if self.gemini_binary else None,
            'active_sessions': len(self.active_sessions),
            'session_ids': list(self.active_sessions.keys()),
            'configuration': self.default_config,
            'connection_test': self._test_gemini_connection()
        }

# Global wrapper instance
gemini_wrapper = GeminiCLIWrapper()

async def test_gemini_wrapper():
    """Test the Gemini CLI wrapper functionality"""
    print("🧪 Testing Gemini CLI Wrapper")
    print("=" * 40)
    
    # Check status
    status = gemini_wrapper.get_status()
    print(f"📊 Gemini Binary Available: {status['gemini_binary_available']}")
    print(f"📂 Gemini Path: {status['gemini_path']}")
    
    if not status['gemini_binary_available']:
        print("❌ Gemini CLI not available")
        return
    
    # Test simple execution
    print("\n🔤 Testing simple prompt execution...")
    response = await gemini_wrapper.execute_prompt("What is 2+2?")
    
    if response.success:
        print(f"✅ Response: {response.content[:100]}...")
        print(f"⏱️ Execution time: {response.execution_time:.2f}s")
    else:
        print(f"❌ Failed: {response.error_message}")
    
    # Test session-based execution
    print("\n💬 Testing session-based execution...")
    session_id = gemini_wrapper.create_session()
    
    response1 = await gemini_wrapper.execute_in_session(
        session_id, "Remember that my name is TreeQuest"
    )
    response2 = await gemini_wrapper.execute_in_session(
        session_id, "What is my name?"
    )
    
    if response2.success:
        print(f"✅ Session response: {response2.content[:100]}...")
    else:
        print(f"❌ Session failed: {response2.error_message}")
    
    gemini_wrapper.close_session(session_id)
    
    print("\n📊 Final status:")
    print(json.dumps(gemini_wrapper.get_status(), indent=2))

if __name__ == "__main__":
    asyncio.run(test_gemini_wrapper())