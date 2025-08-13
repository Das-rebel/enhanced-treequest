# Contributing to Enhanced TreeQuest

We welcome contributions to Enhanced TreeQuest! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/enhanced-treequest.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Make your changes
5. Test your changes: `python -m pytest tests/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📋 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- API keys for testing (at least OpenAI recommended)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-treequest.git
cd enhanced-treequest

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
# Add your API keys to .env

# Run tests to verify setup
python -m pytest tests/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=enhanced_treequest

# Run specific test file
python -m pytest tests/test_provider_integration.py

# Run with verbose output
python -m pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Follow the existing test patterns

Example test structure:
```python
import pytest
from enhanced_treequest_controller import fast_execution

@pytest.mark.asyncio
async def test_basic_execution():
    """Test basic task execution functionality"""
    result = await fast_execution("Hello, world!")
    assert result.success
    assert result.content
    assert result.provider
```

## 🏗️ Code Style

### Python Style Guidelines

- Follow PEP 8
- Use Black for code formatting: `black .`
- Use isort for import sorting: `isort .`
- Use type hints where appropriate
- Maximum line length: 88 characters

### Code Quality

- Run linting: `flake8 .`
- Type checking: `mypy .`
- Security scanning: `bandit -r .`

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## 🌟 Types of Contributions

### 🐛 Bug Reports

When filing a bug report, please include:

- **Clear title** describing the issue
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, etc.)
- **Error messages** and stack traces
- **API providers** being used

Use the bug report template:

```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Enhanced TreeQuest version:
- API providers:

## Additional Context
Any other relevant information
```

### ✨ Feature Requests

For feature requests, please include:

- **Clear description** of the feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

### 🔧 Code Contributions

#### Adding New AI Providers

To add a new AI provider:

1. Create a wrapper class implementing the `ProviderWrapper` interface
2. Add provider configuration to `universal_api_key_manager.py`
3. Update provider routing logic in `enhanced_treequest_controller.py`
4. Add comprehensive tests
5. Update documentation

Example provider wrapper:
```python
class NewProviderWrapper(ProviderWrapper):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = NewProviderClient(api_key=api_key)
    
    async def execute(self, prompt: str, **kwargs) -> TaskResult:
        # Implementation here
        pass
    
    def estimate_cost(self, prompt: str) -> float:
        # Cost estimation logic
        pass
    
    def check_health(self) -> HealthStatus:
        # Health check implementation
        pass
```

#### Improving Core Functionality

- **Performance improvements**: Benchmark before and after
- **New execution strategies**: Add to `ExecutionStrategy` enum
- **Enhanced routing**: Improve task-specific provider selection
- **Better error handling**: Add resilience and recovery mechanisms

## 📖 Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Update docstrings for all functions
- Add type hints
- Update README.md for significant changes

### API Documentation

Follow Google-style docstrings:

```python
async def fast_execution(prompt: str, **kwargs) -> TaskResult:
    """Execute a task with optimal provider selection.
    
    Args:
        prompt: The task prompt to execute
        **kwargs: Additional execution parameters
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            provider: Force specific provider
    
    Returns:
        TaskResult containing response, provider info, and metrics
    
    Raises:
        ProviderException: When all providers fail
        InvalidPromptException: When prompt is invalid
    
    Example:
        >>> result = await fast_execution("Hello, world!")
        >>> print(f"Response: {result.content}")
    """
```

## 🔒 Security Guidelines

### API Key Security

- Never commit API keys to the repository
- Use environment variables for sensitive data
- Validate and sanitize all inputs
- Implement rate limiting for production use

### Code Security

- Use `bandit` for security scanning
- Validate all user inputs
- Implement proper error handling
- Follow secure coding practices

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers get started
- Collaborate effectively

### Communication

- Use clear, professional language
- Be patient with questions
- Provide helpful feedback
- Report issues constructively

## 🎯 Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on multiple environments
4. **Documentation** review
5. **Final approval** and merge

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors graph
- Special mentions for significant contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: maintainers@enhanced-treequest.dev

## 🗺️ Roadmap

See our [project roadmap](https://github.com/yourusername/enhanced-treequest/projects) for planned features and improvements.

---

Thank you for contributing to Enhanced TreeQuest! 🌳✨