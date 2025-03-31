# Contributing to SynaptiCode

Thank you for your interest in contributing to SynaptiCode! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

Before submitting a bug report, please check existing issues to avoid duplicates. When reporting a bug, include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Screenshots or code snippets if applicable
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

When suggesting enhancements, include:

- A clear, descriptive title
- Step-by-step description of the enhancement
- Any relevant examples or references

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

### Coding Guidelines

- Follow PEP 8 style guide
- Write docstrings for all functions, classes, and modules
- Include unit tests for new functionality
- Use type hints where appropriate

## Development Setup

```bash
# Clone the repository
git clone https://github.com/admsolvemprobler/SynaptiCode.git
cd SynaptiCode

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

## Running Tests

```bash
pytest
```

## License

By contributing to SynaptiCode, you agree that your contributions will be licensed under the project's MIT license.
