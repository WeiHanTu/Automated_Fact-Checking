# Contributing to Fact-checking BERT

Thank you for your interest in contributing to the Fact-checking BERT project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check existing issues**: Before creating a new issue, check if the bug has already been reported.
2. **Create a detailed issue**: Include:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Error messages or logs

### Suggesting Enhancements

1. **Check existing issues**: Look for similar feature requests.
2. **Create a feature request**: Include:
   - Clear description of the enhancement
   - Use cases and benefits
   - Implementation suggestions (if any)

### Code Contributions

#### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/fact-checking-bert.git
   cd fact-checking-bert
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

#### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and linting**
   ```bash
   make test
   make lint
   make format
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions small and focused

### Code Formatting

We use `black` for code formatting and `isort` for import sorting:

```bash
make format
```

### Linting

We use `flake8` for linting and `mypy` for type checking:

```bash
make lint
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Update inline comments for complex logic

## 🧪 Testing

### Running Tests

```bash
make test
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)
- Mock external dependencies

### Test Structure

```
tests/
├── test_data_processing.py
├── test_models.py
├── test_utils.py
└── conftest.py
```

## 📦 Pull Request Guidelines

### Before Submitting

1. **Ensure tests pass**
   ```bash
   make test
   ```

2. **Check code quality**
   ```bash
   make lint
   make format
   ```

3. **Update documentation**
   - Update README.md if needed
   - Add docstrings for new functions
   - Update inline comments

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

Examples:
feat(models): add custom optimizer support
fix(data): resolve tokenization issue
docs(readme): update installation instructions
test(utils): add tests for clippyadagrad
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🚀 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `setup.py`
2. **Update CHANGELOG.md**
3. **Create release tag**
4. **Deploy to PyPI** (if applicable)

## 📞 Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers for private matters

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Fact-checking BERT project! 