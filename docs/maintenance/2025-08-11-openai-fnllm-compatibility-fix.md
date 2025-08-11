# GraphRAG Dependency Fixes and Prevention

## 🚨 Issue Summary

GraphRAG was experiencing CLI failures due to version incompatibility between:
- `fnllm` version 0.3.1 (latest stable)
- `openai` version 1.99.x (incompatible)

**Error:** `ImportError: cannot import name 'ChatCompletionMessageToolCall' from 'openai.types.chat.chat_completion_message_tool_call'`

## ✅ Solution Implemented

### 1. Updated `pyproject.toml`
- **openai**: Changed from `^1.68.0` to `>=1.68.0,<1.80.0`
- **fnllm**: Changed from `^0.3.0` to `>=0.3.0,<0.4.0`
- Added detailed comments explaining version constraints

### 2. Created `constraints.txt`
- Pins critical dependency versions
- Prevents automatic upgrades to incompatible versions
- Covers all major dependency categories

### 3. Created `requirements-dev.txt`
- Development dependencies with version constraints
- Separates runtime vs development needs

### 4. Updated `README.md`
- Added dependency management section
- Clear installation instructions
- Compatibility warnings

### 5. Created Installation Scripts
- `scripts/install-deps.sh` (Unix/Linux/macOS)
- `scripts/install-deps.py` (Cross-platform)
- Ensures proper dependency installation

## 🔧 How to Use

### For Users
```bash
# Option 1: Use the installation script
./scripts/install-deps.sh

# Option 2: Install with constraints manually
pip install -r constraints.txt

# Option 3: Use Poetry (recommended)
poetry install
```

### For Developers
```bash
# Install with development dependencies
./scripts/install-deps.sh --dev

# Or manually
pip install -r requirements-dev.txt -r constraints.txt
```

## 🚫 What to Avoid

1. **Don't upgrade openai beyond 1.79.x** - breaks fnllm compatibility
2. **Don't remove version constraints** - allows incompatible upgrades
3. **Don't mix pip and poetry** without constraints
4. **Don't install from system Python** - use virtual environment

## 🔍 Monitoring

To check for potential issues:
```bash
# Verify current versions
pip list | grep -E "(openai|fnllm)"

# Expected versions:
# openai 1.68.0-1.79.x
# fnllm 0.3.x
```

## 🚀 Future Prevention

1. **CI/CD Integration**: Add dependency checks to build pipelines
2. **Automated Testing**: Test CLI functionality after dependency updates
3. **Version Pinning**: Consider pinning exact versions for critical dependencies
4. **Dependency Updates**: Review and test before updating major versions

## 📝 Notes

- The `fnllm` package is actively developed but has limited version compatibility
- `openai` package has breaking changes between major versions
- GraphRAG CLI functionality depends on both packages working together
- Virtual environments are essential for dependency isolation

## 🆘 Troubleshooting

If you encounter CLI failures:

1. Check package versions: `pip list | grep -E "(openai|fnllm)"`
2. Verify virtual environment is active
3. Reinstall with constraints: `pip install -r constraints.txt --force-reinstall`
4. Check for conflicting packages in global environment
5. Report issues with version information included
