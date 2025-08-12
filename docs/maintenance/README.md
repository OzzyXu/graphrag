# Maintenance Documentation

This folder contains documentation for maintenance tasks, bug fixes, and dependency management issues.

## 📋 Maintenance Records

### 2025-08-11: OpenAI-FNLLM Compatibility Fix
- **File**: [2025-08-11-openai-fnllm-compatibility-fix.md](./2025-08-11-openai-fnllm-compatibility-fix.md)
- **Issue**: CLI failures due to version incompatibility between `fnllm` and `openai` packages
- **Solution**: Updated dependency constraints and created version pinning files
- **Status**: ✅ Resolved

## 🔧 How to Add New Maintenance Records

When documenting a new maintenance task or fix:

1. **Create a new file** with the format: `YYYY-MM-DD-description-of-issue.md`
2. **Update this README** with a new entry
3. **Include**:
   - Issue description
   - Solution implemented
   - Files modified
   - Status (Resolved/In Progress/Blocked)
   - Any relevant commands or code snippets

## 📁 File Naming Convention

- **Format**: `YYYY-MM-DD-descriptive-name.md`
- **Examples**:
  - `2025-08-11-openai-fnllm-compatibility-fix.md`
  - `2025-08-12-python-3-13-upgrade-blocker.md`
  - `2025-08-13-azure-dependency-update.md`

## 🎯 Purpose

This folder serves as a historical record of:
- Dependency issues and their resolutions
- Breaking changes and migration steps
- Performance optimizations
- Security updates
- Infrastructure changes

## 📖 Related Documentation

- [DEVELOPING.md](../../DEVELOPING.md) - Development setup and guidelines
- [breaking-changes.md](../../breaking-changes.md) - Breaking changes documentation
- [CHANGELOG.md](../../CHANGELOG.md) - Version change history
