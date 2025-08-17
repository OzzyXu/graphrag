#!/usr/bin/env python3
"""Test import of causal_search module."""

import sys
sys.path.append('..')

try:
    import graphrag.query.structured_search.causal_search
    print("✓ causal_search module imported successfully!")
    
    from graphrag.query.structured_search.causal_search.search import CausalSearch
    print("✓ CausalSearch class imported successfully!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
