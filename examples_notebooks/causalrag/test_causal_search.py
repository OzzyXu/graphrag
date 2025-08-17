#!/usr/bin/env python3
"""Simple test for causal search functionality."""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from graphrag.query.structured_search.causal_search.search import CausalSearch, CausalSearchError

async def test_causal_search_basic():
    """Test basic causal search functionality."""
    print("Testing CausalSearch basic functionality...")
    
    try:
        # Test that we can create the class (without actual model)
        search = CausalSearch(
            model=None,  # Will cause error if we try to use it
            context_builder=None,
            s_parameter=3,
            max_context_tokens=12000
        )
        
        print(f"✓ CausalSearch class created successfully")
        print(f"✓ s_parameter: {search.s_parameter}")
        print(f"✓ max_context_tokens: {search.max_context_tokens}")
        print(f"✓ Causal discovery prompt loaded: {len(search.causal_discovery_prompt)} characters")
        print(f"✓ Causal summary prompt loaded: {len(search.causal_summary_prompt)} characters")
        
        # Test prompt loading
        assert "Network Data" in search.causal_discovery_prompt
        assert "Causal Summary" in search.causal_summary_prompt
        
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_prompt_loading():
    """Test prompt loading functionality."""
    print("\nTesting prompt loading...")
    
    try:
        # Test with default prompts
        search = CausalSearch(
            model=None,
            context_builder=None
        )
        
        # Check that prompts contain expected content
        discovery_prompt = search.causal_discovery_prompt
        summary_prompt = search.causal_summary_prompt
        
        assert "causal discovery" in discovery_prompt.lower()
        assert "causal summary" in summary_prompt.lower()
        assert "{graph_data}" in discovery_prompt
        assert "{causal_summary}" in summary_prompt
        
        print("✓ Prompt loading tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Prompt loading test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Causal Search Test Suite")
    print("=" * 40)
    
    tests = [
        test_causal_search_basic(),
        test_prompt_loading(),
    ]
    
    results = await asyncio.gather(*tests)
    
    print("\n" + "=" * 40)
    if all(results):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
