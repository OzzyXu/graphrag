#!/usr/bin/env python3
"""
Comprehensive test suite for Causal Search functionality.

This script tests all the major features of the causal search method:
1. Configuration loading and validation
2. Prompt loading from GraphRAG structure
3. Causal search engine creation
4. Output file generation with configurable options
5. Query-specific file naming
6. Error handling and edge cases
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from graphrag.config.load_config import load_config
    from graphrag.query.factory import get_causal_search_engine
    from graphrag.query.structured_search.causal_search.search import CausalSearch, CausalSearchError
    from graphrag.vector_stores.base import BaseVectorStore
    from graphrag.language_model.manager import ModelManager
    from graphrag.language_model.protocol.base import ChatModel, EmbeddingModel
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this from the project root and all dependencies are installed")
    sys.exit(1)


class MockChatModel(ChatModel):
    """Mock chat model for testing purposes."""
    
    def __init__(self, responses: List[str] = None):
        self.responses = responses or ["Mock causal report", "Mock final response"]
        self.call_count = 0
    
    async def achat(self, prompt: str, history: List[Dict[str, str]] = None, 
                    model_parameters: Dict[str, Any] = None) -> str:
        """Mock chat completion."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        logger.info(f"Mock chat model called with prompt: {prompt[:100]}...")
        return response
    
    async def astream_chat(self, prompt: str, history: List[Dict[str, str]] = None,
                          model_parameters: Dict[str, Any] = None):
        """Mock streaming chat completion."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        yield response


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing purposes."""
    
    async def aembed_text(self, text: str) -> List[float]:
        """Mock text embedding."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # Mock 5-dimensional embedding


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing purposes."""
    
    def __init__(self):
        self.entities = {}
    
    def filter_by_id(self, entity_keys):
        """Mock entity filtering."""
        pass
    
    async def connect(self):
        """Mock connection."""
        pass
    
    async def load_documents(self, documents):
        """Mock document loading."""
        pass
    
    async def search_by_id(self, entity_keys):
        """Mock search by ID."""
        pass
    
    async def similarity_search_by_text(self, text, k=10):
        """Mock similarity search by text."""
        return []
    
    async def similarity_search_by_vector(self, vector, k=10):
        """Mock similarity search by vector."""
        return []


class MockLocalContextBuilder:
    """Mock context builder for testing purposes."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    async def build_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock context building."""
        return {
            "entities": [{"id": "1", "title": "Test Entity"}],
            "relationships": [{"source": "1", "target": "2", "description": "Test Relationship"}],
            "text_units": [{"id": "1", "text": "Test text unit"}],
            "community_reports": [{"community_id": "1", "summary": "Test community"}]
        }
    
    def get_context_records(self, **kwargs):
        """Mock context records retrieval."""
        return {
            "entities": [{"id": "1", "title": "Test Entity"}],
            "relationships": [{"source": "1", "target": "2", "description": "Test Relationship"}],
            "text_units": [{"id": "1", "text": "Test text unit"}],
            "community_reports": [{"community_id": "1", "summary": "Test community"}]
        }


def test_configuration_loading():
    """Test that causal search configuration can be loaded properly."""
    logger.info("🧪 Testing configuration loading...")
    
    try:
        # Try to load configuration from current directory
        try:
            config = load_config(Path("ragtest").resolve(), "settings.yaml")
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            # Try alternative approach
            try:
                config = load_config(Path("ragtest").resolve(), Path("settings.yaml"))
            except Exception as e2:
                logger.error(f"❌ Alternative configuration loading also failed: {e2}")
                return False
        
        # Check if causal_search section exists
        if not hasattr(config, 'causal_search'):
            logger.error("❌ Configuration missing causal_search section")
            return False
        
        # Check required fields
        required_fields = [
            's_parameter', 'top_k_mapped_entities', 'top_k_relationships',
            'text_unit_prop', 'community_prop', 'max_context_tokens',
            'chat_model_id', 'embedding_model_id', 'save_network_data',
            'save_causal_report', 'output_folder'
        ]
        
        for field in required_fields:
            if not hasattr(config.causal_search, field):
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        logger.info("✅ Configuration loading test passed")
        logger.info(f"   s_parameter: {config.causal_search.s_parameter}")
        logger.info(f"   save_network_data: {config.causal_search.save_network_data}")
        logger.info(f"   save_causal_report: {config.causal_search.save_causal_report}")
        logger.info(f"   output_folder: {config.causal_search.output_folder}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False


def test_prompt_loading():
    """Test that prompts can be loaded from GraphRAG structure."""
    logger.info("🧪 Testing prompt loading...")
    
    try:
        # Test importing prompts
        from graphrag.prompts.query.causal_search import CAUSAL_DISCOVERY_PROMPT, CAUSAL_SUMMARY_PROMPT
        
        # Check prompt content
        if not CAUSAL_DISCOVERY_PROMPT or len(CAUSAL_DISCOVERY_PROMPT.strip()) < 100:
            logger.error("❌ Causal discovery prompt is too short or empty")
            return False
        
        if not CAUSAL_SUMMARY_PROMPT or len(CAUSAL_SUMMARY_PROMPT.strip()) < 100:
            logger.error("❌ Causal summary prompt is too short or empty")
            return False
        
        # Check for required placeholders
        if "{graph_data}" not in CAUSAL_DISCOVERY_PROMPT:
            logger.error("❌ Causal discovery prompt missing {graph_data} placeholder")
            return False
        
        if "{causal_summary}" not in CAUSAL_SUMMARY_PROMPT:
            logger.error("❌ Causal summary prompt missing {causal_summary} placeholder")
            return False
        
        logger.info("✅ Prompt loading test passed")
        logger.info(f"   Discovery prompt length: {len(CAUSAL_DISCOVERY_PROMPT)}")
        logger.info(f"   Summary prompt length: {len(CAUSAL_SUMMARY_PROMPT)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt loading failed: {e}")
        return False


def test_causal_search_creation():
    """Test that causal search engine can be created."""
    logger.info("🧪 Testing causal search engine creation...")
    
    try:
        # Create mock models
        mock_chat_model = MockChatModel()
        mock_embedding_model = MockEmbeddingModel()
        mock_vector_store = MockVectorStore()
        
        # Create causal search instance
        causal_search = CausalSearch(
            model=mock_chat_model,
            context_builder=MockLocalContextBuilder(),
            s_parameter=3,
            max_context_tokens=12000
        )
        
        # Check basic attributes
        if causal_search.s_parameter != 3:
            logger.error(f"❌ s_parameter not set correctly: {causal_search.s_parameter}")
            return False
        
        if causal_search.max_context_tokens != 12000:
            logger.error(f"❌ max_context_tokens not set correctly: {causal_search.max_context_tokens}")
            return False
        
        # Check prompt loading
        if not causal_search.causal_discovery_prompt:
            logger.error("❌ Causal discovery prompt not loaded")
            return False
        
        if not causal_search.causal_summary_prompt:
            logger.error("❌ Causal summary prompt not loaded")
            return False
        
        logger.info("✅ Causal search engine creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Causal search engine creation failed: {e}")
        return False


def test_query_id_generation():
    """Test query ID generation for file naming."""
    logger.info("🧪 Testing query ID generation...")
    
    try:
        # Create causal search instance
        causal_search = CausalSearch(
            model=MockChatModel(),
            context_builder=MockLocalContextBuilder(),
            s_parameter=3,
            max_context_tokens=12000
        )
        
        # Test query ID generation
        query1 = "What are the causal relationships?"
        query2 = "What are the causal relationships?"  # Same query
        query3 = "Different query about causality"
        
        id1 = causal_search._generate_query_id(query1)
        id2 = causal_search._generate_query_id(query2)
        id3 = causal_search._generate_query_id(query3)
        
        # Check format (should be hash_timestamp)
        if not id1.count('_') == 1:
            logger.error(f"❌ Query ID format incorrect: {id1}")
            return False
        
        # Same query should have same hash part
        hash1 = id1.split('_')[0]
        hash2 = id2.split('_')[0]
        if hash1 != hash2:
            logger.error(f"❌ Same query should have same hash: {hash1} vs {hash2}")
            return False
        
        # Different queries should have different hashes
        hash3 = id3.split('_')[0]
        if hash1 == hash3:
            logger.error(f"❌ Different queries should have different hashes: {hash1} vs {hash3}")
            return False
        
        logger.info("✅ Query ID generation test passed")
        logger.info(f"   Query 1 ID: {id1}")
        logger.info(f"   Query 2 ID: {id2}")
        logger.info(f"   Query 3 ID: {id3}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Query ID generation test failed: {e}")
        return False


async def test_output_saving():
    """Test output saving functionality with different configurations."""
    logger.info("🧪 Testing output saving...")
    
    try:
        # Test 1: Output saving disabled
        causal_search = CausalSearch(
            model=MockChatModel(),
            context_builder=MockLocalContextBuilder(),
            s_parameter=3,
            max_context_tokens=12000
        )
        
        # Set context builder params to disable output saving
        causal_search.context_builder_params = {
            'save_network_data': False,
            'save_causal_report': False,
            'output_folder': 'causal_search'
        }
        
        # Try to save outputs (should not create files)
        await causal_search._save_outputs("test data", "test report", "test query")
        
        # Check that no files were created
        outputs_dir = Path("data/outputs/causal_search")
        if outputs_dir.exists():
            logger.warning("⚠️  Output directory created even when saving disabled")
        
        # Test 2: Output saving enabled
        causal_search.context_builder_params = {
            'save_network_data': True,
            'save_causal_report': True,
            'output_folder': 'causal_search'
        }
        
        # Save outputs
        await causal_search._save_outputs("test network data", "test causal report", "test query")
        
        # Check that files were created
        if not outputs_dir.exists():
            logger.error("❌ Output directory not created when saving enabled")
            return False
        
        # Check for files with query-specific naming
        files = list(outputs_dir.glob("*"))
        if len(files) < 2:
            logger.error(f"❌ Expected at least 2 output files, found {len(files)}")
            return False
        
        # Check file names contain query ID
        query_id = causal_search._generate_query_id("test query")
        expected_files = [
            f"causal_search_network_data_{query_id}.json",
            f"causal_search_report_{query_id}.md"
        ]
        
        for expected_file in expected_files:
            if not (outputs_dir / expected_file).exists():
                logger.error(f"❌ Expected file not found: {expected_file}")
                return False
        
        logger.info("✅ Output saving test passed")
        logger.info(f"   Created {len(files)} files in {outputs_dir}")
        
        # Clean up test files
        import shutil
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
            logger.info("   Cleaned up test output directory")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Output saving test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling in causal search."""
    logger.info("🧪 Testing error handling...")
    
    try:
        # Test 1: Invalid model parameters
        try:
            causal_search = CausalSearch(
                model=None,  # Invalid model
                context_builder=MockLocalContextBuilder(),
                s_parameter=3,
                max_context_tokens=12000
            )
            # Try to use the model to trigger the error
            await causal_search._generate_causal_report("test data")
            logger.error("❌ Should have failed when using invalid model")
            return False
        except Exception as e:
            logger.info(f"✅ Correctly handled invalid model ({type(e).__name__})")
        
        # Test 2: CausalSearchError exception
        causal_search = CausalSearch(
            model=MockChatModel(),
            context_builder=MockLocalContextBuilder(),
            s_parameter=3,
            max_context_tokens=12000
        )
        
        # Test custom exception
        try:
            raise CausalSearchError("Test error message")
        except CausalSearchError as e:
            if "Test error message" not in str(e):
                logger.error(f"❌ Exception message not preserved: {e}")
                return False
            logger.info("✅ CausalSearchError exception working correctly")
        
        logger.info("✅ Error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False


async def test_full_search_workflow():
    """Test the complete causal search workflow."""
    logger.info("🧪 Testing full search workflow...")
    
    try:
        # Create causal search instance
        causal_search = CausalSearch(
            model=MockChatModel(),
            context_builder=MockLocalContextBuilder(),
            s_parameter=3,
            max_context_tokens=12000
        )
        
        # Set context builder params for output saving
        causal_search.context_builder_params = {
            'save_network_data': True,
            'save_causal_report': True,
            'output_folder': 'causal_search'
        }
        
        # Mock the placeholder methods
        async def mock_extract_extended_nodes(query, top_k, **kwargs):
            return ["node1", "node2", "node3"]
        
        async def mock_extract_graph_information(nodes, **kwargs):
            # Return an object with context_records and context_chunks attributes
            class MockContextResult:
                def __init__(self):
                    self.context_records = {
                        "entities": ["entity1"], 
                        "relationships": ["rel1"]
                    }
                    self.context_chunks = "Mock context chunks for testing"
            return MockContextResult()
        
        def mock_format_network_data(context):
            return json.dumps({"test": "data"})
        
        # Replace placeholder methods
        causal_search._extract_extended_nodes = mock_extract_extended_nodes
        causal_search._extract_graph_information = mock_extract_graph_information
        causal_search._format_network_data_for_causal_prompt = mock_format_network_data
        
        # Run search
        start_time = time.time()
        result = await causal_search.search("Test causal query")
        completion_time = time.time() - start_time
        
        # Check result structure
        if not hasattr(result, 'response'):
            logger.error("❌ Search result missing response")
            return False
        
        if not hasattr(result, 'completion_time'):
            logger.error("❌ Search result missing completion_time")
            return False
        
        logger.info("✅ Full search workflow test passed")
        logger.info(f"   Response length: {len(result.response)}")
        logger.info(f"   Completion time: {completion_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full search workflow test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Causal Search Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Prompt Loading", test_prompt_loading),
        ("Engine Creation", test_causal_search_creation),
        ("Query ID Generation", test_query_id_generation),
        ("Output Saving", test_output_saving),
        ("Error Handling", test_error_handling),
        ("Full Workflow", test_full_search_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Causal search is working correctly.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
