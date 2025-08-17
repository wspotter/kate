"""
Simple test runner for Kate LLM RAG system tests.
"""

import os
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_basic_tests() -> bool:
    """Run basic functionality tests."""
    print("Kate LLM Desktop Client - RAG Pipeline Test Suite")
    print("=" * 50)
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "total": 0
    }
    
    # Import test classes
    try:
        from tests.test_rag_pipeline import (
            TestConfigurationValidation,
            TestDocumentProcessing,
            TestErrorHandling,
            TestEvaluationMetrics,
            TestRAGPipelineIntegration,
            TestResponseGeneration,
            TestRetrievalQuality,
            TestVectorOperations,
            test_integration_workflow,
        )
    except ImportError as e:
        print(f"❌ Failed to import test classes: {e}")
        return False
    
    # Test classes to run
    test_classes = [
        ("RAG Pipeline Integration", TestRAGPipelineIntegration),
        ("Document Processing", TestDocumentProcessing),
        ("Vector Operations", TestVectorOperations),
        ("Retrieval Quality", TestRetrievalQuality),
        ("Response Generation", TestResponseGeneration),
        ("Evaluation Metrics", TestEvaluationMetrics),
        ("Error Handling", TestErrorHandling),
        ("Configuration Validation", TestConfigurationValidation)
    ]
    
    # Run class-based tests
    for class_name, test_class in test_classes:
        try:
            instance = test_class()
            methods = [method for method in dir(instance) if method.startswith('test_')]
            
            class_passed = 0
            class_total = len(methods)
            
            for method in methods:
                try:
                    getattr(instance, method)()
                    class_passed += 1
                except Exception as e:
                    print(f"  ✗ {class_name}.{method}: {e}")
                    
            if class_passed == class_total:
                print(f"✓ {class_name} Tests: PASSED ({class_passed}/{class_total})")
                test_results["passed"] += 1
            else:
                print(f"✗ {class_name} Tests: FAILED ({class_passed}/{class_total})")
                test_results["failed"] += 1
                
        except Exception as e:
            print(f"✗ {class_name} Tests: FAILED - {e}")
            test_results["failed"] += 1
            
        test_results["total"] += 1
    
    # Run integration workflow test
    try:
        test_integration_workflow()
        print("✓ Integration Workflow Test: PASSED")
        test_results["passed"] += 1
    except Exception as e:
        print(f"✗ Integration Workflow Test: FAILED - {e}")
        test_results["failed"] += 1
    test_results["total"] += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Total Test Suites: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print(f"Success Rate: {(test_results['passed'] / test_results['total']) * 100:.1f}%")
    
    if test_results["failed"] == 0:
        print("\n🎉 All tests passed! RAG pipeline is working correctly.")
        return True
    else:
        print(f"\n⚠️  {test_results['failed']} test suite(s) failed. Please review and fix issues.")
        return False


def validate_rag_system() -> bool:
    """Validate key components of the RAG system exist."""
    print("\nValidating RAG System Components:")
    print("-" * 40)
    
    # Get the parent directory (project root)
    project_root = Path(__file__).parent.parent
    
    components = {
        "Database Models": "app/database/models.py",
        "Database Manager": "app/database/manager.py",
        "Embedding Service": "app/services/embedding_service.py",
        "Document Processor": "app/services/document_processor.py",
        "Vector Store": "app/services/vector_store.py",
        "Retrieval Service": "app/services/retrieval_service.py",
        "RAG Integration": "app/services/rag_integration_service.py",
        "RAG Evaluation": "app/services/rag_evaluation_service.py",
        "Background Processing": "app/services/background_processing_service.py",
        "Chat Area UI": "app/ui/components/chat_area.py",
        "Message Bubbles UI": "app/ui/components/message_bubble.py",
        "Document Manager UI": "app/ui/components/document_manager.py",
        "Progress Indicators UI": "app/ui/components/progress_indicators.py",
        "Evaluation Dashboard UI": "app/ui/components/evaluation_dashboard.py",
        "Main Window": "app/ui/main_window.py"
    }
    
    all_present = True
    for component, filepath in components.items():
        full_path = project_root / filepath
        if full_path.exists():
            print(f"✓ {component}")
        else:
            print(f"✗ {component} - Missing: {filepath}")
            all_present = False
            
    return all_present


def run_detailed_tests() -> bool:
    """Run detailed tests with individual method reporting."""
    print("\nRunning Detailed Test Analysis:")
    print("-" * 40)
    
    try:
        from tests.test_rag_pipeline import (
            TestConfigurationValidation,
            TestDocumentProcessing,
            TestErrorHandling,
            TestEvaluationMetrics,
            TestRAGPipelineIntegration,
            TestResponseGeneration,
            TestRetrievalQuality,
            TestVectorOperations,
        )
    except ImportError as e:
        print(f"❌ Failed to import test classes: {e}")
        return False
    
    test_classes = [
        TestRAGPipelineIntegration,
        TestDocumentProcessing,
        TestVectorOperations,
        TestRetrievalQuality,
        TestResponseGeneration,
        TestEvaluationMetrics,
        TestErrorHandling,
        TestConfigurationValidation
    ]
    
    total_methods = 0
    passed_methods = 0
    
    for test_class in test_classes:
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        print(f"\n{test_class.__name__}:")
        for method in methods:
            total_methods += 1
            try:
                getattr(instance, method)()
                print(f"  ✓ {method}")
                passed_methods += 1
            except Exception as e:
                print(f"  ✗ {method}: {e}")
    
    print(f"\nDetailed Results: {passed_methods}/{total_methods} methods passed")
    return passed_methods == total_methods


def main() -> bool:
    """Main test execution."""
    print("Starting Kate LLM RAG System Test Suite...")
    
    # Validate system components
    components_valid = validate_rag_system()
    
    if not components_valid:
        print("\n❌ System validation failed. Some components are missing.")
        return False
        
    print("\n✅ All system components present.")
    
    # Run basic tests
    tests_passed = run_basic_tests()
    
    # If basic tests fail, run detailed analysis
    if not tests_passed:
        print("\n" + "=" * 50)
        print("Running detailed analysis to identify specific issues...")
        run_detailed_tests()
    
    if tests_passed:
        print("\n🎉 RAG system testing completed successfully!")
        print("The Kate LLM Desktop Client RAG pipeline is ready for production use.")
        return True
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)