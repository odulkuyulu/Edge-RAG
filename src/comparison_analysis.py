"""
Comparison Script: Original vs Enhanced RAG System
Shows the improvements in chunking and retrieval accuracy
"""

import time
import sys
import os
sys.path.append('src')

from indexer import index_document as original_index, load_documents
from enhanced_indexer import enhanced_index_document
from retriever import search_documents as original_search
from enhanced_retriever import enhanced_search_documents

def run_comparison():
    print("🔬 RAG System Enhancement Comparison")
    print("=" * 60)
    
    # Test queries for comparison
    test_queries = [
        {
            "query": "What is the Microsoft G42 partnership about?",
            "language": "english",
            "expected_content": ["partnership", "microsoft", "g42", "investment", "billion"]
        },
        {
            "query": "How much did Microsoft invest?",
            "language": "english", 
            "expected_content": ["1.5 billion", "investment", "microsoft"]
        },
        {
            "query": "ما هي المواد القانونية المهمة؟",
            "language": "arabic",
            "expected_content": ["مادة", "قانون", "المادة"]
        }
    ]
    
    print("\n📊 Comparison Results:")
    print("-" * 60)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {test['query']}")
        print("." * 50)
        
        # Test original system
        print("\n📋 Original System:")
        try:
            start_time = time.time()
            original_result = original_search(
                test['query'], 
                language=test['language'],
                embed_model="bge-m3",
                generation_model="qwen2.5:0.5b"
            )
            original_time = time.time() - start_time
            
            original_response = original_result.get('response', 'No response')
            original_sources = len(original_result.get('sources', []))
            
            print(f"⏱️  Response Time: {original_time:.2f}s")
            print(f"📄 Sources Found: {original_sources}")
            print(f"💬 Response Preview: {original_response[:100]}...")
            
            # Check content coverage
            response_lower = original_response.lower()
            covered_content = sum(1 for content in test['expected_content'] if content.lower() in response_lower)
            coverage = covered_content / len(test['expected_content']) if test['expected_content'] else 0
            print(f"🎯 Content Coverage: {coverage:.1%} ({covered_content}/{len(test['expected_content'])})")
            
        except Exception as e:
            print(f"❌ Original system error: {e}")
            original_time = 0
            original_sources = 0
            coverage = 0
        
        # Test enhanced system
        print("\n🚀 Enhanced System:")
        try:
            start_time = time.time()
            enhanced_result = enhanced_search_documents(
                test['query'],
                language=test['language'],
                embed_model="bge-m3", 
                generation_model="qwen2.5:0.5b",
                max_results=10
            )
            enhanced_time = time.time() - start_time
            
            enhanced_response = enhanced_result.get('response', 'No response')
            enhanced_sources = len(enhanced_result.get('sources', []))
            confidence = enhanced_result.get('confidence', 0.0)
            context_quality = enhanced_result.get('context_quality', 'unknown')
            
            print(f"⏱️  Response Time: {enhanced_time:.2f}s")
            print(f"📄 Sources Found: {enhanced_sources}")
            print(f"🎯 Confidence: {confidence:.1%}")
            print(f"🏆 Context Quality: {context_quality}")
            print(f"💬 Response Preview: {enhanced_response[:100]}...")
            
            # Check content coverage
            response_lower = enhanced_response.lower()
            enhanced_covered = sum(1 for content in test['expected_content'] if content.lower() in response_lower)
            enhanced_coverage = enhanced_covered / len(test['expected_content']) if test['expected_content'] else 0
            print(f"🎯 Content Coverage: {enhanced_coverage:.1%} ({enhanced_covered}/{len(test['expected_content'])})")
            
            # Show source details
            if enhanced_result.get('sources'):
                print("\n📖 Enhanced Source Breakdown:")
                for j, source in enumerate(enhanced_result['sources'][:3], 1):
                    doc_type = source.get('doc_type', 'unknown')
                    importance = source.get('importance_score', 0)
                    final_score = source.get('final_score', 0)
                    print(f"   {j}. {source.get('source', 'Unknown')} ({doc_type}) - "
                          f"Importance: {importance:.2f}, Score: {final_score:.3f}")
            
        except Exception as e:
            print(f"❌ Enhanced system error: {e}")
            enhanced_time = 0
            enhanced_sources = 0
            enhanced_coverage = 0
            confidence = 0
        
        # Comparison summary
        print(f"\n📈 Improvement Summary:")
        if original_sources > 0 and enhanced_sources > 0:
            source_improvement = ((enhanced_sources - original_sources) / original_sources) * 100
            print(f"   📄 Sources: {source_improvement:+.1f}% change")
        
        if coverage > 0 and enhanced_coverage > 0:
            coverage_improvement = ((enhanced_coverage - coverage) / coverage) * 100
            print(f"   🎯 Content Coverage: {coverage_improvement:+.1f}% improvement")
        
        print(f"   🎯 Confidence Score: {confidence:.1%} (Enhanced only)")
        print(f"   🏆 Context Quality: {context_quality} (Enhanced only)")

def show_chunking_comparison():
    print("\n\n🔧 Chunking Strategy Comparison")
    print("=" * 60)
    
    # Load a sample document
    docs = load_documents()
    if not docs:
        print("❌ No documents found for comparison")
        return
    
    # Find the AI document for comparison
    ai_doc = None
    for doc in docs:
        if 'ai_en.txt' in doc['filename']:
            ai_doc = doc
            break
    
    if not ai_doc:
        print("❌ AI document not found for chunking comparison")
        return
    
    print(f"\n📄 Document: {ai_doc['filename']}")
    print(f"📏 Total Length: {len(ai_doc['text'])} characters")
    
    # Original chunking (from indexer.py)
    import textwrap
    original_chunks = textwrap.wrap(ai_doc['text'], 1000)  # CHUNK_SIZE from indexer
    
    print(f"\n📋 Original Chunking:")
    print(f"   📊 Number of chunks: {len(original_chunks)}")
    print(f"   📏 Avg chunk size: {sum(len(chunk) for chunk in original_chunks) / len(original_chunks):.0f} chars")
    print(f"   📑 Chunk size range: {min(len(chunk) for chunk in original_chunks)} - {max(len(chunk) for chunk in original_chunks)} chars")
    
    # Enhanced chunking
    from enhanced_indexer import EnhancedDocumentProcessor
    processor = EnhancedDocumentProcessor()
    profile = processor.detect_document_type(ai_doc['text'], ai_doc['filename'])
    enhanced_chunks = processor.intelligent_chunking(ai_doc['text'], profile, 'english')
    
    print(f"\n🚀 Enhanced Chunking:")
    print(f"   📊 Number of chunks: {len(enhanced_chunks)}")
    print(f"   🎯 Document type: {profile.name}")
    print(f"   📏 Target chunk size: {profile.chunk_size} chars")
    print(f"   🔄 Overlap ratio: {profile.overlap_ratio:.1%}")
    
    chunk_sizes = [chunk['word_count'] * 5 for chunk in enhanced_chunks]  # Estimate char count
    print(f"   📏 Avg chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} chars (estimated)")
    print(f"   📑 Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} chars (estimated)")
    
    # Show semantic types
    semantic_types = {}
    for chunk in enhanced_chunks:
        sem_type = chunk.get('semantic_type', 'unknown')
        semantic_types[sem_type] = semantic_types.get(sem_type, 0) + 1
    
    print(f"   🏷️  Semantic types:")
    for sem_type, count in semantic_types.items():
        print(f"      • {sem_type}: {count} chunks")
    
    # Show importance distribution
    importance_scores = [chunk.get('importance_score', 0) for chunk in enhanced_chunks]
    avg_importance = sum(importance_scores) / len(importance_scores)
    high_importance = sum(1 for score in importance_scores if score > 0.7)
    
    print(f"   ⭐ Importance analysis:")
    print(f"      • Average importance: {avg_importance:.2f}")
    print(f"      • High importance chunks: {high_importance}/{len(enhanced_chunks)}")
    
    # Sample chunk comparison
    print(f"\n📋 Sample Chunk Comparison:")
    print(f"   Original Chunk 1 (first 100 chars): {original_chunks[0][:100]}...")
    print(f"   Enhanced Chunk 1 (first 100 chars): {enhanced_chunks[0]['text'][:100]}...")
    print(f"   Enhanced Chunk 1 Type: {enhanced_chunks[0]['semantic_type']}")
    print(f"   Enhanced Chunk 1 Importance: {enhanced_chunks[0]['importance_score']:.2f}")

if __name__ == "__main__":
    print("🚀 Starting RAG System Enhancement Analysis...")
    
    # Run the comparison
    run_comparison()
    
    # Show chunking improvements
    show_chunking_comparison()
    
    print("\n\n🎉 Comparison Complete!")
    print("\n💡 Key Improvements:")
    print("   1. 🧠 Intelligent document type detection")
    print("   2. 📏 Adaptive chunk sizing based on content type")
    print("   3. 🔄 Smart overlapping for context continuity")
    print("   4. ⭐ Importance scoring for better ranking")
    print("   5. 🏷️  Semantic type awareness")
    print("   6. 🎯 Multi-factor relevance scoring")
    print("   7. 📊 Confidence and quality metrics")
    print("   8. 🌍 Enhanced multilingual support")
