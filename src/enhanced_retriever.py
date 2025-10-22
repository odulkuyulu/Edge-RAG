"""
Enhanced Retriever for Edge RAG System
Leverages intelligent chunking metadata for improved search accuracy
"""

import os, json, requests, ollama, numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from dataclasses import dataclass
import re

# Import from existing modules
from indexer import generate_embedding, detect_language
from retriever import extract_entities_azure, extract_key_phrases_azure

load_dotenv()

@dataclass
class SearchContext:
    """Context information for enhanced search"""
    query_type: str  # question, lookup, comparison, etc.
    focus_entities: List[str]
    focus_keywords: List[str]
    requires_structure: bool
    language_preference: str

class EnhancedRetriever:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY", "")
        )
        
        # Query pattern analysis
        self.question_patterns = [
            r'\bwhat\b', r'\bhow\b', r'\bwhen\b', r'\bwhere\b', r'\bwhy\b', r'\bwho\b',
            r'\bماذا\b', r'\bكيف\b', r'\bمتى\b', r'\bأين\b', r'\bلماذا\b', r'\bمن\b'
        ]
        
        self.lookup_patterns = [
            r'\bfind\b', r'\bshow\b', r'\blist\b', r'\btell me about\b',
            r'\bأجد\b', r'\bأظهر\b', r'\bقائمة\b', r'\bأخبرني عن\b'
        ]
        
        self.comparison_patterns = [
            r'\bcompare\b', r'\bdifference\b', r'\bbetter\b', r'\bversus\b', r'\bvs\b',
            r'\bقارن\b', r'\bالفرق\b', r'\bأفضل\b', r'\bمقابل\b'
        ]

    def analyze_query_context(self, query: str) -> SearchContext:
        """Analyze query to understand search intent and context"""
        query_lower = query.lower()
        
        # Determine query type
        query_type = "lookup"  # default
        if any(re.search(pattern, query_lower) for pattern in self.question_patterns):
            query_type = "question"
        elif any(re.search(pattern, query_lower) for pattern in self.comparison_patterns):
            query_type = "comparison"
        
        # Extract focus entities and keywords using Azure
        language = detect_language(query)
        entities = extract_entities_azure(query, language)
        key_phrases = extract_key_phrases_azure(query, language)
        
        # Flatten entities
        focus_entities = []
        for entity_list in entities.values():
            focus_entities.extend(entity_list)
        
        # Determine if structural information is needed
        structure_indicators = [
            'article', 'section', 'part', 'clause', 'investment', 'partnership',
            'مادة', 'فصل', 'بند', 'قسم', 'استثمار', 'شراكة'
        ]
        requires_structure = any(indicator in query_lower for indicator in structure_indicators)
        
        return SearchContext(
            query_type=query_type,
            focus_entities=focus_entities,
            focus_keywords=key_phrases,
            requires_structure=requires_structure,
            language_preference=language
        )

    def enhanced_search(self, 
                       query: str,
                       doc_type: Optional[str] = None,
                       language: Optional[str] = None,
                       importance_threshold: float = 0.0,
                       max_results: int = 10,
                       embed_model: str = "bge-m3") -> List[Dict[str, Any]]:
        """Enhanced search with context-aware ranking"""
        
        # Analyze query context
        search_context = self.analyze_query_context(query)
        
        # Use detected language or provided language
        query_language = language or search_context.language_preference
        
        # Generate query embedding
        query_embedding = generate_embedding(query, query_language, embed_model, role="query")
        
        # Build collection name
        lang_code = "ar" if query_language == "arabic" else "en"
        collection_name = f"rag_docs_{lang_code}_{embed_model.replace('/', '_').replace(':', '_')}"
        
        # Build enhanced filters
        filters = []
        
        # Document type filter
        if doc_type:
            filters.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        
        # Importance threshold filter
        if importance_threshold > 0:
            filters.append(FieldCondition(key="importance_score", range=Range(gte=importance_threshold)))
        
        # Language filter
        filters.append(FieldCondition(key="language", match=MatchValue(value=query_language)))
        
        # Structure preference filter
        if search_context.requires_structure:
            filters.append(FieldCondition(key="has_structure", match=MatchValue(value=True)))
        
        # Perform search
        search_filter = Filter(must=filters) if filters else None
        
        try:
            # Get more results than needed for re-ranking
            raw_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=max_results * 3,  # Get 3x results for intelligent re-ranking
                with_payload=True
            )
            
            # Enhanced re-ranking with context awareness
            reranked_results = self._enhanced_rerank(query, raw_results, search_context)
            
            # Return top results
            return reranked_results[:max_results]
            
        except Exception as e:
            print(f"[ENHANCED_SEARCH] Error: {e}")
            return []

    def _enhanced_rerank(self, query: str, results: List, search_context: SearchContext) -> List[Dict[str, Any]]:
        """Advanced re-ranking based on multiple factors"""
        
        enhanced_results = []
        query_lower = query.lower()
        
        for result in results:
            payload = result.payload
            base_score = result.score
            
            # Initialize relevance multiplier
            relevance_multiplier = 1.0
            
            # Factor 1: Importance score boost
            importance = payload.get('importance_score', 0.5)
            relevance_multiplier += importance * 0.25
            
            # Factor 2: Document type relevance
            doc_type = payload.get('doc_type', 'general')
            if self._is_doc_type_relevant(query_lower, doc_type, search_context):
                relevance_multiplier += 0.2
            
            # Factor 3: Semantic type relevance
            semantic_type = payload.get('semantic_type', 'content')
            if self._is_semantic_type_relevant(semantic_type, search_context):
                relevance_multiplier += 0.15
            
            # Factor 4: Entity matching boost
            entities = payload.get('entities', {})
            entity_boost = self._calculate_entity_boost(entities, search_context.focus_entities)
            relevance_multiplier += entity_boost
            
            # Factor 5: Key phrase matching boost
            key_phrases = payload.get('key_phrases', [])
            phrase_boost = self._calculate_phrase_boost(key_phrases, search_context.focus_keywords, query_lower)
            relevance_multiplier += phrase_boost
            
            # Factor 6: Structural information boost
            if search_context.requires_structure and payload.get('has_structure', False):
                relevance_multiplier += 0.15
                
                # Additional boost for structural markers that match query
                structural_markers = payload.get('structural_markers', [])
                if any(marker.lower() in query_lower for marker in structural_markers):
                    relevance_multiplier += 0.1
            
            # Factor 7: Content quality indicators
            word_count = payload.get('word_count', 0)
            if 50 <= word_count <= 300:  # Sweet spot for informative chunks
                relevance_multiplier += 0.05
            elif word_count < 30:  # Penalty for very short chunks
                relevance_multiplier *= 0.8
            
            # Factor 8: Query type specific boosts
            if search_context.query_type == "question":
                # Boost chunks that likely contain answers
                text = payload.get('text', '').lower()
                if any(word in text for word in ['because', 'due to', 'result', 'لأن', 'نتيجة', 'بسبب']):
                    relevance_multiplier += 0.1
            
            elif search_context.query_type == "comparison":
                # Boost chunks that contain comparative language
                text = payload.get('text', '').lower()
                if any(word in text for word in ['compared', 'versus', 'different', 'مقارنة', 'مختلف']):
                    relevance_multiplier += 0.1
            
            # Calculate final score
            final_score = base_score * relevance_multiplier
            
            # Prepare enhanced result
            enhanced_result = {
                'text': payload.get('text', ''),
                'source': payload.get('source', ''),
                'doc_type': doc_type,
                'semantic_type': semantic_type,
                'importance_score': importance,
                'similarity_score': base_score,
                'relevance_multiplier': relevance_multiplier,
                'final_score': final_score,
                'chunk_info': {
                    'chunk_id': payload.get('chunk_id', 0),
                    'total_chunks': payload.get('total_chunks', 1),
                    'word_count': word_count,
                    'has_structure': payload.get('has_structure', False),
                    'structural_markers': payload.get('structural_markers', [])
                },
                'entities': entities,
                'key_phrases': key_phrases,
                'metadata': payload.get('metadata', {}),
                'match_reasons': self._explain_match(payload, search_context, query_lower)
            }
            
            enhanced_results.append(enhanced_result)
        
        # Sort by final score
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return enhanced_results

    def _is_doc_type_relevant(self, query: str, doc_type: str, context: SearchContext) -> bool:
        """Check if document type is relevant to query"""
        type_keywords = {
            'business_partnership': ['partnership', 'investment', 'business', 'microsoft', 'g42', 'azure', 'ai'],
            'legal_arabic': ['law', 'legal', 'article', 'قانون', 'مادة', 'legal'],
            'events_business': ['event', 'conference', 'meeting', 'أحداث', 'مؤتمر'],
            'form_document': ['form', 'details', 'bank', 'account', 'استمارة', 'بنك']
        }
        
        keywords = type_keywords.get(doc_type, [])
        return any(keyword in query for keyword in keywords)

    def _is_semantic_type_relevant(self, semantic_type: str, context: SearchContext) -> bool:
        """Check if semantic type matches query requirements"""
        if context.requires_structure:
            return 'structured' in semantic_type or 'section' in semantic_type
        
        if context.query_type == "question":
            return semantic_type in ['structured_section', 'semantic_paragraph']
        
        return True  # Most semantic types are generally relevant

    def _calculate_entity_boost(self, entities: Dict[str, List[str]], focus_entities: List[str]) -> float:
        """Calculate boost based on entity matching"""
        if not focus_entities:
            return 0.0
        
        matched_entities = 0
        total_entities = 0
        
        for entity_list in entities.values():
            total_entities += len(entity_list)
            for entity in entity_list:
                if any(focus_entity.lower() in entity.lower() or entity.lower() in focus_entity.lower() 
                      for focus_entity in focus_entities):
                    matched_entities += 1
        
        if total_entities == 0:
            return 0.0
        
        return min(matched_entities / len(focus_entities), 0.2)  # Cap at 0.2

    def _calculate_phrase_boost(self, key_phrases: List[str], focus_keywords: List[str], query: str) -> float:
        """Calculate boost based on key phrase matching"""
        boost = 0.0
        
        # Exact phrase matches in key phrases
        for phrase in key_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in query:
                boost += 0.05
            
            # Partial matches with focus keywords
            for keyword in focus_keywords:
                if keyword.lower() in phrase_lower:
                    boost += 0.02
        
        return min(boost, 0.15)  # Cap at 0.15

    def _explain_match(self, payload: Dict, context: SearchContext, query: str) -> List[str]:
        """Generate explanation for why this chunk matches"""
        reasons = []
        
        importance = payload.get('importance_score', 0.5)
        if importance > 0.7:
            reasons.append(f"High importance content (score: {importance:.2f})")
        
        if payload.get('has_structure', False):
            reasons.append("Contains structured information")
        
        # Entity matches
        entities = payload.get('entities', {})
        for entity_list in entities.values():
            for entity in entity_list:
                if any(focus_entity.lower() in entity.lower() for focus_entity in context.focus_entities):
                    reasons.append(f"Entity match: {entity}")
                    break
        
        # Key phrase matches
        key_phrases = payload.get('key_phrases', [])
        for phrase in key_phrases:
            if phrase.lower() in query:
                reasons.append(f"Key phrase match: {phrase}")
        
        doc_type = payload.get('doc_type', 'general')
        if self._is_doc_type_relevant(query, doc_type, context):
            reasons.append(f"Relevant document type: {doc_type}")
        
        return reasons

    def generate_enhanced_response(self, 
                                 query: str, 
                                 search_results: List[Dict[str, Any]], 
                                 generation_model: str = "qwen2.5:0.5b") -> Dict[str, Any]:
        """Generate response with enhanced context and source attribution"""
        
        if not search_results:
            language = detect_language(query)
            no_results_msg = ("لم أجد معلومات ذات صلة بسؤالك في الوثائق المتاحة." 
                            if language == "arabic" 
                            else "I couldn't find relevant information for your query in the available documents.")
            return {
                'response': no_results_msg,
                'confidence': 0.0,
                'sources_used': 0,
                'context_quality': 'no_results'
            }
        
        # Build rich context from search results
        context_parts = []
        source_info = []
        
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            source = result.get('source', 'Unknown')
            doc_type = result.get('doc_type', 'general')
            semantic_type = result.get('semantic_type', 'content')
            importance = result.get('importance_score', 0.5)
            text = result.get('text', '')
            
            # Add chunk context information
            chunk_info = result.get('chunk_info', {})
            chunk_id = chunk_info.get('chunk_id', 0)
            total_chunks = chunk_info.get('total_chunks', 1)
            
            context_header = f"[Source {i+1}: {source} ({doc_type}) - Chunk {chunk_id+1}/{total_chunks}, Importance: {importance:.2f}]"
            context_parts.append(f"{context_header}\n{text}\n")
            
            source_info.append({
                'source': source,
                'doc_type': doc_type,
                'semantic_type': semantic_type,
                'importance': importance,
                'chunk_info': chunk_info,
                'match_reasons': result.get('match_reasons', [])
            })
        
        context = "\n".join(context_parts)
        
        # Enhanced prompt with context awareness
        language = detect_language(query)
        
        if language == "arabic":
            prompt = f"""بناءً على الوثائق المرفقة، أجب على السؤال التالي بدقة ووضوح:

السؤال: {query}

السياق من الوثائق:
{context}

تعليمات:
- استخدم المعلومات من الوثائق المرفقة فقط
- اذكر مصدر المعلومات عند الإشارة لنقاط محددة
- إذا لم تجد إجابة دقيقة، اذكر ذلك بوضوح
- كن موجزاً ومفيداً
- رتب المعلومات بشكل منطقي

الإجابة:"""
        else:
            prompt = f"""Based on the provided documents, answer the following question accurately and clearly:

Question: {query}

Context from documents:
{context}

Instructions:
- Use only information from the provided documents
- Cite sources when referencing specific points
- If you cannot find a precise answer, state this clearly
- Be concise and helpful
- Organize information logically

Answer:"""
        
        try:
            # Generate response
            response = ollama.generate(model=generation_model, prompt=prompt)
            generated_text = response['response']
            
            # Calculate confidence based on search results quality
            avg_score = sum(r.get('final_score', 0) for r in search_results[:3]) / min(3, len(search_results))
            confidence = min(avg_score * 1.2, 1.0)  # Boost confidence slightly, cap at 1.0
            
            # Determine context quality
            high_importance_count = sum(1 for r in search_results if r.get('importance_score', 0) > 0.7)
            if high_importance_count >= 2:
                context_quality = 'high'
            elif high_importance_count >= 1:
                context_quality = 'medium'
            else:
                context_quality = 'low'
            
            return {
                'response': generated_text,
                'confidence': confidence,
                'sources_used': len(search_results),
                'context_quality': context_quality,
                'source_details': source_info
            }
            
        except Exception as e:
            print(f"[GENERATION] Error: {e}")
            error_msg = ("عذراً، حدث خطأ في توليد الإجابة." 
                        if language == "arabic" 
                        else "Sorry, there was an error generating the response.")
            return {
                'response': error_msg,
                'confidence': 0.0,
                'sources_used': 0,
                'context_quality': 'error'
            }

def enhanced_search_documents(query: str, 
                            language: str = None, 
                            embed_model: str = "bge-m3", 
                            generation_model: str = "qwen2.5:0.5b",
                            doc_type: str = None,
                            importance_threshold: float = 0.0,
                            max_results: int = 10) -> Dict[str, Any]:
    """Main enhanced search function"""
    
    retriever = EnhancedRetriever()
    
    # Perform enhanced search
    search_results = retriever.enhanced_search(
        query=query,
        language=language,
        doc_type=doc_type,
        importance_threshold=importance_threshold,
        max_results=max_results,
        embed_model=embed_model
    )
    
    # Generate enhanced response
    response_data = retriever.generate_enhanced_response(query, search_results, generation_model)
    
    # Prepare final result
    result = {
        'response': response_data['response'],
        'confidence': response_data['confidence'],
        'context_quality': response_data['context_quality'],
        'sources': [
            {
                'text': result['text'][:300] + '...' if len(result['text']) > 300 else result['text'],
                'source': result['source'],
                'doc_type': result['doc_type'],
                'semantic_type': result['semantic_type'],
                'similarity_score': round(result['similarity_score'], 3),
                'final_score': round(result['final_score'], 3),
                'importance_score': round(result['importance_score'], 3),
                'relevance_multiplier': round(result['relevance_multiplier'], 3),
                'chunk_info': result['chunk_info'],
                'match_reasons': result['match_reasons'][:3]  # Limit to top 3 reasons
            }
            for result in search_results[:5]
        ],
        'search_stats': {
            'total_results': len(search_results),
            'sources_used': response_data['sources_used'],
            'avg_importance': round(sum(r.get('importance_score', 0) for r in search_results[:5]) / min(5, len(search_results)), 3) if search_results else 0,
            'query_language': language or detect_language(query)
        }
    }
    
    return result

if __name__ == "__main__":
    # Test the enhanced retriever
    test_queries = [
        "What is the Microsoft G42 partnership about?",
        "كم استثمرت مايكروسوفت في شراكة G42؟",
        "What are the key aspects of the investment?",
        "ما هي المواد القانونية المهمة في القانون؟"
    ]
    
    retriever = EnhancedRetriever()
    
    for query in test_queries:
        print(f"\n🔍 Testing query: {query}")
        result = enhanced_search_documents(query, max_results=3)
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"📋 Sources: {result['search_stats']['sources_used']}")
        print(f"💡 Response preview: {result['response'][:100]}...")
