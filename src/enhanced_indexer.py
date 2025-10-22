"""
Enhanced Document Processor for Edge RAG System
Customized for specific document types in the data folder:
- AI/Tech business documents (ai_en.txt)
- Legal documents (قانون رقم (5) لسنة 1970.pdf)
- Events/Business documents (eventss.pdf, الأحداث.pdf)
- Forms (Bank Details Form.pdf)
"""

import os, uuid, json, re, nltk
from typing import List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Import from existing modules
from indexer import (
    client, generate_embedding, detect_language, extract_entities, 
    extract_key_phrases, analyze_sentiment, ensure_collection, 
    process_with_document_intelligence, PointStruct
)

# Download NLTK data if needed
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

@dataclass
class DocumentProfile:
    """Document type profile with specific processing rules"""
    name: str
    indicators: List[str]
    chunk_size: int
    overlap_ratio: float
    structure_patterns: List[str]
    importance_keywords: List[str]
    semantic_preservers: List[str]  # Patterns that should not be split

# Define document profiles based on your actual data
DOCUMENT_PROFILES = {
    'business_partnership': DocumentProfile(
        name='business_partnership',
        indicators=[
            'partnership', 'investment', 'microsoft', 'g42', 'strategic', 
            'billion', 'azure', 'cloud', 'ai', 'artificial intelligence',
            'collaboration', 'enterprise', 'sectors'
        ],
        chunk_size=600,
        overlap_ratio=0.15,
        structure_patterns=[
            r'Key aspects.*?:',
            r'The investment.*?:',
            r'The partnership.*?:',
            r'[A-Z][^.!?]*partnership[^.!?]*[.!?]',
        ],
        importance_keywords=[
            'investment', 'billion', 'partnership', 'strategic', 'microsoft',
            'ai', 'azure', 'cloud', 'development'
        ],
        semantic_preservers=[
            r'\$[\d,]+(?:\.\d+)?\s*billion',
            r'Microsoft.*?partnership',
            r'Key aspects.*?:.*?(?=\n\n|\n[A-Z])',
        ]
    ),
    
    'legal_arabic': DocumentProfile(
        name='legal_arabic',
        indicators=[
            'قانون', 'مادة', 'رقم', 'لسنة', 'المادة', 'الفصل', 'البند',
            'يجب', 'يحق', 'وفقا', 'بموجب', 'تطبق', 'تعتبر'
        ],
        chunk_size=400,
        overlap_ratio=0.2,
        structure_patterns=[
            r'مادة\s*\d+',
            r'المادة\s*\d+',
            r'الفصل\s*\d+',
            r'البند\s*\d+',
            r'قانون\s*رقم.*?لسنة\s*\d+'
        ],
        importance_keywords=[
            'قانون', 'مادة', 'المادة', 'يجب', 'يحق', 'وفقا', 'بموجب'
        ],
        semantic_preservers=[
            r'مادة\s*\d+.*?(?=مادة\s*\d+|\Z)',
            r'المادة\s*\d+.*?(?=المادة\s*\d+|\Z)',
            r'قانون\s*رقم.*?لسنة\s*\d+',
        ]
    ),
    
    'events_business': DocumentProfile(
        name='events_business',
        indicators=[
            'event', 'conference', 'summit', 'meeting', 'workshop',
            'participants', 'agenda', 'speakers', 'registration',
            'أحداث', 'مؤتمر', 'قمة', 'ورشة', 'مشاركين'
        ],
        chunk_size=500,
        overlap_ratio=0.1,
        structure_patterns=[
            r'Event.*?:',
            r'Date.*?:',
            r'Time.*?:',
            r'Location.*?:',
            r'Agenda.*?:',
        ],
        importance_keywords=[
            'event', 'date', 'time', 'location', 'speakers', 'agenda',
            'أحداث', 'تاريخ', 'وقت', 'مكان', 'متحدثين'
        ],
        semantic_preservers=[
            r'Event.*?:.*?(?=\n[A-Z]|\Z)',
            r'Date.*?:.*?(?=\n|\Z)',
        ]
    ),
    
    'form_document': DocumentProfile(
        name='form_document',
        indicators=[
            'form', 'details', 'bank', 'account', 'information',
            'name', 'address', 'phone', 'email', 'number',
            'استمارة', 'تفاصيل', 'بنك', 'حساب', 'معلومات'
        ],
        chunk_size=300,
        overlap_ratio=0.05,
        structure_patterns=[
            r'[A-Z][^:]*:',
            r'Name.*?:',
            r'Address.*?:',
            r'Phone.*?:',
            r'Account.*?:',
        ],
        importance_keywords=[
            'name', 'address', 'phone', 'account', 'bank', 'details',
            'اسم', 'عنوان', 'هاتف', 'حساب', 'بنك'
        ],
        semantic_preservers=[
            r'[A-Z][^:]*:.*?(?=\n[A-Z][^:]*:|\Z)',
        ]
    )
}

class EnhancedDocumentProcessor:
    def __init__(self):
        self.profiles = DOCUMENT_PROFILES
        
    def detect_document_type(self, text: str, filename: str) -> DocumentProfile:
        """Detect document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Score each profile
        scores = {}
        
        for profile_name, profile in self.profiles.items():
            score = 0
            
            # Check filename indicators
            if any(indicator in filename_lower for indicator in profile.indicators):
                score += 2
            
            # Check content indicators
            for indicator in profile.indicators:
                if indicator in text_lower:
                    score += 1
                    
            # Boost for specific patterns
            for pattern in profile.structure_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 3
                    
            scores[profile_name] = score
        
        # Return the highest scoring profile, or default to business_partnership
        best_profile = max(scores, key=scores.get) if scores else 'business_partnership'
        
        # Special handling for your specific files
        if 'قانون' in filename_lower or 'law' in filename_lower:
            best_profile = 'legal_arabic'
        elif 'event' in filename_lower or 'أحداث' in filename_lower:
            best_profile = 'events_business'  
        elif 'bank' in filename_lower or 'form' in filename_lower:
            best_profile = 'form_document'
        elif 'ai' in filename_lower or 'microsoft' in text_lower or 'g42' in text_lower:
            best_profile = 'business_partnership'
            
        print(f"[ENHANCED] Detected document type: {best_profile} for '{filename}' (score: {scores.get(best_profile, 0)})")
        return self.profiles[best_profile]

    def intelligent_chunking(self, text: str, profile: DocumentProfile, language: str) -> List[Dict[str, Any]]:
        """Advanced chunking based on document profile"""
        
        # Step 1: Try structure-aware chunking
        structured_chunks = self._structure_aware_chunking(text, profile)
        
        # Step 2: If structure chunking didn't work well, fall back to semantic chunking
        if not structured_chunks or len(structured_chunks) < 2:
            structured_chunks = self._semantic_sentence_chunking(text, profile, language)
        
        # Step 3: Post-process chunks
        enhanced_chunks = []
        for i, chunk_data in enumerate(structured_chunks):
            enhanced_chunk = {
                'text': chunk_data['text'],
                'chunk_id': i,
                'total_chunks': len(structured_chunks),
                'doc_type': profile.name,
                'language': language,
                'semantic_type': chunk_data.get('semantic_type', 'content'),
                'importance_score': self._calculate_importance_score(chunk_data['text'], profile),
                'word_count': len(chunk_data['text'].split()),
                'has_structure': chunk_data.get('has_structure', False),
                'structural_markers': chunk_data.get('structural_markers', []),
                'overlap_info': self._calculate_overlap_info(i, structured_chunks, profile)
            }
            enhanced_chunks.append(enhanced_chunk)
        
        print(f"[ENHANCED] Created {len(enhanced_chunks)} intelligent chunks using {profile.name} profile")
        return enhanced_chunks

    def _structure_aware_chunking(self, text: str, profile: DocumentProfile) -> List[Dict[str, Any]]:
        """Chunk based on document structure patterns"""
        chunks = []
        
        # Find structural boundaries
        boundaries = []
        for pattern in profile.structure_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                boundaries.append({
                    'position': match.start(),
                    'end': match.end(),
                    'pattern': pattern,
                    'text': match.group()
                })
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x['position'])
        
        if not boundaries:
            return []
        
        # Create chunks based on boundaries
        current_pos = 0
        
        for i, boundary in enumerate(boundaries):
            # Get text from current position to this boundary
            if boundary['position'] > current_pos:
                pre_text = text[current_pos:boundary['position']].strip()
                if pre_text and len(pre_text) > 50:  # Only keep substantial content
                    chunks.append({
                        'text': pre_text,
                        'semantic_type': 'preamble',
                        'has_structure': False
                    })
            
            # Get text from this boundary to next boundary (or end)
            next_pos = boundaries[i + 1]['position'] if i + 1 < len(boundaries) else len(text)
            section_text = text[boundary['position']:next_pos].strip()
            
            if section_text:
                # Further split if too long
                if len(section_text) > profile.chunk_size * 1.5:
                    sub_chunks = self._split_large_section(section_text, profile)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'text': sub_chunk,
                            'semantic_type': f'structured_section_{j}',
                            'has_structure': True,
                            'structural_markers': [boundary['text']]
                        })
                else:
                    chunks.append({
                        'text': section_text,
                        'semantic_type': 'structured_section',
                        'has_structure': True,
                        'structural_markers': [boundary['text']]
                    })
            
            current_pos = next_pos
        
        return chunks

    def _semantic_sentence_chunking(self, text: str, profile: DocumentProfile, language: str) -> List[Dict[str, Any]]:
        """Fallback to intelligent sentence-based chunking"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > profile.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'semantic_type': 'semantic_paragraph',
                    'has_structure': False,
                    'sentence_count': len(current_sentences)
                })
                
                # Start new chunk with overlap
                overlap_size = int(len(current_sentences) * profile.overlap_ratio)
                if overlap_size > 0:
                    overlap_sentences = current_sentences[-overlap_size:]
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'semantic_type': 'semantic_paragraph',
                'has_structure': False,
                'sentence_count': len(current_sentences)
            })
        
        return chunks

    def _split_large_section(self, text: str, profile: DocumentProfile) -> List[str]:
        """Split large sections while preserving semantic meaning"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) > profile.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with small overlap
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def _calculate_importance_score(self, text: str, profile: DocumentProfile) -> float:
        """Calculate importance score based on profile-specific keywords"""
        score = 0.5  # Base score
        text_lower = text.lower()
        
        # Boost for profile-specific important keywords
        keyword_count = sum(1 for keyword in profile.importance_keywords if keyword in text_lower)
        if profile.importance_keywords:
            score += (keyword_count / len(profile.importance_keywords)) * 0.3
        
        # Boost for structural markers
        if any(pattern for pattern in profile.structure_patterns if re.search(pattern, text, re.IGNORECASE)):
            score += 0.2
        
        # Boost for numbers (often important in business/legal docs)
        if re.search(r'\d+', text):
            score += 0.1
        
        # Boost for monetary amounts or percentages
        if re.search(r'\$[\d,]+|\d+%|billion|million', text, re.IGNORECASE):
            score += 0.15
        
        # Boost for legal/formal language
        formal_indicators = ['shall', 'hereby', 'whereas', 'pursuant', 'يجب', 'وفقا', 'بموجب']
        if any(indicator in text_lower for indicator in formal_indicators):
            score += 0.1
            
        return min(score, 1.0)

    def _calculate_overlap_info(self, chunk_id: int, all_chunks: List[Dict], profile: DocumentProfile) -> Dict[str, Any]:
        """Calculate overlap information for better retrieval"""
        overlap_info = {
            'has_previous_overlap': chunk_id > 0,
            'has_next_overlap': chunk_id < len(all_chunks) - 1,
            'overlap_ratio': profile.overlap_ratio
        }
        
        # Calculate semantic similarity with adjacent chunks
        current_text = all_chunks[chunk_id]['text']
        current_words = set(word_tokenize(current_text.lower()))
        
        if chunk_id > 0:
            prev_words = set(word_tokenize(all_chunks[chunk_id - 1]['text'].lower()))
            overlap_info['prev_word_overlap'] = len(current_words & prev_words) / len(current_words | prev_words) if current_words | prev_words else 0
        
        if chunk_id < len(all_chunks) - 1:
            next_words = set(word_tokenize(all_chunks[chunk_id + 1]['text'].lower()))
            overlap_info['next_word_overlap'] = len(current_words & next_words) / len(current_words | next_words) if current_words | next_words else 0
        
        return overlap_info

def enhanced_index_document(file_path: str, content: Union[str, bytes], 
                          metadata: Dict[str, Any] = None, 
                          embed_model: str = "bge-m3") -> bool:
    """Enhanced document indexing with intelligent chunking"""
    
    processor = EnhancedDocumentProcessor()
    
    try:
        metadata = metadata or {}
        if "source" not in metadata:
            metadata["source"] = os.path.basename(file_path)

        # Get text content
        if file_path.lower().endswith(".pdf"):
            full_path = os.path.join("data", file_path) if not os.path.isabs(file_path) else file_path
            di = process_with_document_intelligence(full_path)
            text = di["text"]
            metadata.update(di)
        else:
            text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)

        if not text or not text.strip():
            print(f"[ENHANCED] No text content for '{file_path}'")
            return False

        # Detect document type and language
        profile = processor.detect_document_type(text, file_path)
        language = detect_language(text)
        
        print(f"[ENHANCED] Processing '{file_path}' with {profile.name} profile in {language}")

        # Intelligent chunking
        chunks = processor.intelligent_chunking(text, profile, language)
        
        # Enhanced metadata for the entire document
        doc_metadata = {
            'doc_type': profile.name,
            'language': language,
            'total_length': len(text),
            'total_chunks': len(chunks),
            'chunk_size': profile.chunk_size,
            'overlap_ratio': profile.overlap_ratio,
            'processing_method': 'enhanced_intelligent'
        }
        metadata.update(doc_metadata)

        # Process each chunk with enhanced features
        successful_chunks = 0
        for chunk_data in chunks:
            try:
                chunk_text = chunk_data['text']
                
                # Generate embeddings
                vec = generate_embedding(chunk_text, language, model=embed_model, role="passage")
                coll = ensure_collection(client, language, embed_model, vec)
                
                # Extract enhanced features for this chunk
                ents_raw = extract_entities(chunk_text, language)
                ents_by_cat = {}
                for e in ents_raw:
                    ents_by_cat.setdefault(e["category"], []).append(e["text"])
                
                phrases = extract_key_phrases(chunk_text, language)
                sentiment = analyze_sentiment(chunk_text, language)

                # Enhanced payload with rich metadata
                payload = {
                    "text": chunk_text,
                    "source": metadata.get("source"),
                    "language": language,
                    "doc_type": profile.name,
                    "chunk_id": chunk_data['chunk_id'],
                    "total_chunks": chunk_data['total_chunks'],
                    "semantic_type": chunk_data['semantic_type'],
                    "importance_score": chunk_data['importance_score'],
                    "word_count": chunk_data['word_count'],
                    "has_structure": chunk_data['has_structure'],
                    "structural_markers": chunk_data.get('structural_markers', []),
                    "overlap_info": chunk_data['overlap_info'],
                    "entities": ents_by_cat,
                    "key_phrases": phrases,
                    "sentiment": sentiment,
                    "metadata": metadata,
                    "enhanced_processing": True
                }
                
                # Store in Qdrant with enhanced metadata
                client.upsert(
                    collection_name=coll,
                    points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)]
                )
                successful_chunks += 1
                
            except Exception as chunk_error:
                print(f"[ENHANCED] Error processing chunk {chunk_data['chunk_id']}: {chunk_error}")
                continue
        
        success_rate = successful_chunks / len(chunks) if chunks else 0
        print(f"[ENHANCED] Successfully indexed {successful_chunks}/{len(chunks)} chunks ({success_rate:.1%}) for '{file_path}'")
        return successful_chunks > 0
        
    except Exception as e:
        print(f"[ENHANCED] Index error for '{file_path}': {e}")
        return False

if __name__ == "__main__":
    # Enhanced indexing of all documents
    from indexer import load_documents
    
    print("🚀 Starting Enhanced Document Indexing...")
    
    docs = load_documents()
    if not docs:
        print("No documents found in data/ folder")
        exit()
    
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}/{len(docs)}] Processing: {doc['filename']}")
        success = enhanced_index_document(doc["filename"], doc["text"], embed_model="bge-m3")
        
        if success:
            print(f"✅ Enhanced indexing successful for '{doc['filename']}'")
        else:
            print(f"❌ Enhanced indexing failed for '{doc['filename']}'")
    
    print("\n🎉 Enhanced Document Indexing Complete!")
