"""
RAG evaluation service for measuring response quality and performance metrics.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..database.models import Document, DocumentChunk
from .embedding_service import EmbeddingService

# Lazy import to avoid hanging during startup
_SentenceTransformer = None

def _lazy_import_sentence_transformer():
    """Lazy import SentenceTransformer only when needed."""
    global _SentenceTransformer
    
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
            logger.info("SentenceTransformer imported successfully for evaluation")
        except ImportError as e:
            logger.warning(f"SentenceTransformer not available for evaluation: {e}")
            _SentenceTransformer = False
    
    return _SentenceTransformer


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    RELEVANCE_SCORE = "relevance_score"
    COHERENCE_SCORE = "coherence_score"
    COMPLETENESS_SCORE = "completeness_score"
    CITATION_ACCURACY = "citation_accuracy"
    RESPONSE_TIME = "response_time"
    TOKEN_USAGE = "token_usage"
    RETRIEVAL_PRECISION = "retrieval_precision"
    RETRIEVAL_RECALL = "retrieval_recall"
    ANSWER_QUALITY = "answer_quality"
    FACTUAL_ACCURACY = "factual_accuracy"


@dataclass
class RetrievalContext:
    """Context information for retrieved documents."""
    document_chunks: List[DocumentChunk]
    similarity_scores: List[float]
    retrieval_query: str
    total_retrieved: int
    retrieval_time: float


@dataclass
class ResponseEvaluation:
    """Evaluation results for a RAG response."""
    evaluation_id: str
    timestamp: datetime
    query: str
    response: str
    retrieval_context: RetrievalContext
    
    # Core metrics
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    citation_accuracy: float = 0.0
    answer_quality: float = 0.0
    factual_accuracy: float = 0.0
    
    # Performance metrics
    response_time: float = 0.0
    token_usage: int = 0
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    
    # Detailed analysis
    cited_sources: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    irrelevant_content: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Overall score
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary."""
        return {
            'evaluation_id': self.evaluation_id,
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'response': self.response,
            'metrics': {
                'relevance_score': self.relevance_score,
                'coherence_score': self.coherence_score,
                'completeness_score': self.completeness_score,
                'citation_accuracy': self.citation_accuracy,
                'answer_quality': self.answer_quality,
                'factual_accuracy': self.factual_accuracy,
                'response_time': self.response_time,
                'token_usage': self.token_usage,
                'retrieval_precision': self.retrieval_precision,
                'retrieval_recall': self.retrieval_recall,
                'overall_score': self.overall_score,
                'confidence_score': self.confidence_score
            },
            'analysis': {
                'cited_sources': self.cited_sources,
                'missing_information': self.missing_information,
                'irrelevant_content': self.irrelevant_content
            },
            'retrieval_stats': {
                'total_retrieved': self.retrieval_context.total_retrieved,
                'retrieval_time': self.retrieval_context.retrieval_time,
                'avg_similarity': np.mean(self.retrieval_context.similarity_scores) if self.retrieval_context.similarity_scores else 0.0
            }
        }


class RAGEvaluationService:
    """Service for evaluating RAG system performance and response quality."""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.logger = logger.bind(component="RAGEvaluationService")
        
        # Evaluation history
        self.evaluation_history: List[ResponseEvaluation] = []
        
        # Configuration
        self.config = {
            'relevance_threshold': 0.7,
            'coherence_weight': 0.2,
            'completeness_weight': 0.2,
            'citation_weight': 0.15,
            'answer_quality_weight': 0.25,
            'factual_accuracy_weight': 0.2,
            'min_citation_score': 0.6,
            'max_response_time': 5.0,  # seconds
            'similarity_threshold': 0.75
        }
        
        # Load evaluation model (lightweight for real-time evaluation)
        
    async def _ensure_model_loaded(self) -> None:
        """Ensure the evaluation model is loaded."""
        if not self._model_loaded:
            SentenceTransformer = _lazy_import_sentence_transformer()
            if SentenceTransformer and SentenceTransformer is not False:
                try:
                    
                    self.logger.info("Evaluation model loaded successfully")
                except Exception as e:
                    self.logger.warning(f"Could not load evaluation model: {e}")
                    self.evaluation_model = None
            else:
                self.evaluation_model = None
            self._model_loaded = True
        try:
            
            self.logger.info("Evaluation model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load evaluation model: {e}")
            self.evaluation_model = None
            
    async def evaluate_response(self, 
                               query: str,
                               response: str,
                               retrieval_context: RetrievalContext,
                               response_time: float,
                               token_usage: int = 0) -> ResponseEvaluation:
        """Evaluate a RAG response comprehensively."""
        evaluation_id = f"eval_{int(time.time() * 1000)}"
        
        evaluation = ResponseEvaluation(
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow(),
            query=query,
            response=response,
            retrieval_context=retrieval_context,
            response_time=response_time,
            token_usage=token_usage
        )
        
        try:
            # Evaluate different aspects
            evaluation.relevance_score = await self._evaluate_relevance(query, response, retrieval_context)
            evaluation.coherence_score = await self._evaluate_coherence(response)
            evaluation.completeness_score = await self._evaluate_completeness(query, response, retrieval_context)
            evaluation.citation_accuracy = await self._evaluate_citation_accuracy(response, retrieval_context)
            evaluation.answer_quality = await self._evaluate_answer_quality(query, response)
            evaluation.factual_accuracy = await self._evaluate_factual_accuracy(response, retrieval_context)
            
            # Performance metrics
            evaluation.retrieval_precision = self._calculate_retrieval_precision(retrieval_context)
            evaluation.retrieval_recall = self._calculate_retrieval_recall(retrieval_context)
            
            # Detailed analysis
            evaluation.cited_sources = self._extract_cited_sources(response)
            evaluation.missing_information = self._identify_missing_information(query, response, retrieval_context)
            evaluation.irrelevant_content = self._identify_irrelevant_content(response, retrieval_context)
            
            # Calculate overall score
            evaluation.overall_score = self._calculate_overall_score(evaluation)
            evaluation.confidence_score = self._calculate_confidence_score(evaluation)
            
            # Store evaluation
            self.evaluation_history.append(evaluation)
            
            self.logger.info(f"Evaluation completed: {evaluation_id}, Overall Score: {evaluation.overall_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            evaluation.overall_score = 0.0
            
        return evaluation
        
    async def _evaluate_relevance(self, query: str, response: str, context: RetrievalContext) -> float:
        """Evaluate how relevant the response is to the query."""
        if not self.evaluation_model:
            return 0.7  # Default score
            
        try:
            # Calculate semantic similarity between query and response
            query_embedding = self.evaluation_model.encode([query])
            response_embedding = self.evaluation_model.encode([response])
            
            similarity = np.dot(query_embedding[0], response_embedding[0]) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(response_embedding[0])
            )
            
            # Consider retrieved context relevance
            context_scores = context.similarity_scores if context.similarity_scores else [0.5]
            avg_context_relevance = np.mean(context_scores)
            
            # Combine similarity scores
            relevance_score = 0.6 * float(similarity) + 0.4 * avg_context_relevance
            
            return min(max(relevance_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error evaluating relevance: {e}")
            return 0.5
            
    async def _evaluate_coherence(self, response: str) -> float:
        """Evaluate the coherence and readability of the response."""
        try:
            # Basic coherence checks
            sentences = re.split(r'[.!?]+', response.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) == 0:
                return 0.0
                
            # Check for proper sentence structure
            complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
            sentence_completeness = complete_sentences / len(sentences)
            
            # Check for logical flow (simple heuristics)
            transition_words = ['however', 'therefore', 'furthermore', 'additionally', 
                             'moreover', 'consequently', 'thus', 'hence', 'nevertheless']
            transition_score = min(sum(1 for word in transition_words if word in response.lower()) / 5, 1.0)
            
            # Check for repetition
            words = response.lower().split()
            unique_words = len(set(words))
            repetition_penalty = 1.0 if len(words) == 0 else unique_words / len(words)
            
            # Combine coherence factors
            coherence_score = (
                0.4 * sentence_completeness +
                0.3 * transition_score +
                0.3 * repetition_penalty
            )
            
            return min(max(coherence_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error evaluating coherence: {e}")
            return 0.5
            
    async def _evaluate_completeness(self, query: str, response: str, context: RetrievalContext) -> float:
        """Evaluate if the response completely addresses the query."""
        try:
            # Extract key concepts from query
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            query_words = {w for w in query_words if len(w) > 2}  # Filter short words
            
            # Check coverage in response
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            
            if not query_words:
                return 1.0
                
            coverage = len(query_words.intersection(response_words)) / len(query_words)
            
            # Consider available information in context
            context_text = ' '.join([chunk.content for chunk in context.document_chunks])
            context_words = set(re.findall(r'\b\w+\b', context_text.lower()))
            
            available_coverage = len(query_words.intersection(context_words)) / len(query_words)
            
            # Adjust completeness based on available information
            if available_coverage > 0:
                completeness_score = coverage / available_coverage
            else:
                completeness_score = coverage
                
            return min(max(completeness_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error evaluating completeness: {e}")
            return 0.5
            
    async def _evaluate_citation_accuracy(self, response: str, context: RetrievalContext) -> float:
        """Evaluate accuracy of citations and source references."""
        try:
            # Find citation patterns in response
            citation_patterns = [
                r'\[(\d+)\]',  # [1], [2], etc.
                r'\(([^)]+)\)',  # (Source name)
                r'according to ([^,\.]+)',  # according to X
                r'as stated in ([^,\.]+)',  # as stated in X
            ]
            
            citations_found = []
            for pattern in citation_patterns:
                citations_found.extend(re.findall(pattern, response, re.IGNORECASE))
                
            if not citations_found:
                # No citations found - check if sources were available
                if context.document_chunks:
                    return 0.3  # Penalty for not citing available sources
                else:
                    return 1.0  # No sources to cite
                    
            # Check if citations correspond to actual sources
            valid_citations = 0
            for citation in citations_found:
                # Simple validation - check if citation refers to available documents
                citation_lower = str(citation).lower()
                for chunk in context.document_chunks:
                    if (citation_lower in chunk.document.filename.lower() or 
                        citation_lower in chunk.document.title.lower()):
                        valid_citations += 1
                        break
                        
            citation_accuracy = valid_citations / len(citations_found) if citations_found else 0.0
            
            return min(max(citation_accuracy, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error evaluating citations: {e}")
            return 0.5
            
    async def _evaluate_answer_quality(self, query: str, response: str) -> float:
        """Evaluate overall answer quality using multiple factors."""
        try:
            # Length appropriateness
            response_length = len(response.split())
            if response_length < 10:
                length_score = response_length / 10
            elif response_length > 500:
                length_score = max(0.5, 1.0 - (response_length - 500) / 1000)
            else:
                length_score = 1.0
                
            # Structure quality
            has_intro = any(word in response[:100].lower() for word in ['yes', 'no', 'based on', 'according to'])
            has_conclusion = any(word in response[-100:].lower() for word in ['therefore', 'thus', 'in conclusion', 'overall'])
            structure_score = (0.5 * has_intro + 0.5 * has_conclusion)
            
            # Informativeness (unique meaningful words)
            words = re.findall(r'\b\w{4,}\b', response.lower())  # Words with 4+ characters
            unique_meaningful_words = len(set(words))
            informativeness_score = min(unique_meaningful_words / 20, 1.0)
            
            # Combine quality factors
            quality_score = (
                0.3 * length_score +
                0.3 * structure_score +
                0.4 * informativeness_score
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error evaluating answer quality: {e}")
            return 0.5
            
    async def _evaluate_factual_accuracy(self, response: str, context: RetrievalContext) -> float:
        """Evaluate factual accuracy against source documents."""
        try:
            if not context.document_chunks:
                return 0.5  # Cannot verify without sources
                
            # Extract factual statements from response (simplified)
            response_sentences = re.split(r'[.!?]+', response.strip())
            response_sentences = [s.strip() for s in response_sentences if s.strip()]
            
            # Check against source content
            source_text = ' '.join([chunk.content for chunk in context.document_chunks])
            
            if not self.evaluation_model:
                # Simple text matching fallback
                matching_score = 0.0
                for sentence in response_sentences:
                    words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
                    source_words = set(re.findall(r'\b\w{4,}\b', source_text.lower()))
                    if words:
                        overlap = len(words.intersection(source_words)) / len(words)
                        matching_score += overlap
                        
                return min(matching_score / len(response_sentences) if response_sentences else 0.0, 1.0)
            else:
                # Semantic similarity based accuracy
                if not response_sentences:
                    return 0.0
                    
                response_embeddings = self.evaluation_model.encode(response_sentences)
                source_embedding = self.evaluation_model.encode([source_text])
                
                similarities = [
                    np.dot(resp_emb, source_embedding[0]) / (
                        np.linalg.norm(resp_emb) * np.linalg.norm(source_embedding[0])
                    ) for resp_emb in response_embeddings
                ]
                
                avg_similarity = np.mean(similarities)
                return min(max(float(avg_similarity), 0.0), 1.0)
                
        except Exception as e:
            self.logger.error(f"Error evaluating factual accuracy: {e}")
            return 0.5
            
    def _calculate_retrieval_precision(self, context: RetrievalContext) -> float:
        """Calculate precision of retrieved documents."""
        if not context.similarity_scores:
            return 0.0
            
        # Consider documents above similarity threshold as relevant
        relevant_docs = sum(1 for score in context.similarity_scores 
                          if score >= self.config['similarity_threshold'])
        
        return relevant_docs / len(context.similarity_scores) if context.similarity_scores else 0.0
        
    def _calculate_retrieval_recall(self, context: RetrievalContext) -> float:
        """Calculate recall of retrieved documents (simplified)."""
        # This is simplified - in practice would need ground truth relevant documents
        # For now, estimate based on retrieval quality
        if not context.similarity_scores:
            return 0.0
            
        avg_similarity = np.mean(context.similarity_scores)
        
        # Estimate recall based on average similarity and number retrieved
        estimated_recall = min(avg_similarity * (len(context.similarity_scores) / 10), 1.0)
        
        return max(estimated_recall, 0.0)
        
    def _extract_cited_sources(self, response: str) -> List[str]:
        """Extract cited sources from response text."""
        citations = []
        
        # Pattern for citations
        patterns = [
            r'\[([^\]]+)\]',
            r'\(([^)]+)\)',
            r'according to ([^,\.]+)',
            r'as mentioned in ([^,\.]+)',
            r'from ([^,\.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            citations.extend(matches)
            
        return list(set(citations))  # Remove duplicates
        
    def _identify_missing_information(self, query: str, response: str, context: RetrievalContext) -> List[str]:
        """Identify information gaps in the response."""
        missing = []
        
        # Extract key concepts from query
        query_concepts = re.findall(r'\b\w{4,}\b', query.lower())
        response_concepts = re.findall(r'\b\w{4,}\b', response.lower())
        
        # Find missing concepts
        for concept in query_concepts:
            if concept not in response_concepts:
                # Check if it's available in context
                context_text = ' '.join([chunk.content for chunk in context.document_chunks])
                if concept in context_text.lower():
                    missing.append(f"Missing information about: {concept}")
                    
        return missing
        
    def _identify_irrelevant_content(self, response: str, context: RetrievalContext) -> List[str]:
        """Identify potentially irrelevant content in response."""
        irrelevant = []
        
        # Simple check for content not supported by sources
        response_sentences = re.split(r'[.!?]+', response.strip())
        context_text = ' '.join([chunk.content for chunk in context.document_chunks]).lower()
        
        for sentence in response_sentences:
            if len(sentence.strip()) < 10:
                continue
                
            # Check if sentence has support in context
            sentence_words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
            context_words = set(re.findall(r'\b\w{4,}\b', context_text))
            
            if sentence_words and len(sentence_words.intersection(context_words)) / len(sentence_words) < 0.3:
                irrelevant.append(sentence.strip())
                
        return irrelevant
        
    def _calculate_overall_score(self, evaluation: ResponseEvaluation) -> float:
        """Calculate weighted overall score."""
        config = self.config
        
        score = (
            config['coherence_weight'] * evaluation.coherence_score +
            config['completeness_weight'] * evaluation.completeness_score +
            config['citation_weight'] * evaluation.citation_accuracy +
            config['answer_quality_weight'] * evaluation.answer_quality +
            config['factual_accuracy_weight'] * evaluation.factual_accuracy
        )
        
        # Apply penalties
        if evaluation.response_time > config['max_response_time']:
            time_penalty = min(0.2, (evaluation.response_time - config['max_response_time']) / 10)
            score *= (1.0 - time_penalty)
            
        return min(max(score, 0.0), 1.0)
        
    def _calculate_confidence_score(self, evaluation: ResponseEvaluation) -> float:
        """Calculate confidence in the evaluation."""
        # Based on availability of context and model certainty
        factors = [
            evaluation.retrieval_context.total_retrieved > 0,  # Has context
            len(evaluation.retrieval_context.similarity_scores) > 2,  # Multiple sources
            evaluation.citation_accuracy > 0.5,  # Good citations
            evaluation.response_time < self.config['max_response_time'],  # Reasonable time
        ]
        
        confidence = sum(factors) / len(factors)
        
        # Adjust based on score consistency
        scores = [
            evaluation.relevance_score,
            evaluation.coherence_score,
            evaluation.completeness_score,
            evaluation.answer_quality,
            evaluation.factual_accuracy
        ]
        
        score_variance = np.var(scores)
        consistency_factor = max(0.5, 1.0 - score_variance)
        
        return confidence * consistency_factor
        
    def get_evaluation_summary(self, num_recent: int = 10) -> Dict[str, Any]:
        """Get summary of recent evaluations."""
        recent_evals = self.evaluation_history[-num_recent:] if self.evaluation_history else []
        
        if not recent_evals:
            return {
                'total_evaluations': 0,
                'average_scores': {},
                'performance_trends': {}
            }
            
        # Calculate averages
        avg_scores = {
            'overall_score': np.mean([e.overall_score for e in recent_evals]),
            'relevance_score': np.mean([e.relevance_score for e in recent_evals]),
            'coherence_score': np.mean([e.coherence_score for e in recent_evals]),
            'completeness_score': np.mean([e.completeness_score for e in recent_evals]),
            'citation_accuracy': np.mean([e.citation_accuracy for e in recent_evals]),
            'answer_quality': np.mean([e.answer_quality for e in recent_evals]),
            'factual_accuracy': np.mean([e.factual_accuracy for e in recent_evals]),
            'confidence_score': np.mean([e.confidence_score for e in recent_evals])
        }
        
        # Performance metrics
        performance = {
            'avg_response_time': np.mean([e.response_time for e in recent_evals]),
            'avg_token_usage': np.mean([e.token_usage for e in recent_evals]),
            'avg_retrieval_precision': np.mean([e.retrieval_precision for e in recent_evals]),
            'avg_retrieval_recall': np.mean([e.retrieval_recall for e in recent_evals])
        }
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'recent_evaluations': len(recent_evals),
            'average_scores': avg_scores,
            'performance_metrics': performance,
            'evaluation_trend': self._calculate_trend(recent_evals)
        }
        
    def _calculate_trend(self, evaluations: List[ResponseEvaluation]) -> str:
        """Calculate performance trend."""
        if len(evaluations) < 3:
            return "insufficient_data"
            
        # Compare first half vs second half
        mid_point = len(evaluations) // 2
        first_half_avg = np.mean([e.overall_score for e in evaluations[:mid_point]])
        second_half_avg = np.mean([e.overall_score for e in evaluations[mid_point:]])
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
            
    def export_evaluations(self, filepath: str) -> None:
        """Export evaluation history to JSON file."""
        try:
            data = {
                'evaluations': [eval.to_dict() for eval in self.evaluation_history],
                'summary': self.get_evaluation_summary(),
                'export_timestamp': datetime.utcnow().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            self.logger.info(f"Exported {len(self.evaluation_history)} evaluations to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting evaluations: {e}")