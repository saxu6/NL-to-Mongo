"""
Human-in-the-Loop Feedback System for MongoDB Query Translator.
Captures user feedback, analyzes patterns, and drives model improvement.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import redis
from redis import Redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import get_logger

logger = get_logger(__name__)

class FeedbackType(Enum):
    """Types of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"

class FeedbackCategory(Enum):
    """Categories of feedback."""
    QUERY_ACCURACY = "query_accuracy"
    RESPONSE_TIME = "response_time"
    RESULT_RELEVANCE = "result_relevance"
    QUERY_UNDERSTANDING = "query_understanding"
    SYSTEM_USABILITY = "system_usability"

class FeedbackPriority(Enum):
    """Priority levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Feedback:
    """User feedback record."""
    feedback_id: str
    user_id: str
    session_id: str
    query_id: str
    feedback_type: FeedbackType
    feedback_category: FeedbackCategory
    rating: Optional[int] = None  # 1-5 scale
    comment: Optional[str] = None
    correction: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    processed: bool = False
    processed_at: Optional[datetime] = None

@dataclass
class FeedbackAnalysis:
    """Analysis results for feedback."""
    analysis_id: str
    feedback_id: str
    sentiment_score: float
    topic_cluster: int
    key_phrases: List[str]
    improvement_suggestions: List[str]
    confidence_score: float
    analysis_timestamp: datetime

@dataclass
class FeedbackInsights:
    """Insights derived from feedback analysis."""
    insight_id: str
    insight_type: str
    description: str
    affected_queries: List[str]
    recommended_actions: List[str]
    priority: FeedbackPriority
    confidence: float
    created_at: datetime

class FeedbackSystem:
    """
    Comprehensive feedback system for continuous model improvement.
    
    Features:
    - Multi-modal feedback collection
    - Real-time feedback analysis
    - Pattern recognition and clustering
    - Automated insight generation
    - Integration with model retraining
    - Feedback-driven prompt optimization
    """
    
    def __init__(self, 
                 mongodb_uri: str,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 database_name: str = "feedback_system"):
        """
        Initialize the feedback system.
        
        Args:
            mongodb_uri: MongoDB connection string
            redis_host: Redis host for caching
            redis_port: Redis port
            database_name: Database name for feedback data
        """
        self.mongodb_uri = mongodb_uri
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.database_name = database_name
        
        # Initialize connections
        self._init_mongodb()
        self._init_redis()
        
        # Initialize analysis components
        self._init_analysis_components()
        
        logger.info("Feedback system initialized")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client[self.database_name]
            
            # Collections
            self.feedback_collection = self.db.feedback
            self.analysis_collection = self.db.feedback_analysis
            self.insights_collection = self.db.feedback_insights
            self.patterns_collection = self.db.feedback_patterns
            self.improvements_collection = self.db.improvements
            
            # Create indexes
            self._create_indexes()
            
            logger.info("MongoDB connection established for feedback system")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for feedback system")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_analysis_components(self):
        """Initialize components for feedback analysis."""
        try:
            # Initialize TF-IDF vectorizer for text analysis
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize clustering for topic analysis
            self.topic_clusterer = KMeans(n_clusters=10, random_state=42)
            
            logger.info("Analysis components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize analysis components: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for feedback system."""
        try:
            # Feedback collection indexes
            self.feedback_collection.create_index("feedback_id", unique=True)
            self.feedback_collection.create_index("user_id")
            self.feedback_collection.create_index("query_id")
            self.feedback_collection.create_index("feedback_type")
            self.feedback_collection.create_index("timestamp")
            self.feedback_collection.create_index("processed")
            
            # Analysis collection indexes
            self.analysis_collection.create_index("analysis_id", unique=True)
            self.analysis_collection.create_index("feedback_id")
            self.analysis_collection.create_index("topic_cluster")
            self.analysis_collection.create_index("analysis_timestamp")
            
            # Insights collection indexes
            self.insights_collection.create_index("insight_id", unique=True)
            self.insights_collection.create_index("insight_type")
            self.insights_collection.create_index("priority")
            self.insights_collection.create_index("created_at")
            
            logger.info("Database indexes created for feedback system")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def submit_feedback(self, 
                       user_id: str,
                       session_id: str,
                       query_id: str,
                       feedback_type: FeedbackType,
                       feedback_category: FeedbackCategory,
                       rating: Optional[int] = None,
                       comment: Optional[str] = None,
                       correction: Optional[Dict[str, Any]] = None,
                       suggestion: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit user feedback.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query_id: Query identifier
            feedback_type: Type of feedback
            feedback_category: Category of feedback
            rating: Rating score (1-5)
            comment: Text comment
            correction: Correction data
            suggestion: Improvement suggestion
            metadata: Additional metadata
            
        Returns:
            Feedback ID
        """
        try:
            # Generate feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Determine priority based on feedback type and content
            priority = self._determine_priority(feedback_type, comment, correction)
            
            # Create feedback record
            feedback = Feedback(
                feedback_id=feedback_id,
                user_id=user_id,
                session_id=session_id,
                query_id=query_id,
                feedback_type=feedback_type,
                feedback_category=feedback_category,
                rating=rating,
                comment=comment,
                correction=correction,
                suggestion=suggestion,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
                priority=priority,
                processed=False
            )
            
            # Store feedback
            self.feedback_collection.insert_one(asdict(feedback))
            
            # Cache recent feedback for quick access
            if self.redis_client:
                cache_key = f"feedback:{feedback_id}"
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    json.dumps(asdict(feedback), default=str)
                )
            
            # Trigger analysis if high priority
            if priority in [FeedbackPriority.HIGH, FeedbackPriority.CRITICAL]:
                self._analyze_feedback_async(feedback_id)
            
            logger.info(f"Feedback submitted: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise
    
    def _determine_priority(self, 
                           feedback_type: FeedbackType,
                           comment: Optional[str],
                           correction: Optional[Dict[str, Any]]) -> FeedbackPriority:
        """Determine feedback priority based on content."""
        try:
            # High priority indicators
            high_priority_keywords = [
                "error", "broken", "wrong", "incorrect", "failed", 
                "not working", "bug", "issue", "problem"
            ]
            
            critical_priority_keywords = [
                "security", "data loss", "privacy", "breach", 
                "unauthorized", "hack", "exploit"
            ]
            
            # Check comment for priority keywords
            if comment:
                comment_lower = comment.lower()
                
                if any(keyword in comment_lower for keyword in critical_priority_keywords):
                    return FeedbackPriority.CRITICAL
                
                if any(keyword in comment_lower for keyword in high_priority_keywords):
                    return FeedbackPriority.HIGH
            
            # Check feedback type
            if feedback_type == FeedbackType.CORRECTION:
                return FeedbackPriority.HIGH
            
            if feedback_type == FeedbackType.THUMBS_DOWN:
                return FeedbackPriority.MEDIUM
            
            return FeedbackPriority.LOW
            
        except Exception as e:
            logger.error(f"Failed to determine priority: {e}")
            return FeedbackPriority.MEDIUM
    
    def _analyze_feedback_async(self, feedback_id: str):
        """Analyze feedback asynchronously."""
        try:
            # This would typically be done in a background task queue
            # For now, we'll do it synchronously
            self.analyze_feedback(feedback_id)
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback asynchronously: {e}")
    
    def analyze_feedback(self, feedback_id: str) -> Optional[FeedbackAnalysis]:
        """
        Analyze a specific feedback item.
        
        Args:
            feedback_id: Feedback ID to analyze
            
        Returns:
            Feedback analysis results
        """
        try:
            # Get feedback
            feedback_doc = self.feedback_collection.find_one({"feedback_id": feedback_id})
            if not feedback_doc:
                logger.error(f"Feedback {feedback_id} not found")
                return None
            
            feedback = Feedback(**feedback_doc)
            
            # Perform analysis
            analysis = self._perform_feedback_analysis(feedback)
            
            # Store analysis
            self.analysis_collection.insert_one(asdict(analysis))
            
            # Mark feedback as processed
            self.feedback_collection.update_one(
                {"feedback_id": feedback_id},
                {
                    "$set": {
                        "processed": True,
                        "processed_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            logger.info(f"Analyzed feedback: {feedback_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            return None
    
    def _perform_feedback_analysis(self, feedback: Feedback) -> FeedbackAnalysis:
        """Perform detailed analysis of feedback."""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Sentiment analysis (simplified)
            sentiment_score = self._analyze_sentiment(feedback.comment or "")
            
            # Topic clustering
            topic_cluster = self._cluster_feedback_topic(feedback)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(feedback.comment or "")
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(feedback)
            
            # Calculate confidence score
            confidence_score = self._calculate_analysis_confidence(feedback)
            
            analysis = FeedbackAnalysis(
                analysis_id=analysis_id,
                feedback_id=feedback.feedback_id,
                sentiment_score=sentiment_score,
                topic_cluster=topic_cluster,
                key_phrases=key_phrases,
                improvement_suggestions=improvement_suggestions,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now(timezone.utc)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to perform feedback analysis: {e}")
            raise
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (simplified implementation)."""
        try:
            if not text:
                return 0.0
            
            # Simple sentiment analysis based on keywords
            positive_words = ["good", "great", "excellent", "perfect", "amazing", "love", "like"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "wrong", "error"]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / total_words
            return max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
            
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return 0.0
    
    def _cluster_feedback_topic(self, feedback: Feedback) -> int:
        """Cluster feedback into topics."""
        try:
            # Combine feedback text for clustering
            text_parts = []
            
            if feedback.comment:
                text_parts.append(feedback.comment)
            
            if feedback.suggestion:
                text_parts.append(feedback.suggestion)
            
            if feedback.correction:
                text_parts.append(str(feedback.correction))
            
            combined_text = " ".join(text_parts)
            
            if not combined_text:
                return 0  # Default cluster
            
            # This is a simplified clustering - in practice, you'd use more sophisticated methods
            # For now, return a hash-based cluster
            cluster_id = hash(combined_text) % 10
            return abs(cluster_id)
            
        except Exception as e:
            logger.error(f"Failed to cluster feedback topic: {e}")
            return 0
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            if not text:
                return []
            
            # Simple key phrase extraction (in practice, use more sophisticated NLP)
            words = text.lower().split()
            
            # Filter out common words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top phrases
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:5] if freq > 1]
            
        except Exception as e:
            logger.error(f"Failed to extract key phrases: {e}")
            return []
    
    def _generate_improvement_suggestions(self, feedback: Feedback) -> List[str]:
        """Generate improvement suggestions based on feedback."""
        try:
            suggestions = []
            
            # Generate suggestions based on feedback type
            if feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                suggestions.append("Review query understanding for this type of request")
                suggestions.append("Check if additional training data is needed")
            
            if feedback.feedback_type == FeedbackType.CORRECTION:
                suggestions.append("Update model training with corrected examples")
                suggestions.append("Review prompt engineering for this query pattern")
            
            if feedback.feedback_category == FeedbackCategory.QUERY_ACCURACY:
                suggestions.append("Improve query parsing and understanding")
                suggestions.append("Add more examples to training data")
            
            if feedback.feedback_category == FeedbackCategory.RESPONSE_TIME:
                suggestions.append("Optimize model inference performance")
                suggestions.append("Review caching strategies")
            
            # Add suggestions based on comment content
            if feedback.comment:
                comment_lower = feedback.comment.lower()
                
                if "slow" in comment_lower or "timeout" in comment_lower:
                    suggestions.append("Investigate performance bottlenecks")
                
                if "wrong" in comment_lower or "incorrect" in comment_lower:
                    suggestions.append("Review model accuracy for this query type")
                
                if "confusing" in comment_lower or "unclear" in comment_lower:
                    suggestions.append("Improve response clarity and formatting")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate improvement suggestions: {e}")
            return []
    
    def _calculate_analysis_confidence(self, feedback: Feedback) -> float:
        """Calculate confidence score for analysis."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on feedback completeness
            if feedback.comment:
                confidence += 0.2
            
            if feedback.rating:
                confidence += 0.1
            
            if feedback.correction:
                confidence += 0.2
            
            # Decrease confidence for very short comments
            if feedback.comment and len(feedback.comment) < 10:
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate analysis confidence: {e}")
            return 0.5
    
    def generate_insights(self, hours: int = 24) -> List[FeedbackInsights]:
        """
        Generate insights from recent feedback.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of generated insights
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Get recent feedback
            recent_feedback = list(self.feedback_collection.find({
                "timestamp": {"$gte": cutoff_time},
                "processed": True
            }))
            
            if not recent_feedback:
                return []
            
            # Get analysis for recent feedback
            feedback_ids = [f["feedback_id"] for f in recent_feedback]
            analyses = list(self.analysis_collection.find({
                "feedback_id": {"$in": feedback_ids}
            }))
            
            # Generate insights
            insights = []
            
            # Insight 1: Common negative feedback patterns
            negative_feedback = [f for f in recent_feedback if f.get("feedback_type") == "thumbs_down"]
            if len(negative_feedback) > 5:
                insight = self._create_pattern_insight(
                    "negative_feedback_pattern",
                    "High volume of negative feedback detected",
                    [f["query_id"] for f in negative_feedback],
                    ["Review common failure patterns", "Improve model training"],
                    FeedbackPriority.HIGH
                )
                insights.append(insight)
            
            # Insight 2: Performance issues
            performance_feedback = [f for f in recent_feedback 
                                  if f.get("feedback_category") == "response_time"]
            if len(performance_feedback) > 3:
                insight = self._create_pattern_insight(
                    "performance_issues",
                    "Multiple performance-related complaints",
                    [f["query_id"] for f in performance_feedback],
                    ["Optimize model inference", "Review system resources"],
                    FeedbackPriority.MEDIUM
                )
                insights.append(insight)
            
            # Insight 3: Topic clustering insights
            topic_insights = self._generate_topic_insights(analyses)
            insights.extend(topic_insights)
            
            # Store insights
            for insight in insights:
                self.insights_collection.insert_one(asdict(insight))
            
            logger.info(f"Generated {len(insights)} insights from recent feedback")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    def _create_pattern_insight(self, 
                               insight_type: str,
                               description: str,
                               affected_queries: List[str],
                               recommended_actions: List[str],
                               priority: FeedbackPriority) -> FeedbackInsights:
        """Create a pattern-based insight."""
        return FeedbackInsights(
            insight_id=str(uuid.uuid4()),
            insight_type=insight_type,
            description=description,
            affected_queries=affected_queries,
            recommended_actions=recommended_actions,
            priority=priority,
            confidence=0.8,  # Base confidence for pattern insights
            created_at=datetime.now(timezone.utc)
        )
    
    def _generate_topic_insights(self, analyses: List[Dict[str, Any]]) -> List[FeedbackInsights]:
        """Generate insights based on topic clustering."""
        try:
            insights = []
            
            # Group analyses by topic cluster
            cluster_groups = {}
            for analysis in analyses:
                cluster = analysis.get("topic_cluster", 0)
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(analysis)
            
            # Generate insights for clusters with multiple items
            for cluster, cluster_analyses in cluster_groups.items():
                if len(cluster_analyses) >= 3:  # Minimum threshold for insights
                    # Get common key phrases
                    all_phrases = []
                    for analysis in cluster_analyses:
                        all_phrases.extend(analysis.get("key_phrases", []))
                    
                    # Find most common phrases
                    phrase_freq = {}
                    for phrase in all_phrases:
                        phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
                    
                    common_phrases = [phrase for phrase, freq in phrase_freq.items() if freq >= 2]
                    
                    if common_phrases:
                        insight = FeedbackInsights(
                            insight_id=str(uuid.uuid4()),
                            insight_type="topic_cluster",
                            description=f"Common feedback topic: {', '.join(common_phrases[:3])}",
                            affected_queries=[a["feedback_id"] for a in cluster_analyses],
                            recommended_actions=[
                                "Review model performance for this topic",
                                "Consider additional training data",
                                "Update prompt engineering"
                            ],
                            priority=FeedbackPriority.MEDIUM,
                            confidence=0.7,
                            created_at=datetime.now(timezone.utc)
                        )
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate topic insights: {e}")
            return []
    
    def get_feedback_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent feedback."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Get feedback statistics
            total_feedback = self.feedback_collection.count_documents({
                "timestamp": {"$gte": cutoff_time}
            })
            
            positive_feedback = self.feedback_collection.count_documents({
                "timestamp": {"$gte": cutoff_time},
                "feedback_type": "thumbs_up"
            })
            
            negative_feedback = self.feedback_collection.count_documents({
                "timestamp": {"$gte": cutoff_time},
                "feedback_type": "thumbs_down"
            })
            
            # Get average rating
            rating_feedback = list(self.feedback_collection.find({
                "timestamp": {"$gte": cutoff_time},
                "rating": {"$exists": True}
            }))
            
            avg_rating = 0.0
            if rating_feedback:
                ratings = [f["rating"] for f in rating_feedback if f.get("rating")]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
            
            # Get feedback by category
            category_stats = {}
            for category in FeedbackCategory:
                count = self.feedback_collection.count_documents({
                    "timestamp": {"$gte": cutoff_time},
                    "feedback_category": category.value
                })
                category_stats[category.value] = count
            
            # Get priority distribution
            priority_stats = {}
            for priority in FeedbackPriority:
                count = self.feedback_collection.count_documents({
                    "timestamp": {"$gte": cutoff_time},
                    "priority": priority.value
                })
                priority_stats[priority.value] = count
            
            summary = {
                "time_range_hours": hours,
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "negative_feedback": negative_feedback,
                "satisfaction_rate": positive_feedback / total_feedback if total_feedback > 0 else 0,
                "average_rating": avg_rating,
                "category_distribution": category_stats,
                "priority_distribution": priority_stats,
                "processed_feedback": self.feedback_collection.count_documents({
                    "timestamp": {"$gte": cutoff_time},
                    "processed": True
                })
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get feedback summary: {e}")
            return {}
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Get actionable improvement recommendations based on feedback."""
        try:
            # Get recent insights
            recent_insights = list(self.insights_collection.find({
                "created_at": {"$gte": datetime.now(timezone.utc) - timedelta(days=7)}
            }).sort("created_at", -1))
            
            # Get unprocessed high-priority feedback
            high_priority_feedback = list(self.feedback_collection.find({
                "priority": {"$in": ["high", "critical"]},
                "processed": False
            }).sort("timestamp", -1).limit(10))
            
            recommendations = []
            
            # Convert insights to recommendations
            for insight in recent_insights:
                recommendation = {
                    "type": "insight_based",
                    "title": insight["description"],
                    "priority": insight["priority"],
                    "confidence": insight["confidence"],
                    "actions": insight["recommended_actions"],
                    "affected_queries": len(insight["affected_queries"]),
                    "created_at": insight["created_at"]
                }
                recommendations.append(recommendation)
            
            # Convert high-priority feedback to recommendations
            for feedback in high_priority_feedback:
                recommendation = {
                    "type": "feedback_based",
                    "title": f"High priority feedback: {feedback['feedback_type']}",
                    "priority": feedback["priority"],
                    "confidence": 0.9,
                    "actions": [
                        "Review and address specific feedback",
                        "Investigate root cause",
                        "Implement corrective measures"
                    ],
                    "affected_queries": 1,
                    "created_at": feedback["timestamp"],
                    "feedback_id": feedback["feedback_id"]
                }
                recommendations.append(recommendation)
            
            # Sort by priority and confidence
            priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda x: (priority_order.get(x["priority"], 0), x["confidence"]),
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get improvement recommendations: {e}")
            return []
    
    def export_feedback_for_training(self, 
                                   days: int = 30,
                                   include_positive: bool = True,
                                   include_negative: bool = True) -> List[Dict[str, Any]]:
        """Export feedback data for model training."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Build query based on requirements
            query = {"timestamp": {"$gte": cutoff_time}}
            
            if not include_positive and not include_negative:
                return []
            
            if not include_positive:
                query["feedback_type"] = {"$ne": "thumbs_up"}
            
            if not include_negative:
                query["feedback_type"] = {"$ne": "thumbs_down"}
            
            # Get feedback with analysis
            feedback_data = list(self.feedback_collection.find(query))
            
            # Enrich with analysis data
            training_data = []
            for feedback in feedback_data:
                analysis = self.analysis_collection.find_one({
                    "feedback_id": feedback["feedback_id"]
                })
                
                training_record = {
                    "feedback_id": feedback["feedback_id"],
                    "user_query": feedback.get("metadata", {}).get("user_query", ""),
                    "generated_query": feedback.get("metadata", {}).get("generated_query", ""),
                    "feedback_type": feedback["feedback_type"],
                    "feedback_category": feedback["feedback_category"],
                    "rating": feedback.get("rating"),
                    "comment": feedback.get("comment"),
                    "correction": feedback.get("correction"),
                    "suggestion": feedback.get("suggestion"),
                    "sentiment_score": analysis.get("sentiment_score") if analysis else 0,
                    "key_phrases": analysis.get("key_phrases", []) if analysis else [],
                    "improvement_suggestions": analysis.get("improvement_suggestions", []) if analysis else [],
                    "timestamp": feedback["timestamp"]
                }
                
                training_data.append(training_record)
            
            logger.info(f"Exported {len(training_data)} feedback records for training")
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to export feedback for training: {e}")
            return []
    
    def cleanup_old_data(self, days_old: int = 90) -> int:
        """Clean up old feedback data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            # Clean up old feedback (keep only processed feedback)
            feedback_result = self.feedback_collection.delete_many({
                "timestamp": {"$lt": cutoff_date},
                "processed": True,
                "priority": {"$in": ["low", "medium"]}
            })
            
            # Clean up old analysis
            analysis_result = self.analysis_collection.delete_many({
                "analysis_timestamp": {"$lt": cutoff_date}
            })
            
            # Clean up old insights
            insights_result = self.insights_collection.delete_many({
                "created_at": {"$lt": cutoff_date}
            })
            
            total_cleaned = (feedback_result.deleted_count + 
                           analysis_result.deleted_count + 
                           insights_result.deleted_count)
            
            logger.info(f"Cleaned up {total_cleaned} old feedback records")
            return total_cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
