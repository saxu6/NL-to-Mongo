"""
A/B Testing Framework for Model Comparison.
Enables controlled experiments to compare model performance in production.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import hashlib

import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import mlflow
from mlflow.tracking import MlflowClient

from utils.logger import get_logger

logger = get_logger(__name__)

class ExperimentStatus(Enum):
    """A/B test experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    """Type of model variant."""
    CONTROL = "control"
    TREATMENT = "treatment"

@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    model_name: str
    control_version: str
    treatment_version: str
    traffic_split: float  # 0.0 to 1.0
    duration_days: int
    success_metrics: List[str]
    minimum_sample_size: int
    statistical_significance: float  # 0.95 for 95%
    created_at: datetime
    created_by: str

@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    experiment_id: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    total_users: int
    control_users: int
    treatment_users: int
    metrics: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, float]
    winner: Optional[str]
    confidence: float
    recommendation: str

class ABTestingFramework:
    """
    A/B Testing framework for model comparison in production.
    
    Features:
    - Traffic splitting and user assignment
    - Statistical significance testing
    - Real-time experiment monitoring
    - Automated experiment management
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 mongodb_uri: str,
                 mlflow_tracking_uri: str = None,
                 database_name: str = "ab_testing"):
        """
        Initialize the A/B testing framework.
        
        Args:
            mongodb_uri: MongoDB connection string
            mlflow_tracking_uri: MLflow tracking server URI
            database_name: Database name for A/B testing data
        """
        self.mongodb_uri = mongodb_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.database_name = database_name
        
        # Initialize connections
        self._init_mongodb()
        self._init_mlflow()
        
        logger.info("A/B Testing framework initialized")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client[self.database_name]
            
            # Collections
            self.experiments_collection = self.db.experiments
            self.assignments_collection = self.db.user_assignments
            self.events_collection = self.db.experiment_events
            self.results_collection = self.db.experiment_results
            
            # Create indexes
            self._create_indexes()
            
            logger.info("MongoDB connection established for A/B testing")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    def _init_mlflow(self):
        """Initialize MLflow connection."""
        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            
            logger.info("MLflow connection established for A/B testing")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_client = None
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        try:
            # Experiments collection
            self.experiments_collection.create_index("experiment_id", unique=True)
            self.experiments_collection.create_index("status")
            self.experiments_collection.create_index("created_at")
            
            # User assignments collection
            self.assignments_collection.create_index([("user_id", 1), ("experiment_id", 1)], unique=True)
            self.assignments_collection.create_index("experiment_id")
            self.assignments_collection.create_index("variant")
            
            # Events collection
            self.events_collection.create_index([("experiment_id", 1), ("user_id", 1)])
            self.events_collection.create_index("timestamp")
            self.events_collection.create_index("event_type")
            
            logger.info("Database indexes created for A/B testing")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def create_experiment(self, 
                         name: str,
                         description: str,
                         model_name: str,
                         control_version: str,
                         treatment_version: str,
                         traffic_split: float = 0.5,
                         duration_days: int = 7,
                         success_metrics: List[str] = None,
                         minimum_sample_size: int = 1000,
                         statistical_significance: float = 0.95,
                         created_by: str = "system") -> str:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            model_name: Name of the model being tested
            control_version: Control model version
            treatment_version: Treatment model version
            traffic_split: Traffic split ratio (0.0 to 1.0)
            duration_days: Experiment duration in days
            success_metrics: List of success metrics to track
            minimum_sample_size: Minimum sample size for statistical power
            statistical_significance: Statistical significance level
            created_by: Creator of the experiment
            
        Returns:
            Experiment ID
        """
        try:
            # Validate inputs
            if not 0.0 <= traffic_split <= 1.0:
                raise ValueError("Traffic split must be between 0.0 and 1.0")
            
            if success_metrics is None:
                success_metrics = ["accuracy", "user_satisfaction", "response_time"]
            
            # Generate experiment ID
            experiment_id = self._generate_experiment_id(name)
            
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                model_name=model_name,
                control_version=control_version,
                treatment_version=treatment_version,
                traffic_split=traffic_split,
                duration_days=duration_days,
                success_metrics=success_metrics,
                minimum_sample_size=minimum_sample_size,
                statistical_significance=statistical_significance,
                created_at=datetime.now(timezone.utc),
                created_by=created_by
            )
            
            # Store experiment
            self.experiments_collection.insert_one({
                "experiment_id": experiment_id,
                "config": asdict(config),
                "status": ExperimentStatus.DRAFT.value,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            })
            
            logger.info(f"Created A/B test experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an A/B test experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Success status
        """
        try:
            # Get experiment
            experiment = self.experiments_collection.find_one({"experiment_id": experiment_id})
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Validate experiment can be started
            if experiment["status"] != ExperimentStatus.DRAFT.value:
                raise ValueError(f"Experiment {experiment_id} is not in draft status")
            
            # Update status
            self.experiments_collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$set": {
                        "status": ExperimentStatus.RUNNING.value,
                        "start_date": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            # Log experiment start
            self._log_experiment_event(experiment_id, "experiment_started", {
                "status": ExperimentStatus.RUNNING.value
            })
            
            logger.info(f"Started A/B test experiment: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            return False
    
    def assign_user_to_variant(self, 
                              experiment_id: str, 
                              user_id: str) -> str:
        """
        Assign a user to a variant (control or treatment).
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            
        Returns:
            Variant assignment (control or treatment)
        """
        try:
            # Check if user is already assigned
            existing_assignment = self.assignments_collection.find_one({
                "experiment_id": experiment_id,
                "user_id": user_id
            })
            
            if existing_assignment:
                return existing_assignment["variant"]
            
            # Get experiment configuration
            experiment = self.experiments_collection.find_one({"experiment_id": experiment_id})
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment["status"] != ExperimentStatus.RUNNING.value:
                return "control"  # Default to control if experiment not running
            
            config = experiment["config"]
            traffic_split = config["traffic_split"]
            
            # Determine variant based on user ID hash for consistency
            user_hash = int(hashlib.md5(f"{user_id}{experiment_id}".encode()).hexdigest(), 16)
            variant = "treatment" if (user_hash % 100) < (traffic_split * 100) else "control"
            
            # Store assignment
            self.assignments_collection.insert_one({
                "experiment_id": experiment_id,
                "user_id": user_id,
                "variant": variant,
                "assigned_at": datetime.now(timezone.utc)
            })
            
            # Log assignment
            self._log_experiment_event(experiment_id, "user_assigned", {
                "user_id": user_id,
                "variant": variant
            })
            
            return variant
            
        except Exception as e:
            logger.error(f"Failed to assign user to variant: {e}")
            return "control"  # Default to control on error
    
    def log_experiment_event(self, 
                           experiment_id: str,
                           user_id: str,
                           event_type: str,
                           event_data: Dict[str, Any] = None) -> bool:
        """
        Log an event for an experiment.
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            event_type: Type of event
            event_data: Additional event data
            
        Returns:
            Success status
        """
        try:
            # Get user's variant assignment
            assignment = self.assignments_collection.find_one({
                "experiment_id": experiment_id,
                "user_id": user_id
            })
            
            if not assignment:
                logger.warning(f"User {user_id} not assigned to experiment {experiment_id}")
                return False
            
            # Log event
            event_doc = {
                "experiment_id": experiment_id,
                "user_id": user_id,
                "variant": assignment["variant"],
                "event_type": event_type,
                "event_data": event_data or {},
                "timestamp": datetime.now(timezone.utc)
            }
            
            self.events_collection.insert_one(event_doc)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log experiment event: {e}")
            return False
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        Get results for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment results or None
        """
        try:
            # Get experiment
            experiment = self.experiments_collection.find_one({"experiment_id": experiment_id})
            if not experiment:
                return None
            
            # Get user assignments
            assignments = list(self.assignments_collection.find({"experiment_id": experiment_id}))
            
            # Get events
            events = list(self.events_collection.find({"experiment_id": experiment_id}))
            
            # Calculate metrics
            metrics = self._calculate_experiment_metrics(assignments, events, experiment["config"])
            
            # Calculate statistical significance
            significance = self._calculate_statistical_significance(metrics)
            
            # Determine winner
            winner, confidence = self._determine_winner(metrics, significance)
            
            # Create result
            result = ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus(experiment["status"]),
                start_date=experiment.get("start_date"),
                end_date=experiment.get("end_date"),
                total_users=len(assignments),
                control_users=len([a for a in assignments if a["variant"] == "control"]),
                treatment_users=len([a for a in assignments if a["variant"] == "treatment"]),
                metrics=metrics,
                statistical_significance=significance,
                winner=winner,
                confidence=confidence,
                recommendation=self._generate_recommendation(metrics, significance, winner)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return None
    
    def _calculate_experiment_metrics(self, 
                                    assignments: List[Dict],
                                    events: List[Dict],
                                    config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each variant."""
        try:
            metrics = {}
            success_metrics = config["success_metrics"]
            
            for variant in ["control", "treatment"]:
                variant_assignments = [a for a in assignments if a["variant"] == variant]
                variant_events = [e for e in events if e["variant"] == variant]
                
                variant_metrics = {}
                
                # Calculate basic metrics
                variant_metrics["total_users"] = len(variant_assignments)
                variant_metrics["total_events"] = len(variant_events)
                
                # Calculate success metrics
                for metric in success_metrics:
                    if metric == "accuracy":
                        # Calculate accuracy based on successful events
                        successful_events = [e for e in variant_events if e.get("event_data", {}).get("success", False)]
                        variant_metrics[metric] = len(successful_events) / len(variant_events) if variant_events else 0
                    
                    elif metric == "user_satisfaction":
                        # Calculate average satisfaction score
                        satisfaction_scores = [e.get("event_data", {}).get("satisfaction_score", 0) 
                                             for e in variant_events if "satisfaction_score" in e.get("event_data", {})]
                        variant_metrics[metric] = np.mean(satisfaction_scores) if satisfaction_scores else 0
                    
                    elif metric == "response_time":
                        # Calculate average response time
                        response_times = [e.get("event_data", {}).get("response_time", 0) 
                                        for e in variant_events if "response_time" in e.get("event_data", {})]
                        variant_metrics[metric] = np.mean(response_times) if response_times else 0
                
                metrics[variant] = variant_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate experiment metrics: {e}")
            return {}
    
    def _calculate_statistical_significance(self, 
                                          metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate statistical significance for each metric."""
        try:
            significance = {}
            
            for metric_name in ["accuracy", "user_satisfaction", "response_time"]:
                if metric_name in metrics["control"] and metric_name in metrics["treatment"]:
                    control_value = metrics["control"][metric_name]
                    treatment_value = metrics["treatment"][metric_name]
                    
                    # Simple t-test simulation (in practice, use scipy.stats)
                    # This is a simplified version for demonstration
                    if control_value > 0 and treatment_value > 0:
                        # Calculate p-value (simplified)
                        diff = abs(treatment_value - control_value)
                        pooled_std = (control_value + treatment_value) / 2
                        t_stat = diff / (pooled_std / np.sqrt(2))
                        
                        # Approximate p-value (simplified)
                        p_value = max(0.001, min(0.999, 1 - abs(t_stat) / 3))
                        significance[metric_name] = 1 - p_value
                    else:
                        significance[metric_name] = 0.5
            
            return significance
            
        except Exception as e:
            logger.error(f"Failed to calculate statistical significance: {e}")
            return {}
    
    def _determine_winner(self, 
                         metrics: Dict[str, Dict[str, float]],
                         significance: Dict[str, float]) -> Tuple[Optional[str], float]:
        """Determine the winning variant."""
        try:
            # Key metrics for determining winner
            key_metrics = ["accuracy", "user_satisfaction"]
            
            control_wins = 0
            treatment_wins = 0
            total_comparisons = 0
            
            for metric in key_metrics:
                if metric in metrics["control"] and metric in metric["treatment"]:
                    control_value = metrics["control"][metric]
                    treatment_value = metrics["treatment"][metric]
                    
                    # Check if difference is statistically significant
                    if significance.get(metric, 0) > 0.95:
                        if treatment_value > control_value:
                            treatment_wins += 1
                        else:
                            control_wins += 1
                        total_comparisons += 1
            
            if total_comparisons == 0:
                return None, 0.5
            
            if treatment_wins > control_wins:
                return "treatment", treatment_wins / total_comparisons
            elif control_wins > treatment_wins:
                return "control", control_wins / total_comparisons
            else:
                return None, 0.5
                
        except Exception as e:
            logger.error(f"Failed to determine winner: {e}")
            return None, 0.0
    
    def _generate_recommendation(self, 
                               metrics: Dict[str, Dict[str, float]],
                               significance: Dict[str, float],
                               winner: Optional[str]) -> str:
        """Generate recommendation based on results."""
        try:
            if winner is None:
                return "No significant difference found. Continue with current model."
            
            if winner == "treatment":
                return "Treatment variant shows significant improvement. Recommend promotion to production."
            else:
                return "Control variant performs better. Recommend keeping current model."
                
        except Exception as e:
            logger.error(f"Failed to generate recommendation: {e}")
            return "Unable to generate recommendation."
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{name_hash}"
    
    def _log_experiment_event(self, 
                            experiment_id: str, 
                            event_type: str, 
                            event_data: Dict[str, Any]):
        """Log an experiment management event."""
        try:
            self.events_collection.insert_one({
                "experiment_id": experiment_id,
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now(timezone.utc),
                "system_event": True
            })
        except Exception as e:
            logger.error(f"Failed to log experiment event: {e}")
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B test experiment."""
        try:
            # Update experiment status
            self.experiments_collection.update_one(
                {"experiment_id": experiment_id},
                {
                    "$set": {
                        "status": ExperimentStatus.COMPLETED.value,
                        "end_date": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            # Log experiment end
            self._log_experiment_event(experiment_id, "experiment_stopped", {
                "status": ExperimentStatus.COMPLETED.value
            })
            
            logger.info(f"Stopped A/B test experiment: {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            return False
    
    def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments."""
        try:
            experiments = list(self.experiments_collection.find({
                "status": ExperimentStatus.RUNNING.value
            }))
            
            return experiments
            
        except Exception as e:
            logger.error(f"Failed to get active experiments: {e}")
            return []
    
    def cleanup_completed_experiments(self, days_old: int = 30) -> int:
        """Clean up old completed experiments."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            # Find experiments to cleanup
            experiments_to_cleanup = list(self.experiments_collection.find({
                "status": ExperimentStatus.COMPLETED.value,
                "end_date": {"$lt": cutoff_date}
            }))
            
            cleaned_count = 0
            for experiment in experiments_to_cleanup:
                experiment_id = experiment["experiment_id"]
                
                # Archive experiment
                self.experiments_collection.update_one(
                    {"experiment_id": experiment_id},
                    {"$set": {"status": ExperimentStatus.ARCHIVED.value}}
                )
                
                cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} completed experiments")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup experiments: {e}")
            return 0
