"""
Continuous Improvement and Retraining Pipeline for MongoDB Query Translator.
Automated model retraining based on feedback, performance metrics, and data drift.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncio
import threading
import time

import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import redis
from redis import Redis

from mlops.feedback.feedback_system import FeedbackSystem
from mlops.monitoring.model_monitoring import ModelMonitor
from mlops.registry.model_registry import ModelRegistry, ModelStage
from utils.logger import get_logger

logger = get_logger(__name__)

class RetrainingTrigger(Enum):
    """Triggers for model retraining."""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEEDBACK_THRESHOLD = "feedback_threshold"
    MANUAL = "manual"

class RetrainingStatus(Enum):
    """Status of retraining process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class RetrainingConfig:
    """Configuration for retraining pipeline."""
    model_name: str
    current_version: str
    trigger: RetrainingTrigger
    trigger_threshold: float
    training_data_days: int
    validation_split: float
    test_split: float
    min_improvement_threshold: float
    max_training_time_hours: int
    notification_emails: List[str]
    auto_deploy: bool
    created_at: datetime
    created_by: str

@dataclass
class RetrainingJob:
    """Retraining job record."""
    job_id: str
    config: RetrainingConfig
    status: RetrainingStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    training_metrics: Optional[Dict[str, float]]
    validation_metrics: Optional[Dict[str, float]]
    improvement_metrics: Optional[Dict[str, float]]
    new_model_version: Optional[str]
    deployment_info: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime

@dataclass
class TrainingDataset:
    """Training dataset for retraining."""
    dataset_id: str
    model_name: str
    training_samples: int
    validation_samples: int
    test_samples: int
    data_sources: List[str]
    created_at: datetime
    metadata: Dict[str, Any]

class RetrainingPipeline:
    """
    Comprehensive retraining pipeline for continuous model improvement.
    
    Features:
    - Automated retraining triggers
    - Data collection and preparation
    - Model training and validation
    - Performance comparison
    - Automated deployment
    - Rollback capabilities
    - A/B testing integration
    """
    
    def __init__(self, 
                 mongodb_uri: str,
                 mlflow_tracking_uri: str = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 database_name: str = "retraining_pipeline"):
        """
        Initialize the retraining pipeline.
        
        Args:
            mongodb_uri: MongoDB connection string
            mlflow_tracking_uri: MLflow tracking server URI
            redis_host: Redis host for caching
            redis_port: Redis port
            database_name: Database name for retraining data
        """
        self.mongodb_uri = mongodb_uri
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.database_name = database_name
        
        # Initialize connections
        self._init_mongodb()
        self._init_redis()
        self._init_mlflow()
        
        # Initialize MLOps components
        self._init_mlops_components()
        
        # Pipeline state
        self.active_jobs = {}
        self.retraining_threads = {}
        self.is_running = False
        
        logger.info("Retraining pipeline initialized")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client[self.database_name]
            
            # Collections
            self.retraining_configs_collection = self.db.retraining_configs
            self.retraining_jobs_collection = self.db.retraining_jobs
            self.training_datasets_collection = self.db.training_datasets
            self.retraining_triggers_collection = self.db.retraining_triggers
            
            # Create indexes
            self._create_indexes()
            
            logger.info("MongoDB connection established for retraining pipeline")
            
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
            logger.info("Redis connection established for retraining pipeline")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_mlflow(self):
        """Initialize MLflow connection."""
        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            logger.info("MLflow connection established for retraining pipeline")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_client = None
    
    def _init_mlops_components(self):
        """Initialize MLOps components."""
        try:
            # Initialize feedback system
            self.feedback_system = FeedbackSystem(
                mongodb_uri=self.mongodb_uri,
                redis_host=self.redis_host,
                redis_port=self.redis_port
            )
            
            # Initialize model monitor
            self.model_monitor = ModelMonitor(
                mongodb_uri=self.mongodb_uri,
                redis_host=self.redis_host,
                redis_port=self.redis_port,
                mlflow_tracking_uri=self.mlflow_tracking_uri
            )
            
            # Initialize model registry
            self.model_registry = ModelRegistry(
                tracking_uri=self.mlflow_tracking_uri,
                experiment_name="retraining-pipeline"
            )
            
            logger.info("MLOps components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLOps components: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for retraining pipeline."""
        try:
            # Retraining configs
            self.retraining_configs_collection.create_index("model_name")
            self.retraining_configs_collection.create_index("trigger")
            
            # Retraining jobs
            self.retraining_jobs_collection.create_index("job_id", unique=True)
            self.retraining_jobs_collection.create_index("model_name")
            self.retraining_jobs_collection.create_index("status")
            self.retraining_jobs_collection.create_index("created_at")
            
            # Training datasets
            self.training_datasets_collection.create_index("dataset_id", unique=True)
            self.training_datasets_collection.create_index("model_name")
            self.training_datasets_collection.create_index("created_at")
            
            # Retraining triggers
            self.retraining_triggers_collection.create_index("model_name")
            self.retraining_triggers_collection.create_index("trigger_type")
            self.retraining_triggers_collection.create_index("timestamp")
            
            logger.info("Database indexes created for retraining pipeline")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def setup_retraining_config(self, 
                               model_name: str,
                               current_version: str,
                               trigger: RetrainingTrigger,
                               trigger_threshold: float = 0.1,
                               training_data_days: int = 30,
                               validation_split: float = 0.2,
                               test_split: float = 0.1,
                               min_improvement_threshold: float = 0.05,
                               max_training_time_hours: int = 4,
                               notification_emails: List[str] = None,
                               auto_deploy: bool = False,
                               created_by: str = "system") -> str:
        """
        Setup retraining configuration for a model.
        
        Args:
            model_name: Name of the model
            current_version: Current model version
            trigger: Retraining trigger type
            trigger_threshold: Threshold for triggering retraining
            training_data_days: Days of data to use for training
            validation_split: Validation data split ratio
            test_split: Test data split ratio
            min_improvement_threshold: Minimum improvement required for deployment
            max_training_time_hours: Maximum training time
            notification_emails: Email addresses for notifications
            auto_deploy: Whether to auto-deploy improved models
            created_by: Creator of the configuration
            
        Returns:
            Configuration ID
        """
        try:
            config_id = str(uuid.uuid4())
            
            config = RetrainingConfig(
                model_name=model_name,
                current_version=current_version,
                trigger=trigger,
                trigger_threshold=trigger_threshold,
                training_data_days=training_data_days,
                validation_split=validation_split,
                test_split=test_split,
                min_improvement_threshold=min_improvement_threshold,
                max_training_time_hours=max_training_time_hours,
                notification_emails=notification_emails or [],
                auto_deploy=auto_deploy,
                created_at=datetime.now(timezone.utc),
                created_by=created_by
            )
            
            # Store configuration
            config_doc = {
                "config_id": config_id,
                "config": asdict(config),
                "active": True,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.retraining_configs_collection.replace_one(
                {"model_name": model_name, "trigger": trigger.value},
                config_doc,
                upsert=True
            )
            
            logger.info(f"Retraining configuration setup for {model_name}")
            return config_id
            
        except Exception as e:
            logger.error(f"Failed to setup retraining configuration: {e}")
            raise
    
    def start_retraining_monitoring(self):
        """Start monitoring for retraining triggers."""
        try:
            if self.is_running:
                logger.warning("Retraining monitoring already running")
                return
            
            self.is_running = True
            
            # Start monitoring thread
            monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            monitoring_thread.start()
            
            logger.info("Retraining monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start retraining monitoring: {e}")
            raise
    
    def stop_retraining_monitoring(self):
        """Stop monitoring for retraining triggers."""
        try:
            self.is_running = False
            logger.info("Retraining monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop retraining monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop for retraining triggers."""
        try:
            while self.is_running:
                try:
                    # Check all retraining triggers
                    self._check_retraining_triggers()
                    
                    # Check active retraining jobs
                    self._check_active_jobs()
                    
                    # Sleep for monitoring interval
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in retraining monitoring loop: {e}")
                    time.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Retraining monitoring loop failed: {e}")
    
    def _check_retraining_triggers(self):
        """Check for retraining trigger conditions."""
        try:
            # Get active retraining configurations
            configs = list(self.retraining_configs_collection.find({"active": True}))
            
            for config_doc in configs:
                config = RetrainingConfig(**config_doc["config"])
                
                # Check if retraining is already in progress
                active_job = self.retraining_jobs_collection.find_one({
                    "config.model_name": config.model_name,
                    "status": {"$in": ["pending", "in_progress"]}
                })
                
                if active_job:
                    continue  # Skip if already retraining
                
                # Check trigger conditions
                should_retrain = False
                trigger_reason = ""
                
                if config.trigger == RetrainingTrigger.SCHEDULED:
                    should_retrain, trigger_reason = self._check_scheduled_trigger(config)
                
                elif config.trigger == RetrainingTrigger.PERFORMANCE_DEGRADATION:
                    should_retrain, trigger_reason = self._check_performance_trigger(config)
                
                elif config.trigger == RetrainingTrigger.DATA_DRIFT:
                    should_retrain, trigger_reason = self._check_data_drift_trigger(config)
                
                elif config.trigger == RetrainingTrigger.CONCEPT_DRIFT:
                    should_retrain, trigger_reason = self._check_concept_drift_trigger(config)
                
                elif config.trigger == RetrainingTrigger.FEEDBACK_THRESHOLD:
                    should_retrain, trigger_reason = self._check_feedback_trigger(config)
                
                if should_retrain:
                    self._trigger_retraining(config, trigger_reason)
                
        except Exception as e:
            logger.error(f"Failed to check retraining triggers: {e}")
    
    def _check_scheduled_trigger(self, config: RetrainingConfig) -> Tuple[bool, str]:
        """Check if scheduled retraining should be triggered."""
        try:
            # Check if enough time has passed since last retraining
            last_job = self.retraining_jobs_collection.find_one({
                "config.model_name": config.model_name,
                "status": "completed"
            }, sort=[("end_time", -1)])
            
            if not last_job:
                return True, "No previous retraining found"
            
            last_retraining = last_job["end_time"]
            time_since_last = datetime.now(timezone.utc) - last_retraining
            
            # Trigger if more than 7 days have passed (configurable)
            if time_since_last.days >= 7:
                return True, f"Scheduled retraining (last: {time_since_last.days} days ago)"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Failed to check scheduled trigger: {e}")
            return False, ""
    
    def _check_performance_trigger(self, config: RetrainingConfig) -> Tuple[bool, str]:
        """Check if performance degradation should trigger retraining."""
        try:
            # Get current model performance
            dashboard_data = self.model_monitor.get_monitoring_dashboard_data(
                config.model_name, config.current_version, hours=24
            )
            
            if not dashboard_data or "metrics" not in dashboard_data:
                return False, "No performance data available"
            
            # Check accuracy degradation
            recent_metrics = dashboard_data["metrics"]
            if recent_metrics:
                latest_metrics = recent_metrics[-1].get("metrics", {})
                current_accuracy = latest_metrics.get("accuracy", 0.0)
                
                # Get baseline accuracy
                baseline_accuracy = 0.85  # This should come from model registry
                
                accuracy_degradation = baseline_accuracy - current_accuracy
                
                if accuracy_degradation > config.trigger_threshold:
                    return True, f"Performance degradation: {accuracy_degradation:.3f} > {config.trigger_threshold}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Failed to check performance trigger: {e}")
            return False, ""
    
    def _check_data_drift_trigger(self, config: RetrainingConfig) -> Tuple[bool, str]:
        """Check if data drift should trigger retraining."""
        try:
            # Get recent drift reports
            drift_reports = list(self.model_monitor.drift_reports_collection.find({
                "model_name": config.model_name,
                "drift_type": "data_drift",
                "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(hours=24)}
            }))
            
            if not drift_reports:
                return False, "No drift data available"
            
            # Check for significant drift
            for report in drift_reports:
                drift_scores = report.get("drift_scores", {})
                max_drift = max(drift_scores.values()) if drift_scores else 0
                
                if max_drift > config.trigger_threshold:
                    return True, f"Data drift detected: {max_drift:.3f} > {config.trigger_threshold}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Failed to check data drift trigger: {e}")
            return False, ""
    
    def _check_concept_drift_trigger(self, config: RetrainingConfig) -> Tuple[bool, str]:
        """Check if concept drift should trigger retraining."""
        try:
            # Get recent concept drift reports
            drift_reports = list(self.model_monitor.drift_reports_collection.find({
                "model_name": config.model_name,
                "drift_type": "concept_drift",
                "timestamp": {"$gte": datetime.now(timezone.utc) - timedelta(hours=24)}
            }))
            
            if not drift_reports:
                return False, "No concept drift data available"
            
            # Check for significant concept drift
            for report in drift_reports:
                drift_scores = report.get("drift_scores", {})
                concept_drift_score = drift_scores.get("concept_drift_score", 0)
                
                if concept_drift_score > config.trigger_threshold:
                    return True, f"Concept drift detected: {concept_drift_score:.3f} > {config.trigger_threshold}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Failed to check concept drift trigger: {e}")
            return False, ""
    
    def _check_feedback_trigger(self, config: RetrainingConfig) -> Tuple[bool, str]:
        """Check if feedback threshold should trigger retraining."""
        try:
            # Get recent feedback summary
            feedback_summary = self.feedback_system.get_feedback_summary(hours=24)
            
            if not feedback_summary:
                return False, "No feedback data available"
            
            # Check negative feedback ratio
            total_feedback = feedback_summary.get("total_feedback", 0)
            negative_feedback = feedback_summary.get("negative_feedback", 0)
            
            if total_feedback > 0:
                negative_ratio = negative_feedback / total_feedback
                
                if negative_ratio > config.trigger_threshold:
                    return True, f"High negative feedback: {negative_ratio:.3f} > {config.trigger_threshold}"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Failed to check feedback trigger: {e}")
            return False, ""
    
    def _trigger_retraining(self, config: RetrainingConfig, trigger_reason: str):
        """Trigger a retraining job."""
        try:
            job_id = str(uuid.uuid4())
            
            # Create retraining job
            job = RetrainingJob(
                job_id=job_id,
                config=config,
                status=RetrainingStatus.PENDING,
                start_time=None,
                end_time=None,
                training_metrics=None,
                validation_metrics=None,
                improvement_metrics=None,
                new_model_version=None,
                deployment_info=None,
                error_message=None,
                created_at=datetime.now(timezone.utc)
            )
            
            # Store job
            self.retraining_jobs_collection.insert_one(asdict(job))
            
            # Log trigger
            self.retraining_triggers_collection.insert_one({
                "job_id": job_id,
                "model_name": config.model_name,
                "trigger_type": config.trigger.value,
                "trigger_reason": trigger_reason,
                "timestamp": datetime.now(timezone.utc)
            })
            
            # Start retraining in background
            retraining_thread = threading.Thread(
                target=self._execute_retraining_job,
                args=(job_id,),
                daemon=True
            )
            retraining_thread.start()
            
            self.retraining_threads[job_id] = retraining_thread
            
            logger.info(f"Triggered retraining job {job_id} for {config.model_name}: {trigger_reason}")
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining: {e}")
    
    def _execute_retraining_job(self, job_id: str):
        """Execute a retraining job."""
        try:
            # Get job
            job_doc = self.retraining_jobs_collection.find_one({"job_id": job_id})
            if not job_doc:
                logger.error(f"Retraining job {job_id} not found")
                return
            
            job = RetrainingJob(**job_doc)
            
            # Update status to in progress
            self._update_job_status(job_id, RetrainingStatus.IN_PROGRESS)
            
            # Step 1: Collect training data
            training_dataset = self._collect_training_data(job.config)
            
            # Step 2: Train new model
            training_metrics, validation_metrics = self._train_new_model(
                job.config, training_dataset
            )
            
            # Step 3: Evaluate model performance
            improvement_metrics = self._evaluate_model_improvement(
                job.config, training_metrics, validation_metrics
            )
            
            # Step 4: Register new model if improved
            new_model_version = None
            if self._should_deploy_model(improvement_metrics, job.config):
                new_model_version = self._register_new_model(
                    job.config, training_metrics, validation_metrics
                )
                
                # Step 5: Deploy if auto-deploy is enabled
                deployment_info = None
                if job.config.auto_deploy:
                    deployment_info = self._deploy_new_model(
                        job.config, new_model_version
                    )
            
            # Update job with results
            self._update_job_results(
                job_id, training_metrics, validation_metrics, 
                improvement_metrics, new_model_version, deployment_info
            )
            
            # Update status to completed
            self._update_job_status(job_id, RetrainingStatus.COMPLETED)
            
            logger.info(f"Retraining job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Retraining job {job_id} failed: {e}")
            self._update_job_error(job_id, str(e))
            self._update_job_status(job_id, RetrainingStatus.FAILED)
    
    def _collect_training_data(self, config: RetrainingConfig) -> TrainingDataset:
        """Collect training data for retraining."""
        try:
            dataset_id = str(uuid.uuid4())
            
            # Get feedback data for training
            feedback_data = self.feedback_system.export_feedback_for_training(
                days=config.training_data_days,
                include_positive=True,
                include_negative=True
            )
            
            # Get query logs for training
            query_logs = self._get_query_logs_for_training(config.training_data_days)
            
            # Combine and prepare training data
            training_data = self._prepare_training_data(feedback_data, query_logs)
            
            # Split data
            train_data, val_data, test_data = self._split_training_data(
                training_data, config.validation_split, config.test_split
            )
            
            # Create dataset record
            dataset = TrainingDataset(
                dataset_id=dataset_id,
                model_name=config.model_name,
                training_samples=len(train_data),
                validation_samples=len(val_data),
                test_samples=len(test_data),
                data_sources=["feedback_system", "query_logs"],
                created_at=datetime.now(timezone.utc),
                metadata={
                    "training_data_days": config.training_data_days,
                    "validation_split": config.validation_split,
                    "test_split": config.test_split
                }
            )
            
            # Store dataset
            self.training_datasets_collection.insert_one(asdict(dataset))
            
            logger.info(f"Collected training dataset {dataset_id} with {len(training_data)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to collect training data: {e}")
            raise
    
    def _get_query_logs_for_training(self, days: int) -> List[Dict[str, Any]]:
        """Get query logs for training."""
        try:
            # This would query your query logs collection
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Failed to get query logs: {e}")
            return []
    
    def _prepare_training_data(self, 
                              feedback_data: List[Dict[str, Any]], 
                              query_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare training data from feedback and query logs."""
        try:
            training_data = []
            
            # Process feedback data
            for feedback in feedback_data:
                training_record = {
                    "user_query": feedback.get("user_query", ""),
                    "generated_query": feedback.get("generated_query", ""),
                    "label": 1 if feedback.get("feedback_type") == "thumbs_up" else 0,
                    "rating": feedback.get("rating", 0),
                    "comment": feedback.get("comment", ""),
                    "source": "feedback"
                }
                training_data.append(training_record)
            
            # Process query logs
            for log in query_logs:
                training_record = {
                    "user_query": log.get("user_query", ""),
                    "generated_query": log.get("generated_query", ""),
                    "label": 1 if log.get("success", False) else 0,
                    "rating": 0,
                    "comment": "",
                    "source": "query_log"
                }
                training_data.append(training_record)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            return []
    
    def _split_training_data(self, 
                           training_data: List[Dict[str, Any]], 
                           validation_split: float, 
                           test_split: float) -> Tuple[List, List, List]:
        """Split training data into train/validation/test sets."""
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(training_data)
            
            # Split into train and temp (validation + test)
            train_df, temp_df = train_test_split(
                df, test_size=(validation_split + test_split), random_state=42
            )
            
            # Split temp into validation and test
            val_size = validation_split / (validation_split + test_split)
            val_df, test_df = train_test_split(
                temp_df, test_size=(1 - val_size), random_state=42
            )
            
            return train_df.to_dict('records'), val_df.to_dict('records'), test_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to split training data: {e}")
            return [], [], []
    
    def _train_new_model(self, 
                        config: RetrainingConfig, 
                        dataset: TrainingDataset) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train a new model."""
        try:
            # This is a placeholder for actual model training
            # In practice, you would:
            # 1. Load the current model
            # 2. Fine-tune on new data
            # 3. Evaluate on validation set
            
            # Simulate training metrics
            training_metrics = {
                "accuracy": 0.87 + np.random.normal(0, 0.02),
                "precision": 0.84 + np.random.normal(0, 0.02),
                "recall": 0.89 + np.random.normal(0, 0.02),
                "f1_score": 0.86 + np.random.normal(0, 0.02),
                "training_loss": 0.12 + np.random.normal(0, 0.01)
            }
            
            validation_metrics = {
                "accuracy": 0.85 + np.random.normal(0, 0.02),
                "precision": 0.82 + np.random.normal(0, 0.02),
                "recall": 0.87 + np.random.normal(0, 0.02),
                "f1_score": 0.84 + np.random.normal(0, 0.02),
                "validation_loss": 0.15 + np.random.normal(0, 0.01)
            }
            
            logger.info(f"Trained new model for {config.model_name}")
            return training_metrics, validation_metrics
            
        except Exception as e:
            logger.error(f"Failed to train new model: {e}")
            raise
    
    def _evaluate_model_improvement(self, 
                                   config: RetrainingConfig, 
                                   training_metrics: Dict[str, float], 
                                   validation_metrics: Dict[str, float]) -> Dict[str, float]:
        """Evaluate model improvement over baseline."""
        try:
            # Get baseline metrics from current model
            baseline_metrics = self._get_baseline_metrics(config.model_name, config.current_version)
            
            # Calculate improvement
            improvement_metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1_score"]:
                if metric in validation_metrics and metric in baseline_metrics:
                    improvement = validation_metrics[metric] - baseline_metrics[metric]
                    improvement_metrics[f"{metric}_improvement"] = improvement
            
            # Calculate overall improvement score
            improvements = [v for v in improvement_metrics.values()]
            improvement_metrics["overall_improvement"] = np.mean(improvements) if improvements else 0
            
            return improvement_metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model improvement: {e}")
            return {}
    
    def _get_baseline_metrics(self, model_name: str, model_version: str) -> Dict[str, float]:
        """Get baseline metrics for comparison."""
        try:
            # This would retrieve metrics from the model registry
            # For now, return default baseline metrics
            return {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            }
            
        except Exception as e:
            logger.error(f"Failed to get baseline metrics: {e}")
            return {}
    
    def _should_deploy_model(self, 
                           improvement_metrics: Dict[str, float], 
                           config: RetrainingConfig) -> bool:
        """Determine if the new model should be deployed."""
        try:
            overall_improvement = improvement_metrics.get("overall_improvement", 0)
            return overall_improvement >= config.min_improvement_threshold
            
        except Exception as e:
            logger.error(f"Failed to determine deployment decision: {e}")
            return False
    
    def _register_new_model(self, 
                           config: RetrainingConfig, 
                           training_metrics: Dict[str, float], 
                           validation_metrics: Dict[str, float]) -> str:
        """Register the new model in the model registry."""
        try:
            # Register model with MLflow
            model_version = self.model_registry.register_model(
                model_name=config.model_name,
                model_path="model_artifacts",  # This would be the actual model path
                description=f"Retrained model triggered by {config.trigger.value}",
                tags={"retraining_trigger": config.trigger.value},
                metrics=validation_metrics,
                parameters={"training_data_days": config.training_data_days}
            )
            
            # Promote to staging
            self.model_registry.promote_model(
                model_name=config.model_name,
                version=model_version,
                stage=ModelStage.STAGING,
                comment="Automated retraining - staging for validation"
            )
            
            logger.info(f"Registered new model version {model_version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Failed to register new model: {e}")
            raise
    
    def _deploy_new_model(self, config: RetrainingConfig, model_version: str) -> Dict[str, Any]:
        """Deploy the new model to production."""
        try:
            # Deploy model
            deployment_info = self.model_registry.deploy_model(
                model_name=config.model_name,
                version=model_version,
                stage=ModelStage.PRODUCTION
            )
            
            logger.info(f"Deployed new model version {model_version}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to deploy new model: {e}")
            raise
    
    def _update_job_status(self, job_id: str, status: RetrainingStatus):
        """Update retraining job status."""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.now(timezone.utc)
            }
            
            if status == RetrainingStatus.IN_PROGRESS:
                update_data["start_time"] = datetime.now(timezone.utc)
            elif status in [RetrainingStatus.COMPLETED, RetrainingStatus.FAILED, RetrainingStatus.CANCELLED]:
                update_data["end_time"] = datetime.now(timezone.utc)
            
            self.retraining_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def _update_job_results(self, 
                           job_id: str, 
                           training_metrics: Dict[str, float], 
                           validation_metrics: Dict[str, float], 
                           improvement_metrics: Dict[str, float], 
                           new_model_version: str, 
                           deployment_info: Dict[str, Any]):
        """Update retraining job with results."""
        try:
            update_data = {
                "training_metrics": training_metrics,
                "validation_metrics": validation_metrics,
                "improvement_metrics": improvement_metrics,
                "new_model_version": new_model_version,
                "deployment_info": deployment_info,
                "updated_at": datetime.now(timezone.utc)
            }
            
            self.retraining_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.error(f"Failed to update job results: {e}")
    
    def _update_job_error(self, job_id: str, error_message: str):
        """Update retraining job with error message."""
        try:
            self.retraining_jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "error_message": error_message,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to update job error: {e}")
    
    def _check_active_jobs(self):
        """Check status of active retraining jobs."""
        try:
            # Get active jobs
            active_jobs = list(self.retraining_jobs_collection.find({
                "status": {"$in": ["pending", "in_progress"]}
            }))
            
            for job_doc in active_jobs:
                job = RetrainingJob(**job_doc)
                
                # Check if job has exceeded maximum training time
                if job.start_time:
                    training_duration = datetime.now(timezone.utc) - job.start_time
                    max_duration = timedelta(hours=job.config.max_training_time_hours)
                    
                    if training_duration > max_duration:
                        logger.warning(f"Retraining job {job.job_id} exceeded maximum training time")
                        self._update_job_error(job.job_id, "Exceeded maximum training time")
                        self._update_job_status(job.job_id, RetrainingStatus.FAILED)
                
        except Exception as e:
            logger.error(f"Failed to check active jobs: {e}")
    
    def get_retraining_status(self, model_name: str) -> Dict[str, Any]:
        """Get retraining status for a model."""
        try:
            # Get recent retraining jobs
            recent_jobs = list(self.retraining_jobs_collection.find({
                "config.model_name": model_name
            }).sort("created_at", -1).limit(10))
            
            # Get retraining configurations
            configs = list(self.retraining_configs_collection.find({
                "config.model_name": model_name,
                "active": True
            }))
            
            # Get recent triggers
            recent_triggers = list(self.retraining_triggers_collection.find({
                "model_name": model_name
            }).sort("timestamp", -1).limit(5))
            
            status = {
                "model_name": model_name,
                "active_configurations": len(configs),
                "recent_jobs": len(recent_jobs),
                "recent_triggers": len(recent_triggers),
                "last_retraining": recent_jobs[0]["created_at"] if recent_jobs else None,
                "last_trigger": recent_triggers[0]["timestamp"] if recent_triggers else None,
                "monitoring_active": self.is_running
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get retraining status: {e}")
            return {}
    
    def manual_retraining_trigger(self, 
                                 model_name: str, 
                                 trigger_reason: str = "Manual trigger",
                                 created_by: str = "user") -> str:
        """Manually trigger retraining for a model."""
        try:
            # Get retraining configuration
            config_doc = self.retraining_configs_collection.find_one({
                "config.model_name": model_name,
                "active": True
            })
            
            if not config_doc:
                raise ValueError(f"No active retraining configuration found for {model_name}")
            
            config = RetrainingConfig(**config_doc["config"])
            
            # Override trigger type for manual trigger
            config.trigger = RetrainingTrigger.MANUAL
            
            # Trigger retraining
            self._trigger_retraining(config, f"{trigger_reason} by {created_by}")
            
            logger.info(f"Manual retraining triggered for {model_name}")
            return "Retraining triggered successfully"
            
        except Exception as e:
            logger.error(f"Failed to trigger manual retraining: {e}")
            raise
