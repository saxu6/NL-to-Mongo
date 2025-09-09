"""
Comprehensive Model Monitoring System for MongoDB Query Translator.
Tracks model performance, data drift, and system health in real-time.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import threading
import time

import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
from redis import Redis

from utils.logger import get_logger

logger = get_logger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SYSTEM_HEALTH = "system_health"
    BUSINESS_METRICS = "business_metrics"

@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    metric_type: MetricType
    alert_level: AlertLevel
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    model_name: str
    model_version: str
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    model_name: str
    model_version: str
    monitoring_interval: int  # seconds
    alert_thresholds: Dict[str, float]
    drift_thresholds: Dict[str, float]
    performance_metrics: List[str]
    data_quality_checks: List[str]
    business_metrics: List[str]

class ModelMonitor:
    """
    Comprehensive model monitoring system.
    
    Features:
    - Real-time performance tracking
    - Data drift detection
    - Concept drift monitoring
    - System health monitoring
    - Automated alerting
    - Prometheus metrics integration
    - Historical trend analysis
    """
    
    def __init__(self, 
                 mongodb_uri: str,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 mlflow_tracking_uri: str = None,
                 prometheus_port: int = 8001):
        """
        Initialize the model monitoring system.
        
        Args:
            mongodb_uri: MongoDB connection string
            redis_host: Redis host for caching
            redis_port: Redis port
            mlflow_tracking_uri: MLflow tracking server URI
            prometheus_port: Port for Prometheus metrics server
        """
        self.mongodb_uri = mongodb_uri
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.prometheus_port = prometheus_port
        
        # Initialize connections
        self._init_mongodb()
        self._init_redis()
        self._init_mlflow()
        self._init_prometheus()
        
        # Monitoring state
        self.monitoring_configs = {}
        self.active_alerts = {}
        self.monitoring_threads = {}
        self.is_monitoring = False
        
        logger.info("Model monitoring system initialized")
    
    def _init_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            self.mongo_client = MongoClient(self.mongodb_uri)
            self.db = self.mongo_client.get_default_database()
            
            # Collections
            self.monitoring_configs_collection = self.db.monitoring_configs
            self.alerts_collection = self.db.alerts
            self.metrics_collection = self.db.monitoring_metrics
            self.drift_reports_collection = self.db.drift_reports
            
            # Create indexes
            self._create_indexes()
            
            logger.info("MongoDB connection established for monitoring")
            
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
            logger.info("Redis connection established for monitoring")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_mlflow(self):
        """Initialize MLflow connection."""
        try:
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)
            logger.info("MLflow connection established for monitoring")
            
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_client = None
    
    def _init_prometheus(self):
        """Initialize Prometheus metrics."""
        try:
            # Define Prometheus metrics
            self.request_counter = Counter(
                'model_requests_total',
                'Total number of model requests',
                ['model_name', 'model_version', 'status']
            )
            
            self.request_duration = Histogram(
                'model_request_duration_seconds',
                'Model request duration',
                ['model_name', 'model_version']
            )
            
            self.model_accuracy = Gauge(
                'model_accuracy',
                'Model accuracy',
                ['model_name', 'model_version']
            )
            
            self.data_drift_score = Gauge(
                'data_drift_score',
                'Data drift score',
                ['model_name', 'feature_name']
            )
            
            self.concept_drift_score = Gauge(
                'concept_drift_score',
                'Concept drift score',
                ['model_name', 'model_version']
            )
            
            self.system_health = Gauge(
                'system_health_score',
                'System health score',
                ['component']
            )
            
            # Start Prometheus server
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for monitoring."""
        try:
            # Monitoring configs
            self.monitoring_configs_collection.create_index("model_name")
            self.monitoring_configs_collection.create_index("model_version")
            
            # Alerts
            self.alerts_collection.create_index("model_name")
            self.alerts_collection.create_index("alert_level")
            self.alerts_collection.create_index("timestamp")
            self.alerts_collection.create_index("resolved")
            
            # Metrics
            self.metrics_collection.create_index([("model_name", 1), ("timestamp", 1)])
            self.metrics_collection.create_index("metric_type")
            
            # Drift reports
            self.drift_reports_collection.create_index([("model_name", 1), ("timestamp", 1)])
            self.drift_reports_collection.create_index("drift_type")
            
            logger.info("Database indexes created for monitoring")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    def setup_monitoring(self, 
                        model_name: str,
                        model_version: str,
                        config: MonitoringConfig) -> bool:
        """
        Setup monitoring for a model.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            config: Monitoring configuration
            
        Returns:
            Success status
        """
        try:
            # Store monitoring configuration
            config_doc = {
                "model_name": model_name,
                "model_version": model_version,
                "config": asdict(config),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "active": True
            }
            
            self.monitoring_configs_collection.replace_one(
                {"model_name": model_name, "model_version": model_version},
                config_doc,
                upsert=True
            )
            
            # Store in memory
            self.monitoring_configs[f"{model_name}:{model_version}"] = config
            
            logger.info(f"Monitoring setup for model {model_name}:{model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return False
    
    def start_monitoring(self, model_name: str, model_version: str) -> bool:
        """
        Start monitoring for a model.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            
        Returns:
            Success status
        """
        try:
            model_key = f"{model_name}:{model_version}"
            
            if model_key not in self.monitoring_configs:
                logger.error(f"Monitoring not configured for {model_key}")
                return False
            
            if model_key in self.monitoring_threads:
                logger.warning(f"Monitoring already running for {model_key}")
                return True
            
            # Start monitoring thread
            config = self.monitoring_configs[model_key]
            thread = threading.Thread(
                target=self._monitoring_loop,
                args=(model_name, model_version, config),
                daemon=True
            )
            thread.start()
            
            self.monitoring_threads[model_key] = thread
            self.is_monitoring = True
            
            logger.info(f"Started monitoring for model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False
    
    def stop_monitoring(self, model_name: str, model_version: str) -> bool:
        """
        Stop monitoring for a model.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            
        Returns:
            Success status
        """
        try:
            model_key = f"{model_name}:{model_version}"
            
            if model_key in self.monitoring_threads:
                # Note: In a real implementation, you'd need a way to stop the thread gracefully
                del self.monitoring_threads[model_key]
                logger.info(f"Stopped monitoring for model {model_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
            return False
    
    def _monitoring_loop(self, 
                        model_name: str, 
                        model_version: str, 
                        config: MonitoringConfig):
        """Main monitoring loop for a model."""
        try:
            while self.is_monitoring:
                try:
                    # Collect metrics
                    self._collect_performance_metrics(model_name, model_version, config)
                    self._detect_data_drift(model_name, model_version, config)
                    self._detect_concept_drift(model_name, model_version, config)
                    self._check_system_health(model_name, model_version, config)
                    
                    # Check for alerts
                    self._check_alerts(model_name, model_version, config)
                    
                    # Sleep for monitoring interval
                    time.sleep(config.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop for {model_name}:{model_version}: {e}")
                    time.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Monitoring loop failed for {model_name}:{model_version}: {e}")
    
    def _collect_performance_metrics(self, 
                                   model_name: str, 
                                   model_version: str, 
                                   config: MonitoringConfig):
        """Collect performance metrics for the model."""
        try:
            # Get recent predictions and their outcomes
            recent_predictions = self._get_recent_predictions(model_name, model_version)
            
            if not recent_predictions:
                return
            
            # Calculate performance metrics
            metrics = {}
            
            for metric_name in config.performance_metrics:
                if metric_name == "accuracy":
                    correct_predictions = sum(1 for p in recent_predictions if p.get("correct", False))
                    metrics[metric_name] = correct_predictions / len(recent_predictions)
                
                elif metric_name == "latency":
                    latencies = [p.get("latency", 0) for p in recent_predictions]
                    metrics[metric_name] = np.mean(latencies)
                
                elif metric_name == "throughput":
                    # Calculate requests per minute
                    time_window = 60  # seconds
                    recent_requests = [p for p in recent_predictions 
                                     if (datetime.now(timezone.utc) - p.get("timestamp", datetime.now(timezone.utc))).seconds < time_window]
                    metrics[metric_name] = len(recent_requests)
            
            # Store metrics
            self._store_metrics(model_name, model_version, MetricType.PERFORMANCE, metrics)
            
            # Update Prometheus metrics
            if "accuracy" in metrics:
                self.model_accuracy.labels(
                    model_name=model_name, 
                    model_version=model_version
                ).set(metrics["accuracy"])
            
            logger.debug(f"Collected performance metrics for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    def _detect_data_drift(self, 
                          model_name: str, 
                          model_version: str, 
                          config: MonitoringConfig):
        """Detect data drift in input features."""
        try:
            # Get baseline data (training data)
            baseline_data = self._get_baseline_data(model_name, model_version)
            
            # Get current data (recent predictions)
            current_data = self._get_current_data(model_name, model_version)
            
            if not baseline_data or not current_data:
                return
            
            # Convert to DataFrames
            baseline_df = pd.DataFrame(baseline_data)
            current_df = pd.DataFrame(current_data)
            
            # Calculate drift scores for each feature
            drift_scores = {}
            
            for feature in config.drift_thresholds.keys():
                if feature in baseline_df.columns and feature in current_df.columns:
                    # Calculate statistical distance (simplified)
                    baseline_mean = baseline_df[feature].mean()
                    baseline_std = baseline_df[feature].std()
                    current_mean = current_df[feature].mean()
                    
                    if baseline_std > 0:
                        drift_score = abs(current_mean - baseline_mean) / baseline_std
                        drift_scores[feature] = drift_score
                        
                        # Update Prometheus metric
                        self.data_drift_score.labels(
                            model_name=model_name,
                            feature_name=feature
                        ).set(drift_score)
            
            # Store drift report
            if drift_scores:
                self._store_drift_report(model_name, model_version, "data_drift", drift_scores)
            
            logger.debug(f"Detected data drift for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to detect data drift: {e}")
    
    def _detect_concept_drift(self, 
                             model_name: str, 
                             model_version: str, 
                             config: MonitoringConfig):
        """Detect concept drift in model performance."""
        try:
            # Get recent predictions with outcomes
            recent_predictions = self._get_recent_predictions(model_name, model_version)
            
            if len(recent_predictions) < 100:  # Need sufficient data
                return
            
            # Calculate concept drift score based on performance degradation
            baseline_accuracy = self._get_baseline_accuracy(model_name, model_version)
            current_accuracy = sum(1 for p in recent_predictions if p.get("correct", False)) / len(recent_predictions)
            
            if baseline_accuracy > 0:
                concept_drift_score = (baseline_accuracy - current_accuracy) / baseline_accuracy
                
                # Update Prometheus metric
                self.concept_drift_score.labels(
                    model_name=model_name,
                    model_version=model_version
                ).set(concept_drift_score)
                
                # Store drift report
                self._store_drift_report(
                    model_name, 
                    model_version, 
                    "concept_drift", 
                    {"concept_drift_score": concept_drift_score}
                )
            
            logger.debug(f"Detected concept drift for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to detect concept drift: {e}")
    
    def _check_system_health(self, 
                           model_name: str, 
                           model_version: str, 
                           config: MonitoringConfig):
        """Check system health metrics."""
        try:
            health_metrics = {}
            
            # Check database connectivity
            try:
                self.db.admin.command('ping')
                health_metrics["database_health"] = 1.0
            except:
                health_metrics["database_health"] = 0.0
            
            # Check Redis connectivity
            if self.redis_client:
                try:
                    self.redis_client.ping()
                    health_metrics["redis_health"] = 1.0
                except:
                    health_metrics["redis_health"] = 0.0
            else:
                health_metrics["redis_health"] = 0.0
            
            # Check MLflow connectivity
            if self.mlflow_client:
                try:
                    self.mlflow_client.search_experiments()
                    health_metrics["mlflow_health"] = 1.0
                except:
                    health_metrics["mlflow_health"] = 0.0
            else:
                health_metrics["mlflow_health"] = 0.0
            
            # Calculate overall system health
            overall_health = np.mean(list(health_metrics.values()))
            health_metrics["overall_health"] = overall_health
            
            # Update Prometheus metrics
            for component, score in health_metrics.items():
                self.system_health.labels(component=component).set(score)
            
            # Store metrics
            self._store_metrics(model_name, model_version, MetricType.SYSTEM_HEALTH, health_metrics)
            
            logger.debug(f"Checked system health for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    def _check_alerts(self, 
                     model_name: str, 
                     model_version: str, 
                     config: MonitoringConfig):
        """Check for alert conditions."""
        try:
            # Get recent metrics
            recent_metrics = self._get_recent_metrics(model_name, model_version)
            
            if not recent_metrics:
                return
            
            # Check each alert threshold
            for metric_name, threshold in config.alert_thresholds.items():
                if metric_name in recent_metrics:
                    current_value = recent_metrics[metric_name]
                    
                    # Determine alert level
                    alert_level = AlertLevel.INFO
                    if current_value > threshold * 1.5:
                        alert_level = AlertLevel.CRITICAL
                    elif current_value > threshold:
                        alert_level = AlertLevel.WARNING
                    
                    # Create alert if threshold exceeded
                    if current_value > threshold:
                        self._create_alert(
                            model_name=model_name,
                            model_version=model_version,
                            metric_type=MetricType.PERFORMANCE,
                            alert_level=alert_level,
                            message=f"{metric_name} exceeded threshold: {current_value:.3f} > {threshold:.3f}",
                            threshold=threshold,
                            current_value=current_value
                        )
            
            logger.debug(f"Checked alerts for {model_name}:{model_version}")
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    def _create_alert(self, 
                     model_name: str,
                     model_version: str,
                     metric_type: MetricType,
                     alert_level: AlertLevel,
                     message: str,
                     threshold: float,
                     current_value: float) -> str:
        """Create a new alert."""
        try:
            alert_id = f"{model_name}:{model_version}:{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            alert = Alert(
                alert_id=alert_id,
                metric_type=metric_type,
                alert_level=alert_level,
                message=message,
                threshold=threshold,
                current_value=current_value,
                timestamp=datetime.now(timezone.utc),
                model_name=model_name,
                model_version=model_version
            )
            
            # Store alert
            self.alerts_collection.insert_one(asdict(alert))
            
            # Store in memory
            self.active_alerts[alert_id] = alert
            
            # Send notification (placeholder)
            self._send_alert_notification(alert)
            
            logger.warning(f"Created alert: {alert_id} - {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return ""
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification (placeholder for actual implementation)."""
        try:
            # In practice, this would send notifications via:
            # - Slack webhooks
            # - Email
            # - PagerDuty
            # - SMS
            
            logger.warning(f"ALERT [{alert.alert_level.value.upper()}]: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    def _get_recent_predictions(self, 
                               model_name: str, 
                               model_version: str, 
                               hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent predictions for the model."""
        try:
            # This would query your prediction logs
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent predictions: {e}")
            return []
    
    def _get_baseline_data(self, model_name: str, model_version: str) -> List[Dict[str, Any]]:
        """Get baseline training data for drift detection."""
        try:
            # This would retrieve the original training data
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Failed to get baseline data: {e}")
            return []
    
    def _get_current_data(self, model_name: str, model_version: str) -> List[Dict[str, Any]]:
        """Get current input data for drift detection."""
        try:
            # This would retrieve recent input data
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Failed to get current data: {e}")
            return []
    
    def _get_baseline_accuracy(self, model_name: str, model_version: str) -> float:
        """Get baseline accuracy for concept drift detection."""
        try:
            # This would retrieve the original model accuracy
            # For now, return a default value
            return 0.85
            
        except Exception as e:
            logger.error(f"Failed to get baseline accuracy: {e}")
            return 0.85
    
    def _get_recent_metrics(self, 
                           model_name: str, 
                           model_version: str, 
                           hours: int = 1) -> Dict[str, float]:
        """Get recent metrics for alert checking."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            recent_metrics = list(self.metrics_collection.find({
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", -1).limit(1))
            
            if recent_metrics:
                return recent_metrics[0].get("metrics", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get recent metrics: {e}")
            return {}
    
    def _store_metrics(self, 
                      model_name: str, 
                      model_version: str, 
                      metric_type: MetricType, 
                      metrics: Dict[str, float]):
        """Store metrics in the database."""
        try:
            metric_doc = {
                "model_name": model_name,
                "model_version": model_version,
                "metric_type": metric_type.value,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc)
            }
            
            self.metrics_collection.insert_one(metric_doc)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def _store_drift_report(self, 
                           model_name: str, 
                           model_version: str, 
                           drift_type: str, 
                           drift_scores: Dict[str, float]):
        """Store drift report in the database."""
        try:
            drift_doc = {
                "model_name": model_name,
                "model_version": model_version,
                "drift_type": drift_type,
                "drift_scores": drift_scores,
                "timestamp": datetime.now(timezone.utc)
            }
            
            self.drift_reports_collection.insert_one(drift_doc)
            
        except Exception as e:
            logger.error(f"Failed to store drift report: {e}")
    
    def get_monitoring_dashboard_data(self, 
                                    model_name: str, 
                                    model_version: str, 
                                    hours: int = 24) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Get metrics
            metrics = list(self.metrics_collection.find({
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", 1))
            
            # Get alerts
            alerts = list(self.alerts_collection.find({
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", -1))
            
            # Get drift reports
            drift_reports = list(self.drift_reports_collection.find({
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": {"$gte": cutoff_time}
            }).sort("timestamp", -1))
            
            dashboard_data = {
                "model_name": model_name,
                "model_version": model_version,
                "time_range_hours": hours,
                "metrics": metrics,
                "alerts": alerts,
                "drift_reports": drift_reports,
                "summary": {
                    "total_alerts": len(alerts),
                    "critical_alerts": len([a for a in alerts if a["alert_level"] == "critical"]),
                    "warning_alerts": len([a for a in alerts if a["alert_level"] == "warning"]),
                    "data_drift_detected": len([d for d in drift_reports if d["drift_type"] == "data_drift"]),
                    "concept_drift_detected": len([d for d in drift_reports if d["drift_type"] == "concept_drift"])
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = None) -> bool:
        """Resolve an alert."""
        try:
            # Update alert in database
            self.alerts_collection.update_one(
                {"alert_id": alert_id},
                {
                    "$set": {
                        "resolved": True,
                        "resolution_notes": resolution_notes,
                        "resolved_at": datetime.now(timezone.utc)
                    }
                }
            )
            
            # Remove from active alerts
            if alert_id in self.active_alerts:
                del self.active_alerts[alert_id]
            
            logger.info(f"Resolved alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old monitoring data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            # Clean up old metrics
            metrics_result = self.metrics_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean up old alerts
            alerts_result = self.alerts_collection.delete_many({
                "timestamp": {"$lt": cutoff_date},
                "resolved": True
            })
            
            # Clean up old drift reports
            drift_result = self.drift_reports_collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            total_cleaned = metrics_result.deleted_count + alerts_result.deleted_count + drift_result.deleted_count
            
            logger.info(f"Cleaned up {total_cleaned} old monitoring records")
            return total_cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
