"""
MLflow Model Registry for MongoDB Query Translator.
Comprehensive model versioning, staging, and deployment management.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from pymongo import MongoClient

from utils.logger import get_logger

logger = get_logger(__name__)

class ModelStage(Enum):
    """Model stages in the registry."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class ModelStatus(Enum):
    """Model status indicators."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    PENDING = "pending"

@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    model_name: str
    version: str
    stage: ModelStage
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    description: str
    tags: Dict[str, str]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    signature: Optional[Dict[str, Any]] = None
    input_example: Optional[Dict[str, Any]] = None

@dataclass
class ModelComparison:
    """Model comparison results."""
    baseline_model: str
    candidate_model: str
    baseline_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    improvement: Dict[str, float]
    recommendation: str
    confidence: float

class ModelRegistry:
    """
    Comprehensive model registry for managing ML models with MLflow.
    
    Features:
    - Model versioning and staging
    - Performance tracking and comparison
    - Automated model promotion
    - A/B testing support
    - Model lineage and metadata
    - Deployment management
    """
    
    def __init__(self, 
                 tracking_uri: str = None,
                 registry_uri: str = None,
                 experiment_name: str = "mongodb-query-translator"):
        """
        Initialize the model registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        
        # Initialize MLflow
        self._init_mlflow()
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        
        logger.info(f"Model registry initialized with experiment: {experiment_name}")
    
    def _init_mlflow(self):
        """Initialize MLflow configuration."""
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
            
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            logger.info("MLflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def register_model(self, 
                      model_name: str,
                      model_path: str,
                      run_id: str = None,
                      description: str = None,
                      tags: Dict[str, str] = None,
                      metrics: Dict[str, float] = None,
                      parameters: Dict[str, Any] = None,
                      signature: Dict[str, Any] = None,
                      input_example: Dict[str, Any] = None) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model artifacts
            run_id: MLflow run ID
            description: Model description
            tags: Model tags
            metrics: Model metrics
            parameters: Model parameters
            signature: Model signature
            input_example: Input example
            
        Returns:
            Model version
        """
        try:
            # Start MLflow run if not provided
            if run_id is None:
                with mlflow.start_run() as run:
                    run_id = run.info.run_id
                    
                    # Log parameters and metrics
                    if parameters:
                        mlflow.log_params(parameters)
                    if metrics:
                        mlflow.log_metrics(metrics)
                    if tags:
                        mlflow.set_tags(tags)
                    
                    # Log model
                    model_uri = mlflow.log_model(
                        artifact_path="model",
                        model_path=model_path,
                        signature=signature,
                        input_example=input_example
                    ).model_uri
            
            # Register model
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_path,
                run_id=run_id,
                description=description,
                tags=tags or {}
            )
            
            # Add metadata
            metadata = ModelMetadata(
                model_name=model_name,
                version=model_version.version,
                stage=ModelStage.NONE,
                status=ModelStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                description=description or "",
                tags=tags or {},
                metrics=metrics or {},
                parameters=parameters or {},
                signature=signature,
                input_example=input_example
            )
            
            # Store metadata in model version
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="metadata",
                value=json.dumps(asdict(metadata), default=str)
            )
            
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions with metadata
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            model_versions = []
            for version in versions:
                # Get metadata
                metadata_tag = version.tags.get("metadata")
                metadata = json.loads(metadata_tag) if metadata_tag else {}
                
                model_versions.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "status": version.status,
                    "created_at": version.creation_timestamp,
                    "description": version.description,
                    "tags": version.tags,
                    "metadata": metadata
                })
            
            return model_versions
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    def promote_model(self, 
                     model_name: str, 
                     version: str, 
                     stage: ModelStage,
                     comment: str = None) -> bool:
        """
        Promote a model to a specific stage.
        
        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage
            comment: Promotion comment
            
        Returns:
            Success status
        """
        try:
            # Transition model stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value,
                archive_existing_versions=True
            )
            
            # Add comment if provided
            if comment:
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="promotion_comment",
                    value=comment
                )
            
            # Update metadata
            self._update_model_metadata(model_name, version, {
                "stage": stage.value,
                "status": ModelStatus.ACTIVE.value,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Promoted model {model_name} version {version} to {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def compare_models(self, 
                      model_name: str,
                      baseline_stage: ModelStage,
                      candidate_stage: ModelStage) -> ModelComparison:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the model
            baseline_stage: Baseline model stage
            candidate_stage: Candidate model stage
            
        Returns:
            Model comparison results
        """
        try:
            # Get baseline model
            baseline_version = self.client.get_latest_versions(
                model_name, stages=[baseline_stage.value]
            )[0]
            
            # Get candidate model
            candidate_version = self.client.get_latest_versions(
                model_name, stages=[candidate_stage.value]
            )[0]
            
            # Get metrics for both models
            baseline_metrics = self._get_model_metrics(baseline_version)
            candidate_metrics = self._get_model_metrics(candidate_version)
            
            # Calculate improvement
            improvement = {}
            for metric in baseline_metrics:
                if metric in candidate_metrics:
                    improvement[metric] = candidate_metrics[metric] - baseline_metrics[metric]
            
            # Determine recommendation
            recommendation, confidence = self._evaluate_model_performance(
                baseline_metrics, candidate_metrics
            )
            
            comparison = ModelComparison(
                baseline_model=f"{model_name}:{baseline_version.version}",
                candidate_model=f"{model_name}:{candidate_version.version}",
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                improvement=improvement,
                recommendation=recommendation,
                confidence=confidence
            )
            
            logger.info(f"Model comparison completed: {recommendation}")
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def _get_model_metrics(self, model_version) -> Dict[str, float]:
        """Get metrics for a model version."""
        try:
            # Get run metrics
            run = self.client.get_run(model_version.run_id)
            metrics = run.data.metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {}
    
    def _evaluate_model_performance(self, 
                                   baseline_metrics: Dict[str, float],
                                   candidate_metrics: Dict[str, float]) -> tuple:
        """
        Evaluate model performance and provide recommendation.
        
        Args:
            baseline_metrics: Baseline model metrics
            candidate_metrics: Candidate model metrics
            
        Returns:
            Tuple of (recommendation, confidence)
        """
        try:
            # Key metrics to evaluate
            key_metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            improvements = []
            degradations = []
            
            for metric in key_metrics:
                if metric in baseline_metrics and metric in candidate_metrics:
                    diff = candidate_metrics[metric] - baseline_metrics[metric]
                    if diff > 0.01:  # 1% improvement threshold
                        improvements.append(diff)
                    elif diff < -0.01:  # 1% degradation threshold
                        degradations.append(abs(diff))
            
            # Calculate confidence based on consistency
            total_changes = len(improvements) + len(degradations)
            if total_changes == 0:
                return "no_change", 0.5
            
            improvement_ratio = len(improvements) / total_changes
            
            if improvement_ratio >= 0.7:
                return "promote", improvement_ratio
            elif improvement_ratio <= 0.3:
                return "reject", 1 - improvement_ratio
            else:
                return "investigate", 0.5
                
        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
            return "error", 0.0
    
    def _update_model_metadata(self, 
                              model_name: str, 
                              version: str, 
                              updates: Dict[str, Any]):
        """Update model metadata."""
        try:
            # Get current metadata
            metadata_tag = self.client.get_model_version_tag(
                name=model_name,
                version=version,
                key="metadata"
            )
            
            if metadata_tag:
                metadata = json.loads(metadata_tag.value)
                metadata.update(updates)
                
                # Update tag
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="metadata",
                    value=json.dumps(metadata, default=str)
                )
                
        except Exception as e:
            logger.error(f"Failed to update model metadata: {e}")
    
    def deploy_model(self, 
                    model_name: str, 
                    version: str = None,
                    stage: ModelStage = None,
                    deployment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deploy a model to production.
        
        Args:
            model_name: Name of the model
            version: Model version (if None, uses latest from stage)
            stage: Model stage (if None, uses production)
            deployment_config: Deployment configuration
            
        Returns:
            Deployment information
        """
        try:
            if stage is None:
                stage = ModelStage.PRODUCTION
            
            # Get model version
            if version is None:
                model_version = self.client.get_latest_versions(
                    model_name, stages=[stage.value]
                )[0]
                version = model_version.version
            
            # Get model URI
            model_uri = f"models:/{model_name}/{version}"
            
            # Deploy model (this would integrate with your deployment system)
            deployment_info = self._deploy_to_production(
                model_uri, deployment_config or {}
            )
            
            # Update deployment metadata
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="deployment_info",
                value=json.dumps(deployment_info, default=str)
            )
            
            logger.info(f"Deployed model {model_name} version {version}")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    def _deploy_to_production(self, 
                             model_uri: str, 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model to production environment.
        
        Args:
            model_uri: Model URI
            config: Deployment configuration
            
        Returns:
            Deployment information
        """
        # This is a placeholder for actual deployment logic
        # In practice, you would:
        # 1. Update Kubernetes deployments
        # 2. Update API configurations
        # 3. Run health checks
        # 4. Update load balancers
        
        deployment_info = {
            "model_uri": model_uri,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "deployment_config": config,
            "status": "deployed",
            "endpoint": f"https://api.company.com/models/{model_uri.split('/')[-1]}"
        }
        
        return deployment_info
    
    def get_model_performance_history(self, 
                                     model_name: str,
                                     days: int = 30) -> Dict[str, Any]:
        """
        Get model performance history.
        
        Args:
            model_name: Name of the model
            days: Number of days to look back
            
        Returns:
            Performance history data
        """
        try:
            # Get all versions
            versions = self.get_model_versions(model_name)
            
            # Filter by date range
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            history = {
                "model_name": model_name,
                "period_days": days,
                "versions": [],
                "performance_trends": {}
            }
            
            for version in versions:
                if version["created_at"] >= cutoff_date:
                    history["versions"].append(version)
            
            # Calculate trends
            if len(history["versions"]) > 1:
                history["performance_trends"] = self._calculate_performance_trends(
                    history["versions"]
                )
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return {}
    
    def _calculate_performance_trends(self, versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trends from version history."""
        try:
            trends = {}
            
            # Sort by creation date
            sorted_versions = sorted(versions, key=lambda x: x["created_at"])
            
            # Calculate trends for key metrics
            key_metrics = ["accuracy", "precision", "recall", "f1_score"]
            
            for metric in key_metrics:
                values = []
                for version in sorted_versions:
                    if metric in version.get("metadata", {}).get("metrics", {}):
                        values.append(version["metadata"]["metrics"][metric])
                
                if len(values) > 1:
                    # Calculate trend (simple linear regression slope)
                    x = np.arange(len(values))
                    y = np.array(values)
                    slope = np.polyfit(x, y, 1)[0]
                    
                    trends[metric] = {
                        "trend": "improving" if slope > 0 else "declining",
                        "slope": float(slope),
                        "latest_value": float(values[-1]),
                        "change_percentage": float((values[-1] - values[0]) / values[0] * 100)
                    }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to calculate performance trends: {e}")
            return {}
    
    def cleanup_old_models(self, 
                          model_name: str,
                          keep_versions: int = 5,
                          keep_stages: List[ModelStage] = None) -> int:
        """
        Clean up old model versions.
        
        Args:
            model_name: Name of the model
            keep_versions: Number of versions to keep
            keep_stages: Stages to preserve
            
        Returns:
            Number of versions cleaned up
        """
        try:
            if keep_stages is None:
                keep_stages = [ModelStage.PRODUCTION, ModelStage.STAGING]
            
            # Get all versions
            versions = self.get_model_versions(model_name)
            
            # Filter versions to clean up
            versions_to_cleanup = []
            for version in versions:
                if (version["stage"] not in [stage.value for stage in keep_stages] and
                    version["status"] != "ACTIVE"):
                    versions_to_cleanup.append(version)
            
            # Sort by creation date and keep only the most recent ones
            versions_to_cleanup.sort(key=lambda x: x["created_at"], reverse=True)
            versions_to_cleanup = versions_to_cleanup[keep_versions:]
            
            # Archive versions
            cleaned_count = 0
            for version in versions_to_cleanup:
                try:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=version["version"],
                        stage=ModelStage.ARCHIVED.value
                    )
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to archive version {version['version']}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old model versions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return 0
