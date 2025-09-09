"""
Airflow DAG for Model Retraining Pipeline.
Automated model retraining based on feedback and performance metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

import sys
import os
sys.path.append('/opt/airflow/dags')

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import json

from mlops.feature_store.feature_store import FeatureStore
from utils.logger import get_logger

logger = get_logger(__name__)

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['mlops-team@company.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'catchup': False,
}

# DAG definition
dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining based on feedback and performance',
    schedule_interval='@weekly',  # Run weekly
    max_active_runs=1,
    tags=['mlops', 'retraining', 'model-update'],
)

def check_retraining_conditions(**context) -> Dict[str, Any]:
    """Check if retraining conditions are met."""
    try:
        # Get feedback data from the last week
        feedback_threshold = int(Variable.get("FEEDBACK_THRESHOLD", default_var="100"))
        performance_threshold = float(Variable.get("PERFORMANCE_THRESHOLD", default_var="0.85"))
        
        # Check feedback volume
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get feedback data (assuming it's stored in a feedback collection)
        feedback_collection = feature_store.db.feedback
        recent_feedback = feedback_collection.count_documents({
            "created_at": {"$gte": datetime.now() - timedelta(days=7)}
        })
        
        # Get current model performance
        current_performance = get_current_model_performance()
        
        retraining_conditions = {
            "feedback_count": recent_feedback,
            "feedback_threshold": feedback_threshold,
            "current_performance": current_performance,
            "performance_threshold": performance_threshold,
            "should_retrain": recent_feedback >= feedback_threshold or current_performance < performance_threshold,
            "reason": []
        }
        
        if recent_feedback >= feedback_threshold:
            retraining_conditions["reason"].append(f"High feedback volume: {recent_feedback}")
        
        if current_performance < performance_threshold:
            retraining_conditions["reason"].append(f"Low performance: {current_performance}")
        
        logger.info(f"Retraining conditions: {retraining_conditions}")
        return retraining_conditions
        
    except Exception as e:
        logger.error(f"Failed to check retraining conditions: {e}")
        raise

def get_current_model_performance() -> float:
    """Get current model performance metrics."""
    try:
        # This would typically query your monitoring system
        # For now, we'll simulate based on recent feedback
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get recent feedback scores
        feedback_collection = feature_store.db.feedback
        recent_feedback = list(feedback_collection.find({
            "created_at": {"$gte": datetime.now() - timedelta(days=7)},
            "rating": {"$exists": True}
        }))
        
        if not recent_feedback:
            return 0.9  # Default performance if no feedback
        
        # Calculate average rating
        ratings = [f["rating"] for f in recent_feedback if "rating" in f]
        if ratings:
            return sum(ratings) / len(ratings)
        
        return 0.9
        
    except Exception as e:
        logger.error(f"Failed to get current performance: {e}")
        return 0.9

def get_feature_store_config() -> Dict[str, Any]:
    """Get feature store configuration."""
    return {
        'mongodb_uri': Variable.get("MONGODB_URI"),
        'redis_host': Variable.get("REDIS_HOST", default_var="localhost"),
        'redis_port': int(Variable.get("REDIS_PORT", default_var="6379")),
        'mlflow_tracking_uri': Variable.get("MLFLOW_TRACKING_URI", default_var=None),
    }

def prepare_training_data(**context) -> Dict[str, Any]:
    """Prepare training data from feedback and query logs."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get feedback data
        feedback_collection = feature_store.db.feedback
        feedback_data = list(feedback_collection.find({
            "created_at": {"$gte": datetime.now() - timedelta(days=30)}
        }))
        
        # Get query logs
        query_logs_collection = feature_store.db.query_logs
        query_logs = list(query_logs_collection.find({
            "created_at": {"$gte": datetime.now() - timedelta(days=30)}
        }))
        
        # Prepare training dataset
        training_data = []
        
        for feedback in feedback_data:
            if "query_id" in feedback and "rating" in feedback:
                # Find corresponding query log
                query_log = next(
                    (q for q in query_logs if q.get("_id") == feedback["query_id"]), 
                    None
                )
                
                if query_log:
                    training_data.append({
                        "user_query": query_log.get("user_query", ""),
                        "generated_mql": query_log.get("generated_mql", {}),
                        "success": query_log.get("success", False),
                        "rating": feedback["rating"],
                        "feedback_type": feedback.get("feedback_type", "general")
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        if len(df) == 0:
            raise ValueError("No training data available")
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        data_info = {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "positive_feedback_ratio": (df["rating"] > 3).mean() if "rating" in df.columns else 0,
            "success_rate": df["success"].mean() if "success" in df.columns else 0
        }
        
        logger.info(f"Training data prepared: {data_info}")
        
        # Store in XCom
        return {
            "data_info": data_info,
            "train_data": train_df.to_dict('records'),
            "test_data": test_df.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        raise

def retrain_model(**context) -> Dict[str, Any]:
    """Retrain the model with new data."""
    try:
        # Get training data from previous task
        training_data = context['task_instance'].xcom_pull(task_ids='prepare_training_data')
        train_data = pd.DataFrame(training_data["train_data"])
        test_data = pd.DataFrame(training_data["test_data"])
        
        # Set up MLflow experiment
        mlflow.set_experiment("model-retraining")
        
        with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "training_samples": len(train_data),
                "test_samples": len(test_data),
                "retraining_trigger": "scheduled",
                "model_type": "query_translator"
            })
            
            # For this example, we'll create a simple model
            # In practice, you would retrain your actual LLM or fine-tune it
            
            # Simulate model training
            model_metrics = train_simple_model(train_data, test_data)
            
            # Log metrics
            mlflow.log_metrics(model_metrics)
            
            # Save model
            model_info = save_retrained_model(model_metrics)
            
            logger.info(f"Model retraining completed: {model_metrics}")
            
            return {
                "model_metrics": model_metrics,
                "model_info": model_info,
                "training_data_info": training_data["data_info"]
            }
            
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise

def train_simple_model(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, float]:
    """Train a simple model (placeholder for actual model training)."""
    # This is a placeholder - in practice, you would:
    # 1. Fine-tune your LLM on the new data
    # 2. Train a classification model for query success prediction
    # 3. Update prompt templates based on feedback
    
    # Simulate training metrics
    metrics = {
        "accuracy": 0.85 + np.random.normal(0, 0.02),
        "precision": 0.82 + np.random.normal(0, 0.02),
        "recall": 0.88 + np.random.normal(0, 0.02),
        "f1_score": 0.85 + np.random.normal(0, 0.02),
        "training_loss": 0.15 + np.random.normal(0, 0.01),
        "validation_loss": 0.18 + np.random.normal(0, 0.01)
    }
    
    return metrics

def save_retrained_model(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Save the retrained model to MLflow."""
    try:
        # Create a simple model wrapper
        class RetrainedQueryTranslator(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):
                # This would contain your actual retrained model logic
                return {"prediction": "retrained_model_output"}
        
        # Log the model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="retrained_model",
            python_model=RetrainedQueryTranslator(),
            registered_model_name="mongodb-query-translator"
        )
        
        return {
            "model_uri": model_info.model_uri,
            "model_version": "latest",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to save retrained model: {e}")
        raise

def validate_retrained_model(**context) -> Dict[str, Any]:
    """Validate the retrained model against test data."""
    try:
        # Get model info from previous task
        retraining_results = context['task_instance'].xcom_pull(task_ids='retrain_model')
        
        # Get test data
        training_data = context['task_instance'].xcom_pull(task_ids='prepare_training_data')
        test_data = pd.DataFrame(training_data["test_data"])
        
        # Load the retrained model
        model_uri = retraining_results["model_info"]["model_uri"]
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Validate on test data
        validation_results = validate_model_performance(model, test_data)
        
        # Check if model meets quality thresholds
        quality_threshold = float(Variable.get("MODEL_QUALITY_THRESHOLD", default_var="0.8"))
        model_quality_acceptable = validation_results["overall_score"] >= quality_threshold
        
        validation_summary = {
            "validation_results": validation_results,
            "quality_threshold": quality_threshold,
            "model_quality_acceptable": model_quality_acceptable,
            "recommendation": "deploy" if model_quality_acceptable else "reject"
        }
        
        logger.info(f"Model validation completed: {validation_summary}")
        return validation_summary
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise

def validate_model_performance(model, test_data: pd.DataFrame) -> Dict[str, Any]:
    """Validate model performance on test data."""
    # This is a placeholder for actual model validation
    # In practice, you would run the model on test data and calculate metrics
    
    validation_results = {
        "accuracy": 0.87 + np.random.normal(0, 0.01),
        "precision": 0.84 + np.random.normal(0, 0.01),
        "recall": 0.89 + np.random.normal(0, 0.01),
        "f1_score": 0.86 + np.random.normal(0, 0.01),
        "overall_score": 0.86 + np.random.normal(0, 0.01),
        "test_samples": len(test_data)
    }
    
    return validation_results

def deploy_model(**context) -> str:
    """Deploy the validated model to production."""
    try:
        # Get validation results
        validation_results = context['task_instance'].xcom_pull(task_ids='validate_retrained_model')
        
        if not validation_results["model_quality_acceptable"]:
            raise ValueError("Model quality below threshold, deployment rejected")
        
        # Get model info
        retraining_results = context['task_instance'].xcom_pull(task_ids='retrain_model')
        model_uri = retraining_results["model_info"]["model_uri"]
        
        # Deploy model (this would typically involve updating your serving infrastructure)
        deployment_result = deploy_to_production(model_uri)
        
        logger.info(f"Model deployed successfully: {deployment_result}")
        return deployment_result
        
    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise

def deploy_to_production(model_uri: str) -> str:
    """Deploy model to production environment."""
    # This is a placeholder for actual deployment logic
    # In practice, you would:
    # 1. Update your model serving endpoint
    # 2. Update Kubernetes deployments
    # 3. Update API configurations
    # 4. Run health checks
    
    return f"Model deployed from {model_uri}"

def send_retraining_notification(**context) -> str:
    """Send notification about retraining results."""
    try:
        # Get all results
        retraining_conditions = context['task_instance'].xcom_pull(task_ids='check_retraining_conditions')
        retraining_results = context['task_instance'].xcom_pull(task_ids='retrain_model')
        validation_results = context['task_instance'].xcom_pull(task_ids='validate_retrained_model')
        
        # Prepare notification message
        message = f"""
        🤖 Model Retraining Pipeline Completed
        
        *Retraining Trigger:*
        {', '.join(retraining_conditions['reason'])}
        
        *Training Results:*
        • Training samples: {retraining_results['training_data_info']['train_samples']}
        • Test samples: {retraining_results['training_data_info']['test_samples']}
        • Model accuracy: {retraining_results['model_metrics']['accuracy']:.3f}
        
        *Validation Results:*
        • Overall score: {validation_results['validation_results']['overall_score']:.3f}
        • Quality acceptable: {validation_results['model_quality_acceptable']}
        • Recommendation: {validation_results['recommendation']}
        """
        
        logger.info(f"Retraining notification: {message}")
        return message
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return "Notification failed"

# Task definitions
with TaskGroup("evaluation", dag=dag) as eval_group:
    check_conditions = PythonOperator(
        task_id='check_retraining_conditions',
        python_callable=check_retraining_conditions,
        provide_context=True,
    )

with TaskGroup("training", dag=dag) as training_group:
    prepare_data = PythonOperator(
        task_id='prepare_training_data',
        python_callable=prepare_training_data,
        provide_context=True,
    )
    
    retrain_model_task = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model,
        provide_context=True,
    )

with TaskGroup("validation", dag=dag) as validation_group:
    validate_model = PythonOperator(
        task_id='validate_retrained_model',
        python_callable=validate_retrained_model,
        provide_context=True,
    )

with TaskGroup("deployment", dag=dag) as deployment_group:
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        provide_context=True,
    )

with TaskGroup("notification", dag=dag) as notification_group:
    send_notification = PythonOperator(
        task_id='send_retraining_notification',
        python_callable=send_retraining_notification,
        provide_context=True,
    )
    
    slack_notification = SlackWebhookOperator(
        task_id='slack_retraining_notification',
        http_conn_id='slack_webhook',
        message='''
        🤖 Model Retraining Pipeline Completed
        
        *Results:*
        {{ task_instance.xcom_pull(task_ids='send_retraining_notification') }}
        ''',
        trigger_rule='all_success',
    )

# Task dependencies
eval_group >> training_group >> validation_group >> deployment_group >> notification_group

# Email notification on failure
email_on_failure = EmailOperator(
    task_id='email_retraining_failure',
    to=['mlops-team@company.com'],
    subject='Model Retraining Pipeline Failed',
    html_content='''
    <h2>Model Retraining Pipeline Failure</h2>
    <p>The model retraining pipeline has failed. Please check the Airflow logs for details.</p>
    <p><strong>DAG:</strong> {{ dag.dag_id }}</p>
    <p><strong>Task:</strong> {{ task_instance.task_id }}</p>
    <p><strong>Execution Date:</strong> {{ ds }}</p>
    ''',
    trigger_rule='one_failed',
    dag=dag,
)

# Set up failure notification
eval_group >> email_on_failure
training_group >> email_on_failure
validation_group >> email_on_failure
deployment_group >> email_on_failure
