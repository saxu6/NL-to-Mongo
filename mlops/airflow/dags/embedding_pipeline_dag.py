"""
Airflow DAG for MongoDB Query Translator Embedding Pipeline.
Automated data processing workflow with monitoring and error handling.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

import sys
import os
sys.path.append('/opt/airflow/dags')

from mlops.feature_store.feature_store import FeatureStore
from mlops.feature_store.embedding_pipeline import EmbeddingPipeline
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
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# DAG definition
dag = DAG(
    'mongodb_embedding_pipeline',
    default_args=default_args,
    description='Automated embedding generation for MongoDB Query Translator',
    schedule_interval='@daily',  # Run daily at midnight
    max_active_runs=1,
    tags=['mlops', 'embeddings', 'mongodb'],
)

def get_feature_store_config() -> Dict[str, Any]:
    """Get feature store configuration from Airflow variables."""
    return {
        'mongodb_uri': Variable.get("MONGODB_URI"),
        'redis_host': Variable.get("REDIS_HOST", default_var="localhost"),
        'redis_port': int(Variable.get("REDIS_PORT", default_var="6379")),
        'mlflow_tracking_uri': Variable.get("MLFLOW_TRACKING_URI", default_var=None),
    }

def initialize_feature_store(**context) -> str:
    """Initialize feature store and return status."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Test connections
        stats = feature_store.get_feature_statistics()
        logger.info(f"Feature store initialized successfully: {stats}")
        
        # Store in XCom for downstream tasks
        return "success"
        
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {e}")
        raise

def discover_schemas(**context) -> Dict[str, Any]:
    """Discover schemas for all collections."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get all collections
        collections = feature_store.db.list_collection_names()
        schema_results = {}
        
        for collection_name in collections:
            if collection_name.startswith('system.'):
                continue
                
            logger.info(f"Discovering schema for collection: {collection_name}")
            schema = feature_store.discover_schema(collection_name)
            schema_results[collection_name] = schema
        
        logger.info(f"Schema discovery completed for {len(schema_results)} collections")
        return schema_results
        
    except Exception as e:
        logger.error(f"Schema discovery failed: {e}")
        raise

def process_embeddings(**context) -> Dict[str, Any]:
    """Process embeddings for all collections."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        pipeline = EmbeddingPipeline(feature_store, batch_size=50, max_workers=4)
        
        # Get collections to process
        collections = Variable.get("EMBEDDING_COLLECTIONS", deserialize_json=True, default_var=[])
        if not collections:
            # Process all collections except system collections
            collections = [name for name in feature_store.db.list_collection_names() 
                          if not name.startswith('system.')]
        
        processing_results = {}
        
        for collection_name in collections:
            logger.info(f"Processing embeddings for collection: {collection_name}")
            
            # Check if collection has documents
            doc_count = feature_store.db[collection_name].count_documents({})
            if doc_count == 0:
                logger.info(f"Skipping empty collection: {collection_name}")
                continue
            
            # Process embeddings
            stats = pipeline.process_collection(
                collection_name=collection_name,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            processing_results[collection_name] = {
                'total_documents': stats.total_documents,
                'processed_documents': stats.processed_documents,
                'failed_documents': stats.failed_documents,
                'success_rate': stats.success_rate,
                'duration_seconds': stats.duration
            }
            
            logger.info(f"Completed processing {collection_name}: {stats.success_rate:.1f}% success rate")
        
        return processing_results
        
    except Exception as e:
        logger.error(f"Embedding processing failed: {e}")
        raise

def validate_embeddings(**context) -> Dict[str, Any]:
    """Validate generated embeddings."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get collections to validate
        collections = Variable.get("EMBEDDING_COLLECTIONS", deserialize_json=True, default_var=[])
        if not collections:
            collections = [name for name in feature_store.db.list_collection_names() 
                          if not name.startswith('system.')]
        
        validation_results = {}
        
        for collection_name in collections:
            logger.info(f"Validating embeddings for collection: {collection_name}")
            
            # Check if embeddings exist
            embedding_count = feature_store.embeddings_collection.count_documents({
                "collection_name": collection_name
            })
            
            if embedding_count == 0:
                logger.info(f"No embeddings found for collection: {collection_name}")
                continue
            
            # Validate embeddings
            validation = feature_store.validate_embeddings(collection_name, sample_size=100)
            validation_results[collection_name] = validation
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Embedding validation failed: {e}")
        raise

def generate_processing_report(**context) -> Dict[str, Any]:
    """Generate comprehensive processing report."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get processing results from previous tasks
        processing_results = context['task_instance'].xcom_pull(task_ids='process_embeddings')
        validation_results = context['task_instance'].xcom_pull(task_ids='validate_embeddings')
        
        # Generate reports for each collection
        reports = {}
        collections = Variable.get("EMBEDDING_COLLECTIONS", deserialize_json=True, default_var=[])
        if not collections:
            collections = [name for name in feature_store.db.list_collection_names() 
                          if not name.startswith('system.')]
        
        for collection_name in collections:
            report = feature_store.get_processing_report(collection_name)
            reports[collection_name] = report
        
        # Overall statistics
        overall_stats = {
            'total_collections': len(collections),
            'processing_results': processing_results,
            'validation_results': validation_results,
            'collection_reports': reports,
            'feature_store_stats': feature_store.get_feature_statistics(),
            'generated_at': datetime.now().isoformat()
        }
        
        # Store report in XCom
        return overall_stats
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

def cleanup_old_embeddings(**context) -> str:
    """Clean up old embeddings based on retention policy."""
    try:
        config = get_feature_store_config()
        feature_store = FeatureStore(**config)
        
        # Get retention days from variables
        retention_days = int(Variable.get("EMBEDDING_RETENTION_DAYS", default_var="90"))
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Delete old embeddings
        result = feature_store.embeddings_collection.delete_many({
            "created_at": {"$lt": cutoff_date}
        })
        
        logger.info(f"Cleaned up {result.deleted_count} old embeddings")
        return f"Cleaned up {result.deleted_count} old embeddings"
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise

# Task definitions
with TaskGroup("initialization", dag=dag) as init_group:
    init_feature_store = PythonOperator(
        task_id='initialize_feature_store',
        python_callable=initialize_feature_store,
        provide_context=True,
    )
    
    check_mongodb_connection = BashOperator(
        task_id='check_mongodb_connection',
        bash_command='python -c "from pymongo import MongoClient; import os; client = MongoClient(os.environ[\'MONGODB_URI\']); client.admin.command(\'ping\'); print(\'MongoDB connection successful\')"',
        env={'MONGODB_URI': Variable.get("MONGODB_URI")},
    )

with TaskGroup("data_processing", dag=dag) as processing_group:
    discover_schemas_task = PythonOperator(
        task_id='discover_schemas',
        python_callable=discover_schemas,
        provide_context=True,
    )
    
    process_embeddings_task = PythonOperator(
        task_id='process_embeddings',
        python_callable=process_embeddings,
        provide_context=True,
    )
    
    validate_embeddings_task = PythonOperator(
        task_id='validate_embeddings',
        python_callable=validate_embeddings,
        provide_context=True,
    )

with TaskGroup("reporting", dag=dag) as reporting_group:
    generate_report_task = PythonOperator(
        task_id='generate_processing_report',
        python_callable=generate_processing_report,
        provide_context=True,
    )
    
    # Slack notification (optional)
    slack_notification = SlackWebhookOperator(
        task_id='slack_notification',
        http_conn_id='slack_webhook',
        message='''
        :robot_face: MongoDB Embedding Pipeline Completed
        
        *Processing Results:*
        {% for collection, stats in task_instance.xcom_pull(task_ids='process_embeddings').items() %}
        • {{ collection }}: {{ stats.success_rate }}% success rate ({{ stats.processed_documents }}/{{ stats.total_documents }})
        {% endfor %}
        
        *Duration:* {{ task_instance.xcom_pull(task_ids='process_embeddings') | length }} collections processed
        ''',
        trigger_rule='all_success',
    )

with TaskGroup("cleanup", dag=dag) as cleanup_group:
    cleanup_old_embeddings_task = PythonOperator(
        task_id='cleanup_old_embeddings',
        python_callable=cleanup_old_embeddings,
        provide_context=True,
    )

# Task dependencies
init_group >> processing_group >> reporting_group >> cleanup_group

# Email notification on failure
email_on_failure = EmailOperator(
    task_id='email_on_failure',
    to=['mlops-team@company.com'],
    subject='MongoDB Embedding Pipeline Failed',
    html_content='''
    <h2>MongoDB Embedding Pipeline Failure</h2>
    <p>The embedding pipeline has failed. Please check the Airflow logs for details.</p>
    <p><strong>DAG:</strong> {{ dag.dag_id }}</p>
    <p><strong>Task:</strong> {{ task_instance.task_id }}</p>
    <p><strong>Execution Date:</strong> {{ ds }}</p>
    ''',
    trigger_rule='one_failed',
    dag=dag,
)

# Set up failure notification
init_group >> email_on_failure
processing_group >> email_on_failure
reporting_group >> email_on_failure
cleanup_group >> email_on_failure
