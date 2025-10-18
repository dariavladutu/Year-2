from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from datetime import datetime, timedelta
import logging

# Import your custom functions from the 'pipeline' package
from pipeline.preprocessing import preprocess_new_data
from pipeline.dataset_utils import merge_datasets
from pipeline.training import train_model_pipeline
from pipeline.model_utils import save_model_to_api
from pipeline.inference import predict_new_images

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'root_loop_pipeline',
    default_args=default_args,
    description='Looping pipeline for root segmentation training and prediction',
    schedule_interval='@daily',  # Or use None and trigger manually
    start_date=datetime(2025, 6, 1),
    catchup=False,
) as dag:

    def check_if_20_images_ready():
        import os
        image_dir = "/data/new_images"
        num_images = len([f for f in os.listdir(image_dir) if f.endswith(".png")])
        if num_images < 20:
            logging.warning(f"Only {num_images} new images found. Waiting for 20.")
            raise ValueError("Not enough images yet.")
        logging.info("✅ 20+ new images ready.")

    wait_for_images = PythonOperator(
        task_id='wait_for_images',
        python_callable=check_if_20_images_ready
    )

    preprocess = PythonOperator(
        task_id='preprocess_new_data',
        python_callable=preprocess_new_data
    )

    merge = PythonOperator(
        task_id='merge_datasets',
        python_callable=merge_datasets
    )

    retrain = PythonOperator(
        task_id='retrain_model',
        python_callable=train_model_pipeline
    )

    save_model = PythonOperator(
        task_id='save_model_to_api',
        python_callable=save_model_to_api
    )

    predict = PythonOperator(
        task_id='predict_new_images',
        python_callable=predict_new_images
    )

    loop = TriggerDagRunOperator(
        task_id='restart_loop',
        trigger_dag_id='root_loop_pipeline'
    )

    # Define the DAG flow
    wait_for_images >> preprocess >> merge >> retrain >> save_model >> predict >> loop
