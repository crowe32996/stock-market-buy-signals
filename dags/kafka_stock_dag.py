from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import subprocess
import psycopg2
import logging
import sys
import os
import csv
from dotenv import load_dotenv
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
import pandas as pd



load_dotenv()

POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 26),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kafka_stock_pipeline',
    default_args=default_args,
    description='Run Kafka Producer & Consumer concurrently, then fetch and email results',
    schedule_interval='@daily',
    catchup=False,
    concurrency=4,
    max_active_runs=1,
)

def print_env_info():
    logging.info(f"Python path: {sys.executable}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Environment PATH: {os.environ.get('PATH')}")
    logging.info(f"User: {os.environ.get('USER')}")

def should_run_producer():
    return Variable.get("run_producer", default_var="true").lower() == "true"

def run_producer():
    if not should_run_producer():
        print("Skipping producer task as per config.")
        raise AirflowSkipException("Skipping producer task")
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/producer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=1800
        )
        logging.info(f"[Producer STDOUT]\n{result.stdout}")
        logging.error(f"[Producer STDERR]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Producer script timed out after 30 minutes")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Producer error (non-zero exit): {e.stderr}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running producer: {str(ex)}")
        raise

def should_run_consumer():
    return Variable.get("run_consumer", default_var="true").lower() == "true"

def run_consumer():
    if not should_run_consumer():
        print("Skipping producer task as per config.")
        raise AirflowSkipException("Skipping producer task")
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/consumer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=1800
        )
        logging.info(f"[Consumer STDOUT]\n{result.stdout}")
        logging.error(f"[Consumer STDERR]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Consumer script timed out after 30 minutes")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Consumer error (non-zero exit): {e.stderr}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running consumer: {str(ex)}")
        raise

def run_compute_indicators():
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/compute_indicators.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=1200
        )
        # Will only log if exit code is 0
        logging.info(f"[Compute STDOUT]\n{result.stdout}")
        logging.error(f"[Compute STDERR]\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error("❌ Compute Indicators script failed!")
        logging.error(f"STDOUT:\n{e.stdout}")
        logging.error(f"STDERR:\n{e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logging.error("Compute Indicators script timed out after 20 minutes")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running compute indicators: {str(ex)}")
        raise


def fetch_stock_data():
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT
    )

    df = pd.read_sql("""
        SELECT id, symbol, open_price, close_price, high_price, low_price, volume, date,
               SMA_10, SMA_20, SMA_50, RSI, MACD, Signal, buy_signal_short, buy_signal_long, buy_signal_longer
        FROM stock_data
        ORDER BY date DESC
        LIMIT 100000;
    """, conn)

    conn.close()

    result_file = "/opt/airflow/volumes/output/stock_buy_signals.csv"
    df.to_csv(result_file, index=False)
    return result_file


def run_analysis():
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/analyze_buy_signals.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=600
        )
        logging.info(f"[Analysis STDOUT]\n{result.stdout}")
        logging.error(f"[Analysis STDERR]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Analysis script timed out after 120 seconds")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Analysis error (non-zero exit): {e.stderr}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running analysis: {str(ex)}")
        raise

producer_task = PythonOperator(
    task_id='run_kafka_producer',
    python_callable=run_producer,
    dag=dag,
)

consumer_task = PythonOperator(
    task_id='run_kafka_consumer',
    python_callable=run_consumer,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,  # <-- runs even if upstream skipped
)

compute_indicators_task = PythonOperator(
    task_id='compute_indicators',
    python_callable=run_compute_indicators,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,  
)

# fetch_task = PythonOperator(
#     task_id='fetch_stock_data',
#     python_callable=fetch_stock_data,
#     dag=dag,
#     trigger_rule=TriggerRule.ALL_DONE,
# )

# analysis_task = PythonOperator(
#     task_id='run_analysis_script',
#     python_callable=run_analysis,
#     dag=dag,
#     trigger_rule=TriggerRule.ALL_DONE, 
# )

# email_task = EmailOperator(
#     task_id='send_email',
#     to='cwr321@gmail.com',
#     subject='Stock Data Report',
#     html_content='Attached is the latest stock data.',
#     files=[
#         '/opt/airflow/volumes/output/stock_buy_signals.csv',
#         '/opt/airflow/volumes/output/optimal_holding_periods_vs_baseline.csv',
#         '/opt/airflow/volumes/output/avg_return_short.png',
#         '/opt/airflow/volumes/output/win_rate_short.png',
#         '/opt/airflow/volumes/output/avg_return_long.png',
#         '/opt/airflow/volumes/output/win_rate_long.png',
#         '/opt/airflow/volumes/output/avg_return_longer.png',
#         '/opt/airflow/volumes/output/win_rate_longer.png',
#         '/opt/airflow/volumes/output/avg_return_outperformance_full.png',
#         '/opt/airflow/volumes/output/winrate_improvement_full.png'
#     ],
#     trigger_rule=TriggerRule.ALL_DONE,
#     dag=dag,
# )


producer_task >> consumer_task >> compute_indicators_task #>> fetch_task >> analysis_task >> email_task
