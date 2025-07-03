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

def run_producer():
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/producer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=300
        )
        logging.info(f"[Producer STDOUT]\n{result.stdout}")
        logging.error(f"[Producer STDERR]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Producer script timed out after 300 seconds")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Producer error (non-zero exit): {e.stderr}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running producer: {str(ex)}")
        raise

def run_consumer():
    try:
        print_env_info()
        result = subprocess.run(
            ["python3", "/opt/airflow/dags/scripts/consumer.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=300
        )
        logging.info(f"[Consumer STDOUT]\n{result.stdout}")
        logging.error(f"[Consumer STDERR]\n{result.stderr}")
    except subprocess.TimeoutExpired:
        logging.error("Consumer script timed out after 300 seconds")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Consumer error (non-zero exit): {e.stderr}")
        raise
    except Exception as ex:
        logging.error(f"Unexpected error running consumer: {str(ex)}")
        raise

def fetch_stock_data():
    conn = psycopg2.connect(
        dbname="stock_market_av",
        user="ec2-user",
        password="Lefevre102!",
        host="postgres",
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM stock_data ORDER BY date DESC LIMIT 1000;")
    rows = cursor.fetchall()
    conn.close()

    headers = [
        'id', 'symbol', 'open_price', 'close_price', 'high_price', 'low_price',
        'volume', 'date', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal', 'buy_signal'
    ]

    result_file = "/opt/airflow/volumes/output/stock_buy_signals.csv" 
    with open(result_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

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
            timeout=120
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
)

fetch_task = PythonOperator(
    task_id='fetch_stock_data',
    python_callable=fetch_stock_data,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
)

analysis_task = PythonOperator(
    task_id='run_analysis_script',
    python_callable=run_analysis,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
)

email_task = EmailOperator(
    task_id='send_email',
    to='cwr321@gmail.com',
    subject='Stock Data Report',
    html_content='Attached is the latest stock data.',
    files=[
        '/opt/airflow/volumes/output/stock_buy_signals.csv',
        '/opt/airflow/volumes/output/avg_return_by_buy_signal.png',
        '/opt/airflow/volumes/output/win_rate_by_buy_signal.png'
    ],
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)


producer_task >> consumer_task >> fetch_task >> analysis_task >> email_task