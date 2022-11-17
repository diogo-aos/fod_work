import mlflow
from typing import Optional

def mlflow_log_eval(logs: dict, prefix: str = '', step: Optional[int] = None):
    for k, v in logs.items():
        mlflow.log_metric(f'{prefix}{k}', v, step=step)