from dvclive import Live
import mlflow

mflow.set_tracking_uri("http://localhost:5000")
mlflow.start_run()

with Live(log_mlflow=True) as live:
    live.log_metric("accuracy", 0.92)

mlflow.end_run()

