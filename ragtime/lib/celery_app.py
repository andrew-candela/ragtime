import os
from celery import Celery

app = Celery(
    broker=os.environ["REDIS_HOST"],
    backend=(
        f"db+postgresql://{os.environ['POSTGRES_USER']}"
        f":{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}"
        f"/{os.environ['POSTGRES_DB']}"
    ),
)

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["application/json"],
    broker_connection_retry_on_startup=False,
)
