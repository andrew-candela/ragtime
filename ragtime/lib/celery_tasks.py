from typing import Iterable

from ragtime.lib.celery_app import app
from celery.utils.log import get_task_logger


logger = get_task_logger(__name__)


@app.task
def add(x, y):
    logger.info(f"Adding something: {x, y}")
    return x + y


@app.task
def mul(x, y):
    return x * y


@app.task
def read_str_to_bytes(input_str: str) -> bytes:
    return input_str.encode()


@app.task
def print_bytes(input_bytes: bytes) -> str:

    return input_bytes.decode()


@app.task
def xsum(numbers: Iterable[int | float]):
    return sum(numbers)
