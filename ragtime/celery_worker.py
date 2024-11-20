from ragtime.lib.celery_app import app


def main():
    chain = app.signature(
        "ragtime.lib.celery_tasks.add", kwargs={"x": 36, "y": -1}
    ) | app.signature("ragtime.lib.celery_tasks.add", kwargs={"y": 22})
    return chain()


if __name__ == "__main__":
    result = main()
    print(f"{result=}")
    print(f"{result.state=}")
    print(f"{result.get()=}")
    print(f"{result.state=}")
