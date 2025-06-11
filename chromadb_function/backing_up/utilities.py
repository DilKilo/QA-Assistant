import datetime


def get_current_timestamp() -> str:
    current_timestamp = datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    )

    return current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
