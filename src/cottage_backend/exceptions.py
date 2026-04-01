class BackendError(Exception):
    pass


class BackendConnectionError(BackendError):
    pass


class BackendResponseError(BackendError):
    pass