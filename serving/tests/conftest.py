import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module", autouse=True)
def _app_lifespan():
    """
    Starlette 1.x only runs lifespan (startup/shutdown) when TestClient is used
    as a context manager. This fixture wraps the app in a context manager so the
    lifespan runs (loading the Predictor, setting up DB tables, etc.) before any
    test in the module runs.
    """
    import test_main
    with TestClient(test_main.app) as c:
        test_main.client = c
        yield
