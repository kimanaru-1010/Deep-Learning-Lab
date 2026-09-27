import pytest

from minidl.utils.student import StudentTODO


def pytest_addoption(parser):
    parser.addoption(
        "--run-student", action="store_true", help="Treat unfinished lessons as failures"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-student"):
        return
    for item in items:
        if item.get_closest_marker("student"):
            item.add_marker(
                pytest.mark.xfail(
                    raises=StudentTODO,
                    strict=False,
                    reason="Student exercise still raises an explicit TODO; use --run-student while learning",
                )
            )
