import nox

nox.options.default_venv_backend = "uv"

python_versions = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(name="tests", python=python_versions, reuse_venv=False)
def run_tests(session: nox.Session) -> None:
    """Run all pytest tests in the /tests/ folder."""
    session.run("uv", "sync", "--extra", "geo", "--active")
    session.run("pytest", "-s", "tests/")


NUMPY_VERSIONS = ["2.1", "2.2", "2.3"]

python_versions = ["3.12"]


@nox.session(name="test_numpy", python=python_versions, reuse_venv=False)
@nox.parametrize("numpy", NUMPY_VERSIONS)
def test_numpy(session: nox.Session, numpy: str) -> None:
    """Run tests with specific NumPy versions using uv."""
    # 1. Sync the project environment first
    session.run("uv", "sync", "--extra", "geo", "--active")

    # 2. Override with the specific NumPy version for this session
    # Using 'uv pip install' ensures the specific version is applied
    session.run("uv", "pip", "install", f"numpy=={numpy}")

    # 3. Run your tests
    session.run("pytest", "-s", "tests/")
