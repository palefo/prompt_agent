"""Test set-up: a throw-away PostgreSQL database and studies folder.

Run with scripts/test.sh (inside the app image, next to the db container).
The language model is never called: tests replace app.generate_reply.
"""

import os
import pathlib
import shutil
import sys
import tempfile

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDIES_TMP = pathlib.Path(tempfile.mkdtemp(prefix="studies-"))
shutil.copytree(ROOT / "south_asia", STUDIES_TMP / "south_asia")

TEST_DB = os.environ.get("TEST_POSTGRES_DB", "futures_test")
os.environ.update({
    "STUDIES_DIR": str(STUDIES_TMP),
    "POSTGRES_DB": TEST_DB,
    "ADMIN_PASSWORD": "test-password",
    "FLASK_SECRET_KEY": "test-secret",
    "SESSION_COOKIE_SECURE": "false",
})
os.environ.pop("DATABASE_URL", None)
# The agent is built at import time; it needs credentials but makes no call.
if os.environ.get("USE_AZURE", "false").lower() != "true":
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")

with psycopg.connect(
    make_conninfo(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname="postgres",
        user=os.environ.get("POSTGRES_USER", "futures"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    ),
    autocommit=True,
) as conn:
    conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(TEST_DB)))
    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))

import app as app_module  # noqa: E402
from backend import db as db_module  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    shutil.rmtree(STUDIES_TMP, ignore_errors=True)


@pytest.fixture
def app():
    app_module.app.config["TESTING"] = True
    return app_module.app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db():
    return db_module


class FakeLLM:
    """Scripted stand-in for the agent: asks questions, then completes."""

    def __init__(self):
        self.calls = []
        self.complete_after = 2  # user messages before the interview ends
        self.fail = False

    def __call__(self, history, thread_id, choices=None):
        self.calls.append({"history": list(history), "choices": dict(choices or {})})
        if self.fail:
            raise app_module.AgentUnavailable("simulated outage")
        answered = sum(1 for message in history if message["role"] == "user")
        if self.complete_after is not None and answered >= self.complete_after:
            summary = {"name": (choices or {}).get("name"), "answers": {"q1": "x"}}
            return "Thank you, your perspective helps shape what the future could become.", summary
        return f"Question {answered + 1}?", None


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(app_module, "generate_reply", fake)
    return fake
