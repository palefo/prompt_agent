# AI Futures – Goal Reflection Agent

A Flask + LangGraph study platform: participants give consent, tell us a little
about themselves, pick a use case and a scenario, and then talk to an AI
interviewer. Researchers create studies and download the answers from `/admin`.

- **Deployment (Docker, HTTPS, database, backups):** [DEPLOYMENT.md](DEPLOYMENT.md)
- **User manual with screenshots (participants and admins):** [docs/USER_MANUAL.md](docs/USER_MANUAL.md)

---

## Architecture

```
app.py                      ← Flask entry point (participant flow + /admin)
PromptBasedAgent.py         ← LangGraph agent (Azure OpenAI or OpenAI)
backend/
  agent_service.py          ← calls the agent, detects the final JSON summary
  db.py                     ← PostgreSQL storage of answers and transcripts
  studies.py                ← study configs, images and PDFs on disk
prompts/
  agent.prompt              ← System prompt
web/                        ← Jinja templates, CSS
translations/               ← Babel catalogues (Spanish, Brazilian Portuguese)
south_asia/                 ← bundled study (seeded into the data folder)
Dockerfile, docker-compose.yml, deploy/caddy/  ← production stack
tests/                      ← pytest suite (scripts/test.sh)
```

### Thread IDs (UUID5)

Each conversation is keyed by a **deterministic UUID5** derived from a random
per-browser-session seed: `uuid.uuid5(uuid.NAMESPACE_DNS, seed)`.

---

## Run with Docker (recommended)

```bash
cp .env.example .env        # fill in the secrets
docker compose up -d --build
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the details.

## Local development without Docker

You need Python 3.12 and a PostgreSQL server (for example
`docker compose up -d db` and `POSTGRES_HOST=localhost` with the port published).

```bash
pip install -r requirements.txt
cp .env.example .env        # at minimum: POSTGRES_*, ADMIN_PASSWORD, LLM settings
pybabel compile -d translations
flask --app app.py --debug run
```

Studies are read from and written to `STUDIES_DIR` (default: the project folder).

Run the tests (inside Docker, against a separate test database):

```bash
scripts/test.sh
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | ✅ | Signs the session cookie |
| `ADMIN_PASSWORD` | ✅ | Password required for `/admin` |
| `POSTGRES_PASSWORD` (+ `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_HOST`) or `DATABASE_URL` | ✅ | Database connection |
| `USE_AZURE` | | `true` → Azure OpenAI, `false` → OpenAI |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT` | with Azure | Azure OpenAI settings |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | without Azure | OpenAI settings (default model `gpt-4.1-mini`) |
| `STUDIES_DIR` | | Folder holding the studies (Docker: `/data/studies`) |
| `DOMAIN`, `LANDING_DIR`, `DATA_DIR` | Docker only | See [DEPLOYMENT.md](DEPLOYMENT.md) |

All variables are listed with explanations in [.env.example](.env.example).
Supabase is no longer used.

The admin study builder is protected by `ADMIN_PASSWORD`. The app fails closed
with a `503` response if the variable is missing.

## Study languages

The interface is available in English (`en`), Spanish (`es`) and Brazilian
Portuguese (`pt-BR`). When creating a study, enter one language per line as a
two-letter code, optionally with a region, followed by the display name:

```text
en: English
es: Español
pt-BR: Português (Brasil)
```

The participant page builds its language selector from the active study's
`language_options`. Participants get their browser's language if the study
offers it, otherwise the first one. Translated interface text needs a Babel
catalogue under `translations/<code>/LC_MESSAGES/`, written with an underscore
(`translations/pt_BR/` for `pt-BR`); without one, the interface falls back to
English. Use `pt-BR`, not `pt`: a plain `pt` does not pick up the Brazilian
catalogue. The study texts themselves (use cases, scenarios, questions) are
shown as entered; the AI assistant replies in the selected language.

After changing texts in `app.py` or the templates, update the catalogues:

```bash
pybabel extract -F babel.cfg -o messages.pot .
pybabel update -i messages.pot -d translations
# translate the new entries in every translations/<code>/LC_MESSAGES/messages.po
pybabel compile -d translations
```

The test suite fails if a text has no translation in the `es` or `pt_BR`
catalogue. Both catalogues were machine-written; have a native speaker review
them before running a study in that language.
