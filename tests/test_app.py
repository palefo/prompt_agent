import io
import json
import pathlib
import uuid

import pytest
from babel.messages.extract import extract_from_file
from babel.messages.pofile import read_po
from PIL import Image

from backend import studies
from backend.agent_service import _extract_json_object
from PromptBasedAgent import _load_system_prompt

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDY = "south_asia"


def new_pid():
    return f"TEST-{uuid.uuid4().hex[:10]}"


def start_participant(client, study=STUDY, pid=None):
    pid = pid or new_pid()
    response = client.post(
        f"/{study}/consent", data={"prolific_id": pid, "consent": "on", "pis": "on"}
    )
    assert response.status_code == 302, response.get_data(as_text=True)
    return pid


def fill_entry(client, study=STUDY, **overrides):
    data = {"name": "Priya", "age": "34", "gender": "Woman", "community": "Indian", **overrides}
    return client.post(f"/{study}?view=usecases", data=data)


def reach_chat(client, usecase="one", scenario="one"):
    assert fill_entry(client).status_code == 302
    assert client.post(f"/{STUDY}?view=scenario", data={usecase: ""}).status_code == 302
    assert client.post(f"/{STUDY}/chat", data={scenario: ""}).status_code == 302
    return client.get(f"/{STUDY}/chat")


# ── Infrastructure ──────────────────────────────────────────────────────────

def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_home_never_embeds_itself(client):
    own_host = client.get("/", headers={"Host": "futures.conversational-care.ai"})
    assert b"<embed" not in own_host.data
    other_host = client.get("/", headers={"Host": "example.vercel.app"})
    assert b"<embed" in other_host.data


def test_unknown_study_is_404(client):
    assert client.get("/no_such_study").status_code == 404
    assert client.get("/no_such_study/consent").status_code == 404


def test_only_public_study_files_are_served(client):
    assert client.get(f"/{STUDY}/pis.pdf").status_code == 200
    assert client.get(f"/{STUDY}/usecase-one.jpg").status_code == 200
    assert client.get(f"/{STUDY}/{STUDY}.json").status_code == 404
    assert client.get(f"/{STUDY}/../app.py").status_code == 404


def test_extract_json_object_handles_code_fences():
    text, parsed = _extract_json_object('Thank you!\n```json\n{"a": {"b": "}"}}\n```')
    assert text == "Thank you!"
    assert parsed == {"a": {"b": "}"}}
    assert _extract_json_object("No JSON here") == ("No JSON here", None)


def test_system_prompt_asks_the_study_questions():
    question = "What do you like most about this option for supporting yourself with diabetes?"
    prompt = _load_system_prompt({"agent_language": "Español", "initial": question})
    assert "agent_language := Español" in prompt
    assert "{{" not in prompt
    assert f'"initial": "{question}"' in prompt
    # The study's own question, not a sentence built from the use case title.
    assert "Q1 := initial" in prompt


# ── Participant flow ────────────────────────────────────────────────────────

def test_full_participant_flow(client, fake_llm, db):
    pid = start_participant(client)
    assert client.get(f"/{STUDY}").status_code == 200
    assert fill_entry(client).status_code == 302
    usecases = client.get(f"/{STUDY}?view=usecases").get_data(as_text=True)
    assert 'class="panel" id="cases"' in usecases

    assert client.post(f"/{STUDY}?view=scenario", data={"one": ""}).status_code == 302
    scenarios = client.get(f"/{STUDY}?view=scenario").get_data(as_text=True)
    assert 'class="panel" id="scenarios"' in scenarios
    assert "A personal diabetes companion" in scenarios

    assert client.post(f"/{STUDY}/chat", data={"two": ""}).status_code == 302
    chat = client.get(f"/{STUDY}/chat").get_data(as_text=True)
    assert "Question 1?" in chat
    choices = fake_llm.calls[0]["choices"]
    assert choices["scenario"] == "Safety"
    assert choices["scenario_title"] == "A responsive health support system"
    assert choices["initial"].startswith("What do you like most")
    assert choices["name"] == "Priya"
    assert "user_id" not in choices  # the Prolific ID is not sent to the model

    assert client.post(f"/{STUDY}/chat", data={"message": "First answer"}).status_code == 302
    assert client.post(f"/{STUDY}/chat", data={"message": "Second answer"}).status_code == 302
    done = client.get(f"/{STUDY}/chat").get_data(as_text=True)
    assert "Thank you for taking part!" in done
    assert 'id="composer"' not in done

    row = db.get_answer(pid, STUDY)
    assert (row["name"], row["age"], row["gender"], row["community"]) == ("Priya", 34, "Woman", "Indian")
    assert row["usecase"] == "Taking care of yourself"
    assert row["scenario"] == "Safety"
    assert row["final"].startswith("If we decided to build")
    assert row["agent_language"] == "English"
    assert [m["role"] for m in row["transcript"]] == ["assistant", "user", "assistant", "user", "assistant"]
    assert row["chatbot_summary"]["answers"]["q1"] == "x"
    assert row["completed_at"] is not None

    # Finished participants go straight to the thank-you page; extra messages are ignored.
    survey = client.get(f"/{STUDY}")
    assert survey.status_code == 302 and survey.location.endswith(f"/{STUDY}/chat")
    client.post(f"/{STUDY}/chat", data={"message": "one more"})
    assert len(db.get_answer(pid, STUDY)["transcript"]) == 5


def test_duplicate_prolific_id_is_rejected(client, app):
    pid = start_participant(client)
    other_browser = app.test_client()
    response = other_browser.post(
        f"/{STUDY}/consent", data={"prolific_id": pid, "consent": "on", "pis": "on"}
    )
    assert response.status_code == 409
    assert b"already been used" in response.data


def test_consent_requires_both_checkboxes(client):
    response = client.post(f"/{STUDY}/consent", data={"prolific_id": new_pid(), "consent": "on"})
    assert response.status_code == 400


def test_prolific_parameters_are_kept(client):
    response = client.get(f"/{STUDY}?PROLIFIC_PID=abc123&STUDY_ID=s1&SESSION_ID=x1")
    assert response.status_code == 302
    assert "PROLIFIC_PID=abc123" in response.location
    consent = client.get(response.location).get_data(as_text=True)
    assert 'value="abc123"' in consent


def test_entry_form_is_validated(client, db):
    pid = start_participant(client)
    response = client.post(f"/{STUDY}?view=usecases", data={"name": "Priya", "age": "34"})
    assert response.status_code == 302 and "view=entry" in response.location
    assert db.get_answer(pid, STUDY)["name"] is None
    assert fill_entry(client, age="25.5").status_code == 302
    assert db.get_answer(pid, STUDY)["age"] is None
    assert fill_entry(client, community="Martian").status_code == 302
    assert db.get_answer(pid, STUDY)["community"] is None


def test_back_navigation_and_language_switch_keep_the_step(client, fake_llm):
    start_participant(client)
    fill_entry(client)
    client.post(f"/{STUDY}?view=scenario", data={"two": ""})
    scenarios = client.get(f"/{STUDY}?view=scenario&lang=en").get_data(as_text=True)
    assert "A direct alert service" in scenarios
    back = client.get(f"/{STUDY}?view=usecases").get_data(as_text=True)
    assert 'class="panel" id="cases"' in back
    entry = client.get(f"/{STUDY}?view=entry").get_data(as_text=True)
    assert 'value="Priya"' in entry  # answers are pre-filled when going back


def test_changing_scenario_starts_a_new_conversation(client, fake_llm, db):
    pid = start_participant(client)
    reach_chat(client, scenario="one")
    client.post(f"/{STUDY}/chat", data={"message": "An answer"})
    assert len(db.get_answer(pid, STUDY)["transcript"]) == 3
    client.post(f"/{STUDY}/chat", data={"three": ""})
    assert db.get_answer(pid, STUDY)["transcript"] == []
    assert db.get_answer(pid, STUDY)["scenario"] == "Community & connection"


def test_llm_failure_keeps_the_participant_message(client, fake_llm, db):
    pid = start_participant(client)
    reach_chat(client)
    fake_llm.fail = True
    response = client.post(f"/{STUDY}/chat", data={"message": "My unsent answer"})
    assert response.status_code == 503
    page = response.get_data(as_text=True)
    assert "could not reply" in page
    assert "My unsent answer</textarea>" in page
    assert len(db.get_answer(pid, STUDY)["transcript"]) == 1  # only the opening message


def test_session_cookie_stays_small_in_long_conversations(client, fake_llm, db):
    fake_llm.complete_after = None
    pid = start_participant(client)
    reach_chat(client)
    for number in range(10):
        client.post(f"/{STUDY}/chat", data={"message": f"Answer {number} " + "word " * 200})
    assert len(db.get_answer(pid, STUDY)["transcript"]) == 21
    assert len(client.get_cookie("session").value) < 1000


def test_chat_requires_consent(client):
    response = client.get(f"/{STUDY}/chat")
    assert response.status_code == 302 and "/consent" in response.location


# ── Admin ───────────────────────────────────────────────────────────────────

def admin_login(client):
    client.get("/admin/login")
    with client.session_transaction() as session:
        token = session["csrf_token"]
    response = client.post("/admin/login", data={"password": "test-password", "csrf_token": token})
    assert response.status_code == 302
    return token


def study_form(study_id, **overrides):
    data = {
        "study_id": study_id,
        "header": "Help us design Testville",
        "future_city": "Testville",
        "text": "An intro text.",
        "language_options": "en: English\nes: Español",
        "user_id_list": "u1\nu2",
        "communities": "Alpha\nBeta",
        "completion_url": "https://app.prolific.com/submissions/complete?cc=TEST123",
    }
    for number in (1, 2):
        data[f"usecase_{number}_name"] = f"Use case {number}"
        data[f"usecase_{number}_description"] = f"Description {number}"
        for question in ("initial", "context", "final"):
            data[f"usecase_{number}_question_{question}"] = f"{question} question {number}?"
        for scenario in (1, 2, 3):
            data[f"scenario_{number}_{scenario}_name"] = f"Scenario {scenario}"
            data[f"scenario_{number}_{scenario}_short"] = f"Short {number}.{scenario}"
            data[f"scenario_{number}_{scenario}_detail"] = f"Detail {number}.{scenario}"
            data[f"scenario_{number}_{scenario}_imagine"] = f"Imagine {number}.{scenario}"
    data.update(overrides)
    return data


def image_upload(name):
    buffer = io.BytesIO()
    Image.new("RGBA", (64, 36), (214, 155, 45, 128)).save(buffer, "PNG")
    buffer.seek(0)
    return buffer, name


def all_uploads():
    files = {field: image_upload(f"{field}.png") for field in studies.IMAGE_FILES}
    files["pis_pdf"] = (io.BytesIO(b"%PDF-1.4\n% test\n"), "pis.pdf")
    return files


def test_admin_requires_login(client):
    response = client.get("/admin")
    assert response.status_code == 302 and "/admin/login" in response.location


def test_admin_wrong_password_and_missing_csrf(client):
    client.get("/admin/login")
    assert client.post("/admin/login", data={"password": "test-password"}).status_code == 400
    with client.session_transaction() as session:
        token = session["csrf_token"]
    response = client.post("/admin/login", data={"password": "nope", "csrf_token": token})
    assert b"Incorrect password" in response.data


def test_admin_creates_and_edits_a_study(client):
    token = admin_login(client)
    data = {**study_form("testville"), **all_uploads(), "csrf_token": token}
    response = client.post("/admin", data=data, content_type="multipart/form-data")
    assert response.status_code == 302 and "saved=testville" in response.location

    folder = pathlib.Path(studies.STUDIES_DIR) / "testville"
    assert all((folder / name).is_file() for name in studies.PUBLIC_FILES)
    with Image.open(folder / "city.jpg") as image:
        assert image.format == "JPEG"
    config = json.loads((folder / "testville.json").read_text())
    assert config["completion_url"].endswith("cc=TEST123")
    assert list(config["usecase_two"]["Use case 2"]["scenarios"]) == ["Scenario 1", "Scenario 2", "Scenario 3"]

    # Creating the same ID again must not overwrite the study.
    again = {**study_form("testville", header="Other"), **all_uploads(), "csrf_token": token}
    assert client.post("/admin", data=again, content_type="multipart/form-data").status_code == 409

    # Editing without uploads keeps the files and settings the form does not manage.
    config["extra_setting"] = 1
    (folder / "testville.json").write_text(json.dumps(config))
    edit = {**study_form("testville", header="New header"), "original_study_id": "testville", "csrf_token": token}
    assert client.post("/admin", data=edit, content_type="multipart/form-data").status_code == 302
    config = json.loads((folder / "testville.json").read_text())
    assert config["header"] == "New header" and config["extra_setting"] == 1
    assert (folder / "city.jpg").is_file()

    # The new study works for participants, in Spanish too.
    consent = client.get("/testville/consent?lang=es").get_data(as_text=True)
    assert "Información sobre la participación" in consent
    assert "New header" in consent


def test_admin_rejects_bad_ids_and_files(client):
    token = admin_login(client)
    for bad_id in ("admin", "app"):
        data = {**study_form(bad_id), **all_uploads(), "csrf_token": token}
        assert client.post("/admin", data=data, content_type="multipart/form-data").status_code == 400
    files = all_uploads()
    files["city_image"] = (io.BytesIO(b"<html>not an image</html>"), "city.jpg")
    data = {**study_form("badimage"), **files, "csrf_token": token}
    response = client.post("/admin", data=data, content_type="multipart/form-data")
    assert response.status_code == 400 and b"not a valid image" in response.data
    assert not (pathlib.Path(studies.STUDIES_DIR) / "badimage").exists()
    same_names = study_form("dupes", scenario_1_2_name="Scenario 1")
    data = {**same_names, **all_uploads(), "csrf_token": token}
    assert client.post("/admin", data=data, content_type="multipart/form-data").status_code == 400


def test_admin_responses_export_and_delete(client, fake_llm, db):
    pid = start_participant(client)
    fill_entry(client, name="=HYPERLINK(\"http://x\")")
    token = admin_login(client)

    listing = client.get(f"/admin/responses?study={STUDY}").get_data(as_text=True)
    assert pid in listing
    csv_text = client.get(f"/admin/export.csv?study={STUDY}").get_data(as_text=True)
    assert "'=HYPERLINK" in csv_text  # neutralised for spreadsheets
    exported = json.loads(client.get(f"/admin/export.json?study={STUDY}").data)
    assert any(row["user_id"] == pid for row in exported)

    row = db.get_answer(pid, STUDY)
    assert client.get(f"/admin/responses/{row['id']}").status_code == 200
    response = client.post(
        f"/admin/responses/{row['id']}/delete", data={"csrf_token": token, "study": STUDY}
    )
    assert response.status_code == 302
    assert db.get_answer(pid, STUDY) is None
    # The Prolific ID can be used again after the test response is deleted.
    start_participant(client.application.test_client(), pid=pid)


def test_admin_accepts_language_codes_with_a_region(client):
    token = admin_login(client)
    form = study_form("brasil", language_options="en: English\npt_br: Português (Brasil)")
    data = {**form, **all_uploads(), "csrf_token": token}
    assert client.post("/admin", data=data, content_type="multipart/form-data").status_code == 302
    languages = studies.load_study_config("brasil")["language_options"]
    assert languages[1] == {"code": "pt-BR", "name": "Português (Brasil)"}
    for bad in ("portuguese: Português", "pt-BRA: Português", "pt"):
        data = {**study_form("badlanguage", language_options=bad), **all_uploads(), "csrf_token": token}
        assert client.post("/admin", data=data, content_type="multipart/form-data").status_code == 400, bad


# ── Languages ───────────────────────────────────────────────────────────────

# The consent page heading in each interface language.
CONSENT_HEADINGS = {
    "en": "Participation information",
    "es": "Información sobre la participación",
    "pt-BR": "Informações sobre a participação",
}


@pytest.fixture
def trilingual():
    """The south_asia study, offered in English, Spanish and Brazilian Portuguese."""
    config = studies.load_study_config(STUDY)
    config["language_options"] = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Español"},
        {"code": "pt-BR", "name": "Português (Brasil)"},
    ]
    studies.save_study("trilingual", config, {})
    return "trilingual"


def test_each_interface_language_is_served(client, trilingual):
    for code, heading in CONSENT_HEADINGS.items():
        page = client.get(f"/{trilingual}/consent?lang={code}").get_data(as_text=True)
        assert heading in page, code
        assert f'<html lang="{code}">' in page


def test_language_choice_is_normalised_and_remembered(client, trilingual):
    client.get(f"/{trilingual}/consent?lang=pt_br")
    page = client.get(f"/{trilingual}/consent").get_data(as_text=True)
    assert CONSENT_HEADINGS["pt-BR"] in page
    assert '<html lang="pt-BR">' in page


def test_browser_language_picks_the_interface_language(app, trilingual):
    cases = {"pt-BR,pt;q=0.9": "pt-BR", "pt": "pt-BR", "es-AR,es;q=0.9": "es", "fr": "en"}
    for header, code in cases.items():
        page = app.test_client().get(f"/{trilingual}/consent", headers={"Accept-Language": header})
        assert CONSENT_HEADINGS[code] in page.get_data(as_text=True), header


def test_pages_outside_a_study_are_translated(client):
    page = client.get("/admin/login?lang=pt-BR").get_data(as_text=True)
    assert "Acesso de administração" in page
    assert '<html lang="pt-BR">' in page


def test_participant_flow_in_brazilian_portuguese(client, fake_llm, db, trilingual):
    pid = start_participant(client, study=trilingual)
    entry = client.get(f"/{trilingual}?lang=pt-BR").get_data(as_text=True)
    assert "conte-nos um pouco sobre você" in entry
    assert fill_entry(client, study=trilingual).status_code == 302
    usecases = client.get(f"/{trilingual}?view=usecases").get_data(as_text=True)
    assert "Escolha um caso de uso" in usecases
    assert client.post(f"/{trilingual}?view=scenario", data={"one": ""}).status_code == 302
    assert client.post(f"/{trilingual}/chat", data={"one": ""}).status_code == 302
    chat = client.get(f"/{trilingual}/chat").get_data(as_text=True)
    assert "Digite sua mensagem…" in chat
    # The assistant is told which language to speak.
    assert fake_llm.calls[0]["choices"]["agent_language"] == "Português (Brasil)"
    assert db.get_answer(pid, trilingual)["agent_language"] == "Português (Brasil)"


def _source_messages():
    """Every text marked for translation in app.py and the templates."""
    found = extract_from_file("python", ROOT / "app.py")
    for template in sorted((ROOT / "web").glob("*.html")):
        found += extract_from_file("jinja2.ext:babel_extract", template)
    return {message for _line, message, _comments, _context in found}


def test_every_interface_text_is_translated():
    source = _source_messages()
    assert CONSENT_HEADINGS["en"] in source
    for locale in ("es", "pt_BR"):
        with (ROOT / "translations" / locale / "LC_MESSAGES" / "messages.po").open("rb") as file:
            catalog = read_po(file)
        translated = {message.id for message in catalog if message.string and not message.fuzzy}
        assert not source - translated, (locale, sorted(source - translated))
        assert not list(catalog.check()), locale  # e.g. a missing %(name)s placeholder
