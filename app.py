import csv
import hmac
import io
import json
import logging
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from flask import (  # noqa: E402
    Flask, Response, abort, flash, g, redirect, render_template, request,
    send_from_directory, session, url_for,
)
from flask_babel import Babel, gettext as _  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from backend import db, studies  # noqa: E402
from backend.agent_service import (  # noqa: E402
    AgentUnavailable, generate_reply, make_thread_id, warmup_agent,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("futures")


def _env_flag(name, default="false"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


app = Flask(__name__, template_folder="web", static_folder="web")
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.environ.get(
    "OPENAI_API_KEY", "dev-secret-key"
    )
if not os.environ.get("FLASK_SECRET_KEY"):
    log.warning("FLASK_SECRET_KEY is not set; sessions use an insecure fallback key.")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

app.config['BABEL_DEFAULT_LOCALE'] = 'en'
# Pages outside a study (home, admin, errors). Each language except English
# needs a catalogue in translations/, e.g. translations/pt_BR for pt-BR.
app.config['LANGUAGES'] = {
    'en': 'English',
    'es': 'Español',
    'pt-BR': 'Português (Brasil)',
}
app.config.update(
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=_env_flag("SESSION_COOKIE_SECURE"),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=int(os.environ.get("SESSION_HOURS", "24"))),
)
if _env_flag("BEHIND_PROXY"):
    # Caddy terminates HTTPS; trust its X-Forwarded-* headers.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Panel classes used by index.html to show one step at a time.
view = "panel"
hide = "panel hidden"

USECASE_KEYS = {"one": "usecase_one", "two": "usecase_two"}
SCENARIO_KEYS = ("one", "two", "three")
USECASE_IMAGES = {"one": "usecase-one.jpg", "two": "usecase-two.jpg"}
SCENARIO_IMAGES = {"one": "scenario-one.jpg", "two": "scenario-two.jpg", "three": "scenario-three.jpg"}
GENDER_OPTIONS = ("Woman", "Man", "Non-binary", "Prefer not to say")
EXTRA_COMMUNITY_OPTIONS = ("Other", "Prefer not to say")
MAX_MESSAGE_LENGTH = 4000
# A language ("es") or a language and region ("pt-BR"), see _language_code.
LANGUAGE_CODE = re.compile(r"[a-z]{2}(-[A-Z]{2})?")
# Keys of a study config managed by the admin form (others are preserved on edit).
FORM_CONFIG_KEYS = {
    "header", "future_city", "text", "language_options", "user_id_list",
    "communities", "usecase_one", "usecase_two", "completion_url",
}

def _language_code(value):
    """Write a locale code the way HTML expects it: "PT_br" -> "pt-BR", "ES" -> "es"."""
    language, _sep, region = value.strip().replace("_", "-").partition("-")
    return f"{language.lower()}-{region.upper()}" if region else language.lower()

def _study_languages(config):
    """Return a locale-code map, accepting old display-name-only configs."""
    languages = {}
    for language in config.get("language_options", []):
        if isinstance(language, dict):
            code = _language_code(language.get("code", ""))
            name = language.get("name", "").strip()
        else:
            name = str(language).strip()
            code = {"English": "en", "Spanish": "es", "Español": "es"}.get(name, "")
        if code and name:
            languages[code] = name
    return languages or {"en": "English"}

def load_study_config(study_id):
    """Study config for this request (read from disk once per request)."""
    cache = g.setdefault("study_configs", {})
    if study_id not in cache:
        cache[study_id] = studies.load_study_config(study_id)
    return cache[study_id]

def _active_languages():
    study_id = request.view_args.get("study_id") if request.view_args else None
    if study_id:
        config = load_study_config(study_id)
        if config is not None:
            return _study_languages(config)
    return app.config["LANGUAGES"]

def get_locale():
    languages = _active_languages()
    # 1. If the user picked a language via ?lang=xx, store and use it.
    requested_language = _language_code(request.args.get('lang', ''))
    if requested_language in languages:
        session['lang'] = requested_language
        return session['lang']
    # 2. Otherwise reuse their stored choice.
    if session.get('lang') in languages:
        return session['lang']
    # 3. Otherwise fall back to the browser's preferred language.
    return request.accept_languages.best_match(languages.keys()) or next(iter(languages))

# Babel names locales "pt_BR" where HTML and the study configs use "pt-BR".
babel = Babel(app, locale_selector=lambda: get_locale().replace("-", "_"))

# Create the answers table if needed and initialise the agent once at start-up.
db.init_db()
warmup_agent()

# ── Helpers ──────────────────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc)

def list_studies():
    counts = db.study_counts()
    result = []
    for study_id in studies.list_study_ids():
        config = studies.load_study_config(study_id) or {}
        result.append({
            "id": study_id,
            "header": config.get("header", study_id),
            "url": url_for("survey", study_id=study_id),
            "external_url": url_for("survey", study_id=study_id, _external=True),
            "counts": counts.get(study_id, {"started": 0, "completed": 0}),
            "missing_files": sorted(
                name for name in studies.PUBLIC_FILES if not studies.file_path(study_id, name)
            ),
        })
    return result

def study_form_values(study_id, config):
    values = {
        "study_id": study_id,
        "header": config.get("header", ""),
        "future_city": config.get("future_city", ""),
        "text": config.get("text", ""),
        "completion_url": config.get("completion_url", ""),
        "language_options": "\n".join(
            f"{language['code']}: {language['name']}"
            if isinstance(language, dict) else str(language)
            for language in config.get("language_options", [])
        ),
        "user_id_list": "\n".join(config.get("user_id_list", [])),
        "communities": "\n".join(config.get("communities", [])),
    }
    for usecase_number, usecase_key in ((1, "usecase_one"), (2, "usecase_two")):
        usecase_name = next(iter(config.get(usecase_key, {})), "")
        usecase = config.get(usecase_key, {}).get(usecase_name, {})
        values[f"usecase_{usecase_number}_name"] = usecase_name
        values[f"usecase_{usecase_number}_description"] = usecase.get("description", "")
        questions = usecase.get("questions", {})
        for question in ("initial", "context", "final"):
            values[f"usecase_{usecase_number}_question_{question}"] = questions.get(question, "")
        scenarios = usecase.get("scenarios", {})
        for scenario_number, (scenario_name, scenario_values) in enumerate(scenarios.items(), 1):
            scenario_values = list(scenario_values) + ["", "", ""]
            values[f"scenario_{usecase_number}_{scenario_number}_name"] = scenario_name
            values[f"scenario_{usecase_number}_{scenario_number}_short"] = scenario_values[0]
            values[f"scenario_{usecase_number}_{scenario_number}_detail"] = scenario_values[1]
            values[f"scenario_{usecase_number}_{scenario_number}_imagine"] = scenario_values[2]
    return values

def _usecase(config, which):
    """(name, data) of use case "one" or "two" in a study config."""
    block = config.get(USECASE_KEYS[which]) or {}
    name = next(iter(block), "")
    return name, block.get(name) or {}

def _find_usecase(config, usecase_name):
    for which in USECASE_KEYS:
        name, usecase = _usecase(config, which)
        if usecase_name and name == usecase_name:
            return which, usecase
    return None, {}

def _scenario_values(usecase, scenario_name):
    """[short description, what it does, imagine sentence] of a scenario."""
    values = (usecase.get("scenarios") or {}).get(scenario_name or "", [])
    return (list(values) + ["", "", ""])[:3]

def _usecase_cards(config):
    tags = {"one": _("Use case one"), "two": _("Use case two")}
    cards = []
    for which in USECASE_KEYS:
        name, usecase = _usecase(config, which)
        cards.append({
            "key": which, "name": name, "tag": tags[which],
            "description": usecase.get("description", ""), "image": USECASE_IMAGES[which],
        })
    return cards

def _scenario_cards(usecase):
    tags = {"one": _("Scenario one"), "two": _("Scenario two"), "three": _("Scenario three")}
    cards = []
    for key, name in zip(SCENARIO_KEYS, usecase.get("scenarios") or {}):
        short, detail, imagine = _scenario_values(usecase, name)
        cards.append({
            "key": key, "name": name, "tag": tags[key], "title": short,
            "detail": detail, "imagine": imagine, "image": SCENARIO_IMAGES[key],
        })
    return cards

def _participant_id(study_id):
    """Prolific ID of the participant who consented to this study in this browser."""
    if session.get("participant_consent") == study_id:
        return session.get("prolific_id")
    return None

def _forget_participant():
    for key in ("participant_consent", "prolific_id", "session_seed", "study_id"):
        session.pop(key, None)

def _to_consent(study_id):
    # Keep the parameters Prolific appends to the study link.
    prolific_args = {
        key: request.args[key]
        for key in ("PROLIFIC_PID", "STUDY_ID", "SESSION_ID") if request.args.get(key)
    }
    return redirect(url_for("consent", study_id=study_id, **prolific_args))

def _thread_id():
    if "session_seed" not in session:
        session["session_seed"] = str(uuid.uuid4())
    return make_thread_id(session["session_seed"])

def sync_agent_language(current_user_id, study_id, answer):
    locale = get_locale()
    agent_language = _active_languages().get(locale, locale)
    if answer.get("agent_language") != agent_language:
        db.update_answer(current_user_id, study_id, agent_language=agent_language)
        answer["agent_language"] = agent_language
    return agent_language

def _agent_choices(answer, config):
    """Values substituted into / appended to the system prompt."""
    _which, usecase = _find_usecase(config, answer.get("usecase"))
    short, detail, imagine = _scenario_values(usecase, answer.get("scenario"))
    choices = {
        key: answer.get(key) or ""
        for key in ("name", "agent_language", "usecase", "scenario", "initial", "context", "final")
    }
    choices.update(
        usecase_description=usecase.get("description", ""),
        scenario_title=short,
        scenario_description=detail,
        scenario_imagine=imagine,
    )
    return choices

def _split_lines(value):
    return [line.strip() for line in value.splitlines() if line.strip()]

def _parse_languages(value):
    languages = []
    for line in _split_lines(value):
        code, separator, name = line.partition(":")
        code = _language_code(code)
        name = name.strip()
        if not separator or not LANGUAGE_CODE.fullmatch(code) or not name:
            raise ValueError(_(
                "Languages must use the format 'code: Language name', "
                "e.g. 'es: Español' or 'pt-BR: Português (Brasil)'."
            ))
        languages.append({"code": code, "name": name})
    if not languages:
        raise ValueError(_("At least one language is required."))
    if len({language["code"] for language in languages}) != len(languages):
        raise ValueError(_("Language codes must be unique."))
    return languages

def _study_form_data(form):

    def usecase(number):
        name = form[f"usecase_{number}_name"].strip()
        scenarios = {}
        for scenario_number in (1, 2, 3):
            scenarios[form[f"scenario_{number}_{scenario_number}_name"].strip()] = [
                form[f"scenario_{number}_{scenario_number}_short"].strip(),
                form[f"scenario_{number}_{scenario_number}_detail"].strip(),
                form[f"scenario_{number}_{scenario_number}_imagine"].strip(),
            ]
        if len(scenarios) != 3:
            raise ValueError(_("The three scenarios of a use case need different names."))
        return {
            name: {
                "description": form[f"usecase_{number}_description"].strip(),
                "scenarios": scenarios,
                "questions": {
                    "initial": form[f"usecase_{number}_question_initial"].strip(),
                    "context": form[f"usecase_{number}_question_context"].strip(),
                    "final": form[f"usecase_{number}_question_final"].strip(),
                },
            }
        }

    if form["usecase_1_name"].strip() == form["usecase_2_name"].strip():
        raise ValueError(_("The two use cases need different titles."))

    config = {
        "header": form["header"].strip(),
        "future_city": form["future_city"].strip(),
        "text": form["text"].strip(),
        "language_options": _parse_languages(form["language_options"]),
        "user_id_list": _split_lines(form["user_id_list"]),
        "communities": _split_lines(form["communities"]),
        "usecase_one": usecase(1),
        "usecase_two": usecase(2),
    }
    completion_url = form.get("completion_url", "").strip()
    if completion_url:
        parsed = urlparse(completion_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(_("The completion URL must be a full link starting with https://"))
        config["completion_url"] = completion_url
    return config

def _save_entry_form(current_user_id, study_id, config):
    """Validate and store the "about you" form. Returns an error message or None."""
    name = request.form.get("name", "").strip()
    gender = request.form.get("gender", "")
    community = request.form.get("community", "")
    try:
        age = int(request.form.get("age", ""))
    except ValueError:
        age = None
    if not name or len(name) > 100:
        return _("Please enter your name.")
    if age is None or not 1 <= age <= 120:
        return _("Please enter a valid age.")
    if gender not in GENDER_OPTIONS:
        return _("Please select your gender.")
    if community not in [*config.get("communities", []), *EXTRA_COMMUNITY_OPTIONS]:
        return _("Please select your community.")
    db.update_answer(
        current_user_id, study_id, name=name, age=age, gender=gender, community=community
    )
    return None

# ── Admin helpers ────────────────────────────────────────────────────────────

def csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = session["csrf_token"] = secrets.token_urlsafe(32)
    return token

def _check_csrf():
    sent = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not sent or not expected or not hmac.compare_digest(sent, expected):
        abort(400, description=_("This form has expired. Please reload the page and try again."))

def admin_required(view_function):
    @wraps(view_function)
    def wrapper(*args, **kwargs):
        if not ADMIN_PASSWORD:
            return "Admin password is not configured.", 503
        if not session.get("admin_authenticated"):
            target = request.full_path.rstrip("?") if request.method == "GET" else None
            return redirect(url_for("admin_login", next=target))
        return view_function(*args, **kwargs)
    return wrapper

def _safe_next(target):
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return None

def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, datetime):
        value = value.isoformat()
    # Stop spreadsheet programs from running participant text as a formula.
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        value = "'" + value
    return value

@app.template_global()
def study_file_url(study_id, filename):
    """URL of a study image/PDF (with a cache-busting version), or None if missing."""
    path = studies.file_path(study_id, filename)
    if not path:
        return None
    return url_for("study_asset", study_id=study_id, filename=filename, v=int(os.path.getmtime(path)))

app.jinja_env.globals["csrf_token"] = csrf_token

@app.context_processor
def inject_languages():
    # Makes `languages` and `current_language` available in every template.
    return {
        'languages': _active_languages(),
        'current_language': get_locale(),
    }

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/healthz")
def healthz():
    try:
        db.ping()
    except Exception:
        log.exception("Health check failed")
        return {"status": "error", "database": "unavailable"}, 503
    return {"status": "ok"}

@app.route("/", methods=["GET"])
def home():
    landing_url = os.environ.get("LANDING_URL", "https://futures.conversational-care.ai/").strip()
    if landing_url and urlparse(landing_url).netloc == request.host:
        landing_url = ""  # never embed this site inside itself
    return render_template("home.html", landing_url=landing_url)

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if not ADMIN_PASSWORD:
        return "Admin password is not configured.", 503
    if request.method == "POST":
        _check_csrf()
        password = request.form.get("password", "")
        if hmac.compare_digest(password.encode(), ADMIN_PASSWORD.encode()):
            session["admin_authenticated"] = True
            session.permanent = True
            return redirect(_safe_next(request.args.get("next")) or url_for("admin"))
        time.sleep(1)  # slow down password guessing
        flash(_("Incorrect password."), "error")
    return render_template("admin_login.html")

@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    _check_csrf()
    session.pop("admin_authenticated", None)
    return redirect(url_for("admin_login"))

@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "GET":
        edit_study_id = secure_filename(request.args.get("edit", "").strip().lower())
        edit_config = studies.load_study_config(edit_study_id) if edit_study_id else None
        form = study_form_values(edit_study_id, edit_config) if edit_config else None
        saved_study = request.args.get("saved", "")
        return render_template(
            "admin.html",
            form=form,
            editing_study=edit_study_id if edit_config else None,
            studies=list_studies(),
            created_study=saved_study if studies.load_study_config(saved_study) else None,
        )

    _check_csrf()
    original_study_id = secure_filename(request.form.get("original_study_id", "").strip().lower())
    study_id = original_study_id or secure_filename(request.form.get("study_id", "").strip().lower())
    existing_config = studies.load_study_config(study_id)

    def form_error(message, status=400):
        flash(message, "error")
        return render_template(
            "admin.html", form=request.form, editing_study=original_study_id or None,
            studies=list_studies(),
        ), status

    if not request.form.get("header", "").strip():
        return form_error(_("Study ID and header are required."))
    if not studies.is_valid_study_id(study_id):
        return form_error(_(
            "The study ID may only use lowercase letters, numbers, - and _, "
            "and cannot be a reserved word such as admin or app."
        ))
    if original_study_id and existing_config is None:
        return form_error(_("This study no longer exists."), 404)
    if not original_study_id and existing_config is not None:
        return form_error(_(
            "A study with this ID already exists. Choose another ID or edit the existing study."
        ), 409)

    required_fields = [
        "future_city", "text", "language_options", "user_id_list", "communities",
    ]
    for usecase_number in (1, 2):
        required_fields += [
            f"usecase_{usecase_number}_name", f"usecase_{usecase_number}_description",
            f"usecase_{usecase_number}_question_initial",
            f"usecase_{usecase_number}_question_context",
            f"usecase_{usecase_number}_question_final",
        ]
        for scenario_number in (1, 2, 3):
            required_fields += [
                f"scenario_{usecase_number}_{scenario_number}_name",
                f"scenario_{usecase_number}_{scenario_number}_short",
                f"scenario_{usecase_number}_{scenario_number}_detail",
                f"scenario_{usecase_number}_{scenario_number}_imagine",
            ]

    missing = [field for field in required_fields if not request.form.get(field, "").strip()]
    image_uploads = {
        field: request.files.get(field) for field in studies.IMAGE_FILES
    }
    missing_images = [] if existing_config else [
        field for field, upload in image_uploads.items() if not upload or not upload.filename
    ]
    pis_upload = request.files.get("pis_pdf")
    has_pis = studies.file_path(study_id, studies.PIS_FILE) is not None
    missing_pis = not has_pis and (not pis_upload or not pis_upload.filename)
    invalid_pis = pis_upload and pis_upload.filename and not pis_upload.filename.lower().endswith(".pdf")
    if missing:
        return form_error(_("Please complete every field."))
    if missing_images:
        return form_error(_("Upload all six study images."))
    if missing_pis:
        return form_error(_("Upload the participant information sheet as a PDF."))
    if invalid_pis:
        return form_error(_("The participant information sheet must be a PDF."))

    try:
        study_config = _study_form_data(request.form)
        files = {}
        for field, upload in image_uploads.items():
            if upload and upload.filename:
                files[studies.IMAGE_FILES[field]] = studies.prepare_image(upload)
        if pis_upload and pis_upload.filename:
            files[studies.PIS_FILE] = studies.prepare_pdf(pis_upload)
    except ValueError as error:
        return form_error(str(error))

    if existing_config:
        # Keep settings that the form does not manage.
        preserved = {k: v for k, v in existing_config.items() if k not in FORM_CONFIG_KEYS}
        study_config = {**preserved, **study_config}
    studies.save_study(study_id, study_config, files)

    if existing_config:
        flash(_("Study '%(study)s' was updated.", study=study_id), "success")
    else:
        flash(_("Study '%(study)s' was created.", study=study_id), "success")
    return redirect(url_for("admin", saved=study_id))

@app.route("/admin/responses")
@admin_required
def admin_responses():
    study_filter = request.args.get("study", "").strip() or None
    return render_template(
        "admin_responses.html",
        rows=db.list_answers(study_filter),
        study_filter=study_filter,
        study_ids=sorted(set(studies.list_study_ids()) | set(db.study_counts())),
    )

@app.route("/admin/responses/<int:answer_id>")
@admin_required
def admin_response(answer_id):
    row = db.get_answer_by_id(answer_id)
    if row is None:
        abort(404)
    summary = row.get("chatbot_summary")
    return render_template(
        "admin_response.html",
        row=row,
        summary_json=json.dumps(summary, indent=2, ensure_ascii=False) if summary is not None else "",
    )

@app.route("/admin/responses/<int:answer_id>/delete", methods=["POST"])
@admin_required
def admin_response_delete(answer_id):
    _check_csrf()
    row = db.get_answer_by_id(answer_id)
    if row is not None:
        db.delete_answer(answer_id)
        flash(_("Deleted the response of %(user)s.", user=row["user_id"]), "success")
    return redirect(url_for("admin_responses", study=request.form.get("study") or None))

@app.route("/admin/export.<fmt>")
@admin_required
def admin_export(fmt):
    if fmt not in ("csv", "json"):
        abort(404)
    study_filter = request.args.get("study", "").strip() or None
    rows = db.list_answers(study_filter)
    filename = f"responses-{study_filter or 'all'}-{_now():%Y%m%d-%H%M%S}.{fmt}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if fmt == "json":
        body = json.dumps(
            [{column: row.get(column) for column in db.EXPORT_COLUMNS} for row in rows],
            indent=2, ensure_ascii=False, default=str,
        )
        return Response(body, mimetype="application/json", headers=headers)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(db.EXPORT_COLUMNS)
    for row in rows:
        writer.writerow([_csv_cell(row.get(column)) for column in db.EXPORT_COLUMNS])
    # The BOM makes Excel open the file as UTF-8.
    return Response("﻿" + output.getvalue(), mimetype="text/csv", headers=headers)

@app.route("/<study_id>/consent", methods=["GET", "POST"])
def consent(study_id):
    current_study_config = load_study_config(study_id)
    if current_study_config is None:
        abort(404)
    if _participant_id(study_id):
        return redirect(url_for("survey", study_id=study_id))

    if request.method == "POST":
        prolific_id = request.form.get("prolific_id", "").strip()
        if (not prolific_id or request.form.get("consent") != "on"
                or request.form.get("pis") != "on"):
            flash(_("Please enter your Prolific ID and consent to participate."), "error")
            return render_template(
                "consent.html", study=study_id, study_config=current_study_config,
                prolific_id=prolific_id,
            ), 400
        if len(prolific_id) > 100:
            flash(_("Please check your Prolific ID."), "error")
            return render_template(
                "consent.html", study=study_id, study_config=current_study_config,
                prolific_id=prolific_id,
            ), 400

        if db.user_id_exists(prolific_id) or not db.create_answer(
            prolific_id, study_id, consent=True
        ):
            flash(_("This Prolific ID has already been used."), "error")
            return render_template(
                "consent.html", study=study_id, study_config=current_study_config,
                prolific_id=prolific_id,
            ), 409

        session.permanent = True
        session["participant_consent"] = study_id
        session["prolific_id"] = prolific_id
        session["session_seed"] = str(uuid.uuid4())
        return redirect(url_for("survey", study_id=study_id))

    return render_template(
        "consent.html",
        study=study_id,
        study_config=current_study_config,
        prolific_id=request.args.get("PROLIFIC_PID", "").strip())

@app.route("/<study_id>", methods=["GET", "POST"])
def survey(study_id):
    current_study_config = load_study_config(study_id)
    if current_study_config is None:
        abort(404)
    current_user_id = _participant_id(study_id)
    if not current_user_id:
        return _to_consent(study_id)
    answer = db.get_answer(current_user_id, study_id)
    if answer is None:  # e.g. the response was deleted from the admin page
        _forget_participant()
        return _to_consent(study_id)
    if answer.get("completed_at"):
        return redirect(url_for("chat", study_id=study_id))

    current_view = request.args.get("view", "entry")
    if current_view not in ("entry", "usecases", "scenario"):
        current_view = "entry"

    # Every POST stores its step and redirects, so reloading never re-submits.
    if request.method == "POST":
        if current_view == "usecases":  # the "about you" form was submitted
            error = _save_entry_form(current_user_id, study_id, current_study_config)
            if error:
                flash(error, "error")
                return redirect(url_for("survey", study_id=study_id, view="entry"))
        elif current_view == "scenario":  # a use case was chosen
            which = next((key for key in USECASE_KEYS if key in request.form), None)
            if which is None:
                return redirect(url_for("survey", study_id=study_id, view="usecases"))
            usecase_choice, usecase = _usecase(current_study_config, which)
            usecase_questions = usecase.get("questions") or {}
            fields = {
                "usecase": usecase_choice,
                "initial": usecase_questions.get("initial", ""),
                "context": usecase_questions.get("context", ""),
                "final": usecase_questions.get("final", ""),
            }
            if usecase_choice != answer.get("usecase"):
                fields.update(scenario=None, transcript=[], chatbot_summary=None)
            db.update_answer(current_user_id, study_id, **fields)
        return redirect(url_for("survey", study_id=study_id, view=current_view))

    # Only show a step once the previous one is done.
    if current_view != "entry" and not answer.get("name"):
        current_view = "entry"
    _which, chosen_usecase = _find_usecase(current_study_config, answer.get("usecase"))
    if current_view == "scenario" and not chosen_usecase:
        current_view = "usecases"

    sync_agent_language(current_user_id, study_id, answer)
    return render_template(
        "index.html",
        study = study_id,
        header = current_study_config.get("header", ""),
        future_city = current_study_config.get("future_city", ""),
        text = current_study_config.get("text", ""),
        entry_status = view if current_view == "entry" else hide,
        usecase_status = view if current_view == "usecases" else hide,
        scenario_status = view if current_view == "scenario" else hide,
        communities = current_study_config.get("communities", []),
        answer = answer,
        usecases = _usecase_cards(current_study_config),
        chosen_usecase = answer.get("usecase") if chosen_usecase else "",
        scenarios = _scenario_cards(chosen_usecase) if chosen_usecase else [],
    )

def _render_chat(study_id, config, answer, usecase, transcript, draft="", error=None, status=200):
    short, _detail, imagine = _scenario_values(usecase, answer.get("scenario"))
    return render_template(
        "chat.html",
        chat_history=transcript,
        thread_id=_thread_id(),
        study = study_id,
        header = config.get("header", ""),
        future_city = config.get("future_city", ""),
        text = config.get("text", ""),
        scenario = answer.get("scenario") or "",
        scenario_title = short,
        scenario_imagine = imagine,
        usecase = answer.get("usecase") or "",
        communities = config.get("communities", []),
        completed = answer.get("completed_at") is not None,
        completion_url = config.get("completion_url", ""),
        draft = draft,
        error = error,
    ), status

@app.route("/<study_id>/chat", methods=["GET", "POST"])
def chat(study_id):
    current_chat_study = load_study_config(study_id)
    if current_chat_study is None:
        abort(404)
    current_user_id = _participant_id(study_id)
    if not current_user_id:
        return _to_consent(study_id)
    answer = db.get_answer(current_user_id, study_id)
    if answer is None:
        _forget_participant()
        return _to_consent(study_id)
    session['study_id'] = study_id
    sync_agent_language(current_user_id, study_id, answer)
    _which, usecase = _find_usecase(current_chat_study, answer.get("usecase"))
    completed = answer.get("completed_at") is not None
    chat_url = url_for("chat", study_id=study_id, _anchor="chat")

    if request.method == "POST":
        if completed:
            return redirect(chat_url)

        # Coming from the scenario cards: store the choice and start a fresh chat.
        choice = next((key for key in SCENARIO_KEYS if key in request.form), None)
        if choice is not None:
            scenario_names = list((usecase.get("scenarios") or {}).keys())
            index = SCENARIO_KEYS.index(choice)
            if index >= len(scenario_names):
                return redirect(url_for("survey", study_id=study_id, view="usecases"))
            scenario_choice = scenario_names[index]
            if scenario_choice != answer.get("scenario"):
                db.update_answer(
                    current_user_id, study_id,
                    scenario=scenario_choice, transcript=[], chatbot_summary=None,
                )
                session["session_seed"] = str(uuid.uuid4())
            return redirect(chat_url)

        # New user input
        message = (request.form.get("message") or "").strip()[:MAX_MESSAGE_LENGTH]
        if not message or not answer.get("scenario"):
            return redirect(chat_url)
        history = list(answer.get("transcript") or [])
        history.append({"role": "user", "content": message, "at": _now().isoformat()})
        try:
            response_text, response_json = generate_reply(
                history, _thread_id(), _agent_choices(answer, current_chat_study)
            )
        except AgentUnavailable:
            return _render_chat(
                study_id, current_chat_study, answer, usecase,
                answer.get("transcript") or [], draft=message,
                error=_("Sorry, the assistant could not reply just now. Please press Send again."),
                status=503,
            )
        if response_text:
            history.append({"role": "assistant", "content": response_text, "at": _now().isoformat()})
        fields = {"transcript": history}
        # Structured JSON means the interview is complete.
        if response_json is not None:
            fields.update(chatbot_summary=response_json, completed_at=_now())
        db.update_answer(current_user_id, study_id, **fields)
        return redirect(chat_url)

    if not usecase:
        return redirect(url_for("survey", study_id=study_id, view="usecases"))
    if not answer.get("scenario"):
        return redirect(url_for("survey", study_id=study_id, view="scenario"))

    transcript = answer.get("transcript") or []
    error = None
    if not transcript and not completed:
        # The assistant opens the conversation.
        try:
            initial_text, _initial_json = generate_reply(
                [], _thread_id(), _agent_choices(answer, current_chat_study)
            )
        except AgentUnavailable:
            error = _("Sorry, the assistant is not available right now. Please reload the page in a moment.")
        else:
            transcript = [{"role": "assistant", "content": initial_text, "at": _now().isoformat()}]
            db.update_answer(current_user_id, study_id, transcript=transcript)
    return _render_chat(study_id, current_chat_study, answer, usecase, transcript, error=error)

@app.route("/<study_id>/chat/clear", methods=['GET','POST'])
def clear_chat(study_id):
    if load_study_config(study_id) is None:
        abort(404)
    current_user_id = _participant_id(study_id)
    if not current_user_id:
        return _to_consent(study_id)
    answer = db.get_answer(current_user_id, study_id)
    if answer and not answer.get("completed_at"):
        db.update_answer(current_user_id, study_id, transcript=[], chatbot_summary=None)
    session["session_seed"] = str(uuid.uuid4())
    return redirect(url_for("chat", study_id=study_id))

@app.route("/<study_id>/clear", methods=['GET','POST'])
def clear_session(study_id):
    if load_study_config(study_id) is None:
        abort(404)
    _forget_participant()
    return redirect(url_for("consent", study_id=study_id))

@app.route("/<study_id>/<path:filename>")
def study_asset(study_id, filename):
    path = studies.file_path(study_id, filename)
    if path is None or load_study_config(study_id) is None:
        abort(404)
    # Versioned URLs (?v=mtime) can be cached for long; they change on upload.
    max_age = 7 * 24 * 3600 if request.args.get("v") else 300
    return send_from_directory(studies.study_dir(study_id), filename, max_age=max_age)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon")

@app.errorhandler(400)
def bad_request(error):
    return render_template(
        "error.html", title=_("Something is not right"),
        message=getattr(error, "description", None) or _("Please go back and try again."),
    ), 400

@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html", title=_("Page not found"),
        message=_("We could not find this page. Please check the link you were given."),
    ), 404

@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "error.html", title=_("Upload too large"),
        message=_("The files are too large (%(size)s MB in total at most). Please use smaller images.",
                  size=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)),
    ), 413

@app.errorhandler(500)
def server_error(_error):
    return render_template(
        "error.html", title=_("Something went wrong"),
        message=_("Please try again in a moment."),
    ), 500

if __name__ == "__main__":
    app.run( use_reloader=False) #debug=True,
