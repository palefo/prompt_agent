"""Study configuration and asset storage on the local filesystem.

Each study lives in ``STUDIES_DIR/<study_id>/`` and contains ``<study_id>.json``,
the six study images and the participant information sheet (``pis.pdf``).

``STUDIES_DIR`` defaults to the project folder (as before). In Docker it points
to a host folder mounted into the container, so studies created from /admin
survive rebuilds, restarts and ``docker compose down``.

Run ``python -m backend.studies seed`` to copy the studies bundled with the code
(``SEED_STUDIES``) into ``STUDIES_DIR`` when they are not there yet.
"""

import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES_DIR = os.path.abspath(os.environ.get("STUDIES_DIR") or PROJECT_ROOT)

# Study IDs become URL prefixes, so they must not clash with other routes.
RESERVED_STUDY_IDS = {
    "admin", "app", "web", "static", "healthz", "favicon.ico", "robots.txt",
}
_STUDY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

IMAGE_FILES = {
    "city_image": "city.jpg",
    "usecase_one_image": "usecase-one.jpg",
    "usecase_two_image": "usecase-two.jpg",
    "scenario_1_image": "scenario-one.jpg",
    "scenario_2_image": "scenario-two.jpg",
    "scenario_3_image": "scenario-three.jpg",
}
PIS_FILE = "pis.pdf"
PUBLIC_FILES = frozenset(IMAGE_FILES.values()) | {PIS_FILE}
MAX_IMAGE_SIDE = 2000


def is_valid_study_id(study_id: str | None) -> bool:
    return bool(study_id) and bool(_STUDY_ID.match(study_id)) and study_id not in RESERVED_STUDY_IDS


def study_dir(study_id: str) -> str:
    return os.path.join(STUDIES_DIR, study_id)


def _config_path(study_id: str) -> str:
    return os.path.join(study_dir(study_id), f"{study_id}.json")


def load_study_config(study_id: str | None) -> dict | None:
    if not is_valid_study_id(study_id):
        return None
    path = _config_path(study_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def list_study_ids() -> list[str]:
    if not os.path.isdir(STUDIES_DIR):
        return []
    return [
        name for name in sorted(os.listdir(STUDIES_DIR))
        if is_valid_study_id(name) and os.path.isfile(_config_path(name))
    ]


def file_path(study_id: str, filename: str) -> str | None:
    """Absolute path of a public study file, or None if it does not exist."""
    if filename not in PUBLIC_FILES or not is_valid_study_id(study_id):
        return None
    path = os.path.join(study_dir(study_id), filename)
    return path if os.path.isfile(path) else None


def _atomic_write(path: str, data: bytes) -> None:
    directory = os.path.dirname(path)
    handle, tmp_path = tempfile.mkstemp(dir=directory, prefix=".upload-")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def prepare_image(upload) -> bytes:
    """Validate an uploaded image and return it as an optimised JPEG.

    Raises ValueError if the file is not an image Pillow can read.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(upload.stream) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                image = image.convert("RGBA")
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            output = io.BytesIO()
            image.save(output, "JPEG", quality=85, optimize=True, progressive=True)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{upload.filename} is not a valid image.") from exc


def prepare_pdf(upload) -> bytes:
    data = upload.read()
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"{upload.filename} is not a valid PDF file.")
    return data


def save_study(study_id: str, config: dict, files: dict[str, bytes]) -> None:
    """Write the study config and any already-validated files to disk."""
    directory = study_dir(study_id)
    os.makedirs(directory, exist_ok=True)
    for filename, data in files.items():
        if filename not in PUBLIC_FILES:
            raise ValueError(f"Unexpected study file {filename}")
        _atomic_write(os.path.join(directory, filename), data)
    payload = json.dumps(config, indent=2, ensure_ascii=False).encode("utf-8")
    _atomic_write(_config_path(study_id), payload)


def seed_studies(names: list[str], source_root: str = PROJECT_ROOT) -> list[str]:
    """Copy bundled studies into STUDIES_DIR unless a study with that ID exists."""
    os.makedirs(STUDIES_DIR, exist_ok=True)
    seeded = []
    for name in names:
        source = os.path.join(source_root, name)
        target = study_dir(name)
        if not is_valid_study_id(name) or not os.path.isfile(os.path.join(source, f"{name}.json")):
            log.warning("Skipping seed study %r: no %s.json in %s", name, name, source)
            continue
        if os.path.abspath(source) == os.path.abspath(target) or os.path.exists(target):
            continue
        shutil.copytree(source, target)
        seeded.append(name)
    return seeded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if sys.argv[1:] != ["seed"]:
        sys.exit("usage: python -m backend.studies seed")
    wanted = [n.strip() for n in os.environ.get("SEED_STUDIES", "south_asia").split(",") if n.strip()]
    source = os.environ.get("SEED_STUDIES_FROM", PROJECT_ROOT)
    copied = seed_studies(wanted, source)
    print(f"Studies directory: {STUDIES_DIR}")
    print(f"Seeded: {', '.join(copied) if copied else 'nothing new'}")
    print(f"Available studies: {', '.join(list_study_ids()) or 'none'}")
