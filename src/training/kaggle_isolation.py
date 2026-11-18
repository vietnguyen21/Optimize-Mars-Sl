"""Create a project-only venv that borrows the image's protected CUDA stack."""
import json
from pathlib import Path
import sys
import venv

from src.training.kaggle_distributions import (
    active_distributions, ownership_index, protected_file_plan, verified_origins,
)


APPLICATION_DIR = 'application-wheels'
APPLICATION_RECEIPT = 'application-lock.sha256'
LAYOUT = 'image-runtime-application-overlay-v1'
PROVENANCE = 'kaggle-image.json'


def protected_import_paths(distributions, protected):
    """Run the retired bridge's strict ownership diagnostic without linking files."""
    ownership = ownership_index(distributions)
    for name in protected:
        verified_origins(name, distributions[name], ownership=ownership, allowed=protected)
    links, _ = protected_file_plan(ownership, protected, source=True)
    return links


def verify_bridge_files(bridge, protected, ownership):
    links, _ = protected_file_plan(ownership, protected)
    actual = {path for path in bridge.rglob('*') if path.is_symlink() or not path.is_dir()}
    if actual != set(links.values()) or any(not path.is_symlink() or not path.is_file() for path in actual):
        raise RuntimeError('Protected bridge has missing or unauthorized files')


def image_site_paths():
    return list(dict.fromkeys(str(Path(entry).resolve()) for entry in sys.path if entry and
                             Path(entry).name in {'site-packages', 'dist-packages'}))


def create_modern_venv(destination, *, baseline_output=None):
    from src.training.modern_environment import capture_image_baseline, canonical_json

    if not (sys.flags.isolated and sys.flags.dont_write_bytecode) or sys.prefix != sys.base_prefix:
        raise RuntimeError('Capture the image with the Kaggle interpreter using -I -B')
    destination = Path(destination).resolve() if destination else None
    if destination and (not destination.is_relative_to(Path('/kaggle/working')) or destination.exists()):
        raise RuntimeError('Choose a new directory under /kaggle/working for the modern venv')
    image = capture_image_baseline(image_site_paths())
    if baseline_output:
        path = Path(baseline_output)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'image-baseline.json').write_text(canonical_json(image))
    if destination:
        return create_application_venv(destination, image)
    return None


def create_application_venv(destination, image):
    from src.training.modern_environment import canonical_json

    destination.mkdir(parents=True)
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(destination)
    (destination / APPLICATION_DIR).mkdir()
    (destination / PROVENANCE).write_text(canonical_json(image))
    return destination / 'bin' / 'python'


def isolated_image():
    """Require the controlled interpreter. Image packages remain visible."""
    prefix = Path(sys.prefix)
    provenance = prefix / PROVENANCE
    if not (sys.flags.isolated and sys.flags.dont_write_bytecode and sys.prefix != sys.base_prefix
            and provenance.is_file() and (prefix / APPLICATION_DIR).is_dir()):
        raise RuntimeError('Run modern environment operations with the Kaggle application venv Python using -I -B')
    image = json.loads(provenance.read_text())
    if image.get('layout') != LAYOUT:
        raise RuntimeError('Recreate this venv using the image-runtime application layout')
    actual = [site for site in image_site_paths() if not Path(site).is_relative_to(prefix.resolve())]
    if actual != image['image_sites']:
        raise RuntimeError('Visible image site-packages differ from the captured interpreter')
    return image


def activate_application():
    """Add the approved prefix as a plain import path, without executing .pth files."""
    image = isolated_image()
    application = Path(sys.prefix) / APPLICATION_DIR
    from src.training.modern_environment import verify_application_prefix

    if any(application.iterdir()) and not (Path(sys.prefix) / APPLICATION_RECEIPT).is_file():
        raise RuntimeError('Application prefix has files without a reviewed installation receipt')
    verify_application_prefix(application, image)
    if str(application) not in sys.path:
        sys.path.insert(0, str(application))
    return image
