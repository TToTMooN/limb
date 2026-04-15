"""Upload datasets to cloud storage (S3, GCS, or HuggingFace Hub).

Auth follows standard SDK credential chains — no limb-specific secrets.
  - S3: AWS_PROFILE, AWS_ACCESS_KEY_ID, or IAM role
  - GCS: GOOGLE_APPLICATION_CREDENTIALS, gcloud auth, or service account
  - HuggingFace: HF_TOKEN env var or `huggingface-cli login`

Storage target is specified as a URI:
  - s3://bucket/prefix/dataset
  - gs://bucket/prefix/dataset
  - hf://username/dataset-name

Or via a config file at ~/.config/limb/storage.yaml::

    default: s3://my-bucket/datasets
    # Or per-task overrides:
    targets:
      pick_cube: gs://lab-bucket/pick_cube
      pour_water: s3://other-bucket/pour_water

Usage:
    uv run limb upload --source datasets/task_wds --target s3://bucket/prefix
    uv run limb upload --source datasets/task_lerobot --target gs://bucket/prefix
    uv run limb upload --source datasets/task --target hf://username/dataset-name
    uv run limb upload --source datasets/task  # uses default from storage.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from loguru import logger


def _load_storage_config() -> dict:
    """Load storage config from ~/.config/limb/storage.yaml if it exists."""
    config_path = Path.home() / ".config" / "limb" / "storage.yaml"
    if not config_path.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        conf = OmegaConf.load(str(config_path))
        return OmegaConf.to_container(conf, resolve=True)
    except Exception as e:
        logger.warning("Failed to load storage config: {}", e)
        return {}


def _resolve_target(target: Optional[str], task_name: Optional[str] = None) -> str:
    """Resolve upload target from explicit arg, task override, or default config."""
    if target:
        return target

    config = _load_storage_config()

    # Check task-specific override
    if task_name and "targets" in config:
        if task_name in config["targets"]:
            resolved = config["targets"][task_name]
            logger.info("Using task-specific target from storage.yaml: {}", resolved)
            return resolved

    # Use default
    if "default" in config:
        resolved = config["default"]
        logger.info("Using default target from storage.yaml: {}", resolved)
        return resolved

    logger.error(
        "No upload target specified. Pass --target or create ~/.config/limb/storage.yaml with a 'default' key."
    )
    raise SystemExit(1)


def _upload_s3(source: Path, bucket: str, prefix: str) -> None:
    """Upload directory to S3 using boto3."""
    try:
        import boto3
    except ImportError:
        logger.error("boto3 not installed. Run: uv add boto3")
        raise SystemExit(1) from None

    s3 = boto3.client("s3")
    files = sorted(f for f in source.rglob("*") if f.is_file())
    logger.info("Uploading {} files to s3://{}/{}", len(files), bucket, prefix)

    for i, f in enumerate(files):
        key = f"{prefix}/{f.relative_to(source)}"
        s3.upload_file(str(f), bucket, key)
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            logger.info("  {}/{} uploaded", i + 1, len(files))

    logger.info("Upload complete: s3://{}/{}", bucket, prefix)


def _upload_gcs(source: Path, bucket: str, prefix: str) -> None:
    """Upload directory to GCS using google-cloud-storage."""
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: uv add google-cloud-storage")
        raise SystemExit(1) from None

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    files = sorted(f for f in source.rglob("*") if f.is_file())
    logger.info("Uploading {} files to gs://{}/{}", len(files), bucket, prefix)

    for i, f in enumerate(files):
        blob_name = f"{prefix}/{f.relative_to(source)}"
        blob = bucket_obj.blob(blob_name)
        blob.upload_from_filename(str(f))
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            logger.info("  {}/{} uploaded", i + 1, len(files))

    logger.info("Upload complete: gs://{}/{}", bucket, prefix)


def _upload_hf(source: Path, repo_id: str) -> None:
    """Upload directory to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Run: uv add huggingface-hub")
        raise SystemExit(1) from None

    api = HfApi()
    logger.info("Uploading to HuggingFace Hub: {}", repo_id)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(source), repo_id=repo_id, repo_type="dataset")
    logger.info("Uploaded: https://huggingface.co/datasets/{}", repo_id)


def upload(source: str, target: str) -> None:
    """Upload a dataset directory to the specified target URI."""
    source_path = Path(source)
    if not source_path.exists():
        logger.error("Source directory not found: {}", source)
        raise SystemExit(1)

    if target.startswith("s3://"):
        parts = target[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else source_path.name
        _upload_s3(source_path, bucket, prefix)

    elif target.startswith("gs://"):
        parts = target[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else source_path.name
        _upload_gcs(source_path, bucket, prefix)

    elif target.startswith("hf://"):
        repo_id = target[5:]
        _upload_hf(source_path, repo_id)

    else:
        logger.error("Unknown target scheme. Use s3://, gs://, or hf:// prefix.")
        raise SystemExit(1)


@dataclass
class Args:
    """Upload a dataset to cloud storage."""

    source: str  # local dataset directory
    target: Optional[str] = None  # s3://bucket/prefix, gs://bucket/prefix, or hf://user/repo
    task: Optional[str] = None  # task name for storage.yaml lookup


def main(args: Args) -> None:
    resolved = _resolve_target(args.target, args.task)
    upload(args.source, resolved)


if __name__ == "__main__":
    main(tyro.cli(Args))
