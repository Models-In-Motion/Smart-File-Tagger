"""
model_registry.py - Manage MLflow model stage transitions.

Usage:
  python model_registry.py --promote <version>
  python model_registry.py --deploy <version>
  python model_registry.py --rollback <version>
  python model_registry.py --list
"""

from __future__ import annotations

import argparse
import os

import mlflow

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "smart-tagger")


def get_client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.MlflowClient()


def promote_to_staging(version: str) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Model version {version} promoted to Staging.")


def deploy_to_production(version: str) -> None:
    client = get_client()
    current = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in current:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=v.version,
            stage="Archived",
            archive_existing_versions=False,
        )
        print(f"Archived previous production version {v.version}.")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )
    print(f"Model version {version} deployed to Production.")


def rollback(version: str) -> None:
    client = get_client()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Archived",
        archive_existing_versions=False,
    )
    print(f"Model version {version} rolled back to Archived.")


def list_versions() -> None:
    client = get_client()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        print(f"Version {v.version}: stage={v.current_stage}, run_id={v.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", metavar="VERSION", help="Promote version to Staging")
    parser.add_argument("--deploy", metavar="VERSION", help="Deploy version to Production")
    parser.add_argument("--rollback", metavar="VERSION", help="Rollback version to Archived")
    parser.add_argument("--list", action="store_true", help="List all versions")
    args = parser.parse_args()

    if args.promote:
        promote_to_staging(args.promote)
    elif args.deploy:
        deploy_to_production(args.deploy)
    elif args.rollback:
        rollback(args.rollback)
    elif args.list:
        list_versions()
    else:
        parser.print_help()
