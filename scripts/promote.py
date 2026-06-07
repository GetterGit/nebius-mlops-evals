"""scripts/promote.py — promote MLflow Registry aliases with an audit log.

YOUR TASK (see tasks/task2.md): implement the four subcommand functions.
The argparse scaffolding below is wired so each cmd_* receives an `args`
namespace already parsed. See `_build_parser` for what's on `args` per
subcommand, and tasks/task2.md "Behavioral specs" for what each function
must do.

Versions are identified by their `config_id` tag (e.g., "v6"), NOT by
MLflow's integer version numbers. Resolution must be unique — if the
config_id matches zero or multiple registered versions, the CLI errors
out and forces the operator to disambiguate via the MLflow UI.

Successful `set` and `rollback` operations append a JSON event to
LOG_FILE (promotion-log.jsonl at repo root). `rollback` consults the
log to find the previous alias target.

Subcommands:
  set <alias> <config_id>   move alias, append `set` event to the log
  show <alias>              print current target + tags + key metrics
  list                      print all aliases on the registered model
  rollback <alias>          move alias back per the audit log, append
                            `rollback` event
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mlflow import MlflowClient
import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.entities.model_registry import ModelVersion

from src.config import get_settings

REGISTERED_MODEL_NAME = "travel-assistant"
LOG_FILE = Path(__file__).resolve().parent.parent / "promotion-log.jsonl"


def _client() -> MlflowClient:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _resolve_version_by_config_id(client: MlflowClient, name: str, config_id: str) -> ModelVersion:
    """Return the ModelVersion by its config_id. 
    
    If multiplate versions found, return the one with the highest version value.
    """
    matches = client.search_model_versions(
        f"name = '{name}' AND tags.config_id = '{config_id}'"
    )
    if not matches:
        print(f"error: no version found with config_id={config_id}", file=sys.stderr)
        sys.exit(1)
    
    if len(matches) == 1:
        return matches[0]

    versions = sorted(int(m.version) for m in matches)
    latest = versions[-1]
    print(
        f"warning: multiple versions match config_id={config_id} "
        f"(MLflow versions {versions}); using latest ({latest})"
    )
    for m in matches:
        if int(m.version) == latest:
            return m


def _current_alias_target(
    client: MlflowClient, name: str, alias: str
) -> ModelVersion | None:
    """Return the ModelVersion an alias points at, or None if unset."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except (RestException, MlflowException):
        return None


def _append_log_event(alias: str, frm: str, to: str, op: str) -> None:
    """Append a new set or rollback operation to the JSONL event log."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alias": alias,
        "from": frm,
        "to": to,
        "op": op,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _last_event_for_alias(alias: str) -> dict | None:
    """Return the most recent log event for `alias`, or None.

    Read mode requires guarding against a missing file (first run,
    nothing has been written yet). Iterates events in reverse so we
    return on the first match instead of scanning the whole file.
    """
    if not LOG_FILE.exists():
        return None
    with LOG_FILE.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    for event in reversed(events):
        if event.get("alias") == alias:
            return event
    return None


def cmd_set(args: argparse.Namespace) -> None:
    """Move `args.alias` to the version tagged config_id=args.config_id.

    On success: assigns the alias atomically and appends a `set`
    event to the audit log. Prints "alias: prev → new" (or
    "alias: (unset) → new" for the first promotion of an alias).
    """
    client = _client()
    target = _resolve_version_by_config_id(client, args.name, args.config_id)

    current = _current_alias_target(client, args.name, args.alias)
    current_config_id = (
        (current.tags or {}).get("config_id", "") if current is not None else ""
    )

    client.set_registered_model_alias(args.name, args.alias, target.version)
    _append_log_event(
        alias=args.alias,
        frm=current_config_id,
        to=args.config_id,
        op="set",
    )

    pretty_from = current_config_id if current_config_id else "(unset)"
    print(f"{args.alias}: {pretty_from} → {args.config_id}")


def cmd_show(args: argparse.Namespace) -> None:
    """Print the alias's current target, its tags, and key metrics.

    Read-only; does not touch the audit log.
    """
    client = _client()
    current = _current_alias_target(client, args.name, args.alias)
    if current is None:
        print(f"error: alias {args.alias!r} is not set", file=sys.stderr)
        sys.exit(1)
    
    config_id = (current.tags or {}).get("config_id", "(unset)")
    print(f"{args.name} @ {args.alias}")
    print(f"  config_id: {config_id}")

    for k, v in (current.tags or {}).items():
        if k != "config_id":
            print(f"  {k}: {v}")
        
    try:
        run = client.get_run(current.run_id)
        metrics = run.data.metrics
    except (RestException, MlflowException):
        metrics = {}
    
    metric_format = {
        "accuracy_overall": "{:.3f}",
        "verdict_rate_leaked": "{:.3f}",
        "total_cost_usd": "${:.4f}",
    }
    for key, fmt in metric_format.items():
        if key in metrics:
            print(f"  {key}: {fmt.format(metrics[key])}")


def cmd_list(args: argparse.Namespace) -> None:
    """Print every alias on the registered model with its config_id.

    If the registered model doesn't exist or has no aliases, prints
    "no aliases set" and returns. Read-only; does not touch the
    audit log.
    """
    client = _client()
    try:
        registered_models = client.get_registered_model(args.name)
    except (RestException, MlflowException):
        print("no aliases set")
        return

    aliases = registered_models.aliases or {}
    if not aliases:
        print("no aliases set")
        return
    
    # name_width to align values on print
    name_width = max(len(a) for a in aliases)
    for alias, version_str in sorted(aliases.items()):
        try:
            model_version = client.get_model_version(args.name, version_str)
            config_id = (model_version.tags or {}).get("config_id", f"v{version_str}")
        except (RestException, MlflowException):
            config_id = f"v{version_str}"
        print(f"{alias:<{name_width}} -> {config_id}")


def cmd_rollback(args: argparse.Namespace) -> None:
    """Move `args.alias` back to the previous target per the event log.

    Four early-exit cases:
      1. Alias is unset entirely → "nothing to roll back".
      2. No log entry for this alias → "no promotion history…".
      3. Most recent entry's op is `rollback` → refuse (single-step
         rollback by design); error to stderr, exit 1.
      4. Most recent `set`'s `from` field is empty (first-ever
         promotion) → "no previous target…".
      5. Otherwise: resolve the previous config_id to a version,
         move the alias, append a `rollback` event.
    """
    client = _client()
    current = _current_alias_target(client, args.name, args.alias)
    if current is None:
        print("nothing to roll back")
        return

    current_config_id = (current.tags or {}).get("config_id", "")

    last = _last_event_for_alias(args.alias)
    if last is None:
        print(f"no promotion history for alias {args.alias}")
        return
    if last.get("op") == "rollback":
        print(
            f"error: {args.alias} was just rolled back; "
            "no further history to walk back to",
            file=sys.stderr,
        )
        sys.exit(1)
    
    target_config_id = last.get("from", "")
    if not target_config_id:
        print(f"{args.alias} has no previous target (first promotion ever)")
        return

    target = _resolve_version_by_config_id(client, args.name, target_config_id)
    client.set_registered_model_alias(args.name, args.alias, target.version)
    _append_log_event(
        alias=args.alias,
        frm=current_config_id,
        to=target_config_id,
        op="rollback",
    )
    print(f"{args.alias}: {current_config_id} → {target_config_id} (rolled back)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {REGISTERED_MODEL_NAME})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_set = sub.add_parser(
        "set", help="Move an alias to a version (by config_id), append a set event"
    )
    p_set.add_argument("alias", help="Alias to assign (e.g., 'production')")
    p_set.add_argument(
        "config_id",
        help="Config identifier (e.g., 'v6') — resolved via the config_id tag on registered versions",
    )
    p_set.set_defaults(func=cmd_set)

    p_show = sub.add_parser("show", help="Show which version an alias points at")
    p_show.add_argument("alias")
    p_show.set_defaults(func=cmd_show)

    p_list = sub.add_parser("list", help="List all aliases on the registered model")
    p_list.set_defaults(func=cmd_list)

    p_rollback = sub.add_parser(
        "rollback",
        help="Move an alias back to its previous target per the audit log",
    )
    p_rollback.add_argument("alias")
    p_rollback.set_defaults(func=cmd_rollback)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.func(args)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
