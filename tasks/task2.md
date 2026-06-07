# Task 2 — Promotion CLI

**Files:** `scripts/promote.py` — fill in four subcommand function bodies. Argparse scaffolding is already wired.

## What promotion is

MLflow's Model Registry tracks every registered version of `travel-assistant`. Each version has:

- An immutable integer version number (1, 2, 3, ...), auto-assigned by MLflow at registration time.
- Tags, including `config_id` (e.g., `v5`). The operator names the YAML config file (`configs/v5.yaml`); the eval pipeline reads that name and writes it as the `config_id` tag when it registers the version.

On top of versions, **aliases** are string labels pointing at exactly one version. The service follows the `production` alias on startup to decide which version to serve:

```
travel-assistant @ production → MLflow version 12 (tags: config_id=v5, ...)
                  @ staging    → MLflow version 14 (tags: config_id=v6, ...)
```

"Promoting v6 to production" means moving the `production` alias to point at the version whose `config_id` tag is `"v6"`.

## How data ends up in MLflow

The MLflow tracking server (running in docker compose) stores everything: experiments, runs, the model registry. The eval pipeline writes to it; this CLI reads from it.

When `python -m src.eval --config v6` runs, the eval:

1. Creates a new **run** under the `travel-assistant` experiment.
2. Logs **params** to that run (`config_id`, `model`, `guardrail_type`, `judge_model`, `dataset_size`) via `mlflow.log_params(...)`.
3. Logs **metrics** to that run (`accuracy_overall`, `verdict_rate_leaked`, `total_cost_usd`, etc.) via `mlflow.log_metric(...)`. These are the numbers `_compute_metrics` returns.
4. Logs **artifacts** to that run (config.json, predictions.jsonl, prompt files) via `mlflow.log_artifact(...)`.
5. On a full eval (no `--limit`), calls `client.create_model_version(...)` to register the run as a new version of the `travel-assistant` registered model. The new version:
  - Gets an integer version number, auto-assigned.
  - Carries tags (`config_id`, `model`, `guardrail_type`, ...) propagated from the eval's params.
  - Has a `run_id` field pointing back to the eval run that produced it.

This CLI reads from that stored state:

- **Search registered versions by tag.** "Find the version whose `config_id` tag is `v6`."
- **Read a version's `run_id`** and from there pull metrics from the source run. `client.get_run(mv.run_id).data.metrics` is a dict like `{"accuracy_overall": 0.91, ...}`.
- **Assign and read aliases.**

## Identifying versions

The CLI identifies registered versions by their `config_id` tag (`v6`), not by MLflow's integer version number (`12`). The operator types `set production v6`; the CLI searches MLflow for a version whose `config_id` tag equals `"v6"` and assigns the alias to that version.

Possible outcomes of the search:

- **One match:** that's the target. Proceed.
- **Zero matches:** print `"no version found with config_id=v6"` and stop with `sys.exit(1)`.
- **Two or more matches:** the operator ran `python -m src.eval --config v6` more than once and MLflow has multiple registrations tagged `config_id=v6`. Take the one with the highest MLflow integer version number (the most recently registered) and continue. Print a warning to stdout first: `"warning: multiple versions match config_id=v6 (MLflow versions [7, 12]); using latest (12)"`.

## What you're building

Four subcommands. Sample session (assume `eval --config v4`, `v5`, `v6` each ran once):

```
$ python scripts/promote.py list
no aliases set

$ python scripts/promote.py set production v4
production: (unset) → v4

$ python scripts/promote.py show production
travel-assistant @ production
  config_id: v4
  model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
  accuracy_overall: 0.89
  verdict_rate_leaked: 0.04
  total_cost_usd: $0.38

$ python scripts/promote.py set production v5
production: v4 → v5

$ python scripts/promote.py set production v6
production: v5 → v6

$ python scripts/promote.py rollback production
production: v6 → v5 (rolled back)

$ python scripts/promote.py rollback production
error: production was just rolled back; no further history to walk back to

$ python scripts/promote.py set production v99
error: no version found with config_id=v99

$ cat promotion-log.jsonl
{"ts": "2026-05-21T14:32:01Z", "alias": "production", "from": "",   "to": "v4", "op": "set"}
{"ts": "2026-05-21T14:33:15Z", "alias": "production", "from": "v4", "to": "v5", "op": "set"}
{"ts": "2026-05-21T14:34:22Z", "alias": "production", "from": "v5", "to": "v6", "op": "set"}
{"ts": "2026-05-21T14:35:01Z", "alias": "production", "from": "v6", "to": "v5", "op": "rollback"}
```

And the multiplicity warning if `eval --config v6` happened to run twice (the CLI picks the latest and continues):

```
$ python scripts/promote.py set production v6
warning: multiple versions match config_id=v6 (MLflow versions [7, 12]); using latest (12)
production: v5 → v6
```

## The audit log

Every successful `set` and `rollback` appends one line to `promotion-log.jsonl` at the repo root (gitignored — won't accidentally end up in git). Each line is one JSON event:

```json
{"ts": "<ISO 8601 UTC>", "alias": "<name>", "from": "<old_config_id_or_empty>", "to": "<new_config_id>", "op": "set" | "rollback"}
```

`rollback` reads the log to find where to roll back to. It scans the log backward for the most recent entry where `alias == args.alias`:

- No matching entry → print `"no promotion history for alias <alias>"` and stop.
- The entry's `op` is `rollback` → print `"<alias> was just rolled back; no further history to walk back to"` and stop. (Single-step rollback by design.)
- The entry's `op` is `set` and its `from` is empty → print `"<alias> has no previous target (first promotion ever)"` and stop.
- Otherwise → take the entry's `from` (a config_id), find that registered version, move the alias to it.

The log file lives at the repo root. The first `set` creates it; subsequent operations append.

## MLflow client API you'll use

Talk to the tracking server via `mlflow.tracking.MlflowClient`. The methods relevant to this task:

- `client.search_model_versions(filter_string)` — searches registered model versions. Filter syntax is SQL-ish: `"name = 'travel-assistant' AND tags.config_id = 'v6'"`. Returns a list of `ModelVersion` objects (possibly empty, possibly multi-element).
- `client.set_registered_model_alias(name, alias, version)` — assigns an alias to a specific version. Atomic.
- `client.get_model_version_by_alias(name, alias)` — returns the `ModelVersion` an alias currently points at. Raises `mlflow.exceptions.RestException` if the alias is unset.
- `client.get_registered_model(name)` — returns a `RegisteredModel` object whose `.aliases` attribute is a `{alias_name: version_number_str}` dict. Useful for `list`.
- `client.get_run(run_id)` — returns a `Run` object. A `ModelVersion` has a `.run_id` attribute pointing at the eval run; from there, `run.data.metrics` is the dict of MLflow-logged metric values.

A `MlflowClient` has no per-call state, so you can instantiate one at the top of `promote.py` and reuse it across all four subcommands, or instantiate a fresh one inside each command function. Both work; the first is slightly less code.

## Behavioral specs

### `cmd_set(args)`

**Inputs:** `args.alias: str` (e.g., `"production"`), `args.config_id: str` (e.g., `"v6"`).

**Behavior:**

1. Find MLflow versions whose `config_id` tag equals `args.config_id` (using `search_model_versions` with a filter string). Apply the rule from "Identifying versions":
  - One match → use it.
  - Zero matches → print the error and stop via `sys.exit(1)`.
  - Two or more matches → print the warning to stdout and take the one with the highest MLflow integer version number.
2. Look up what `args.alias` currently points at via `get_model_version_by_alias`. If it's unset (the client raises `RestException`), treat the current config_id as the empty string `""`.
3. Assign `args.alias` to the version found in step 1.
4. Append one event to `promotion-log.jsonl`:
  ```json
   {"ts": "...", "alias": "<args.alias>", "from": "<current_config_id_or_empty>", "to": "<args.config_id>", "op": "set"}
  ```

**stdout:** one line, e.g. `"production: v5 → v6"`, or `"production: (unset) → v6"` for a first promotion.

### `cmd_show(args)`

**Inputs:** `args.alias: str`.

**Behavior:** Look up the current target of `args.alias` via `get_model_version_by_alias`. Print its `config_id` tag, other tags, and a few key metrics from its eval run (`client.get_run(mv.run_id).data.metrics`). If the alias is unset, print a clear error and stop via `sys.exit(1)`.

Doesn't touch the log.

### `cmd_list(args)`

**Inputs:** none.

**Behavior:** Print every alias currently set on the registered model, with the `config_id` of the version each one points at:

```
production -> v5
staging    -> v6
```

If no aliases are set, print `"no aliases set"`. Doesn't touch the log.

### `cmd_rollback(args)`

**Inputs:** `args.alias: str`.

**Behavior:**

1. Look up the current target of `args.alias`. If it's unset entirely, print `"nothing to roll back"` and stop.
2. Note the current target's `config_id` — you'll print this in the summary line.
3. Find the most recent entry in `promotion-log.jsonl` where `alias == args.alias`. Apply the four cases from "The audit log" above — three of them stop with an error; only the last (a `set` event with a non-empty `from`) proceeds.
4. Take the entry's `from` (a config_id). Find MLflow versions with that `config_id` tag — same rule as in `cmd_set`. If zero matches, print the error and stop. If multiple, print the warning to stdout and take the latest.
5. Assign `args.alias` to that version.
6. Append one event:
  ```json
   {"ts": "...", "alias": "<args.alias>", "from": "<current_config_id>", "to": "<entry.from>", "op": "rollback"}
  ```

**stdout:** `"production: v6 → v5 (rolled back)"`.

## Grading (40 pts)


| Subcommand                                                                               | Points |
| ---------------------------------------------------------------------------------------- | ------ |
| `list` — prints all aliases with their config_ids, or "no aliases set"                   | 5      |
| `show` — resolves alias, prints config_id + tags + key metrics from the source run       | 7      |
| `set` — version lookup with multiplicity handling, alias assignment, audit log append    | 15     |
| `rollback` — audit log backward scan, all four edge cases, version re-lookup, log append | 13     |


## Verifying your work

You need at least three registered versions; run a couple of evals first:

```bash
python -m src.eval --config v1
python -m src.eval --config v4
python -m src.eval --config v5
```

Walk through the UX session above, then inspect `cat promotion-log.jsonl`. Verify each error path:

- `set production v99` → "no version found".
- Re-run `eval --config v5` to create two registrations, then `set production v5` → multiplicity warning printed to stdout, then proceeds and assigns the alias to the latest v5.
- Two consecutive `rollback production` → second one says "already rolled back".
- `rollback production` immediately after the first-ever `set` → "no previous target (first promotion)".

## Hints

- **First run, log file doesn't exist.** The CLI must handle the missing file (treat it as an empty log); don't crash. The log has no header — line-delimited JSON only.
- `**search_model_versions` filter syntax.** Tags are referenced as `tags.config_id` (or `tag.config_id`; varies by MLflow version). Quoting matters: `"name = 'travel-assistant' AND tags.config_id = 'v6'"`.
- `**get_model_version_by_alias` raises `RestException` when the alias is unset.** Wrap in try/except. The first-promotion case is normal, not an error.
- `datetime.now(timezone.utc).isoformat()` produces the right date and time format.

## Submission

Submit a `*.zip` file with:

1. `**promote.py`** — your implementation.
2. `**promotion-log.jsonl**` — the log from your demo session.

