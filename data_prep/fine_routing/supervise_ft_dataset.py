#!/usr/bin/env python3
"""Supervise ``build_ft_fine_routing_dataset``: notify on abort/crash, auto-restart with ``--resume``.

Runs the dataset builder in a subprocess loop. If the child exits with a non-zero
status or is killed by a signal, optionally POSTs a short message (ntfy / Slack /
generic webhook), waits, then restarts the same command. After the first failure,
``--resume`` is injected if the child argv does not already include it so work
continues from ``*.jsonl`` checkpoints.

Usage::

    cd dr-llm
    python -u -m data_prep.supervise_ft_fine_routing_dataset \\
        --notify-url \"https://ntfy.sh/your_topic\" \\
        --restart-delay 60 \\
        --max-restarts 0 \\
        -- \\
        -m data_prep.build_ft_fine_routing_dataset \\
        --model_name Qwen/Qwen2.5-0.5B-Instruct \\
        --ft_results_dir ft_study_results_v7 \\
        --seed 41 \\
        --benchmarks boolq commonsenseqa \\
        --mcts_num_simulations 250 \\
        --output_dir fine_routing_data_ft_qwen05b_250sims \\
        --resume

Environment (optional, overrides defaults if CLI omitted):

* ``FT_DATASET_SUPERVISE_NOTIFY_URL`` — same as ``--notify-url``
* ``FT_DATASET_SUPERVISE_RESTART_DELAY`` — seconds between retries (float)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("supervise_ft_fine_routing")


def _split_supervisor_and_child(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    if "--" not in argv:
        logger.error(
            "Separate supervisor flags from the child command with ' -- '. "
            "Example: python -m data_prep.supervise_ft_fine_routing_dataset "
            "--notify-url URL -- -- -m data_prep.build_ft_fine_routing_dataset ..."
        )
        sys.exit(2)
    i = list(argv).index("--")
    return list(argv[:i]), list(argv[i + 1 :])


def _ensure_resume(argv: List[str]) -> List[str]:
    out = list(argv)
    if "--resume" not in out:
        out.append("--resume")
    return out


def _notify(url: str, title: str, body: str, timeout_s: float = 20.0) -> None:
    """POST ``body`` to ``url``. Slack incoming webhooks get JSON; ntfy gets Title header."""
    if not url:
        return
    try:
        if "hooks.slack.com" in url:
            payload = json.dumps({"text": f"*{title}*\n{body}"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            data = body.encode("utf-8")
            headers = {}
            if title:
                headers["Title"] = title
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
        logger.info("Notification sent to webhook host")
    except urllib.error.URLError as e:
        logger.warning("Notification failed: %s", e)


def _run_child(cmd: List[str]) -> int:
    logger.info("Starting: %s", " ".join(cmd))
    try:
        p = subprocess.run(cmd, check=False)
        return int(p.returncode)
    except KeyboardInterrupt:
        logger.warning("Supervisor interrupted; stopping child via SIGINT")
        raise


def _build_supervisor_parser() -> argparse.ArgumentParser:
    env_url = os.environ.get("FT_DATASET_SUPERVISE_NOTIFY_URL", "").strip()
    env_delay = os.environ.get("FT_DATASET_SUPERVISE_RESTART_DELAY", "").strip()
    sp = argparse.ArgumentParser(
        description="Supervise FT fine-routing dataset build: notify on failure, auto-retry with --resume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -u -m data_prep.supervise_ft_fine_routing_dataset \\\n"
            "    --notify-url https://ntfy.sh/your_topic --restart-delay 60 -- \\\n"
            "    -m data_prep.build_ft_fine_routing_dataset --model_name Qwen/Qwen2.5-0.5B-Instruct \\\n"
            "    --ft_results_dir ft_study_results_v7 --seed 41 --benchmarks boolq commonsenseqa \\\n"
            "    --mcts_num_simulations 250 --output_dir fine_routing_data_ft_qwen05b_250sims --resume\n"
        ),
    )
    sp.add_argument(
        "--notify-url",
        default=env_url or None,
        help="POST notification (ntfy topic URL, Slack incoming webhook, or plain POST)",
    )
    sp.add_argument(
        "--notify-on-success",
        action="store_true",
        help="Also POST when the child exits 0 (dataset finished)",
    )
    sp.add_argument(
        "--restart-delay",
        type=float,
        default=float(env_delay) if env_delay else 30.0,
        help="Seconds to wait before restarting after a non-zero exit",
    )
    sp.add_argument(
        "--max-restarts",
        type=int,
        default=0,
        help="Max retry rounds after a failure (0 = unlimited)",
    )
    return sp


def main() -> None:
    raw = sys.argv[1:]
    if raw and raw[0] in ("-h", "--help") and "--" not in raw:
        _build_supervisor_parser().print_help()
        return

    sup_argv, child_template = _split_supervisor_and_child(raw)
    sup_args = _build_supervisor_parser().parse_args(sup_argv)

    if not child_template:
        logger.error("No child command after '--'")
        sys.exit(2)

    # Child is either: -m module.name [args...]  or script path ...
    if child_template[0] != "-m":
        logger.error(
            "Expected child to start with '-m module.name' "
            "(e.g. -m data_prep.build_ft_fine_routing_dataset ...)"
        )
        sys.exit(2)

    base_cmd = [sys.executable, "-u"] + child_template
    attempt = 0
    failures = 0

    while True:
        attempt += 1
        cmd = list(base_cmd)
        if attempt > 1:
            # argv positions: [python, -u, -m, MOD, ...args]
            mod_idx = cmd.index("-m") + 2
            rest = cmd[mod_idx:]
            cmd = cmd[:mod_idx] + _ensure_resume(rest)

        rc = _run_child(cmd)
        host = os.uname().nodename if hasattr(os, "uname") else ""

        if rc == 0:
            msg = (
                f"build_ft_fine_routing_dataset finished OK (attempt {attempt}, host={host})"
            )
            logger.info("%s", msg)
            if sup_args.notify_url and sup_args.notify_on_success:
                _notify(
                    sup_args.notify_url,
                    title="FT dataset build: success",
                    body=msg,
                )
            return

        failures += 1
        msg = (
            f"build_ft_fine_routing_dataset exited with code {rc} "
            f"(attempt {attempt}, failures={failures}, host={host}). Will retry after {sup_args.restart_delay}s."
        )
        logger.error("%s", msg)
        if sup_args.notify_url:
            _notify(
                sup_args.notify_url,
                title="FT dataset build: aborted / failed",
                body=msg,
            )

        if sup_args.max_restarts > 0 and failures > sup_args.max_restarts:
            logger.error("Max restarts (%d) exceeded; exiting.", sup_args.max_restarts)
            sys.exit(rc if rc != 0 else 1)

        time.sleep(sup_args.restart_delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Supervisor exiting on KeyboardInterrupt")
        sys.exit(130)
