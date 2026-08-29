#!/usr/bin/env python3
"""Send one compact scientific campaign summary through ServerChan."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


MACE_PERSISTENT_TOTAL = 4 * 5 * 2
NEQUIP_PERSISTENT_TOTAL = 4 * 5 * 2
VAL_RE = re.compile(
    r"\[epoch\s+(?P<epoch>\d+)\s+step\s+(?P<step>\d+).*?\].*?"
    r"\|\s+val\s+loss=.*?Fmae=(?P<fmae>[0-9.eE+-]+)\s+"
    r"Emae=(?P<emae>[0-9.eE+-]+)"
)


def latest_line(path: Path) -> str:
    if not path.is_file():
        return "Kimi supervisor has not written a status yet."
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    return lines[-1] if lines else "Kimi supervisor status is empty."


def marker_counts(root: Path, total: int, pattern: str) -> dict[str, int]:
    done = sum(1 for _ in root.glob(f"{pattern}/DONE")) if root.is_dir() else 0
    failed = sum(1 for _ in root.glob(f"{pattern}/FAILED")) if root.is_dir() else 0
    return {"done": done, "failed": failed, "pending": max(total - done, 0)}


def latest_mace_validation(root: Path) -> str:
    logs = list((root / "mace" / "runs").glob("*/seed*/*/logs/train.log"))
    if not logs:
        return "- MACE-Persistent 尚无 validation 记录"
    path = max(logs, key=lambda item: item.stat().st_mtime)
    rows = []
    for match in VAL_RE.finditer(path.read_text(errors="replace")):
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "fmae": 1000.0 * float(match.group("fmae")),
                "emae": 1000.0 * float(match.group("emae")),
            }
        )
    dataset = path.parents[3].name
    seed = path.parents[2].name.removeprefix("seed")
    mode = path.parents[1].name.capitalize()
    if not rows:
        return f"- MACE-Persistent {dataset}/{mode}/seed {seed}: 初始化或首轮 validation 中"
    current = rows[-1]
    best_f = min(rows, key=lambda row: row["fmae"])
    best_e = min(rows, key=lambda row: row["emae"])
    return (
        f"- MACE-Persistent {dataset}/{mode}/seed {seed}: "
        f"{current['step']}/45000 steps；validation 包络 "
        f"F-MAE {best_f['fmae']:.1f} meV Å⁻¹ @ {best_f['step']}，"
        f"E-MAE {best_e['emae']:.1f} meV atom⁻¹ @ {best_e['step']}"
    )


def active_nequip_rows(snapshot: dict, campaign_root: Path) -> str:
    rows = []
    for task in snapshot.get("started_tasks", []):
        paths = list(
            (
                campaign_root
                / f"seed_{task['seed']}"
                / task["mode"]
                / task["dataset"]
            ).glob("**/metrics_epoch.csv")
        )
        if len(paths) != 1:
            continue
        path = paths[0]
        with path.open(newline="") as handle:
            data = list(csv.DictReader(handle))
        if not data:
            continue
        best = min(data, key=lambda row: float(row["validation_f_mae"]))
        current = data[-1]
        rows.append(
            (
                path.stat().st_mtime,
                f"- NequIP-Final {task['dataset']}/{task['mode']}/seed {task['seed']}: "
                f"epoch {int(float(current['epoch']))}；validation F-MAE 包络 "
                f"{1000.0 * float(best['validation_f_mae']):.1f} meV Å⁻¹"
            )
        )
    if not rows:
        return "- NequIP-Final 当前无可读的活跃 validation 记录"
    rows.sort(reverse=True)
    return "\n".join(value for _, value in rows[:4])


def relative_force_gain(full_value: float, aligned_value: float) -> float:
    if aligned_value == 0.0:
        return 0.0
    return 100.0 * (aligned_value - full_value) / aligned_value


def latest_mace_pair(root: Path) -> str:
    candidates = []
    for aligned_path in (root / "mace" / "runs").glob(
        "*/seed*/aligned/metrics.json"
    ):
        full_path = aligned_path.parent.parent / "full" / "metrics.json"
        if full_path.is_file():
            candidates.append((aligned_path.stat().st_mtime, aligned_path, full_path))
    if not candidates:
        return "- MACE-Persistent 尚无完成的 Full–Aligned 配对"
    _, aligned_path, full_path = max(candidates)
    aligned = json.loads(aligned_path.read_text())
    full = json.loads(full_path.read_text())
    aligned_f = float(
        aligned["checkpoint_selection"]["force_mae_envelope"]
        ["force_mae_mev_per_angstrom"]
    )
    full_f = float(
        full["checkpoint_selection"]["force_mae_envelope"]
        ["force_mae_mev_per_angstrom"]
    )
    gain = relative_force_gain(full_f, aligned_f)
    return (
        f"- MACE-Persistent {aligned['dataset']}/seed {aligned['seed']}: "
        f"Full {full_f:.1f} vs Aligned {aligned_f:.1f} meV Å⁻¹；"
        f"learned phase {gain:+.1f}%"
    )


def latest_nequip_pair(root: Path, label: str) -> str:
    candidates = []
    for aligned_path in root.glob("seed_*/aligned/*/**/metrics.json"):
        if not aligned_path.parent.name.startswith("e3nn_chorus"):
            continue
        relative = aligned_path.relative_to(root).parts
        if len(relative) < 4:
            continue
        seed, _, dataset = relative[:3]
        full_paths = [
            path
            for path in (root / seed / "full" / dataset).glob("**/metrics.json")
            if path.parent.name.startswith("e3nn_chorus")
        ]
        if len(full_paths) == 1:
            candidates.append((aligned_path.stat().st_mtime, aligned_path, full_paths[0]))
    if not candidates:
        return f"- {label} 尚无完成的 Full–Aligned 配对"
    _, aligned_path, full_path = max(candidates)
    aligned = json.loads(aligned_path.read_text())
    full = json.loads(full_path.read_text())
    aligned_f = float(
        aligned["independent_validation_envelopes"]["force_mae"]
        ["force_mae_mev_per_angstrom"]
    )
    full_f = float(
        full["independent_validation_envelopes"]["force_mae"]
        ["force_mae_mev_per_angstrom"]
    )
    parts = aligned_path.relative_to(root).parts
    seed = parts[0].removeprefix("seed_")
    dataset = parts[2]
    gain = relative_force_gain(full_f, aligned_f)
    return (
        f"- {label} {dataset}/seed {seed}: Full {full_f:.1f} vs "
        f"Aligned {aligned_f:.1f} meV Å⁻¹；learned phase {gain:+.1f}%"
    )


def relevant_slurm_rows() -> list[dict[str, str]]:
    command = [
        "squeue", "-u", os.environ.get("USER", "rxlin"), "-h",
        "-o", "%i|%T|%j|%M|%R",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    names = {"nequip-causal-v2", "mace-persist-pair", "nequip-persist-pair"}
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 4)
        if len(fields) == 5 and fields[2] in names:
            rows.append(dict(zip(("job_id", "state", "name", "elapsed", "reason"), fields)))
    return rows


def build_message(
    snapshot: dict,
    kimi_status: str,
    campaign_root: Path,
    mace_persistent_root: Path,
    nequip_persistent_root: Path,
) -> tuple[str, str]:
    counts = snapshot["state_counts"]
    complete = counts["DONE"]
    total = snapshot["total_tasks"]
    mace = marker_counts(
        mace_persistent_root / "mace" / "runs",
        MACE_PERSISTENT_TOTAL,
        "*/seed*/*",
    )
    nequip = marker_counts(
        nequip_persistent_root / "status",
        NEQUIP_PERSISTENT_TOTAL,
        "task_*",
    )
    persistent_done = mace["done"] + nequip["done"]
    failures = counts["FAILED"] + mace["failed"] + nequip["failed"]
    health = "异常" if failures or snapshot["oom_logs"] or snapshot["collapsed_pairs"] else "正常"
    title = (
        f"CHORUS：Final {complete}/{total}，Persistent "
        f"{persistent_done}/{MACE_PERSISTENT_TOTAL + NEQUIP_PERSISTENT_TOTAL} · {health}"
    )
    dataset_rows = "\n".join(
        f"- {name}: {value}/20"
        for name, value in snapshot["dataset_done_out_of_20"].items()
    )
    slurm = relevant_slurm_rows()
    active = [row for row in slurm if row["state"] in {"RUNNING", "COMPLETING"}]
    active_rows = "\n".join(
        f"- {row['job_id']} · {row['name']} · {row['state']} · {row['elapsed']}"
        for row in slurm[:12]
    ) or "- 当前没有运行中的本任务 GPU 作业"
    mace_validation = latest_mace_validation(mace_persistent_root)
    nequip_active = active_nequip_rows(snapshot, campaign_root)
    completed_pairs = "\n".join(
        (
            latest_nequip_pair(campaign_root, "NequIP-Final"),
            latest_mace_pair(mace_persistent_root),
            latest_nequip_pair(nequip_persistent_root, "NequIP-Persistent"),
        )
    )
    body = f"""## CHORUS 因果实验总览

- 时间：{datetime.now().astimezone().isoformat(timespec='seconds')}
- NequIP-Final：**{complete}/{total}**；Full–Aligned 配对 {len(snapshot['full_aligned_pair_checks'])}/20
- MACE-Persistent Full/Aligned：**{mace['done']}/{MACE_PERSISTENT_TOTAL}**
- NequIP-Persistent Full/Aligned：**{nequip['done']}/{NEQUIP_PERSISTENT_TOTAL}**
- 当前相关 GPU 作业：{len(active)} 个运行中
- FAILED：**{failures}**
- OOM：**{len(snapshot['oom_logs'])}**
- Full–Aligned 轨迹重合警报：**{len(snapshot['collapsed_pairs'])}**

### NequIP-Final 数据集进度

{dataset_rows}

### 最新 validation 情况

{mace_validation}
{nequip_active}

### 最近完成的因果配对

{completed_pairs}

### 当前 Slurm

{active_rows}

### Kimi 最近动作

{kimi_status}
"""
    return title, body


def send(sendkey: str, title: str, body: str) -> dict:
    endpoint = f"https://sctapi.ftqq.com/{urllib.parse.quote(sendkey, safe='')}.send"
    payload = urllib.parse.urlencode(
        {"title": title, "desp": body, "short": body[:200]}
    ).encode()
    request = urllib.request.Request(endpoint, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text[:500]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--sendkey-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument(
        "--mace-persistent-root",
        type=Path,
        default=Path("/public/home/sps-xia/rxlin/chorus_persistent_causal_20260826"),
    )
    parser.add_argument(
        "--nequip-persistent-root",
        type=Path,
        default=Path(
            "/public/home/sps-xia/rxlin/nequip_chorus_runs/"
            "causal_persistent_phase_specific_20260826"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.sendkey_file.is_file():
        raise SystemExit(f"ServerChan SendKey file is missing: {args.sendkey_file}")
    sendkey = args.sendkey_file.read_text().strip()
    if not sendkey:
        raise SystemExit("ServerChan SendKey file is empty")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    snapshot_json = args.state_dir / "hourly_snapshot.json"
    snapshot_md = args.state_dir / "hourly_snapshot.md"
    subprocess.run(
        [
            sys.executable,
            str(args.tools_dir / "build_kimi_snapshot.py"),
            "--campaign-root", str(args.campaign_root),
            "--protocol", str(args.tools_dir / "protocol.json"),
            "--json", str(snapshot_json),
            "--markdown", str(snapshot_md),
        ],
        check=True,
    )
    snapshot = json.loads(snapshot_json.read_text())
    kimi_status = latest_line(args.campaign_root / "kimi_supervisor" / "status.log")
    title, body = build_message(
        snapshot,
        kimi_status,
        args.campaign_root,
        args.mace_persistent_root,
        args.nequip_persistent_root,
    )
    if args.dry_run:
        print(title)
        print(body)
        return
    try:
        response = send(sendkey, title, body)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"ServerChan request failed: {type(exc).__name__}: {exc}") from exc

    response_data = response.get("data")
    if not isinstance(response_data, dict):
        response_data = {}
    audit = {
        "sent_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": title,
        "state_counts": snapshot["state_counts"],
        "serverchan_code": response.get("code"),
        # ServerChan's success payload also contains per-delivery read keys.
        # They are unnecessary for monitoring and must not enter audit logs.
        "serverchan_status": response_data.get("error") or response.get("message"),
        "serverchan_errno": response_data.get("errno"),
    }
    (args.state_dir / "last_delivery.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
