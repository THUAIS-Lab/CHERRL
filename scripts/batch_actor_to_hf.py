#!/usr/bin/env python3
"""
把所有 checkpoint 下的 FSDP actor 合并为 HuggingFace 格式 (actor_hf)。
默认：非最新 checkpoint 合并后删除原 actor 以省空间，最新 checkpoint 保留 actor。

用法:
    # 全部转为 actor_hf，只保留最新 checkpoint 的 actor
    PYTHONPATH=/path/to/hackingRubricsRL:$PYTHONPATH python scripts/batch_actor_to_hf.py /path/to/checkpoints

    # 合并后不删除任何 actor
    python scripts/batch_actor_to_hf.py /path/to/checkpoints --keep-all-actors

    # 合并后删除所有 actor（包括最新）
    python scripts/batch_actor_to_hf.py /path/to/checkpoints --delete-all-actors

    # 仅打印将要执行的操作
    python scripts/batch_actor_to_hf.py /path/to/checkpoints --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_actor_dirs(root: Path) -> list[Path]:
    """扫描 root 下所有包含 'actor' 子目录的路径，返回这些 actor 的父目录列表。"""
    root = root.resolve()
    if not root.is_dir():
        return []
    parents_with_actor = []
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        if "actor" in dirnames:
            actor_path = Path(dirpath) / "actor"
            # 只处理看起来像 FSDP 的 actor（存在 fsdp_config.json）
            if (actor_path / "fsdp_config.json").exists():
                parents_with_actor.append(Path(dirpath))
        # 不进入 actor / actor_hf 内部
        dirnames[:] = [d for d in dirnames if d not in ("actor", "actor_hf")]
    return parents_with_actor


def is_latest_checkpoint_dir(step_dir: Path) -> bool:
    """
    判断 step_dir 是否为该 run 下的「最新」checkpoint 目录。
    若同级的 run 目录（step_dir 的父目录）存在 latest_checkpointed_iteration.txt，
    且其内容与 step_dir 名称匹配（如 200 对应 global_step_200），则视为最新。
    """
    run_dir = step_dir.parent
    latest_file = run_dir / "latest_checkpointed_iteration.txt"
    if not latest_file.is_file():
        return False
    try:
        content = latest_file.read_text().strip()
    except OSError:
        return False
    if not content:
        return False
    # 支持 global_step_200 或 200 等命名
    return step_dir.name == f"global_step_{content}" or step_dir.name == content


def main():
    parser = argparse.ArgumentParser(
        description="批量合并 FSDP actor 为 HuggingFace 格式；默认只删除非最新 checkpoint 的 actor"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="要扫描的根目录（如 checkpoints 或某个实验目录）",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp",
        help="legacy_model_merger 的 backend（默认: fsdp）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不实际执行合并和删除",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若 actor_hf 已存在则跳过该目录（默认会覆盖）",
    )
    parser.add_argument(
        "--skip-latest",
        action="store_true",
        help="跳过由 latest_checkpointed_iteration.txt 指向的最新 checkpoint，不转化也不删",
    )
    parser.add_argument(
        "--keep-all-actors",
        action="store_true",
        help="合并后不删除任何 actor（默认只保留最新的 actor）",
    )
    parser.add_argument(
        "--delete-all-actors",
        action="store_true",
        help="合并后删除所有 actor，包括最新 checkpoint 的 actor",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="项目根目录，用于 PYTHONPATH 和 scripts 路径（默认：本脚本所在目录的上级）",
    )
    args = parser.parse_args()

    root = args.root_dir.resolve()
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    # 项目根：用于找 scripts/legacy_model_merger.py 和设置 PYTHONPATH
    if args.project_root is not None:
        project_root = args.project_root.resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent
    merger_script = project_root / "scripts" / "legacy_model_merger.py"
    if not merger_script.is_file():
        print(f"错误: 未找到合并脚本: {merger_script}", file=sys.stderr)
        sys.exit(1)

    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(project_root) not in pythonpath.split(os.pathsep):
        new_pythonpath = f"{project_root}{os.pathsep}{pythonpath}" if pythonpath else str(project_root)
    else:
        new_pythonpath = pythonpath

    env = os.environ.copy()
    env["PYTHONPATH"] = new_pythonpath

    actor_parents = find_actor_dirs(root)
    if not actor_parents:
        print(f"在 {root} 下未找到包含 FSDP actor 的子目录。")
        return

    if args.skip_latest:
        n_before = len(actor_parents)
        actor_parents = [p for p in actor_parents if not is_latest_checkpoint_dir(p)]
        if n_before > len(actor_parents):
            print(f"已跳过 {n_before - len(actor_parents)} 个最新 checkpoint 目录 (--skip-latest)\n")

    print(f"待处理目录数: {len(actor_parents)}\n")
    for p in sorted(actor_parents):
        local_dir = p / "actor"
        target_dir = p / "actor_hf"
        if args.skip_existing and target_dir.is_dir():
            print(f"  [跳过-已存在] {local_dir}")
            continue
        print(f"  {local_dir}")
        print(f"    -> {target_dir}")

    if args.dry_run:
        print("\n[dry-run] 未执行任何合并或删除。")
        return

    failed = []
    for p in sorted(actor_parents):
        local_dir = p / "actor"
        target_dir = p / "actor_hf"
        if args.skip_existing and target_dir.is_dir():
            continue

        cmd = [
            sys.executable,
            str(merger_script),
            "merge",
            "--backend",
            args.backend,
            "--local_dir",
            str(local_dir),
            "--target_dir",
            str(target_dir),
        ]
        print(f"\n执行: {' '.join(cmd)}")
        ret = subprocess.run(cmd, env=env, cwd=str(project_root))
        if ret.returncode != 0:
            failed.append(str(local_dir))
            continue

        # 默认：只删除非最新 checkpoint 的 actor，保留最新的
        if args.keep_all_actors:
            print(f"保留: {local_dir}")
        elif args.delete_all_actors:
            print(f"删除: {local_dir}")
            shutil.rmtree(local_dir)
        elif is_latest_checkpoint_dir(p):
            print(f"保留(最新): {local_dir}")
        else:
            print(f"删除: {local_dir}")
            shutil.rmtree(local_dir)

    if failed:
        print(f"\n以下 {len(failed)} 个目录合并失败，未删除 actor:", file=sys.stderr)
        for d in failed:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)
    print("\n全部完成。")


if __name__ == "__main__":
    main()
