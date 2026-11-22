#!/usr/bin/env python3
"""Read-only scope audit for the historical GLYCAM Fuc/Rha case error.

The historical resolver treated upper-case ``F/H`` as L-Fuc/L-Rha and
lower-case ``f/h`` as D-Fuc/D-Rha.  Standard GLYCAM residue naming uses the
opposite convention.  This script freezes that legacy baseline, reads the
current resolver as a source snapshot, and quantifies affected source names,
completed global pickle pairs, and paired LEaP/metadata evidence.

Only three report files are written, all below ``--output-dir``.  Source SDFs,
resolver code, pickles, and run roots are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"

DEFAULT_SDFS = {
    "mono": ROOT / "data" / "dataset" / "casper_mono.sdf",
    "dimer": ROOT / "data" / "dataset" / "casper_dimer.sdf",
    "trimer": ROOT / "data" / "dataset" / "casper_trimer.sdf",
}
DEFAULT_RESOLVER = MD_ROOT / "scripts" / "start_extended24_glycam_topology_dryrun.py"
DEFAULT_ALL_PICKLES = MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_all_seed1"
DEFAULT_LOW_PICKLES = (
    MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1"
)
DEFAULT_OUTPUT = MD_ROOT / "md_runs_all_remaining_v14_safe" / "audit_glycam_enantiomer_scope_v1"

TARGET_NAMES = ("LFuc", "LRha", "DFuc", "DRha")
CORRECT_CODES = {"DFuc": "F", "LFuc": "f", "DRha": "H", "LRha": "h"}
LEGACY_WRONG_CODES = {"DFuc": "f", "LFuc": "F", "DRha": "h", "LRha": "H"}
TARGET_RESIDUE_RE = re.compile(r"^([0-9ZW])([FfHh])([AB])$")
ANY_RESIDUE_RE = re.compile(r"^([0-9ZW])([A-Za-z])([AB])$")
LINKAGE_RE = re.compile(r"\(1->([0-9])\)")
LEAP_SEQUENCE_RE = re.compile(r"\bsequence\s*\{([^}]*)\}", re.MULTILINE)

REPORT_JSON = "glycam_enantiomer_scope.json"
REPORT_CSV = "glycam_enantiomer_scope.csv"
REPORT_MD = "glycam_enantiomer_scope.md"
RERUN_CSV = "glycam_enantiomer_rerun_167.csv"
REPORT_FILES = {REPORT_JSON, REPORT_CSV, REPORT_MD, RERUN_CSV}


class AuditError(RuntimeError):
    """Raised when an input is incomplete, inconsistent, or changes mid-audit."""


@dataclass(frozen=True)
class LibraryRecord:
    library_index: int
    source_class: str
    source_record_index: int
    source_sdf: Path
    molecule_name: str
    clean_name: str
    target_tokens: tuple[str, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_name(name: str) -> str:
    """Convert CASPER ``(1->N)`` notation to the established pickle stem."""
    clean = LINKAGE_RE.sub(r"1\1", name)
    if "(" in clean or ")" in clean:
        raise AuditError(f"unsupported parentheses remain after compacting name: {name!r}")
    if not clean or "/" in clean or "\x00" in clean:
        raise AuditError(f"unsafe compact molecule name: {name!r}")
    return clean


def read_sdf_records(path: Path, source_class: str, start_index: int) -> list[LibraryRecord]:
    if not path.is_file():
        raise AuditError(f"missing source SDF: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if "$$$$" not in text:
        raise AuditError(f"SDF has no record delimiter: {path}")
    chunks = text.split("$$$$")
    if chunks[-1].strip():
        raise AuditError(f"SDF does not end with a complete $$$$ record: {path}")

    records: list[LibraryRecord] = []
    for source_index, chunk in enumerate(chunks[:-1], start=1):
        lines = chunk.strip().splitlines()
        if not lines:
            raise AuditError(f"empty SDF record {source_index}: {path}")
        name = lines[0].strip()
        if not name:
            raise AuditError(f"blank SDF title at record {source_index}: {path}")
        tokens = tuple(token for token in TARGET_NAMES if token in name)
        records.append(
            LibraryRecord(
                library_index=start_index + len(records),
                source_class=source_class,
                source_record_index=source_index,
                source_sdf=path.resolve(),
                molecule_name=name,
                clean_name=compact_name(name),
                target_tokens=tokens,
            )
        )
    if not records:
        raise AuditError(f"SDF contains no records: {path}")
    return records


def read_library(sdfs: dict[str, Path], expected_size: int) -> list[LibraryRecord]:
    records: list[LibraryRecord] = []
    for source_class in ("mono", "dimer", "trimer"):
        records.extend(read_sdf_records(sdfs[source_class], source_class, len(records) + 1))
    names = [record.molecule_name for record in records]
    clean_names = [record.clean_name for record in records]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count != 1)
    duplicate_clean = sorted(name for name, count in Counter(clean_names).items() if count != 1)
    if duplicate_names:
        raise AuditError(f"duplicate SDF molecule titles: {duplicate_names}")
    if duplicate_clean:
        raise AuditError(f"compact-name collisions: {duplicate_clean}")
    if len(records) != expected_size:
        raise AuditError(f"expected {expected_size} SDF records, found {len(records)}")
    return records


def sequence_target_counter(sequence: Sequence[str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for token in sequence:
        match = TARGET_RESIDUE_RE.fullmatch(token)
        if match:
            result[match.group(2)] += 1
    return result


def expected_target_counter(name: str, mapping: dict[str, str]) -> Counter[str]:
    result: Counter[str] = Counter()
    for target, code in mapping.items():
        result[code] += name.count(target)
    return +result


def swap_enantiomer_case(sequence: Sequence[str]) -> list[str]:
    """Swap only the Fuc/Rha residue-code letter, preserving cap/linkage/anomer."""
    swapped: list[str] = []
    for token in sequence:
        match = TARGET_RESIDUE_RE.fullmatch(token)
        if match:
            token = match.group(1) + match.group(2).swapcase() + match.group(3)
        swapped.append(token)
    return swapped


def normalize_sequence(value: Any) -> list[str] | None:
    if isinstance(value, str):
        tokens = value.split()
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        tokens = list(value)
    else:
        return None
    return tokens or None


def load_resolver_snapshot(path: Path) -> tuple[Callable[[str], list[str] | None], str, str, bytes]:
    """Execute an immutable source snapshot without writing bytecode."""
    if not path.is_file():
        raise AuditError(f"missing resolver source: {path}")
    source = path.read_bytes()
    source_sha = sha256_bytes(source)
    module_name = "_glycam_resolver_audit_snapshot"
    module = types.ModuleType(module_name)
    module.__file__ = str(path.resolve())
    module.__package__ = ""

    previous_dont_write = sys.dont_write_bytecode
    inserted_path = str(path.resolve().parent)
    sys.dont_write_bytecode = True
    sys.path.insert(0, inserted_path)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except Exception as exc:  # pragma: no cover - exact import failure is environment-specific
        raise AuditError(f"cannot load resolver snapshot {path}: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
        if sys.path and sys.path[0] == inserted_path:
            sys.path.pop(0)
        sys.dont_write_bytecode = previous_dont_write

    base_codes = getattr(module, "GLYCAM_BASE_CODES", None)
    resolve = getattr(module, "resolve_glycam_sequence", None)
    if not isinstance(base_codes, dict) or not callable(resolve):
        raise AuditError("resolver lacks GLYCAM_BASE_CODES or resolve_glycam_sequence")
    observed = {target: base_codes.get(target) for target in TARGET_NAMES}
    if observed == CORRECT_CODES:
        state = "corrected_standard_case"
    elif observed == LEGACY_WRONG_CODES:
        state = "legacy_wrong_case"
    else:
        raise AuditError(
            "resolver has mixed/unknown Fuc-Rha mapping; refusing inference: "
            + json.dumps(observed, ensure_ascii=False, sort_keys=True)
        )
    return resolve, state, source_sha, source


def snapshot_pickle_dir(path: Path) -> dict[str, tuple[int, int]]:
    if not path.is_dir():
        raise AuditError(f"missing global pickle directory: {path}")
    result: dict[str, tuple[int, int]] = {}
    for item in sorted(path.glob("*.pickle")):
        if not item.is_file():
            continue
        stat = item.stat()
        result[item.name] = (stat.st_size, stat.st_mtime_ns)
    if not result:
        raise AuditError(f"global pickle directory contains no .pickle files: {path}")
    return result


def discover_run_roots(md_root: Path, explicit: Sequence[Path]) -> list[Path]:
    roots = [path.resolve() for path in explicit]
    if not roots:
        roots = sorted(path.resolve() for path in md_root.glob("md_runs*") if path.is_dir())
    if not roots:
        raise AuditError(f"no run roots found below {md_root}")
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise AuditError(f"missing run roots: {missing}")
    if len(set(roots)) != len(roots):
        raise AuditError("duplicate run roots supplied")
    return roots


def discover_dataset_csvs(md_root: Path) -> list[Path]:
    """Find compact dataset manifests, excluding trajectory/result CSVs."""
    paths = {md_root / "all_sugars_forcefield_batch_feasibility.csv"}
    for directory in md_root.glob("md_*_dataset*"):
        if directory.is_dir():
            paths.update(directory.glob("*.csv"))
    paths.update((md_root / "md_small_sample_dataset_v1").glob("*.csv"))
    result = sorted(path.resolve() for path in paths if path.is_file())
    if not result:
        raise AuditError("no dataset metadata CSVs found")
    return result


def resolve_manifest_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_rerun_rows(
    impacted_rows: Sequence[dict[str, Any]],
    library_by_name: dict[str, LibraryRecord],
    dataset_csvs: Sequence[Path],
    root: Path,
) -> list[dict[str, Any]]:
    """Merge authoritative dataset metadata for the 167 resolvable names."""
    wanted = {
        row["molecule_name"]: row
        for row in impacted_rows
        if row["legacy_resolver_status"] == "resolved_wrong"
    }
    clean_to_name = {row["clean_name"]: name for name, row in wanted.items()}
    candidates: dict[str, list[tuple[Path, dict[str, str]]]] = defaultdict(list)
    for csv_path in dataset_csvs:
        try:
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not {"molecule_name", "clean_name"}.issubset(
                    reader.fieldnames
                ):
                    continue
                for raw in reader:
                    molecule = (raw.get("molecule_name") or "").strip()
                    clean = (raw.get("clean_name") or "").strip()
                    matches: set[str] = set()
                    if molecule in wanted:
                        matches.add(molecule)
                    if clean in clean_to_name:
                        matches.add(clean_to_name[clean])
                    if len(matches) > 1:
                        raise AuditError(
                            f"dataset CSV identity conflict in {csv_path}: molecule={molecule!r}, clean={clean!r}"
                        )
                    if matches:
                        candidates[next(iter(matches))].append((csv_path, raw))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise AuditError(f"cannot read dataset CSV {csv_path}: {exc}") from exc

    output: list[dict[str, Any]] = []
    for priority, source_row in enumerate(
        sorted(wanted.values(), key=lambda item: int(item["library_index"])), start=1
    ):
        name = source_row["molecule_name"]
        record = library_by_name[name]
        matches = candidates.get(name, [])
        if not matches:
            raise AuditError(f"no existing dataset CSV metadata for rerun molecule {name}")

        mmff_paths: set[Path] = set()
        stated_source_classes: set[str] = set()
        stated_source_sdfs: set[Path] = set()
        metadata_sources: list[str] = []
        for csv_path, raw in matches:
            metadata_sources.append(str(csv_path))
            mmff_value = (raw.get("mmff100_pickle") or "").strip()
            if mmff_value:
                mmff_paths.add(resolve_manifest_path(mmff_value, root))
            source_class = (raw.get("source_class") or raw.get("dataset") or "").strip()
            if source_class:
                stated_source_classes.add(source_class)
            source_sdf = (raw.get("source_sdf") or "").strip()
            if source_sdf:
                stated_source_sdfs.add(resolve_manifest_path(source_sdf, root))

        if len(mmff_paths) != 1:
            raise AuditError(f"missing/conflicting mmff100_pickle metadata for {name}: {sorted(map(str, mmff_paths))}")
        mmff_path = next(iter(mmff_paths))
        if not mmff_path.is_file():
            raise AuditError(f"mmff100_pickle metadata points to a missing file for {name}: {mmff_path}")
        if stated_source_classes != {record.source_class}:
            raise AuditError(
                f"missing/conflicting source_class metadata for {name}: {sorted(stated_source_classes)}"
            )
        if stated_source_sdfs and stated_source_sdfs != {record.source_sdf}:
            raise AuditError(
                f"conflicting source_sdf metadata for {name}: {sorted(map(str, stated_source_sdfs))}"
            )
        output.append(
            {
                "priority": priority,
                "molecule_name": name,
                "clean_name": record.clean_name,
                "source_class": record.source_class,
                "source_sdf": str(record.source_sdf),
                "mmff100_pickle": str(mmff_path),
                "mmff100_pickle_exists": True,
                "library_index": record.library_index,
                "legacy_wrong_sequence": " ".join(source_row["legacy_wrong_sequence"]),
                "correct_sequence": " ".join(source_row["correct_sequence"]),
                "rerun_reason": "GLYCAM Fuc/Rha D/L residue-code case correction",
                "metadata_sources": "|".join(sorted(set(metadata_sources))),
            }
        )
    if len(output) != len(wanted):
        raise AuditError(f"rerun manifest row loss: expected {len(wanted)}, built {len(output)}")
    return output


def metadata_paths(run_roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for run_root in run_roots:
        paths.update(run_root.rglob("glycam06j_topology_metadata.json"))
    return sorted(path.resolve() for path in paths if path.is_file())


def file_stat_manifest(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for path in paths:
        stat = path.stat()
        result[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
    return result


def extract_leap_sequence_tokens(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="strict")
    tokens: list[str] = []
    for block in LEAP_SEQUENCE_RE.findall(text):
        tokens.extend(block.split())
    return tokens or None


def locate_leap_input(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    """Prefer the historical filename, then an explicit or unique builder variant."""
    exact = metadata_path.with_name("build_glycam06j.leap")
    if exact.is_file():
        return exact.resolve()
    candidates: set[Path] = set()
    explicit = metadata.get("leap_input")
    if isinstance(explicit, str) and explicit:
        explicit_path = Path(explicit)
        if not explicit_path.is_absolute():
            explicit_path = metadata_path.parent / explicit_path
        if explicit_path.is_file():
            candidates.add(explicit_path.resolve())
    candidates.update(path.resolve() for path in metadata_path.parent.glob("build_glycam06j*.leap") if path.is_file())
    if len(candidates) == 1:
        return next(iter(candidates))
    # Return the canonical missing path so the caller records fail-closed unknown evidence.
    return exact.resolve()


def classify_exact_sequence(
    actual: list[str] | None,
    legacy_sequence: Sequence[str],
    correct_sequence: Sequence[str],
) -> str:
    if actual is None:
        return "missing_or_invalid"
    if actual == list(legacy_sequence):
        return "legacy_wrong"
    if actual == list(correct_sequence):
        return "correct"
    return "legacy_unknown"


def classify_leap_target_case(
    actual: list[str] | None,
    legacy_sequence: Sequence[str],
    correct_sequence: Sequence[str],
) -> str:
    if actual is None:
        return "missing_or_invalid"
    actual_codes = sequence_target_counter(actual)
    legacy_codes = sequence_target_counter(legacy_sequence)
    correct_codes = sequence_target_counter(correct_sequence)
    if actual_codes == legacy_codes and actual_codes != correct_codes:
        return "legacy_wrong"
    if actual_codes == correct_codes and actual_codes != legacy_codes:
        return "correct"
    return "legacy_unknown"


def is_first_letter_legacy(name: str, sequences: Iterable[Sequence[str] | None]) -> bool:
    letters: set[str] = set()
    for sequence in sequences:
        if sequence is None:
            continue
        for token in sequence:
            match = ANY_RESIDUE_RE.fullmatch(token)
            if match:
                letters.add(match.group(2))
    return ("DFuc" in name and "D" in letters) or ("DRha" in name and "R" in letters)


def scan_artifacts(
    run_roots: Sequence[Path],
    impacted: Sequence[LibraryRecord],
    legacy_sequences: dict[str, list[str]],
    correct_sequences: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, tuple[int, int]]]:
    by_name = {record.molecule_name: record for record in impacted}
    by_clean = {record.clean_name: record for record in impacted}
    paths = metadata_paths(run_roots)
    if not paths:
        raise AuditError("no glycam06j_topology_metadata.json files found in run roots")
    manifest_paths: list[Path] = list(paths)
    artifacts: list[dict[str, Any]] = []
    unassociated: list[dict[str, str]] = []

    root_for_path: dict[Path, Path] = {}
    for path in paths:
        matches = [root for root in run_roots if path.is_relative_to(root)]
        if len(matches) != 1:
            raise AuditError(f"metadata path belongs to {len(matches)} run roots: {path}")
        root_for_path[path] = matches[0]

    for metadata_path in paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AuditError(f"invalid topology metadata {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise AuditError(f"topology metadata is not a JSON object: {metadata_path}")

        candidates: set[str] = set()
        molecule_value = metadata.get("molecule_name")
        clean_value = metadata.get("clean_name")
        if isinstance(molecule_value, str) and molecule_value in by_name:
            candidates.add(molecule_value)
        if isinstance(clean_value, str) and clean_value in by_clean:
            candidates.add(by_clean[clean_value].molecule_name)
        molecule_dir = metadata_path.parents[1].name
        if molecule_dir in by_clean:
            candidates.add(by_clean[molecule_dir].molecule_name)
        if len(candidates) > 1:
            raise AuditError(f"conflicting molecule identity in {metadata_path}: {sorted(candidates)}")
        if not candidates:
            displayed = " ".join(
                value for value in (molecule_value, clean_value, molecule_dir) if isinstance(value, str)
            )
            if any(target in displayed for target in TARGET_NAMES):
                unassociated.append(
                    {
                        "metadata_path": str(metadata_path),
                        "molecule_name": str(molecule_value or ""),
                        "clean_name": str(clean_value or molecule_dir),
                    }
                )
            continue

        name = next(iter(candidates))
        if name not in legacy_sequences:
            # Unresolved library names cannot have evidence attributed to this resolver error.
            continue
        legacy_sequence = legacy_sequences[name]
        correct_sequence = correct_sequences[name]
        metadata_sequence = normalize_sequence(metadata.get("glycam_sequence"))
        leap_path = locate_leap_input(metadata_path, metadata)
        if leap_path.is_file():
            manifest_paths.append(leap_path.resolve())
        leap_sequence = extract_leap_sequence_tokens(leap_path)
        metadata_class = classify_exact_sequence(metadata_sequence, legacy_sequence, correct_sequence)
        leap_class = classify_leap_target_case(leap_sequence, legacy_sequence, correct_sequence)
        if metadata_class == "legacy_wrong" and leap_class == "legacy_wrong":
            artifact_class = "wrong_topology_evidence"
            reason = "metadata_exact_legacy_and_leap_target_case_legacy"
        elif metadata_class == "correct" and leap_class == "correct":
            artifact_class = "correct_topology_evidence"
            reason = "metadata_exact_correct_and_leap_target_case_correct"
        else:
            artifact_class = "legacy_unknown"
            if is_first_letter_legacy(name, (metadata_sequence, leap_sequence)):
                reason = "legacy_first_letter_D_or_R_code"
            else:
                reason = "metadata_or_leap_does_not_match_frozen_case_baselines"
        artifacts.append(
            {
                "molecule_name": name,
                "clean_name": by_name[name].clean_name,
                "run_root": str(root_for_path[metadata_path]),
                "metadata_path": str(metadata_path),
                "leap_path": str(leap_path.resolve()),
                "metadata_sequence": metadata_sequence,
                "leap_sequence_tokens": leap_sequence,
                "metadata_class": metadata_class,
                "leap_class": leap_class,
                "artifact_class": artifact_class,
                "reason": reason,
            }
        )

    manifest = file_stat_manifest(manifest_paths)
    return artifacts, unassociated, manifest


def ensure_output_dir_is_safe(output_dir: Path, input_dirs: Sequence[Path]) -> Path:
    output = output_dir.resolve()
    for source in input_dirs:
        source = source.resolve()
        if output == source:
            raise AuditError(f"output directory collides with input directory: {output}")
    if output.exists() and not output.is_dir():
        raise AuditError(f"output path is not a directory: {output}")
    if output.is_dir():
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in REPORT_FILES)
        if unexpected:
            raise AuditError(f"output directory contains unrelated files: {unexpected}")
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "library_index",
        "source_class",
        "source_record_index",
        "source_sdf",
        "molecule_name",
        "clean_name",
        "target_tokens",
        "target_token_occurrences",
        "legacy_resolver_status",
        "legacy_wrong_sequence",
        "correct_sequence",
        "all_pickle_exists",
        "all_pickle_path",
        "low_pickle_exists",
        "low_pickle_path",
        "both_pickles_exist",
        "raw_sdf_title_both_pickles_exist",
        "wrong_topology_evidence",
        "wrong_topology_evidence_count",
        "wrong_metadata_paths",
        "wrong_leap_paths",
        "correct_topology_evidence",
        "legacy_unknown_artifact_count",
        "scope_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "|".join(str(item) for item in row[field])
                        if isinstance(row.get(field), list)
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def write_rerun_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "priority",
        "molecule_name",
        "clean_name",
        "source_class",
        "source_sdf",
        "mmff100_pickle",
        "mmff100_pickle_exists",
        "library_index",
        "legacy_wrong_sequence",
        "correct_sequence",
        "rerun_reason",
        "metadata_sources",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# GLYCAM Fuc/Rha D/L 大小写错误影响范围审计",
        "",
        f"审计结论：375 个 mono/dimer/trimer 全库名称中，**{summary['target_name_total']}** 个含 LFuc、LRha、DFuc 或 DRha；旧 resolver 可解析且会发生大小写映射错误的有 **{summary['legacy_resolver_resolved_wrong']}** 个。",
        "",
        f"其中 **{summary['both_global_pickles']}** 个已经同时存在全局 all 与 low 两份 GLYCAM pickle，**{summary['wrong_topology_evidence']}** 个可由成对 metadata + build_glycam06j.leap 证明使用了旧错误大小写；另有 **{summary['unresolved']}** 个旧 resolver 未解析。",
        "",
        "## 映射基线",
        "",
        "| 糖残基 | 旧错误代码 | 正确 GLYCAM 代码 |",
        "|---|---:|---:|",
    ]
    for target in ("DFuc", "LFuc", "DRha", "LRha"):
        lines.append(f"| {target} | `{LEGACY_WRONG_CODES[target]}` | `{CORRECT_CODES[target]}` |")
    lines.extend(
        [
            "",
            "旧错误拓扑证据采用严格定义：同一构建目录的 `glycam06j_topology_metadata.json` 必须完整等于冻结的旧 resolver 序列，且 `build_glycam06j.leap` 中 Fuc/Rha 残基代码计数也必须符合旧大小写。只满足一项不会计为错误证据。",
            "",
            "## 总计",
            "",
            "| 指标 | 分子数 |",
            "|---|---:|",
            f"| 全库 | {summary['library_total']} |",
            f"| 名称含目标残基 | {summary['target_name_total']} |",
            f"| 旧 resolver 可解析且会错 | {summary['legacy_resolver_resolved_wrong']} |",
            f"| all + low 两份 pickle 均存在 | {summary['both_global_pickles']} |",
            f"| 有严格错误拓扑证据 | {summary['wrong_topology_evidence']} |",
            f"| 已有两份 pickle 且有错误证据 | {summary['both_pickles_and_wrong_evidence']} |",
            f"| 可解析、有错误证据但尚无两份 pickle | {summary['wrong_evidence_without_pickle_pair']} |",
            f"| 未解析 | {summary['unresolved']} |",
            "",
            "## 按数据来源",
            "",
            "| 来源 | 全库 | 含目标名称 | 可解析且会错 | 两份 pickle | 错误证据 | 未解析 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source_class in ("mono", "dimer", "trimer"):
        item = report["source_breakdown"][source_class]
        lines.append(
            f"| {source_class} | {item['library_total']} | {item['target_names']} | "
            f"{item['resolved_wrong']} | {item['both_pickles']} | {item['wrong_evidence']} | {item['unresolved']} |"
        )
    lines.extend(
        [
            "",
            "## 按目标名称片段（类别可重叠）",
            "",
            "| 名称片段 | 含该片段的分子 | 片段出现次数 | 可解析且会错 | 两份 pickle | 错误证据 | 未解析 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target in TARGET_NAMES:
        item = report["token_breakdown"][target]
        lines.append(
            f"| {target} | {item['names_containing']} | {item['occurrences']} | "
            f"{item['resolved_wrong']} | {item['both_pickles']} | {item['wrong_evidence']} | {item['unresolved']} |"
        )
    lines.extend(["", f"## 未解析的 {summary['unresolved']} 个名称", ""])
    for name in report["unresolved_names"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## 不纳入 167 个大小写错误计数的历史异常构建记录",
            "",
            f"共 {summary['legacy_unknown_artifact_records']} 条 `legacy_unknown` 构建记录；其中 {summary['legacy_first_letter_artifact_records']} 条是 v10 的 DFuc→D / DRha→R 首字母旧代码。它们是额外历史异常证据，不用于扩大 167 个分子范围。",
            "",
        ]
    )
    for artifact in report["legacy_unknown_artifacts"]:
        lines.append(
            f"- `{artifact['molecule_name']}` — `{artifact['reason']}` — `{artifact['metadata_path']}`"
        )
    lines.extend(
        [
            "",
            "## 输入快照",
            "",
            f"- resolver 状态：`{report['inputs']['resolver_mapping_state']}`",
            f"- resolver SHA256：`{report['inputs']['resolver_sha256']}`",
            f"- 扫描 run roots：{len(report['inputs']['run_roots'])}",
            f"- 扫描 GLYCAM metadata：{summary['metadata_files_scanned']}",
            f"- 全局 all pickle 总数：{summary['all_pickle_directory_total']}；low pickle 总数：{summary['low_pickle_directory_total']}",
            "",
            f"说明：若错误地用带 `(1->N)` 的原始 SDF 标题直接查 pickle，只会命中 {summary['raw_sdf_title_both_pickles']} 个；生产 pickle 使用 compact `clean_name`，正确的成对命中数是 {summary['both_global_pickles']}。前者不是有效范围计数。",
            "",
            "逐分子路径与分类见同目录审计 CSV；`glycam_enantiomer_rerun_167.csv` 是按全库顺序生成、可供 v15 builder/runner 使用的 167 行重建清单；完整证据与输入哈希见 JSON。本审计没有修改任何 SDF、resolver、pickle、拓扑或运行目录。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sdfs = {
        "mono": args.mono_sdf.resolve(),
        "dimer": args.dimer_sdf.resolve(),
        "trimer": args.trimer_sdf.resolve(),
    }
    library = read_library(sdfs, args.expected_library_size)
    impacted = [record for record in library if record.target_tokens]
    if not impacted:
        raise AuditError("no LFuc/LRha/DFuc/DRha names found in source library")

    resolve, resolver_state, resolver_sha, resolver_source = load_resolver_snapshot(args.resolver)
    correct_sequences: dict[str, list[str]] = {}
    legacy_sequences: dict[str, list[str]] = {}
    for record in impacted:
        raw_sequence = resolve(record.molecule_name)
        if raw_sequence is None:
            continue
        if not isinstance(raw_sequence, list) or not all(isinstance(item, str) for item in raw_sequence):
            raise AuditError(f"resolver returned an invalid sequence for {record.molecule_name!r}")
        expected_mapping = CORRECT_CODES if resolver_state == "corrected_standard_case" else LEGACY_WRONG_CODES
        if sequence_target_counter(raw_sequence) != expected_target_counter(record.molecule_name, expected_mapping):
            raise AuditError(
                f"resolver sequence has mixed/inconsistent target codes for {record.molecule_name}: {raw_sequence}"
            )
        if resolver_state == "corrected_standard_case":
            correct_sequence = list(raw_sequence)
            legacy_sequence = swap_enantiomer_case(raw_sequence)
        else:
            legacy_sequence = list(raw_sequence)
            correct_sequence = swap_enantiomer_case(raw_sequence)
        if legacy_sequence == correct_sequence:
            raise AuditError(f"legacy/correct sequences did not differ for {record.molecule_name}")
        if sequence_target_counter(correct_sequence) != expected_target_counter(record.molecule_name, CORRECT_CODES):
            raise AuditError(f"cannot construct correct sequence for {record.molecule_name}")
        if sequence_target_counter(legacy_sequence) != expected_target_counter(
            record.molecule_name, LEGACY_WRONG_CODES
        ):
            raise AuditError(f"cannot reconstruct frozen legacy sequence for {record.molecule_name}")
        correct_sequences[record.molecule_name] = correct_sequence
        legacy_sequences[record.molecule_name] = legacy_sequence

    all_snapshot_before = snapshot_pickle_dir(args.all_pickle_dir)
    low_snapshot_before = snapshot_pickle_dir(args.low_pickle_dir)
    run_roots = discover_run_roots(args.md_root, args.run_root)
    artifacts, unassociated_artifacts, artifact_manifest_before = scan_artifacts(
        run_roots, impacted, legacy_sequences, correct_sequences
    )

    artifacts_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_name[artifact["molecule_name"]].append(artifact)

    rows: list[dict[str, Any]] = []
    for record in impacted:
        resolved = record.molecule_name in legacy_sequences
        all_path = args.all_pickle_dir.resolve() / f"{record.clean_name}.pickle"
        low_path = args.low_pickle_dir.resolve() / f"{record.clean_name}.pickle"
        raw_all_path = args.all_pickle_dir.resolve() / f"{record.molecule_name}.pickle"
        raw_low_path = args.low_pickle_dir.resolve() / f"{record.molecule_name}.pickle"
        all_exists = all_path.is_file()
        low_exists = low_path.is_file()
        pair_exists = all_exists and low_exists
        molecule_artifacts = artifacts_by_name.get(record.molecule_name, [])
        wrong_artifacts = [
            item for item in molecule_artifacts if item["artifact_class"] == "wrong_topology_evidence"
        ]
        correct_artifacts = [
            item for item in molecule_artifacts if item["artifact_class"] == "correct_topology_evidence"
        ]
        unknown_artifacts = [item for item in molecule_artifacts if item["artifact_class"] == "legacy_unknown"]
        wrong_evidence = bool(wrong_artifacts)
        if not resolved:
            category = "unresolved"
        elif pair_exists and wrong_evidence:
            category = "resolved_wrong_with_pickle_pair_and_wrong_topology_evidence"
        elif wrong_evidence:
            category = "resolved_wrong_with_wrong_topology_evidence_no_pickle_pair"
        elif pair_exists:
            category = "resolved_wrong_with_pickle_pair_no_strict_topology_evidence"
        else:
            category = "resolved_wrong_no_pickle_pair_or_strict_topology_evidence"
        rows.append(
            {
                "library_index": record.library_index,
                "source_class": record.source_class,
                "source_record_index": record.source_record_index,
                "source_sdf": str(record.source_sdf),
                "molecule_name": record.molecule_name,
                "clean_name": record.clean_name,
                "target_tokens": list(record.target_tokens),
                "target_token_occurrences": sum(record.molecule_name.count(token) for token in TARGET_NAMES),
                "legacy_resolver_status": "resolved_wrong" if resolved else "unresolved",
                "legacy_wrong_sequence": legacy_sequences.get(record.molecule_name, []),
                "correct_sequence": correct_sequences.get(record.molecule_name, []),
                "all_pickle_exists": all_exists,
                "all_pickle_path": str(all_path),
                "low_pickle_exists": low_exists,
                "low_pickle_path": str(low_path),
                "both_pickles_exist": pair_exists,
                "raw_sdf_title_both_pickles_exist": raw_all_path.is_file() and raw_low_path.is_file(),
                "wrong_topology_evidence": wrong_evidence,
                "wrong_topology_evidence_count": len(wrong_artifacts),
                "wrong_metadata_paths": [item["metadata_path"] for item in wrong_artifacts],
                "wrong_leap_paths": [item["leap_path"] for item in wrong_artifacts],
                "correct_topology_evidence": bool(correct_artifacts),
                "legacy_unknown_artifact_count": len(unknown_artifacts),
                "scope_category": category,
            }
        )

    unresolved_names = [row["molecule_name"] for row in rows if row["legacy_resolver_status"] == "unresolved"]
    legacy_unknown = [item for item in artifacts if item["artifact_class"] == "legacy_unknown"]
    summary = {
        "library_total": len(library),
        "target_name_total": len(rows),
        "legacy_resolver_resolved_wrong": sum(
            row["legacy_resolver_status"] == "resolved_wrong" for row in rows
        ),
        "all_global_pickle_exists": sum(row["all_pickle_exists"] for row in rows),
        "low_global_pickle_exists": sum(row["low_pickle_exists"] for row in rows),
        "both_global_pickles": sum(row["both_pickles_exist"] for row in rows),
        "raw_sdf_title_both_pickles": sum(
            row["raw_sdf_title_both_pickles_exist"] for row in rows
        ),
        "wrong_topology_evidence": sum(row["wrong_topology_evidence"] for row in rows),
        "wrong_topology_evidence_records": sum(
            item["artifact_class"] == "wrong_topology_evidence" for item in artifacts
        ),
        "both_pickles_and_wrong_evidence": sum(
            row["both_pickles_exist"] and row["wrong_topology_evidence"] for row in rows
        ),
        "wrong_evidence_without_pickle_pair": sum(
            row["wrong_topology_evidence"] and not row["both_pickles_exist"] for row in rows
        ),
        "unresolved": len(unresolved_names),
        "correct_topology_evidence": sum(row["correct_topology_evidence"] for row in rows),
        "legacy_unknown_artifact_records": len(legacy_unknown),
        "legacy_first_letter_artifact_records": sum(
            item["reason"] == "legacy_first_letter_D_or_R_code" for item in legacy_unknown
        ),
        "unassociated_target_artifact_records": len(unassociated_artifacts),
        "metadata_files_scanned": len(metadata_paths(run_roots)),
        "target_related_artifact_records": len(artifacts),
        "all_pickle_directory_total": len(all_snapshot_before),
        "low_pickle_directory_total": len(low_snapshot_before),
    }
    if summary["target_name_total"] != (
        summary["legacy_resolver_resolved_wrong"] + summary["unresolved"]
    ):
        raise AuditError("resolved/unresolved partition does not cover impacted names")
    if summary["both_pickles_and_wrong_evidence"] > summary["both_global_pickles"]:
        raise AuditError("invalid pickle/evidence intersection")
    if summary["wrong_topology_evidence"] > summary["legacy_resolver_resolved_wrong"]:
        raise AuditError("wrong topology evidence was attributed to an unresolved name")

    source_breakdown: dict[str, dict[str, int]] = {}
    for source_class in ("mono", "dimer", "trimer"):
        source_rows = [row for row in rows if row["source_class"] == source_class]
        source_breakdown[source_class] = {
            "library_total": sum(record.source_class == source_class for record in library),
            "target_names": len(source_rows),
            "resolved_wrong": sum(row["legacy_resolver_status"] == "resolved_wrong" for row in source_rows),
            "both_pickles": sum(row["both_pickles_exist"] for row in source_rows),
            "wrong_evidence": sum(row["wrong_topology_evidence"] for row in source_rows),
            "unresolved": sum(row["legacy_resolver_status"] == "unresolved" for row in source_rows),
        }

    token_breakdown: dict[str, dict[str, int]] = {}
    for target in TARGET_NAMES:
        target_rows = [row for row in rows if target in row["molecule_name"]]
        token_breakdown[target] = {
            "names_containing": len(target_rows),
            "occurrences": sum(row["molecule_name"].count(target) for row in target_rows),
            "resolved_wrong": sum(row["legacy_resolver_status"] == "resolved_wrong" for row in target_rows),
            "both_pickles": sum(row["both_pickles_exist"] for row in target_rows),
            "wrong_evidence": sum(row["wrong_topology_evidence"] for row in target_rows),
            "unresolved": sum(row["legacy_resolver_status"] == "unresolved" for row in target_rows),
        }

    dataset_csvs = discover_dataset_csvs(args.md_root)
    library_by_name = {record.molecule_name: record for record in library}
    rerun_rows = build_rerun_rows(rows, library_by_name, dataset_csvs, ROOT)
    if len(rerun_rows) != summary["legacy_resolver_resolved_wrong"]:
        raise AuditError(
            "rerun manifest does not exactly cover legacy-resolver wrong names: "
            f"{len(rerun_rows)} != {summary['legacy_resolver_resolved_wrong']}"
        )
    summary["rerun_manifest_rows"] = len(rerun_rows)

    report = {
        "schema_version": 1,
        "audit_status": "complete_fail_closed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "legacy_baseline": {
            "description": "historical resolver case mapping; frozen for scope reconstruction",
            "wrong_codes": LEGACY_WRONG_CODES,
            "correct_codes": CORRECT_CODES,
        },
        "inputs": {
            "sdfs": {
                key: {"path": str(path), "sha256": sha256_path(path)} for key, path in sdfs.items()
            },
            "resolver": str(args.resolver.resolve()),
            "resolver_sha256": resolver_sha,
            "resolver_mapping_state": resolver_state,
            "all_pickle_dir": str(args.all_pickle_dir.resolve()),
            "low_pickle_dir": str(args.low_pickle_dir.resolve()),
            "run_roots": [str(path) for path in run_roots],
            "dataset_metadata_csvs": [
                {"path": str(path), "sha256": sha256_path(path)} for path in dataset_csvs
            ],
        },
        "summary": summary,
        "source_breakdown": source_breakdown,
        "token_breakdown": token_breakdown,
        "unresolved_names": unresolved_names,
        "legacy_unknown_artifacts": legacy_unknown,
        "unassociated_target_artifacts": unassociated_artifacts,
        "artifacts": artifacts,
        "records": rows,
    }

    stability = {
        "resolver_source": resolver_source,
        "resolver_sha256": resolver_sha,
        "all_pickle_manifest": all_snapshot_before,
        "low_pickle_manifest": low_snapshot_before,
        "artifact_manifest": artifact_manifest_before,
        "metadata_paths": [str(path) for path in metadata_paths(run_roots)],
        "run_roots": run_roots,
        "dataset_csv_manifest": file_stat_manifest(dataset_csvs),
    }
    return report, rows, rerun_rows, stability


def verify_stability(args: argparse.Namespace, stability: dict[str, Any]) -> None:
    if sha256_path(args.resolver) != stability["resolver_sha256"]:
        raise AuditError("resolver source changed during audit")
    if snapshot_pickle_dir(args.all_pickle_dir) != stability["all_pickle_manifest"]:
        raise AuditError("all-pickle directory changed during audit")
    if snapshot_pickle_dir(args.low_pickle_dir) != stability["low_pickle_manifest"]:
        raise AuditError("low-pickle directory changed during audit")
    current_metadata = [str(path) for path in metadata_paths(stability["run_roots"])]
    if current_metadata != stability["metadata_paths"]:
        raise AuditError("topology metadata path set changed during audit")
    current_manifest = file_stat_manifest(Path(path) for path in stability["artifact_manifest"])
    if current_manifest != stability["artifact_manifest"]:
        raise AuditError("topology metadata/LEaP evidence changed during audit")
    current_dataset_manifest = file_stat_manifest(
        Path(path) for path in stability["dataset_csv_manifest"]
    )
    if current_dataset_manifest != stability["dataset_csv_manifest"]:
        raise AuditError("dataset metadata CSVs changed during audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only GLYCAM Fuc/Rha D/L case-error scope audit."
    )
    parser.add_argument("--mono-sdf", type=Path, default=DEFAULT_SDFS["mono"])
    parser.add_argument("--dimer-sdf", type=Path, default=DEFAULT_SDFS["dimer"])
    parser.add_argument("--trimer-sdf", type=Path, default=DEFAULT_SDFS["trimer"])
    parser.add_argument("--resolver", type=Path, default=DEFAULT_RESOLVER)
    parser.add_argument("--all-pickle-dir", type=Path, default=DEFAULT_ALL_PICKLES)
    parser.add_argument("--low-pickle-dir", type=Path, default=DEFAULT_LOW_PICKLES)
    parser.add_argument("--md-root", type=Path, default=MD_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Explicit run root to scan; repeat to override automatic md_runs* discovery.",
    )
    parser.add_argument("--expected-library-size", type=int, default=375)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.expected_library_size <= 0:
            raise AuditError("--expected-library-size must be positive")
        output_dir = ensure_output_dir_is_safe(
            args.output_dir,
            [
                args.all_pickle_dir,
                args.low_pickle_dir,
                args.md_root,
                args.resolver.parent,
                args.mono_sdf.parent,
                args.dimer_sdf.parent,
                args.trimer_sdf.parent,
            ],
        )
        report, rows, rerun_rows, stability = build_report(args)
        verify_stability(args, stability)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / REPORT_JSON
        csv_path = output_dir / REPORT_CSV
        markdown_path = output_dir / REPORT_MD
        rerun_path = output_dir / RERUN_CSV
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_csv(csv_path, rows)
        write_rerun_csv(rerun_path, rerun_rows)
        markdown_path.write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "summary": report["summary"]}, ensure_ascii=False))
        return 0
    except AuditError as exc:
        print(f"AUDIT FAILED CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
