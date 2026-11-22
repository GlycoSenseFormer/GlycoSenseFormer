#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import deque
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = MD_ROOT / "md_runs" / "bDGlc14bDGlcOMe"
MOL = "bDGlc14bDGlcOMe"
REFERENCE_SDF = ROOT / "data" / "dataset" / "casper_dimer.sdf"
GMX = os.environ.get("GMX", "gmx")
GMX_NTOMP = os.environ.get("GMX_NTOMP", "4")
GMX_CPT_MINUTES = os.environ.get("GMX_CPT_MINUTES", "1")
MD_DEVICE = os.environ.get("MD_DEVICE", "gpu").lower()
TOPDIR = Path(os.environ.get("GMX_TOPDIR", "/soft/gromacs-2026.1/share/gromacs/top"))

if MD_DEVICE not in {"gpu", "cpu"}:
    raise ValueError(f"MD_DEVICE must be 'gpu' or 'cpu', got {MD_DEVICE!r}")


AMBER_TIP3P_ATOMTYPES = """\
[ atomtypes ]
; name      at.num  mass     charge ptype  sigma      epsilon
OW           8      16.00    0.0000  A   3.15061e-01  6.36386e-01
HW           1       1.008   0.0000  A   0.00000e+00  0.00000e+00

"""


# Charged-solute energy reruns are finite-molecule single-point calculations.
# A derived cutoff is needed only when the established family cutoff cannot
# include every pair.  The 0.020 nm margin is comfortably larger than the
# 0.001 nm coordinate precision of GRO files; rounding is always upward.
# Keeping these constants independent of force-field family makes the safety
# rule identical for CHARMM, GLYCAM/AMBER, and GROMOS.
CHARGED_RERUN_PROTOCOL_ID = "charged-solute-native-plus-allpairs-coulomb-v3"
CHARGED_RERUN_RETIRED_PROTOCOL_ID = "charged-solute-all-pairs-dynamic-cutoff-v2"
CHARGED_RERUN_SPAN_MARGIN_NM = 0.020
CHARGED_RERUN_CUTOFF_QUANTUM_NM = 0.010
CHARGED_RERUN_MDP = "rerun_solute_energy_isolated_charged_v2.mdp"
CHARGED_NATIVE_DEFFNM = "rerun_solute_energy_native"
CHARGED_COULOMB_DEFFNM = "rerun_solute_energy_coulomb_allpairs"


class GpuFatalError(RuntimeError):
    """A GPU/driver failure for which another GPU attempt is unsafe."""


SCIENTIFIC_MDRUN_PATTERNS = (
    re.compile(r"LINCS WARNING", re.IGNORECASE),
    re.compile(r"Too many LINCS", re.IGNORECASE),
    re.compile(r"constraint error", re.IGNORECASE),
    re.compile(r"non-finite", re.IGNORECASE),
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"pressure scaling more than", re.IGNORECASE),
)


def scientific_mdrun_markers(text: str) -> list[str]:
    return [pattern.pattern for pattern in SCIENTIFIC_MDRUN_PATTERNS if pattern.search(text)]


def run_mdrun(cmd: list[str], cwd: Path) -> None:
    print(f"\n[run] {cwd}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=500)
    output_chunks: list[str] = []
    console_path: Path | None = None
    if "-deffnm" in cmd:
        deffnm_index = cmd.index("-deffnm") + 1
        if deffnm_index < len(cmd):
            console_path = cwd / f"{cmd[deffnm_index]}.console.log"
    console_handle = None
    if console_path is not None:
        mode = "a" if "-append" in cmd else "w"
        console_handle = console_path.open(mode, encoding="utf-8")
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            output_chunks.append(line)
            if console_handle is not None:
                console_handle.write(line)
                console_handle.flush()
    finally:
        if console_handle is not None:
            console_handle.close()
    returncode = proc.wait()
    output = "".join(output_chunks)
    markers = scientific_mdrun_markers(output)
    if returncode != 0 or markers:
        # GROMACS can finish with return code zero after one or more LINCS
        # warnings.  Such a trajectory is a failed scientific gate, not a
        # successful smoke test or a resumable production epoch.
        raise subprocess.CalledProcessError(
            returncode or 86,
            cmd,
            output="".join(tail),
        )


def classify_gpu_failure(error: subprocess.CalledProcessError) -> str:
    output = str(error.output or "").lower()
    fatal_markers = (
        "cuda #700",
        "cudaerrorillegaladdress",
        "illegal memory access",
        "cudastreamsynchronize failed",
        "updateconstraingpu",
        "segmentation fault",
        "sigsegv",
    )
    scientific_markers = (
        "too many lincs warnings",
        "lincs warning",
        "constraint error",
        "non-finite",
        "nan",
        "pressure scaling more than",
    )
    recoverable_markers = (
        "no cuda-capable device",
        "gpu acceleration is not available",
        "cannot run tasks on the gpu",
        "not supported on the gpu",
        "failed to initialize cuda",
        "cuda driver version is insufficient",
    )
    if error.returncode < 0 or any(marker in output for marker in fatal_markers):
        return "gpu_fatal"
    if any(marker in output for marker in scientific_markers):
        return "scientific"
    if any(marker in output for marker in recoverable_markers):
        return "recoverable_gpu_setup"
    return "unknown"


@dataclass
class ForceFieldJob:
    key: str
    label: str
    run_dir: Path
    solute_gro: Path
    water_top: Path
    solute_top: Path
    water_model: str
    family: str
    atom_index_map: Path | None = None
    completed_md: bool = False
    solute_charge: int = 0
    positive_ion: str = "NA"
    negative_ion: str = "CL"
    timestep_ps: float | None = None
    candidate_spacing_ps: float = 25.0
    grompp_warning_policy: str = "family_default"


def grompp_maxwarn(job: ForceFieldJob) -> int:
    if job.grompp_warning_policy == "zero":
        return 0
    if job.grompp_warning_policy != "family_default":
        raise ValueError(
            f"{job.key}: unsupported grompp warning policy "
            f"{job.grompp_warning_policy!r}"
        )
    return 1 if job.family == "gromos" else 0


def run(cmd: list[str], cwd: Path, input_text: str | None = None, maxwarn_ok: bool = False) -> None:
    print(f"\n[run] {cwd}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, input=input_text, text=True)
    if proc.returncode != 0 and not maxwarn_ok:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_grompp(job: ForceFieldJob, args: list[str]) -> None:
    cmd = [GMX, "grompp", *args]
    if "-maxwarn" not in cmd:
        cmd.extend(["-maxwarn", str(grompp_maxwarn(job))])
    print(f"\n[run] {job.run_dir}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=job.run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    warning_count = len(re.findall(r"^WARNING\s+\d+\s+\[", proc.stdout, flags=re.MULTILINE))
    expected_gromos_warning = "The GROMOS force fields have been parametrized with a physically" in proc.stdout
    if job.family == "gromos" and job.grompp_warning_policy == "family_default":
        if warning_count != 1 or not expected_gromos_warning:
            raise RuntimeError(
                f"{job.key}: grompp did not emit exactly the one allow-listed GROMOS twin-range warning"
            )
    elif warning_count:
        raise RuntimeError(f"{job.key}: grompp emitted {warning_count} unexpected warning(s)")


def run_grompp_for_neutralization(job: ForceFieldJob, args: list[str]) -> None:
    cmd = [GMX, "grompp", *args, "-maxwarn", "1"]
    print(f"\n[run] {job.run_dir}", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=job.run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    warning_count = len(re.findall(r"^WARNING\s+\d+\s+\[", proc.stdout, flags=re.MULTILINE))
    net_charge_warning = "You are using Ewald electrostatics in a system with net charge" in proc.stdout
    if warning_count != 1 or not net_charge_warning:
        raise RuntimeError(
            f"{job.key}: neutralization grompp did not emit exactly the one allow-listed net-charge warning"
        )


def copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)


def copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        return
    shutil.copytree(source, dest, symlinks=True)


def read_atom_count(gro: Path) -> int:
    lines = gro.read_text().splitlines()
    return int(lines[1].strip())


def maximum_pair_distance_nm(gro: Path) -> float:
    lines = gro.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    if len(lines) != atom_count + 3:
        raise ValueError(f"invalid GRO atom count in {gro}")
    coordinates = [
        (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        for line in lines[2 : 2 + atom_count]
    ]
    if atom_count < 2 or not all(
        math.isfinite(value) for coordinate in coordinates for value in coordinate
    ):
        raise ValueError(f"invalid/non-finite GRO coordinates in {gro}")
    return max(
        math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))
        for index, left in enumerate(coordinates)
        for right in coordinates[index + 1 :]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame_set(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def minimum_box_length_nm(gro: Path) -> float:
    fields = [float(value) for value in gro.read_text(encoding="utf-8").splitlines()[-1].split()]
    if len(fields) not in (3, 9) or any(value <= 0 for value in fields[:3]):
        raise ValueError(f"invalid GRO box in {gro}")
    if len(fields) == 3:
        return min(fields)
    first = (fields[0], fields[3], fields[4])
    second = (fields[5], fields[1], fields[6])
    third = (fields[7], fields[8], fields[2])

    def cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def norm(vector: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    second_cross_third = cross(second, third)
    volume = abs(sum(first[index] * second_cross_third[index] for index in range(3)))
    if volume <= 0:
        raise ValueError(f"degenerate triclinic GRO box in {gro}")
    heights = (
        volume / norm(second_cross_third),
        volume / norm(cross(third, first)),
        volume / norm(cross(first, second)),
    )
    return min(heights)


def charged_rerun_cutoff_plan(
    job: ForceFieldJob,
    frame_paths: list[Path],
) -> dict[str, object]:
    """Choose one all-frame direct-Coulomb cutoff for an isolated solute.

    The maximum raw-coordinate pair distance is valid because the extraction
    epoch first makes the solute whole with ``trjconv -pbc mol``.  Every frame
    must also retain a box strictly larger than twice the selected cutoff so
    that the periodic rerun cannot select a different molecular image.
    """

    if not frame_paths:
        raise ValueError(f"{job.key}: no frames supplied for charged rerun cutoff")
    frame_distances = [
        (maximum_pair_distance_nm(path), path.name) for path in frame_paths
    ]
    maximum_distance, maximum_distance_frame = max(frame_distances)
    base_cutoff_nm = 1.2 if job.family == "charmm" else 1.0
    required_cutoff_nm = maximum_distance + CHARGED_RERUN_SPAN_MARGIN_NM
    cutoff_units = math.ceil(
        required_cutoff_nm / CHARGED_RERUN_CUTOFF_QUANTUM_NM - 1.0e-12
    )
    rounded_cutoff_nm = round(
        cutoff_units * CHARGED_RERUN_CUTOFF_QUANTUM_NM,
        6,
    )
    cutoff_nm = max(base_cutoff_nm, rounded_cutoff_nm)
    uses_derived_cutoff = cutoff_nm > base_cutoff_nm + 1.0e-12
    actual_margin_nm = cutoff_nm - maximum_distance
    if not math.isfinite(cutoff_nm) or (
        uses_derived_cutoff
        and actual_margin_nm + 1.0e-12 < CHARGED_RERUN_SPAN_MARGIN_NM
    ):
        raise RuntimeError(f"{job.key}: failed to construct a safe charged-solute cutoff")
    minimum_box_length = min(minimum_box_length_nm(path) for path in frame_paths)
    if minimum_box_length <= 2.0 * cutoff_nm:
        raise RuntimeError(
            f"{job.key}: minimum box length {minimum_box_length:.6f} nm is not "
            f"greater than twice the derived direct-Coulomb cutoff {cutoff_nm:.3f} nm"
        )
    return {
        "base_forcefield_cutoff_nm": base_cutoff_nm,
        "uses_derived_cutoff": uses_derived_cutoff,
        "required_span_margin_nm": CHARGED_RERUN_SPAN_MARGIN_NM,
        "cutoff_quantum_nm": CHARGED_RERUN_CUTOFF_QUANTUM_NM,
        "cutoff_nm": cutoff_nm,
        "cutoff_minus_maximum_pair_distance_nm": actual_margin_nm,
        "maximum_pair_distance_nm": maximum_distance,
        "maximum_pair_distance_frame": maximum_distance_frame,
        "minimum_box_length_nm": minimum_box_length,
        "minimum_image_margin_nm": minimum_box_length / 2.0 - cutoff_nm,
    }


def _replace_mdp_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\g<prefix>{value}", text)
    if count != 1:
        raise RuntimeError(f"expected exactly one {key!r} entry in rerun MDP, found {count}")
    return updated


def charged_rerun_mdp_block(job: ForceFieldJob, cutoff_nm: float) -> str:
    """Build an all-pairs Coulomb probe accepted by the Verlet scheme.

    Verlet requires equal finite Coulomb and VdW cutoffs for Cut-off
    electrostatics.  Therefore the probe expands rvdw too, but its VdW and
    total-energy terms are never used.  The final energy retains VdW from a
    separate native-cutoff rerun and takes only Coulomb-(SR) from this probe.
    """

    if not job.solute_charge:
        raise ValueError("charged rerun MDP requested for a neutral solute")
    if not math.isfinite(cutoff_nm) or cutoff_nm <= 0.0:
        raise ValueError(f"invalid charged rerun cutoff: {cutoff_nm!r}")
    text = mdp_block(job, "rerun")
    formatted_cutoff = f"{cutoff_nm:.3f}"
    text = _replace_mdp_value(text, "rlist", formatted_cutoff)
    text = _replace_mdp_value(text, "rcoulomb", formatted_cutoff)
    text = _replace_mdp_value(text, "rvdw", formatted_cutoff)
    return text


def write_exact_text(path: Path, text: str, *, label: str) -> None:
    """Atomically create an immutable derived protocol input."""

    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"existing {label} differs from the requested protocol: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_replace_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_safe_charged_protocol_id_upgrade(
    recorded: dict[str, object],
    expected: dict[str, object],
) -> bool:
    """Allow only the hash-identical outputs-bound v2-ID to v3-ID rename."""

    if (
        recorded.get("provenance_state") != "outputs_bound"
        or expected.get("provenance_state") != "outputs_bound"
        or recorded.get("protocol_id") != CHARGED_RERUN_RETIRED_PROTOCOL_ID
        or expected.get("protocol_id") != CHARGED_RERUN_PROTOCOL_ID
    ):
        return False
    recorded_without_id = dict(recorded)
    expected_without_id = dict(expected)
    recorded_without_id.pop("protocol_id", None)
    expected_without_id.pop("protocol_id", None)
    return recorded_without_id == expected_without_id


def _grompp_rerun_tpr(
    job: ForceFieldJob,
    *,
    mdp_name: str,
    first_frame: Path,
    output_name: str,
) -> Path:
    output = job.run_dir / output_name
    if output.exists():
        return output
    args = [
        "-f",
        mdp_name,
        "-c",
        str(first_frame),
        "-p",
        job.solute_top.name,
        "-o",
        output_name,
    ]
    if job.family != "gromos":
        args.extend(["-maxwarn", "0"])
    run_grompp(job, args)
    if not output.is_file():
        raise RuntimeError(f"{job.key}: grompp did not create {output_name}")
    return output


def _run_rerun_epoch(job: ForceFieldJob, *, tpr: Path, deffnm: str) -> Path:
    edr = job.run_dir / f"{deffnm}.edr"
    log = job.run_dir / f"{deffnm}.log"
    if edr.exists():
        validate_finished_rerun_epoch(tpr, edr, log)
        return edr
    if log.exists():
        raise RuntimeError(f"{job.key}: {deffnm} log exists without EDR; refusing partial epoch")
    run(
        [
            GMX,
            "mdrun",
            "-ntmpi",
            "1",
            "-ntomp",
            GMX_NTOMP,
            "-nb",
            "cpu",
            "-s",
            tpr.name,
            "-rerun",
            "solute_5ns_water_200cand.xtc",
            "-deffnm",
            deffnm,
        ],
        job.run_dir,
    )
    validate_finished_rerun_epoch(tpr, edr, log)
    return edr


def _extract_energy_term(
    job: ForceFieldJob,
    *,
    edr: Path,
    output_name: str,
    term: str,
) -> Path:
    output = job.run_dir / output_name
    if output.exists():
        if not energy_xvg_is_complete(output):
            raise RuntimeError(f"{job.key}: incomplete energy component {output.name}")
        if output.stat().st_mtime_ns < edr.stat().st_mtime_ns:
            raise RuntimeError(f"{job.key}: stale energy component {output.name}")
        return output
    run(
        [GMX, "energy", "-f", edr.name, "-o", output.name],
        job.run_dir,
        input_text=f"{term}\n0\n",
    )
    if not energy_xvg_is_complete(output):
        raise RuntimeError(f"{job.key}: gmx energy did not create 200 points for {term}")
    return output


def _read_energy_series(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "@")):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise RuntimeError(f"invalid energy row in {path}: {line!r}")
        time_ps, energy = float(fields[0]), float(fields[1])
        if not math.isfinite(time_ps) or not math.isfinite(energy):
            raise RuntimeError(f"non-finite energy row in {path}: {line!r}")
        rows.append((time_ps, energy))
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 energy rows in {path}, found {len(rows)}")
    return rows


def _combine_charged_energy_components(
    job: ForceFieldJob,
    *,
    native_potential: Path,
    native_coulomb: Path,
    allpairs_coulomb: Path,
    output: Path,
) -> None:
    potential_rows = _read_energy_series(native_potential)
    native_coulomb_rows = _read_energy_series(native_coulomb)
    allpairs_coulomb_rows = _read_energy_series(allpairs_coulomb)
    lines = [
        '# charged-solute energy decomposition: Potential_native - Coulomb-(SR)_native + Coulomb-(SR)_allpairs',
        '@    title "Charged-solute intramolecular potential"',
        '@    xaxis  label "Time (ps)"',
        '@    yaxis  label "Potential (kJ/mol)"',
    ]
    for native_potential_row, native_coulomb_row, allpairs_coulomb_row in zip(
        potential_rows,
        native_coulomb_rows,
        allpairs_coulomb_rows,
    ):
        times = (
            native_potential_row[0],
            native_coulomb_row[0],
            allpairs_coulomb_row[0],
        )
        if max(times) - min(times) > 1.0e-6:
            raise RuntimeError(f"{job.key}: charged-energy component times do not align")
        combined = native_potential_row[1] - native_coulomb_row[1] + allpairs_coulomb_row[1]
        if not math.isfinite(combined):
            raise RuntimeError(f"{job.key}: non-finite combined charged-solute energy")
        lines.append(f"{times[0]:.6f} {combined:.9f}")
    atomic_replace_text(output, "\n".join(lines) + "\n")
    if not energy_xvg_is_complete(output):
        raise RuntimeError(f"{job.key}: combined charged-solute energy is incomplete")


def rerun_large_charged_solute_energy(
    job: ForceFieldJob,
    *,
    frames_dir: Path,
    frame_paths: list[Path],
    cutoff_plan: dict[str, object],
    forcefield_dir: Path,
    method_path: Path,
    output_xvg: Path,
) -> Path:
    """Replace only native Coulomb-(SR) with an all-pairs Coulomb probe."""

    first_frame = frames_dir / "frame0.gro"
    native_mdp = job.run_dir / "rerun_solute_energy.mdp"
    probe_mdp = job.run_dir / CHARGED_RERUN_MDP
    probe_text = charged_rerun_mdp_block(job, float(cutoff_plan["cutoff_nm"]))
    component_names = (
        f"{CHARGED_NATIVE_DEFFNM}.tpr",
        f"{CHARGED_NATIVE_DEFFNM}.edr",
        f"{CHARGED_NATIVE_DEFFNM}.log",
        f"{CHARGED_COULOMB_DEFFNM}.tpr",
        f"{CHARGED_COULOMB_DEFFNM}.edr",
        f"{CHARGED_COULOMB_DEFFNM}.log",
        "rerun_native_potential.xvg",
        "rerun_native_coulomb_sr.xvg",
        "rerun_allpairs_coulomb_sr.xvg",
    )
    component_artifacts = [job.run_dir / name for name in component_names]
    existing_method = (
        json.loads(method_path.read_text(encoding="utf-8"))
        if method_path.exists()
        else None
    )
    planned_without_outputs = (
        isinstance(existing_method, dict)
        and existing_method.get("provenance_state") == "planned"
        and not output_xvg.exists()
        and not any(path.exists() for path in component_artifacts)
    )
    if probe_mdp.exists() and probe_mdp.read_text(encoding="utf-8") != probe_text:
        if not planned_without_outputs:
            raise RuntimeError(
                f"{job.key}: existing Coulomb-probe MDP differs from the requested protocol"
            )
        # Safe migration of this repair's pre-grompp plan.  No TPR/EDR/XVG
        # exists, so no computed epoch can be mixed with the new decomposition.
        atomic_replace_text(probe_mdp, probe_text)
    else:
        write_exact_text(
            probe_mdp,
            probe_text,
            label="charged-solute all-pairs Coulomb probe MDP",
        )

    solute_xtc = job.run_dir / "solute_5ns_water_200cand.xtc"
    method_body: dict[str, object] = {
        "schema_version": 3,
        "protocol_id": CHARGED_RERUN_PROTOCOL_ID,
        "method": "charged-solute native-potential/all-pairs-Coulomb decomposition",
        "formula": "Potential_native - Coulomb-(SR)_native + Coulomb-(SR)_allpairs",
        "solute_charge_e": job.solute_charge,
        "validated_frame_count": len(frame_paths),
        **cutoff_plan,
        "all_atom_pairs_within_coulomb_probe_cutoff": True,
        "all_boxes_larger_than_twice_coulomb_probe_cutoff": True,
        "coulomb_modifier": "Potential-shift (conformation-independent for the all-pairs probe)",
        "pbc": "xyz; whole solute extracted with trjconv -pbc mol",
        "native_energy_terms": "all bonded, 1-4, and family-native VdW terms",
        "coulomb_replacement": "only Coulomb-(SR) is taken from the expanded-cutoff probe",
        "discarded_probe_terms": "probe Potential and LJ-(SR) are never used",
        "ranking_scope": "within one fixed molecule/topology/force-field candidate set",
        "rerun_neighbor_search": "performed independently for every trajectory frame",
        "native_mdp": native_mdp.name,
        "coulomb_probe_mdp": probe_mdp.name,
        "topology": job.solute_top.name,
        "forcefield_directory": forcefield_dir.name,
        "grompp_maxwarn": grompp_maxwarn(job),
        "input_sha256": {
            "native_mdp": sha256_file(native_mdp),
            "coulomb_probe_mdp": sha256_file(probe_mdp),
            "solute_topology": sha256_file(job.solute_top),
            "forcefield_tree": sha256_tree(forcefield_dir),
            "solute_xtc": sha256_file(solute_xtc),
            "frame_set": sha256_frame_set(frame_paths),
        },
    }
    method_plan = {
        **method_body,
        "provenance_state": "planned",
        "tpr_sha256": None,
        "output_sha256": None,
    }
    if existing_method is None:
        atomic_replace_text(
            method_path,
            json.dumps(method_plan, ensure_ascii=False, indent=2) + "\n",
        )
    elif existing_method != method_plan:
        if planned_without_outputs:
            atomic_replace_text(
                method_path,
                json.dumps(method_plan, ensure_ascii=False, indent=2) + "\n",
            )
        elif existing_method.get("provenance_state") not in {"tpr_bound", "outputs_bound"}:
            raise RuntimeError(f"{job.key}: charged-energy decomposition plan changed")

    native_tpr = _grompp_rerun_tpr(
        job,
        mdp_name=native_mdp.name,
        first_frame=first_frame,
        output_name=f"{CHARGED_NATIVE_DEFFNM}.tpr",
    )
    probe_tpr = _grompp_rerun_tpr(
        job,
        mdp_name=probe_mdp.name,
        first_frame=first_frame,
        output_name=f"{CHARGED_COULOMB_DEFFNM}.tpr",
    )
    method_tpr_bound = {
        **method_body,
        "provenance_state": "tpr_bound",
        "tpr_sha256": {
            "native": sha256_file(native_tpr),
            "coulomb_probe": sha256_file(probe_tpr),
        },
        "output_sha256": None,
    }
    existing_method = json.loads(method_path.read_text(encoding="utf-8"))
    if existing_method == method_plan:
        atomic_replace_text(
            method_path,
            json.dumps(method_tpr_bound, ensure_ascii=False, indent=2) + "\n",
        )
    elif existing_method.get("provenance_state") == "tpr_bound" and existing_method != method_tpr_bound:
        raise RuntimeError(f"{job.key}: charged-energy rerun TPR provenance changed")
    elif existing_method.get("provenance_state") == "outputs_bound":
        recorded_tprs = existing_method.get("tpr_sha256")
        if recorded_tprs != method_tpr_bound["tpr_sha256"]:
            raise RuntimeError(f"{job.key}: charged-energy finalized TPR provenance changed")

    native_edr = _run_rerun_epoch(job, tpr=native_tpr, deffnm=CHARGED_NATIVE_DEFFNM)
    probe_edr = _run_rerun_epoch(job, tpr=probe_tpr, deffnm=CHARGED_COULOMB_DEFFNM)
    native_potential = _extract_energy_term(
        job,
        edr=native_edr,
        output_name="rerun_native_potential.xvg",
        term="Potential",
    )
    native_coulomb = _extract_energy_term(
        job,
        edr=native_edr,
        output_name="rerun_native_coulomb_sr.xvg",
        term="Coulomb-(SR)",
    )
    allpairs_coulomb = _extract_energy_term(
        job,
        edr=probe_edr,
        output_name="rerun_allpairs_coulomb_sr.xvg",
        term="Coulomb-(SR)",
    )
    newest_component = max(
        path.stat().st_mtime_ns
        for path in (native_potential, native_coulomb, allpairs_coulomb)
    )
    if output_xvg.exists():
        if not energy_xvg_is_complete(output_xvg):
            raise RuntimeError(f"{job.key}: combined charged-solute XVG is incomplete")
        if output_xvg.stat().st_mtime_ns < newest_component:
            raise RuntimeError(f"{job.key}: combined charged-solute XVG is stale")
    else:
        _combine_charged_energy_components(
            job,
            native_potential=native_potential,
            native_coulomb=native_coulomb,
            allpairs_coulomb=allpairs_coulomb,
            output=output_xvg,
        )
    method_outputs_bound = {
        **method_body,
        "provenance_state": "outputs_bound",
        "tpr_sha256": method_tpr_bound["tpr_sha256"],
        "output_sha256": {
            "native_edr": sha256_file(native_edr),
            "coulomb_probe_edr": sha256_file(probe_edr),
            "native_potential_xvg": sha256_file(native_potential),
            "native_coulomb_sr_xvg": sha256_file(native_coulomb),
            "allpairs_coulomb_sr_xvg": sha256_file(allpairs_coulomb),
            "combined_potential_xvg": sha256_file(output_xvg),
        },
    }
    existing_method = json.loads(method_path.read_text(encoding="utf-8"))
    if existing_method == method_tpr_bound:
        atomic_replace_text(
            method_path,
            json.dumps(method_outputs_bound, ensure_ascii=False, indent=2) + "\n",
        )
    elif existing_method != method_outputs_bound:
        if is_safe_charged_protocol_id_upgrade(existing_method, method_outputs_bound):
            atomic_replace_text(
                method_path,
                json.dumps(method_outputs_bound, ensure_ascii=False, indent=2) + "\n",
            )
        else:
            raise RuntimeError(f"{job.key}: charged-energy output provenance changed")
    return output_xvg


def energy_xvg_is_complete(path: Path, expected_points: int = 200) -> bool:
    if not path.exists():
        return False
    times: list[float] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                return False
            values = [float(value) for value in fields[:2]]
            if not all(math.isfinite(value) for value in values):
                return False
            times.append(values[0])
    except (OSError, UnicodeError, ValueError):
        return False
    return len(times) == expected_points and all(
        right > left for left, right in zip(times, times[1:])
    )


def validate_finished_rerun_epoch(tpr: Path, edr: Path, log: Path) -> None:
    if not tpr.exists() or not edr.exists() or not log.exists():
        raise RuntimeError("rerun energy epoch lacks TPR, EDR, or log")
    log_tail = log.read_text(encoding="utf-8", errors="replace")[-12000:]
    if "Finished mdrun" not in log_tail:
        raise RuntimeError("rerun energy log lacks Finished mdrun")
    if min(edr.stat().st_mtime_ns, log.stat().st_mtime_ns) < tpr.stat().st_mtime_ns:
        raise RuntimeError("rerun EDR/log are stale relative to TPR")


def write_text_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def insert_before_system(top_text: str, insertion: str) -> str:
    marker = "\n[ system ]"
    if insertion.strip() in top_text:
        return top_text
    if marker not in top_text:
        raise ValueError("Could not find [ system ] in topology")
    return top_text.replace(marker, "\n" + insertion.rstrip() + "\n\n[ system ]", 1)


def insert_before_moleculetype(top_text: str, insertion: str) -> str:
    marker = "\n[ moleculetype ]"
    if insertion.strip() in top_text:
        return top_text
    if marker in top_text:
        return top_text.replace(marker, "\n" + insertion.rstrip() + "\n\n[ moleculetype ]", 1)
    include_marker = "\n#include "
    if include_marker in top_text:
        return top_text.replace(include_marker, "\n" + insertion.rstrip() + "\n\n#include ", 1)
    raise ValueError("Could not find [ moleculetype ] or #include in topology")


def prepare_amber_tip3p_top(source_top: Path, dest_top: Path) -> None:
    top_text = source_top.read_text(encoding="utf-8")
    top_text = insert_before_moleculetype(top_text, AMBER_TIP3P_ATOMTYPES)
    top_text = insert_before_system(top_text, '; Include local AMBER TIP3P water topology\n#include "amber_tip3p.itp"')
    dest_top.write_text(top_text, encoding="utf-8")
    copy_file(TOPDIR / "amber99sb-ildn.ff" / "tip3p.itp", dest_top.parent / "amber_tip3p.itp")


def prepare_gaff2() -> ForceFieldJob:
    source = RUN_ROOT / "gaff2_100conf_test"
    acpype = source / f"{MOL}_gaff2.acpype"
    run_dir = RUN_ROOT / "gaff2_water_5ns_200cand_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_file(acpype / f"{MOL}_gaff2_GMX.gro", run_dir / f"{MOL}_gaff2_GMX.gro")
    copy_file(acpype / f"{MOL}_gaff2_GMX.itp", run_dir / f"{MOL}_gaff2_GMX.itp")
    copy_file(acpype / f"posre_{MOL}_gaff2.itp", run_dir / f"posre_{MOL}_gaff2.itp")
    water_top = run_dir / f"{MOL}_gaff2_water.top"
    if not water_top.exists():
        prepare_amber_tip3p_top(acpype / f"{MOL}_gaff2_GMX.top", water_top)
    solute_top = run_dir / f"{MOL}_gaff2_solute_only.top"
    if not solute_top.exists():
        shutil.copy2(water_top, solute_top)
    return ForceFieldJob(
        key="gaff2",
        label="GAFF2/AM1-BCC",
        run_dir=run_dir,
        solute_gro=run_dir / f"{MOL}_gaff2_GMX.gro",
        water_top=water_top,
        solute_top=solute_top,
        water_model="AMBER TIP3P",
        family="amber",
    )


def prepare_glycam06j() -> ForceFieldJob:
    source = RUN_ROOT / "glycam06j_200cand_test"
    acpype = source / f"{MOL}_GLYCAM06j.amb2gmx"
    run_dir = RUN_ROOT / "glycam06j_water_5ns_200cand_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_file(acpype / f"{MOL}_GLYCAM06j_GMX.gro", run_dir / f"{MOL}_GLYCAM06j_GMX.gro")
    copy_file(acpype / f"posre_{MOL}_GLYCAM06j.itp", run_dir / f"posre_{MOL}_GLYCAM06j.itp")
    water_top = run_dir / f"{MOL}_GLYCAM06j_water.top"
    if not water_top.exists():
        prepare_amber_tip3p_top(acpype / f"{MOL}_GLYCAM06j_GMX.top", water_top)
    solute_top = run_dir / f"{MOL}_GLYCAM06j_solute_only.top"
    if not solute_top.exists():
        shutil.copy2(water_top, solute_top)
    copy_file(source / "sdf_to_glycam_frame_heavy_index_map.json", run_dir / "sdf_to_glycam_frame_heavy_index_map.json")
    return ForceFieldJob(
        key="glycam06j",
        label="GLYCAM06j",
        run_dir=run_dir,
        solute_gro=run_dir / f"{MOL}_GLYCAM06j_GMX.gro",
        water_top=water_top,
        solute_top=solute_top,
        water_model="AMBER TIP3P",
        family="amber",
        atom_index_map=run_dir / "sdf_to_glycam_frame_heavy_index_map.json",
    )


def prepare_gromos56a6carbo() -> ForceFieldJob:
    source = RUN_ROOT / "gromos56a6carbo_200cand_test"
    run_dir = RUN_ROOT / "gromos56a6carbo_water_5ns_200cand_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_file(source / f"{MOL}_GROMOS56A6CARBO_processed.gro", run_dir / f"{MOL}_GROMOS56A6CARBO_processed.gro")
    copy_file(source / f"{MOL}_GROMOS56A6CARBO.top", run_dir / f"{MOL}_GROMOS56A6CARBO_water.top")
    copy_file(source / "posre.itp", run_dir / "posre.itp")
    copy_file(source / "sdf_to_gromos_frame_heavy_index_map.json", run_dir / "sdf_to_gromos_frame_heavy_index_map.json")
    copy_tree(source / "gromos53a6carbo_gcm.ff", run_dir / "gromos53a6carbo_gcm.ff")
    solute_top = run_dir / f"{MOL}_GROMOS56A6CARBO_solute_only.top"
    if not solute_top.exists():
        shutil.copy2(run_dir / f"{MOL}_GROMOS56A6CARBO_water.top", solute_top)
    return ForceFieldJob(
        key="gromos56a6carbo",
        label="GROMOS 56A_CARBO",
        run_dir=run_dir,
        solute_gro=run_dir / f"{MOL}_GROMOS56A6CARBO_processed.gro",
        water_top=run_dir / f"{MOL}_GROMOS56A6CARBO_water.top",
        solute_top=solute_top,
        water_model="SPC",
        family="gromos",
        atom_index_map=run_dir / "sdf_to_gromos_frame_heavy_index_map.json",
    )


def prepare_charmm36_bmm() -> ForceFieldJob:
    source_solute = RUN_ROOT / "charmm36_bmm_safe_100conf_test"
    run_dir = RUN_ROOT / "charmm36_bmm_water_5ns_200cand_test"
    copy_file(source_solute / f"{MOL}_BMM_charmm36.top", run_dir / f"{MOL}_BMM_charmm36_solute_only.top")
    copy_file(source_solute / "sdf_to_charmm_frame_heavy_index_map.json", run_dir / "sdf_to_charmm_frame_heavy_index_map.json")
    return ForceFieldJob(
        key="charmm36_bmm",
        label="CHARMM36-BMM",
        run_dir=run_dir,
        solute_gro=run_dir / f"{MOL}_BMM_charmm36_processed.gro",
        water_top=run_dir / f"{MOL}_BMM_charmm36.top",
        solute_top=run_dir / f"{MOL}_BMM_charmm36_solute_only.top",
        water_model="CHARMM TIP3P",
        family="charmm",
        atom_index_map=run_dir / "sdf_to_charmm_frame_heavy_index_map.json",
        completed_md=True,
    )


def integration_timestep_ps(job: ForceFieldJob) -> float:
    """Return a validated integration step without silently changing duration.

    ``MD_GROMOS_DT_PS`` lets the repair queue conservatively use 1 fs while
    leaving the other force fields on their established protocol.  A global
    ``MD_DT_PS`` override remains available for explicit diagnostic work.
    """

    if job.timestep_ps is not None:
        raw: str | float = job.timestep_ps
    else:
        raw = (
            os.environ.get("MD_GROMOS_DT_PS")
            if job.family == "gromos"
            else None
        ) or os.environ.get("MD_DT_PS", "0.002")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"integration timestep must be numeric, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0.0 or value > 0.002:
        raise ValueError(
            f"integration timestep must be finite and in (0, 0.002] ps, got {value!r}"
        )
    return value


def default_stage_steps(stage: str, timestep_ps: float) -> int:
    if stage == "em":
        return 5000
    duration_ps = {"nvt": 100.0, "npt": 200.0, "md": 5000.0, "rerun": 0.0}[stage]
    exact_steps = duration_ps / timestep_ps
    steps = int(round(exact_steps))
    if not math.isclose(exact_steps, steps, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            f"stage duration {duration_ps:g} ps is not an integer number of "
            f"{timestep_ps:g} ps steps"
        )
    return steps


def mdp_block(job: ForceFieldJob, stage: str) -> str:
    if job.family == "charmm":
        rlist, rcoulomb, rvdw = "1.2", "1.2", "1.2"
        vdw_extra = "vdw-modifier            = Force-switch\nrvdw-switch             = 1.0\nDispCorr                = no"
    else:
        rlist, rcoulomb, rvdw = "1.0", "1.0", "1.0"
        vdw_extra = "DispCorr                = EnerPres"

    nonbonded = f"""\
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = {rlist}
coulombtype             = PME
rcoulomb                = {rcoulomb}
vdwtype                 = Cut-off
rvdw                    = {rvdw}
{vdw_extra}
pbc                     = xyz
"""
    timestep_ps = integration_timestep_ps(job)
    default_steps = default_stage_steps(stage, timestep_ps)
    override = os.environ.get(f"MD_NSTEPS_{stage.upper()}")
    nsteps = int(override) if override is not None else default_steps
    if nsteps < 0:
        raise ValueError(f"MD_NSTEPS_{stage.upper()} must be non-negative")

    if stage == "em":
        return f"""\
integrator              = steep
nsteps                  = {nsteps}
emtol                   = 100.0
emstep                  = 0.01

{nonbonded}
"""

    if stage == "nvt":
        return f"""\
integrator              = md
dt                      = {timestep_ps:.6f}
nsteps                  = {nsteps}

continuation            = no
constraints             = h-bonds
constraint_algorithm    = lincs
lincs_iter              = 2
lincs_order             = 8
lincs_warnangle         = 90

{nonbonded}

tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 1.0
ref_t                   = 300
pcoupl                  = no

gen_vel                 = yes
gen_temp                = 300
gen_seed                = 1

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 5000
nstlog                  = 5000
nstxout-compressed      = 0
"""

    if stage == "npt":
        return f"""\
integrator              = md
dt                      = {timestep_ps:.6f}
nsteps                  = {nsteps}

continuation            = yes
constraints             = h-bonds
constraint_algorithm    = lincs
lincs_iter              = 2
lincs_order             = 8
lincs_warnangle         = 90

{nonbonded}

tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 1.0
ref_t                   = 300
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 5.0
ref_p                   = 1.0
compressibility         = 4.5e-5

gen_vel                 = no

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 5000
nstlog                  = 5000
nstxout-compressed      = 0
"""

    if stage == "md":
        spacing_steps_exact = job.candidate_spacing_ps / timestep_ps
        spacing_steps = int(round(spacing_steps_exact))
        if spacing_steps < 1 or not math.isclose(
            spacing_steps_exact,
            spacing_steps,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise ValueError(
                f"{job.key}: candidate spacing {job.candidate_spacing_ps:g} ps is "
                f"not an integer number of {timestep_ps:g} ps steps"
            )
        expected_candidates = nsteps // spacing_steps
        if nsteps % spacing_steps != 0 or expected_candidates != 200:
            raise ValueError(
                f"{job.key}: production protocol would emit {expected_candidates} "
                "rather than exactly 200 candidate frames"
            )
        return f"""\
integrator              = md
dt                      = {timestep_ps:.6f}
nsteps                  = {nsteps}

continuation            = yes
constraints             = h-bonds
constraint_algorithm    = lincs
lincs_iter              = 2
lincs_order             = 8
lincs_warnangle         = 90

{nonbonded}

tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 1.0
ref_t                   = 300
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 5.0
ref_p                   = 1.0
compressibility         = 4.5e-5

gen_vel                 = no

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = {spacing_steps}
nstlog                  = {spacing_steps}
nstxout-compressed      = {spacing_steps}
compressed-x-precision  = 1000
"""

    if stage == "rerun":
        rerun_nonbonded = nonbonded
        if job.solute_charge:
            # PME's uniform neutralizing background is inappropriate for
            # ranking isolated charged-solute conformers.  For these small
            # uronic-acid solutes we instead use direct pairwise Coulomb
            # interactions and fail later if any atom pair exceeds rcoulomb.
            rerun_nonbonded = rerun_nonbonded.replace(
                "coulombtype             = PME",
                "coulombtype             = Cut-off",
            )
            rerun_nonbonded += "\nverlet-buffer-tolerance = -1"
        return f"""\
integrator              = md
dt                      = {timestep_ps:.6f}
nsteps                  = {nsteps}

continuation            = yes
constraints             = h-bonds
constraint_algorithm    = lincs
lincs_iter              = 2
lincs_order             = 8
lincs_warnangle         = 90

{rerun_nonbonded}

tcoupl                  = no
pcoupl                  = no
gen_vel                 = no

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 1
nstlog                  = 1
nstxout-compressed      = 0
"""

    raise ValueError(stage)


def write_mdps(job: ForceFieldJob) -> None:
    blocks: dict[str, str] = {}
    for stage in ("em", "nvt", "npt", "md", "rerun"):
        suffix = "water" if stage != "rerun" else "solute_energy"
        path = job.run_dir / f"{stage}_{suffix}.mdp"
        block = mdp_block(job, stage)
        blocks[stage] = block
        if path.exists():
            if path.read_text(encoding="utf-8") != block:
                raise RuntimeError(
                    f"{job.key}: existing {path.name} differs from the requested protocol; "
                    "use a clean run variant instead of mixing simulation epochs"
                )
        else:
            path.write_text(block, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "forcefield_key": job.key,
        "family": job.family,
        "integration_timestep_ps": integration_timestep_ps(job),
        "candidate_spacing_ps": job.candidate_spacing_ps,
        "candidate_count": 200,
        "grompp_warning_policy": job.grompp_warning_policy,
        "mdp_sha256": {
            stage: hashlib.sha256(block.encode("utf-8")).hexdigest()
            for stage, block in blocks.items()
        },
    }
    manifest_path = job.run_dir / "water_md_protocol.json"
    if manifest_path.exists():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid protocol manifest: {manifest_path}") from exc
        if recorded != manifest:
            raise RuntimeError(
                f"{job.key}: protocol manifest differs from requested settings; "
                "use a clean run variant"
            )
    else:
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, manifest_path)


def mdrun_finished(job: ForceFieldJob, deffnm: str) -> bool:
    output_gro = job.run_dir / f"{deffnm}.gro"
    if not output_gro.exists():
        return False
    log = job.run_dir / f"{deffnm}.log"
    if not log.exists():
        return False
    tpr = job.run_dir / f"{deffnm}.tpr"
    if not tpr.exists():
        return False
    if min(output_gro.stat().st_mtime_ns, log.stat().st_mtime_ns) < tpr.stat().st_mtime_ns:
        return False
    try:
        tail = log.read_text(errors="ignore")[-12000:]
    except OSError:
        return False
    console_log = job.run_dir / f"{deffnm}.console.log"
    if console_log.exists():
        try:
            console_text = console_log.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return False
        markers = scientific_mdrun_markers(console_text)
        if markers:
            raise RuntimeError(
                f"{job.key} {deffnm}: prior epoch contains scientific marker(s) "
                f"{markers}; use a clean run variant instead of resuming it"
            )
    return "Finished mdrun" in tail


def add_checkpoint_args(cmd: list[str], checkpoint: Path) -> None:
    if checkpoint.exists():
        cmd.extend(["-cpi", checkpoint.name, "-append"])


def mdrun(job: ForceFieldJob, deffnm: str, gpu: bool) -> None:
    if mdrun_finished(job, deffnm):
        print(f"[skip] {job.key} {deffnm} already exists", flush=True)
        return
    gpu = gpu and MD_DEVICE == "gpu"
    if gpu:
        cmd = [
            GMX,
            "mdrun",
            "-ntmpi",
            "1",
            "-ntomp",
            GMX_NTOMP,
            "-nb",
            "gpu",
            "-pme",
            "gpu",
            "-bonded",
            "gpu",
            "-update",
            "cpu",
            "-cpt",
            GMX_CPT_MINUTES,
            "-deffnm",
            deffnm,
        ]
    else:
        cmd = [
            GMX,
            "mdrun",
            "-ntmpi",
            "1",
            "-ntomp",
            GMX_NTOMP,
            "-nb",
            "cpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-cpt",
            GMX_CPT_MINUTES,
            "-deffnm",
            deffnm,
        ]
    checkpoint = job.run_dir / f"{deffnm}.cpt"
    add_checkpoint_args(cmd, checkpoint)
    try:
        run_mdrun(cmd, job.run_dir)
    except subprocess.CalledProcessError as exc:
        if not gpu:
            raise
        failure_kind = classify_gpu_failure(exc)
        if failure_kind == "gpu_fatal":
            raise GpuFatalError(
                f"{job.key} {deffnm}: fatal GPU/driver failure; refusing GPU or CPU retry"
            ) from exc
        if failure_kind == "scientific":
            raise RuntimeError(
                f"{job.key} {deffnm}: scientific/constraint failure; refusing CPU retry"
            ) from exc
        if failure_kind != "recoverable_gpu_setup":
            raise RuntimeError(
                f"{job.key} {deffnm}: unclassified GPU failure; refusing automatic retry"
            ) from exc
        cpu_cmd = [
            GMX,
            "mdrun",
            "-ntmpi",
            "1",
            "-ntomp",
            GMX_NTOMP,
            "-nb",
            "cpu",
            "-pme",
            "cpu",
            "-bonded",
            "cpu",
            "-update",
            "cpu",
            "-cpt",
            GMX_CPT_MINUTES,
            "-deffnm",
            deffnm,
        ]
        add_checkpoint_args(cpu_cmd, checkpoint)
        print(f"[retry-full-cpu] {job.key} {deffnm} after recoverable GPU setup failure", flush=True)
        run_mdrun(cpu_cmd, job.run_dir)


def run_water_md(job: ForceFieldJob) -> None:
    stop_after = os.environ.get("MD_STOP_AFTER_STAGE", "").lower()
    if stop_after and stop_after not in {"em", "nvt", "npt", "md"}:
        raise ValueError(f"invalid MD_STOP_AFTER_STAGE: {stop_after!r}")
    write_mdps(job)
    box = job.run_dir / f"{job.key}_water_box.gro"
    solvated = job.run_dir / f"{job.key}_solvated.gro"
    neutralized = job.run_dir / f"{job.key}_neutralized.gro"
    em_tpr = job.run_dir / "em_water.tpr"
    if mdrun_finished(job, "md_5ns_water_200cand"):
        if job.solute_charge:
            if not neutralized.exists() or not em_tpr.exists() or not job.water_top.exists():
                raise RuntimeError(
                    f"{job.key}: completed charged MD lacks its neutralized EM inputs"
                )
            newest_ionized_input = max(
                neutralized.stat().st_mtime_ns,
                job.water_top.stat().st_mtime_ns,
            )
            if newest_ionized_input >= em_tpr.stat().st_mtime_ns:
                raise RuntimeError(
                    f"{job.key}: completed charged MD has a stale/inconsistent EM TPR"
                )
        print(f"[skip] {job.key} water MD already complete", flush=True)
        return

    if not box.exists():
        run([GMX, "editconf", "-f", str(job.solute_gro.name), "-o", box.name, "-bt", "cubic", "-d", "1.2", "-c"], job.run_dir)
    needs_em_tpr = not em_tpr.exists()
    if job.solute_charge and not neutralized.exists():
        # A TPR without the corresponding neutralized coordinate file may be
        # stale output from an older, non-neutralizing workflow.  Rebuild the
        # complete solvation/ionization setup instead of trusting it.
        needs_em_tpr = True
    if job.solute_charge and em_tpr.exists() and neutralized.exists():
        newest_ionized_input = max(
            neutralized.stat().st_mtime_ns,
            job.water_top.stat().st_mtime_ns,
        )
        if newest_ionized_input >= em_tpr.stat().st_mtime_ns:
            # Covers interruption after genion changed coordinates/topology
            # but before strict grompp atomically replaced the old EM TPR.
            needs_em_tpr = True
    if job.solute_charge and needs_em_tpr:
        stale_downstream = [
            path
            for stage, suffixes in (
                ("em_water", ("cpt", "gro", "log", "edr", "trr", "xtc")),
                ("nvt_water", ("tpr", "cpt", "gro", "log", "edr", "trr", "xtc")),
                ("npt_water", ("tpr", "cpt", "gro", "log", "edr", "trr", "xtc")),
                (
                    "md_5ns_water_200cand",
                    ("tpr", "cpt", "gro", "log", "edr", "trr", "xtc"),
                ),
            )
            for suffix in suffixes
            if (path := job.run_dir / f"{stage}.{suffix}").exists()
        ]
        if stale_downstream:
            raise RuntimeError(
                f"{job.key}: neutralization/EM must be rebuilt but downstream stage artifacts "
                "already exist; refusing to mix simulation epochs: "
                + ", ".join(path.name for path in stale_downstream[:8])
            )
    if needs_em_tpr and solvated.exists() and job.solute_top.exists():
        shutil.copy2(job.solute_top, job.water_top)
    if not solvated.exists() or needs_em_tpr:
        run([GMX, "solvate", "-cp", box.name, "-cs", "spc216.gro", "-o", solvated.name, "-p", job.water_top.name], job.run_dir)

    em_input = solvated
    if job.solute_charge:
        if not neutralized.exists() or needs_em_tpr:
            # Invalidate a potentially charged/stale TPR before genion.  If
            # this process is interrupted, the next invocation must grompp
            # the neutralized coordinates instead of running the old TPR.
            em_tpr.unlink(missing_ok=True)
            ions_mdp = job.run_dir / "ions_water.mdp"
            ions_mdp.write_text(mdp_block(job, "em"), encoding="utf-8")
            run_grompp_for_neutralization(
                job,
                [
                    "-f",
                    ions_mdp.name,
                    "-c",
                    solvated.name,
                    "-p",
                    job.water_top.name,
                    "-o",
                    "ions_water.tpr",
                ],
            )
            run(
                [
                    GMX,
                    "genion",
                    "-s",
                    "ions_water.tpr",
                    "-o",
                    neutralized.name,
                    "-p",
                    job.water_top.name,
                    "-pname",
                    job.positive_ion,
                    "-nname",
                    job.negative_ion,
                    "-seed",
                    "1",
                    "-neutral",
                ],
                job.run_dir,
                input_text="SOL\n",
            )
        em_input = neutralized

    if needs_em_tpr:
        if job.solute_charge:
            pending_tpr = job.run_dir / "em_water.pending.tpr"
            pending_tpr.unlink(missing_ok=True)
            run_grompp(
                job,
                [
                    "-f",
                    "em_water.mdp",
                    "-c",
                    em_input.name,
                    "-p",
                    job.water_top.name,
                    "-o",
                    pending_tpr.name,
                ],
            )
            os.replace(pending_tpr, em_tpr)
        else:
            run_grompp(
                job,
                [
                    "-f",
                    "em_water.mdp",
                    "-c",
                    em_input.name,
                    "-p",
                    job.water_top.name,
                    "-o",
                    em_tpr.name,
                ],
            )
    mdrun(job, "em_water", gpu=False)
    if stop_after == "em":
        return

    if not (job.run_dir / "nvt_water.tpr").exists():
        run_grompp(job, ["-f", "nvt_water.mdp", "-c", "em_water.gro", "-p", job.water_top.name, "-o", "nvt_water.tpr"])
    mdrun(job, "nvt_water", gpu=True)
    if stop_after == "nvt":
        return

    if not (job.run_dir / "npt_water.tpr").exists():
        run_grompp(
            job,
            [
                "-f",
                "npt_water.mdp",
                "-c",
                "nvt_water.gro",
                "-t",
                "nvt_water.cpt",
                "-p",
                job.water_top.name,
                "-o",
                "npt_water.tpr",
            ],
        )
    mdrun(job, "npt_water", gpu=True)
    if stop_after == "npt":
        return

    if not (job.run_dir / "md_5ns_water_200cand.tpr").exists():
        run_grompp(
            job,
            [
                "-f",
                "md_water.mdp",
                "-c",
                "npt_water.gro",
                "-t",
                "npt_water.cpt",
                "-p",
                job.water_top.name,
                "-o",
                "md_5ns_water_200cand.tpr",
            ],
        )
    mdrun(job, "md_5ns_water_200cand", gpu=True)


def write_solute_index(job: ForceFieldJob) -> Path:
    atom_count = read_atom_count(job.solute_gro)
    path = job.run_dir / "solute_only.ndx"
    if not path.exists():
        numbers = " ".join(str(i) for i in range(1, atom_count + 1))
        path.write_text(f"[ Solute ]\n{numbers}\n", encoding="utf-8")
    return path


def extract_solute_frames(job: ForceFieldJob) -> Path:
    frames_dir = job.run_dir / "frames_gro_5ns_water_200cand_solute"
    frames_dir.mkdir(exist_ok=True)
    existing = sorted(frames_dir.glob("frame*.gro"))
    solute_xtc = job.run_dir / "solute_5ns_water_200cand.xtc"
    source_tpr = job.run_dir / "md_5ns_water_200cand.tpr"
    source_xtc = job.run_dir / "md_5ns_water_200cand.xtc"
    manifest_path = job.run_dir / "solute_extraction_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_paths = sorted(frames_dir.glob("frame*.gro"))
        current = {
            "frame_count": len(frame_paths),
            "source_md_tpr_sha256": sha256_file(source_tpr),
            "source_md_xtc_sha256": sha256_file(source_xtc),
            "solute_xtc_sha256": sha256_file(solute_xtc),
            "frame_set_sha256": sha256_frame_set(frame_paths),
        }
        if manifest != current:
            raise RuntimeError(f"{job.key}: solute extraction manifest/input hashes changed")
        if current["frame_count"] != 200:
            raise RuntimeError(f"{job.key}: extraction manifest does not contain 200 frames")
        return frames_dir
    if existing or solute_xtc.exists():
        raise RuntimeError(
            f"{job.key}: derived solute frames/XTC exist without a same-epoch extraction manifest"
        )
    index = write_solute_index(job)
    if len(existing) != 200:
        for frame in existing:
            frame.unlink()
        run(
            [
                GMX,
                "trjconv",
                "-s",
                "md_5ns_water_200cand.tpr",
                "-f",
                "md_5ns_water_200cand.xtc",
                "-o",
                str(frames_dir / "frame.gro"),
                "-sep",
                "-b",
                "25",
                "-pbc",
                "mol",
                "-n",
                index.name,
            ],
            job.run_dir,
            input_text="0\n",
        )
    if not solute_xtc.exists():
        run(
            [
                GMX,
                "trjconv",
                "-s",
                "md_5ns_water_200cand.tpr",
                "-f",
                "md_5ns_water_200cand.xtc",
                "-o",
                solute_xtc.name,
                "-b",
                "25",
                "-pbc",
                "mol",
                "-n",
                index.name,
            ],
            job.run_dir,
            input_text="0\n",
        )
    count = len(list(frames_dir.glob("frame*.gro")))
    if count != 200:
        raise RuntimeError(f"{job.key}: expected 200 frames, found {count}")
    frame_paths = sorted(frames_dir.glob("frame*.gro"))
    manifest = {
        "frame_count": len(frame_paths),
        "source_md_tpr_sha256": sha256_file(source_tpr),
        "source_md_xtc_sha256": sha256_file(source_xtc),
        "solute_xtc_sha256": sha256_file(solute_xtc),
        "frame_set_sha256": sha256_frame_set(frame_paths),
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return frames_dir


def rerun_solute_energy(job: ForceFieldJob, frames_dir: Path) -> Path:
    xvg = job.run_dir / "solute_potential_5ns_water_200cand.xvg"
    method_path = job.run_dir / "solute_energy_method.json"
    first_frame = frames_dir / "frame0.gro"
    rerun_mdp_name = "rerun_solute_energy.mdp"
    if job.solute_charge:
        legacy_outputs = [
            path
            for path in (
                job.run_dir / "rerun_solute_energy.tpr",
                job.run_dir / "rerun_solute_energy.edr",
                xvg,
            )
            if path.exists()
        ]
        if legacy_outputs and not method_path.exists():
            raise RuntimeError(
                f"{job.key}: charged-solute energy output lacks method provenance; "
                "refusing to reuse it"
            )
        frame_paths = sorted(frames_dir.glob("frame*.gro"))
        if len(frame_paths) != 200:
            raise RuntimeError(
                f"{job.key}: charged-solute cutoff validation expected 200 frames, "
                f"found {len(frame_paths)}"
            )
        cutoff_plan = charged_rerun_cutoff_plan(job, frame_paths)
        cutoff_nm = float(cutoff_plan["cutoff_nm"])
        forcefield_dirs = sorted(job.run_dir.glob("*.ff"))
        if len(forcefield_dirs) != 1:
            raise RuntimeError(
                f"{job.key}: expected one local force-field directory, found "
                f"{len(forcefield_dirs)}"
            )
        solute_xtc = job.run_dir / "solute_5ns_water_200cand.xtc"
        if bool(cutoff_plan["uses_derived_cutoff"]):
            return rerun_large_charged_solute_energy(
                job,
                frames_dir=frames_dir,
                frame_paths=frame_paths,
                cutoff_plan=cutoff_plan,
                forcefield_dir=forcefield_dirs[0],
                method_path=method_path,
                output_xvg=xvg,
            )
        else:
            # Preserve the established small-solute method byte-for-byte.
            rerun_mdp = job.run_dir / rerun_mdp_name
            legacy_method = {
                "method": "isolated charged-solute intramolecular potential ranking",
                "electrostatics": "direct Coulomb cut-off; PME/background charge disabled",
                "solute_charge_e": job.solute_charge,
                "cutoff_nm": cutoff_nm,
                "validated_frame_count": len(frame_paths),
                "maximum_pair_distance_nm": cutoff_plan["maximum_pair_distance_nm"],
                "maximum_pair_distance_frame": cutoff_plan["maximum_pair_distance_frame"],
                "minimum_box_length_nm": cutoff_plan["minimum_box_length_nm"],
                "minimum_image_margin_nm": cutoff_plan["minimum_image_margin_nm"],
                "all_atom_pairs_within_cutoff": True,
                "all_boxes_larger_than_twice_cutoff": True,
                "coulomb_modifier": "Potential-shift (GROMACS default)",
                "pbc": "xyz; whole solute extracted with trjconv -pbc mol",
                "mdp": "rerun_solute_energy.mdp",
                "topology": job.solute_top.name,
                "forcefield_directory": forcefield_dirs[0].name,
                "input_sha256": {
                    "rerun_mdp": sha256_file(rerun_mdp),
                    "solute_topology": sha256_file(job.solute_top),
                    "forcefield_tree": sha256_tree(forcefield_dirs[0]),
                    "solute_xtc": sha256_file(solute_xtc),
                    "frame_set": sha256_frame_set(frame_paths),
                },
            }
            if method_path.exists():
                if json.loads(method_path.read_text(encoding="utf-8")) != legacy_method:
                    raise RuntimeError(f"{job.key}: charged-solute energy method provenance changed")
            else:
                write_exact_text(
                    method_path,
                    json.dumps(legacy_method, ensure_ascii=False, indent=2) + "\n",
                    label="charged-solute energy method provenance",
                )
    rerun_tpr = job.run_dir / "rerun_solute_energy.tpr"
    rerun_edr = job.run_dir / "rerun_solute_energy.edr"
    rerun_log = job.run_dir / "rerun_solute_energy.log"
    if xvg.exists():
        try:
            validate_finished_rerun_epoch(rerun_tpr, rerun_edr, rerun_log)
        except RuntimeError as exc:
            raise RuntimeError(f"{job.key}: {exc}") from exc
        if not energy_xvg_is_complete(xvg):
            raise RuntimeError(f"{job.key}: solute-energy XVG is incomplete or invalid")
        if not rerun_edr.exists() or xvg.stat().st_mtime_ns < rerun_edr.stat().st_mtime_ns:
            raise RuntimeError(f"{job.key}: solute-energy XVG is stale relative to EDR")
        return xvg
    if not rerun_tpr.exists():
        grompp_args = [
            "-f",
            rerun_mdp_name,
            "-c",
            str(first_frame),
            "-p",
            job.solute_top.name,
            "-o",
            "rerun_solute_energy.tpr",
        ]
        if job.family != "gromos":
            grompp_args.extend(["-maxwarn", "0"])
        run_grompp(
            job,
            grompp_args,
        )
    if rerun_edr.exists():
        try:
            validate_finished_rerun_epoch(rerun_tpr, rerun_edr, rerun_log)
        except RuntimeError as exc:
            raise RuntimeError(f"{job.key}: {exc}") from exc
    else:
        if rerun_log.exists():
            raise RuntimeError(f"{job.key}: rerun log exists without EDR; refusing partial epoch")
        run(
            [
                GMX,
                "mdrun",
                "-ntmpi",
                "1",
                "-ntomp",
                GMX_NTOMP,
                "-nb",
                "cpu",
                "-s",
                "rerun_solute_energy.tpr",
                "-rerun",
                "solute_5ns_water_200cand.xtc",
                "-deffnm",
                "rerun_solute_energy",
            ],
            job.run_dir,
        )
        if not rerun_edr.exists() or not rerun_log.exists():
            raise RuntimeError(f"{job.key}: rerun mdrun did not create EDR/log")
        log_tail = rerun_log.read_text(encoding="utf-8", errors="replace")[-12000:]
        if "Finished mdrun" not in log_tail:
            raise RuntimeError(f"{job.key}: rerun mdrun log lacks Finished mdrun")
    run([GMX, "energy", "-f", "rerun_solute_energy.edr", "-o", xvg.name], job.run_dir, input_text="Potential\n0\n")
    if not energy_xvg_is_complete(xvg):
        raise RuntimeError(f"{job.key}: gmx energy did not create a complete 200-point XVG")
    return xvg


def select_low_energy(job: ForceFieldJob, frames_dir: Path, energy_xvg: Path) -> Path:
    selected_dir = job.run_dir / "frames_gro_5ns_water_200cand_lowest100_sugar_energy"
    run(
        [
            sys.executable,
            str(MD_ROOT / "scripts" / "select_low_energy_frames.py"),
            "--energy-xvg",
            str(energy_xvg),
            "--frames-dir",
            str(frames_dir),
            "--frame-pattern",
            "frame*.gro",
            "--output-dir",
            str(selected_dir),
            "--candidate-csv",
            str(job.run_dir / f"{job.key}_water_5ns_200_candidate_solute_potential.csv"),
            "--selected-csv",
            str(job.run_dir / f"{job.key}_water_5ns_lowest100_sugar_energy_selected.csv"),
            "--metadata-json",
            str(job.run_dir / f"{job.key}_water_5ns_lowest100_sugar_energy_metadata.json"),
            "--top-n",
            "100",
            "--overwrite",
        ],
        ROOT,
    )
    return selected_dir


def convert_pickle(job: ForceFieldJob, frames_dir: Path, selected_dir: Path) -> tuple[Path, Path]:
    all_pickle = MD_ROOT / f"conformations_md_{job.key}_water_5ns_200cand_all_seed1" / f"{MOL}.pickle"
    low_pickle = MD_ROOT / f"conformations_md_{job.key}_water_5ns_200cand_lowest100_sugar_energy_seed1" / f"{MOL}.pickle"
    for output, source_dir, max_frames in ((all_pickle, frames_dir, "200"), (low_pickle, selected_dir, "100")):
        if output.exists():
            continue
        cmd = [
            sys.executable,
            str(MD_ROOT / "scripts" / "md_frames_to_conformer_pickle.py"),
            "--frames_glob",
            str(source_dir / "*.gro"),
            "--reference_sdf",
            str(REFERENCE_SDF),
            "--molecule_name",
            MOL,
            "--output",
            str(output),
            "--max_frames",
            max_frames,
        ]
        if job.atom_index_map is not None:
            cmd.extend(["--atom_index_map", str(job.atom_index_map)])
        run(cmd, ROOT)
    return all_pickle, low_pickle


def read_energy_metadata(job: ForceFieldJob) -> dict[str, object]:
    path = job.run_dir / f"{job.key}_water_5ns_lowest100_sugar_energy_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def selected_indices(job: ForceFieldJob) -> list[int]:
    path = job.run_dir / f"{job.key}_water_5ns_lowest100_sugar_energy_selected.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [int(row["candidate_index"]) for row in rows]


def write_report(rows: list[dict[str, object]]) -> None:
    report = MD_ROOT / "水环境多力场5ns_200候选_糖自身低能100汇总.md"
    json_out = MD_ROOT / "water_5ns_200cand_sugar_energy_lowest100_summary.json"
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# 水环境多力场 5 ns / 200 候选 / 糖自身低能 100 汇总",
        "",
        "本轮每个力场均使用显式水体系采样 5 ns，每 25 ps 抽 1 帧，共 200 个糖构象。低能 100 的筛选不是水盒子 total Potential，而是把这 200 个糖构象用对应力场的溶质单独 topology 做 `mdrun -rerun`，按糖自身 Potential 从低到高选 100 个。",
        "",
        "| 力场 | 水模型 | 200 候选 | 糖低能100 | 糖 Potential 最低 | 入选最高 | 全体最高 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {water_model} | {n_candidates} | {top_n} | {lowest:.3f} | {highest_selected:.3f} | {highest:.3f} |".format(
                label=row["label"],
                water_model=row["water_model"],
                n_candidates=row["n_candidates"],
                top_n=row["top_n"],
                lowest=row["lowest_potential_kj_mol"],
                highest_selected=row["highest_selected_potential_kj_mol"],
                highest=row["highest_candidate_potential_kj_mol"],
            )
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "| 力场 | 运行目录 | 200 候选 pickle | 糖低能100 pickle | 选择表 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['label']} | `{row['run_dir']}` | `{row['all_pickle']}` | `{row['low_pickle']}` | `{row['selected_csv']}` |"
        )
    lines.extend(
        [
            "",
            "说明：这些低能 100 表示“水中采样得到的糖构象，在对应力场溶质单独势能面上最低的 100 个”。如果要把糖-水相互作用也纳入排序，需要另做 solute-solvent energy decomposition。",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {report}", flush=True)
    print(f"[json] {json_out}", flush=True)


def process_job(job: ForceFieldJob) -> dict[str, object]:
    print(f"\n=== {job.label} ({job.key}) ===", flush=True)
    run_water_md(job)
    frames_dir = extract_solute_frames(job)
    energy_xvg = rerun_solute_energy(job, frames_dir)
    selected_dir = select_low_energy(job, frames_dir, energy_xvg)
    all_pickle, low_pickle = convert_pickle(job, frames_dir, selected_dir)
    metadata = read_energy_metadata(job)
    indices = selected_indices(job)
    return {
        "key": job.key,
        "label": job.label,
        "water_model": job.water_model,
        "run_dir": str(job.run_dir),
        "frames_dir": str(frames_dir),
        "selected_dir": str(selected_dir),
        "energy_xvg": str(energy_xvg),
        "candidate_csv": str(job.run_dir / f"{job.key}_water_5ns_200_candidate_solute_potential.csv"),
        "selected_csv": str(job.run_dir / f"{job.key}_water_5ns_lowest100_sugar_energy_selected.csv"),
        "all_pickle": str(all_pickle),
        "low_pickle": str(low_pickle),
        "n_candidates": metadata["n_candidates"],
        "top_n": metadata["top_n"],
        "lowest_potential_kj_mol": metadata["lowest_potential_kj_mol"],
        "highest_selected_potential_kj_mol": metadata["highest_selected_potential_kj_mol"],
        "highest_candidate_potential_kj_mol": metadata["highest_candidate_potential_kj_mol"],
        "selected_candidate_indices": indices,
    }


def main() -> None:
    jobs = [
        prepare_glycam06j(),
        prepare_gromos56a6carbo(),
        prepare_charmm36_bmm(),
    ]
    rows = [process_job(job) for job in jobs]
    write_report(rows)


if __name__ == "__main__":
    main()
