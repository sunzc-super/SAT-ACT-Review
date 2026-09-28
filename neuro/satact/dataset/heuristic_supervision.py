"""Multi-native index for baseline/replay/agreement SAT-ACT ablations."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, cast

import torch

from satact.dataset.action_records import (
    ActionEvalDatasetError,
    _IndexEntry,
    exact_internal_literal_mask,
    signed_external_to_internal_index,
)
from satact.dataset.action_preferences import (
    ActionEvalShardDatasetSATACT,
    _SATACTOutcome,
    _EVAL_ARRAYS,
    _attach_diagnostics,
    _diagnostics_path,
    _pack_satact_diagnostics,
    _pack_satact_index,
    _unpack_satact_index,
    compare_satact_actions,
    derive_target_satact,
)


SATACT_INDEX_CONTRACTS: Final[tuple[str, ...]] = (
    "preference-only",
    "heuristic-supervision",
)
SATACT_NATIVE_OUTCOMES: Final[tuple[str, ...]] = (
    "baseline",
    "replay",
    "agreement",
)
SATACT_MULTI_NATIVE_INDEX_SCHEMA: Final[str] = (
    "decisiontrace-satact-heuristic-index"
)
SATACT_MULTI_NATIVE_DIAGNOSTICS_SCHEMA: Final[str] = (
    "decisiontrace-actioneval-satact-multi-native-diagnostics"
)
SATACTIndexContract = Literal["preference-only", "heuristic-supervision"]
SATACTNativeOutcome = Literal["baseline", "replay", "agreement"]


def canonicalize_satact_index_contract(value: str) -> SATACTIndexContract:
    if value not in SATACT_INDEX_CONTRACTS:
        raise ValueError(f"unsupported SATACT index contract: {value!r}")
    return cast(SATACTIndexContract, value)


def canonicalize_satact_native_outcome(value: str) -> SATACTNativeOutcome:
    if value not in SATACT_NATIVE_OUTCOMES:
        raise ValueError(f"unsupported SATACT native outcome: {value!r}")
    return cast(SATACTNativeOutcome, value)


def validate_satact_index_selection(
    index_contract: str, native_outcome: str
) -> tuple[SATACTIndexContract, SATACTNativeOutcome]:
    contract = canonicalize_satact_index_contract(index_contract)
    outcome = canonicalize_satact_native_outcome(native_outcome)
    if contract == "preference-only" and outcome != "replay":
        raise ValueError(
            "preference-only indexes support only --satact_native_outcome replay"
        )
    return contract, outcome


def _native_outcomes_agree(
    baseline: _SATACTOutcome | None, replay: _SATACTOutcome | None
) -> bool:
    if baseline is None or replay is None:
        return False
    if baseline.complete != replay.complete:
        return False
    if not baseline.complete:
        return True
    return (
        baseline.status,
        baseline.conflicts,
        baseline.propagations,
        baseline.decisions,
        baseline.restarts,
    ) == (
        replay.status,
        replay.conflicts,
        replay.propagations,
        replay.decisions,
        replay.restarts,
    )


def _materialize_target(
    *,
    state: Mapping[str, Any],
    i2e: Sequence[int],
    eligibility_bits: Sequence[int],
    arrays: Mapping[str, Sequence[int]],
    native_outcome: SATACTNativeOutcome,
) -> dict[str, Any] | None:
    if native_outcome == "replay":
        return derive_target_satact(
            state=state,
            i2e=i2e,
            eligibility_bits=eligibility_bits,
            arrays=arrays,
        )

    lengths = {len(arrays[name]) for name in _EVAL_ARRAYS}
    if len(lengths) != 1:
        raise ActionEvalDatasetError("SATACT rollout array lengths disagree")
    native_external = int(state.get("native_external", 0))
    eligibility = exact_internal_literal_mask(i2e, eligibility_bits)
    native_index = signed_external_to_internal_index(native_external, i2e)
    if native_index is not None and not bool(eligibility[native_index]):
        native_index = None

    baseline: tuple[_SATACTOutcome, int] | None = None
    replay: tuple[_SATACTOutcome, int] | None = None
    nonnative: list[tuple[_SATACTOutcome, int]] = []
    seen_literals: set[int] = set()
    rows = zip(*(arrays[name] for name in _EVAL_ARRAYS))
    for values in rows:
        (
            requested,
            reached,
            locator,
            eligible,
            applied,
            applied_literal,
            eval_native,
            status,
            censored,
            horizon,
            conflicts,
            decisions,
            propagations,
            restarts,
        ) = (int(value) for value in values)
        native_matches = eval_native == native_external
        applied_correct = bool(applied) and native_matches and (
            (requested == 0 and applied_literal == eval_native)
            or (requested != 0 and applied_literal == requested)
        )
        reliable = bool(reached and locator and eligible and applied_correct)
        complete = bool(
            reliable and not censored and not horizon and status in {10, 20}
        )
        outcome = _SATACTOutcome(
            requested=requested,
            status=status,
            conflicts=conflicts,
            propagations=propagations,
            decisions=decisions,
            restarts=restarts,
            reliable=reliable,
            complete=complete,
        )
        if requested == 0:
            if reliable and native_index is not None:
                baseline = (outcome, native_index)
            continue
        if not reliable:
            continue
        literal_index = signed_external_to_internal_index(requested, i2e)
        if literal_index is None or not bool(eligibility[literal_index]):
            raise ActionEvalDatasetError(
                "reliable SATACT rollout lies outside exact eligibility"
            )
        if literal_index in seen_literals:
            raise ActionEvalDatasetError("duplicate SATACT forced evaluation literal")
        seen_literals.add(literal_index)
        item = (outcome, literal_index)
        if literal_index == native_index:
            replay = item
        else:
            nonnative.append(item)

    native: tuple[_SATACTOutcome, int] | None = None
    if native_outcome == "baseline":
        native = baseline
    elif _native_outcomes_agree(
        baseline[0] if baseline is not None else None,
        replay[0] if replay is not None else None,
    ):
        native = replay
    actions = list(nonnative)
    if native is not None:
        actions.append(native)

    solved_statuses = {outcome.status for outcome, _ in actions if outcome.complete}
    if len(solved_statuses) > 1:
        raise ActionEvalDatasetError(
            "SAT/UNSAT rollout inconsistency within one SATACT state"
        )

    better: list[int] = []
    worse: list[int] = []
    rule_mask: list[int] = []
    for left_index in range(len(actions)):
        for right_index in range(left_index + 1, len(actions)):
            order, mask = compare_satact_actions(
                actions[left_index][0], actions[right_index][0]
            )
            if order < 0:
                better.append(actions[left_index][1])
                worse.append(actions[right_index][1])
                rule_mask.append(mask)
            elif order > 0:
                better.append(actions[right_index][1])
                worse.append(actions[left_index][1])
                rule_mask.append(mask)
    if not better:
        return None

    evaluated = sorted(actions, key=lambda item: item[1])
    return {
        "n_literals": 2 * len(i2e),
        "native_literal_index": native_index,
        "pair_better": better,
        "pair_worse": worse,
        "pair_rule_mask": rule_mask,
        "evaluated_literals": [literal for _, literal in evaluated],
        "evaluated_complete": [outcome.complete for outcome, _ in evaluated],
        "evaluated_conflicts": [outcome.conflicts for outcome, _ in evaluated],
        "evaluated_propagations": [
            outcome.propagations for outcome, _ in evaluated
        ],
        "evaluated_decisions": [outcome.decisions for outcome, _ in evaluated],
        "evaluated_restarts": [outcome.restarts for outcome, _ in evaluated],
        "listwise_literals": [],
        "listwise_target": [],
        "evaluated_costs": [None for _ in evaluated],
        "baseline_cost": None,
        "baseline_complete": False,
    }


def _multi_part_worker(
    task: tuple[Any, ...],
    *,
    instance_ids: frozenset[str] | None = None,
) -> tuple[int, dict[str, list[_IndexEntry]]]:
    (
        shard_index,
        stem,
        split_dir,
        reader_kind,
        verify,
        max_header_bytes,
        max_array_bytes,
    ) = task
    if reader_kind == "v2":
        from gen_data.rollout_dataset.compact import CompactArrayReader
    else:
        from gen_data.rollout_dataset.compact import CompactArrayReader
    complete = Path(split_dir) / "shards" / f"{stem}.complete.json"
    entries: dict[str, list[_IndexEntry]] = {
        mode: [] for mode in SATACT_NATIVE_OUTCOMES
    }
    with CompactArrayReader(
        complete,
        verify=bool(verify),
        max_header_bytes=int(max_header_bytes),
        max_array_bytes=int(max_array_bytes),
    ) as reader:
        for record_index, record in enumerate(reader.header["records"]):
            metadata = record.get("metadata", {})
            if metadata.get("kind") == "capture_failure":
                continue
            if metadata.get("kind") != "state":
                raise ActionEvalDatasetError("unknown compact record kind")
            state = metadata.get("state")
            if instance_ids is not None and state["instance"] not in instance_ids:
                continue
            names = ("i2e", "eligibility_bits", *_EVAL_ARRAYS)
            read_many = getattr(reader, "read_arrays", None)
            if read_many is not None:
                loaded = read_many(record, names)
                values = {name: list(loaded[name]) for name in names}
            else:
                values = {
                    name: list(reader.read_array(record, name)) for name in names
                }
            arrays = {name: values[name] for name in _EVAL_ARRAYS}
            for mode in SATACT_NATIVE_OUTCOMES:
                target = _materialize_target(
                    state=state,
                    i2e=values["i2e"],
                    eligibility_bits=values["eligibility_bits"],
                    arrays=arrays,
                    native_outcome=cast(SATACTNativeOutcome, mode),
                )
                if target is None:
                    continue
                entries[mode].append(
                    _IndexEntry(
                        shard_index=int(shard_index),
                        record_index=record_index,
                        record_id=str(record.get("record_id")),
                        instance_id=str(state["instance"]),
                        eligible_state=int(state["eligible_state"]),
                        target=target,
                    )
                )
    return int(shard_index), entries


class ActionEvalShardDatasetSATACTMultiNative(ActionEvalShardDatasetSATACT):
    """One raw scan and one cache containing all three native policies."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        index_cache_path: str | Path,
        native_outcome: SATACTNativeOutcome = "replay",
        **kwargs: Any,
    ) -> None:
        self.native_outcome = canonicalize_satact_native_outcome(native_outcome)
        self._built_by_mode: dict[str, list[_IndexEntry]] | None = None
        super().__init__(
            split_dir,
            index_cache_path=index_cache_path,
            **kwargs,
        )
        self.split_fingerprint = (
            f"{self.split_fingerprint}:satact-index-contract=heuristic-supervision"
            f":satact-native-outcome={self.native_outcome}"
        )

    def _satact_config(self, max_states: int | None) -> dict[str, Any]:
        return {
            "schema": SATACT_MULTI_NATIVE_INDEX_SCHEMA,
            "split_dir": str(self.split_dir),
            "manifest_split": self.manifest_split,
            "parts": list(self._parts),
            "max_states": max_states,
        }

    def _load_explicit_cached_index(
        self, path: Path, index_config: Mapping[str, Any]
    ) -> list[_IndexEntry] | None:
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        config = self._satact_config(index_config.get("max_states"))
        if (
            payload.get("schema") != SATACT_MULTI_NATIVE_INDEX_SCHEMA
            or payload.get("index_config") != config
        ):
            return None
        modes = payload.get("modes")
        if not isinstance(modes, Mapping) or self.native_outcome not in modes:
            return None
        entries = _unpack_satact_index(modes[self.native_outcome])
        if self.include_diagnostics:
            diagnostics_path = _diagnostics_path(path)
            if not diagnostics_path.is_file():
                return None
            diagnostics = torch.load(
                diagnostics_path, map_location="cpu", weights_only=False, mmap=True
            )
            diagnostic_modes = diagnostics.get("modes")
            if (
                diagnostics.get("schema")
                != SATACT_MULTI_NATIVE_DIAGNOSTICS_SCHEMA
                or diagnostics.get("index_config") != config
                or not isinstance(diagnostic_modes, Mapping)
                or self.native_outcome not in diagnostic_modes
            ):
                return None
            entries = _attach_diagnostics(
                entries, diagnostic_modes[self.native_outcome]
            )
        return entries

    def _write_explicit_cached_index(
        self,
        path: Path,
        index_config: Mapping[str, Any],
        entries: Sequence[_IndexEntry],
    ) -> None:
        del entries
        if self._built_by_mode is None:
            raise ActionEvalDatasetError("multi-native build results are unavailable")
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self._satact_config(index_config.get("max_states"))
        diagnostics_path = _diagnostics_path(path)
        core_modes = {
            mode: _pack_satact_index(mode_entries, config)
            for mode, mode_entries in self._built_by_mode.items()
        }
        diagnostic_modes = {
            mode: _pack_satact_diagnostics(mode_entries, config)
            for mode, mode_entries in self._built_by_mode.items()
        }
        payloads = (
            (
                path,
                {
                    "schema": SATACT_MULTI_NATIVE_INDEX_SCHEMA,
                    "index_config": config,
                    "modes": core_modes,
                },
            ),
            (
                diagnostics_path,
                {
                    "schema": SATACT_MULTI_NATIVE_DIAGNOSTICS_SCHEMA,
                    "index_config": config,
                    "modes": diagnostic_modes,
                },
            ),
        )
        for destination, payload in payloads:
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp"
            )
            try:
                torch.save(payload, temporary)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

    def _build_index(self, *, max_states: int | None) -> list[_IndexEntry]:
        reader_kind = (
            "v2"
            if self._reader_class.__module__.startswith(
                "gen_data.rollout_dataset"
            )
            else "v1"
        )
        tasks = [
            (
                shard_index,
                stem,
                str(self.split_dir),
                reader_kind,
                False,
                self.max_header_bytes,
                self.max_array_bytes,
            )
            for shard_index, stem in enumerate(self._parts)
        ]
        ordered: dict[str, list[_IndexEntry]] = {
            mode: [] for mode in SATACT_NATIVE_OUTCOMES
        }

        def append_part(
            shard_index: int, entries_by_mode: Mapping[str, Sequence[_IndexEntry]]
        ) -> bool:
            for mode in SATACT_NATIVE_OUTCOMES:
                for entry in entries_by_mode[mode]:
                    if max_states is not None and len(ordered[mode]) >= max_states:
                        break
                    ordered[mode].append(entry)
            self._report_progress(
                phase="index",
                shard_index=shard_index,
                states=len(ordered[self.native_outcome]),
            )
            return max_states is not None and all(
                len(ordered[mode]) >= max_states for mode in SATACT_NATIVE_OUTCOMES
            )

        if self.preprocess_workers == 1:
            for task in tasks:
                shard_index, entries_by_mode = _multi_part_worker(task)
                if append_part(shard_index, entries_by_mode):
                    break
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.preprocess_workers
            ) as executor:
                pending: dict[
                    concurrent.futures.Future[
                        tuple[int, dict[str, list[_IndexEntry]]]
                    ],
                    int,
                ] = {}
                ready: dict[int, dict[str, list[_IndexEntry]]] = {}
                next_task = 0
                next_part = 0
                limit_reached = False
                while next_task < len(tasks) or pending:
                    while (
                        next_task < len(tasks)
                        and len(pending) + len(ready)
                        < 2 * self.preprocess_workers
                    ):
                        future = executor.submit(_multi_part_worker, tasks[next_task])
                        pending[future] = next_task
                        next_task += 1
                    done, _ = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        pending.pop(future)
                        shard_index, entries_by_mode = future.result()
                        ready[shard_index] = entries_by_mode
                    while next_part in ready:
                        entries_by_mode = ready.pop(next_part)
                        if append_part(next_part, entries_by_mode):
                            limit_reached = True
                            break
                        next_part += 1
                    if limit_reached:
                        for future in pending:
                            future.cancel()
                        break
        self._built_by_mode = ordered
        return ordered[self.native_outcome]


__all__ = [
    "ActionEvalShardDatasetSATACTMultiNative",
    "SATACT_INDEX_CONTRACTS",
    "SATACT_MULTI_NATIVE_DIAGNOSTICS_SCHEMA",
    "SATACT_MULTI_NATIVE_INDEX_SCHEMA",
    "SATACT_NATIVE_OUTCOMES",
    "SATACTIndexContract",
    "SATACTNativeOutcome",
    "canonicalize_satact_index_contract",
    "canonicalize_satact_native_outcome",
    "validate_satact_index_selection",
]
