# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""Typed discovery for the GenieSim 3.2 / RoboColiseum benchmark tasks on G2.

The 28 benchmark tasks are organised by leaderboard: ``instruction`` (10),
``spatial`` (8) and ``manipulation`` (10). The ``robust`` board replays the ten
instruction tasks under five generalization axes and is not a separate task
set, so it is not listed here.

A benchmark task's *language* is not known until the GenieSim client sends its
``prompt``: the instruction is sampled per episode. The registry therefore
carries a description only; the adapter substitutes the live prompt when the
episode starts.

``horizon`` is the episode budget in simulator steps, ``step_out`` from each
task's ``instructions.json``. Only the values measured so far are recorded; the
rest use a conservative default and should be filled in as they are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

DEFAULT_HORIZON = 450
FAMILY = "g2_geniesim"

#: GenieSim category tokens and common spellings, mapped to the board name.
BOARD_ALIASES: dict[str, str] = {
    "instruction": "instruction",
    "if": "instruction",
    "spatial": "spatial",
    "manipulation": "manipulation",
    "manip": "manipulation",
}
BOARDS = ("instruction", "spatial", "manipulation")


#: Sub-task names GenieSim reports for a listed task under another name: ``g2op_manip_bimanual_hold_pot.yaml`` sets
#: ``sub_task_name: hold_pot`` (and ``robot_init_states.py`` keys the task as ``hold_pot``).
TASK_NAME_ALIASES: dict[str, str] = {"hold_pot": "bimanual_hold_pot"}


class G2RegistryError(ValueError):
    """Raised when a G2 benchmark task reference is invalid."""


@dataclass(frozen=True, slots=True)
class G2TaskMetadata:
    """Static metadata identifying one GenieSim benchmark task."""

    task_name: str
    board: str
    task_id: int
    description: str
    horizon: int = DEFAULT_HORIZON
    family: str = FAMILY

    @property
    def suite_name(self) -> str:
        return self.board

    @property
    def language(self) -> str:
        """Placeholder until the episode's prompt arrives from the client."""
        return self.description

    @property
    def task_ref(self) -> str:
        return f"{self.board}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "board": self.board,
            "description": self.description,
            "family": self.family,
            "horizon": self.horizon,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
        }


#: ``step_out`` budgets read from GenieSim's ``benchmark/config/llm_task/<task>/*/instructions.json``.
#: Spatial board (problems.json, upstream 6ca11c7): each stage's ``StepOut N`` starts when the stage starts and its
#: last check that can still score is 30 * (N // 30 + 1) rows later (checks run every 30 rows); the horizon is the
#: sum of those over the task's stages plus one check of margin, so the run recorder never stops a program GenieSim
#: would still score (pick: 200 + 200 -> 420; place: 300 + 600 -> 960; sort: 1200 -> 1230; stack: 1500 -> 1530).
#: Manipulation board (C5m, problems.json upstream 6ca11c7), the same rule: open_door 450 -> 480; hold_pot 900 + 900 ->
#: 1860; pour 900 -> 930; stock 300 + 300 + 600 + 600 + 300 -> 2250; take_wrong 300 + 400 -> 750; scoop 900 -> 930; clean
#: 4 x 600 + 300 -> 2850; place_block 200 + 200 + 400 -> 840; sorting 600 + 300 + 3 x 900 -> 3750; continuous: the whole
#: list sits in one ``StepOut 6000`` -> 6030 (each package's chain also has its own 1500).
KNOWN_HORIZONS: dict[str, int] = {
    "open_door": 510,
    "bimanual_hold_pot": 1890,
    "pour_workpiece": 960,
    "stock_and_straighten_shelf": 2280,
    "take_wrong_item_shelf": 780,
    "scoop_popcorn": 960,
    "clean_the_desktop": 2880,
    "place_block_into_box": 870,
    "sorting_packages": 3780,
    "sorting_packages_continuous": 6060,
    "pick_object_absolute_position": 450,
    "pick_object_relative_position": 450,
    "place_beverage_to_anothers_position": 990,
    "place_object_relative_position": 990,
    "sort_cubes_by_size": 1260,
    "sort_number": 1260,
    "stack_bowls": 1560,
    "stack_three_building_blocks": 1560,
}

_TASK_TABLE: tuple[tuple[str, str, str], ...] = (
    # instruction (10)
    ("instruction", "pick_billiards_color", "Pick the billiard ball of the named colour."),
    ("instruction", "pick_block_color", "Pick the block of the named colour."),
    ("instruction", "pick_block_number", "Pick the block carrying the named number."),
    ("instruction", "pick_block_shape", "Pick the block of the named shape."),
    ("instruction", "pick_block_size", "Pick the block of the named size."),
    ("instruction", "pick_common_sense", "Pick the object a common-sense cue identifies."),
    ("instruction", "pick_follow_logic_or", "Pick an object satisfying an OR condition."),
    ("instruction", "pick_object_type", "Pick the object of the named type."),
    ("instruction", "pick_specific_object", "Pick the specifically named object."),
    ("instruction", "straighten_object", "Straighten the named object on the table."),
    # spatial (8)
    ("spatial", "pick_object_absolute_position", "Pick the object at an absolute position."),
    ("spatial", "pick_object_relative_position", "Pick the object at a relative position."),
    ("spatial", "place_beverage_to_anothers_position", "Place a beverage where another one was."),
    ("spatial", "place_object_relative_position", "Place an object relative to another."),
    ("spatial", "sort_cubes_by_size", "Sort the cubes by size."),
    ("spatial", "sort_number", "Sort the numbered objects in order."),
    ("spatial", "stack_bowls", "Stack the bowls."),
    ("spatial", "stack_three_building_blocks", "Stack three building blocks."),
    # manipulation (10)
    ("manipulation", "open_door", "Open the door."),
    ("manipulation", "bimanual_hold_pot", "Hold the pot with both hands and move it."),
    ("manipulation", "pour_workpiece", "Pour the workpiece out of its container."),
    ("manipulation", "stock_and_straighten_shelf", "Stock the shelf and straighten the items."),
    ("manipulation", "take_wrong_item_shelf", "Take the wrong item off the shelf."),
    ("manipulation", "scoop_popcorn", "Scoop popcorn into the container."),
    ("manipulation", "clean_the_desktop", "Clean the desktop."),
    ("manipulation", "place_block_into_box", "Place the block into the box."),
    ("manipulation", "sorting_packages", "Sort the packages."),
    ("manipulation", "sorting_packages_continuous", "Sort packages continuously as they arrive."),
)


def _build_tasks() -> tuple[G2TaskMetadata, ...]:
    tasks: list[G2TaskMetadata] = []
    counters = dict.fromkeys(BOARDS, 0)
    for board, name, description in _TASK_TABLE:
        tasks.append(
            G2TaskMetadata(
                task_name=name,
                board=board,
                task_id=counters[board],
                description=description,
                horizon=KNOWN_HORIZONS.get(name, DEFAULT_HORIZON),
            )
        )
        counters[board] += 1
    return tuple(tasks)


G2_GENIESIM_TASKS: tuple[G2TaskMetadata, ...] = _build_tasks()
TASKS_BY_BOARD: MappingProxyType[str, tuple[G2TaskMetadata, ...]] = MappingProxyType(
    {board: tuple(task for task in G2_GENIESIM_TASKS if task.board == board) for board in BOARDS}
)


class G2TaskRegistry:
    """Resolve the 28 GenieSim benchmark tasks by board + index or by task name."""

    def __init__(self, tasks: tuple[G2TaskMetadata, ...] = G2_GENIESIM_TASKS) -> None:
        self._by_name = {task.task_name: task for task in tasks}
        if len(self._by_name) != len(tasks):
            raise G2RegistryError("G2 task names must be unique")
        self._by_ref = {(task.board, task.task_id): task for task in tasks}

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    @property
    def boards(self) -> tuple[str, ...]:
        return tuple(sorted({task.board for task in self._by_name.values()}))

    def enumerate_tasks(self, board: str | None = None) -> tuple[G2TaskMetadata, ...]:
        if board is None:
            return tuple(self._by_name.values())
        normalized = self._board(board)
        return tuple(task for task in self._by_name.values() if task.board == normalized)

    @staticmethod
    def _board(name: object) -> str:
        key = str(name).strip().lower()
        try:
            return BOARD_ALIASES[key]
        except KeyError as exc:
            raise G2RegistryError(
                f"unknown G2 benchmark board {name!r}; expected one of {', '.join(BOARDS)}"
            ) from exc

    def resolve(
        self,
        task_ref: str | tuple[str, int] | G2TaskMetadata,
        task_id: int | None = None,
    ) -> G2TaskMetadata:
        """Resolve ``(board, index)``, ``"board:index"``, ``"task_name"`` or metadata."""
        if isinstance(task_ref, G2TaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise G2RegistryError("task tuple must contain (board_or_name, task_id)")
            name, tuple_id = task_ref
            if task_id is not None and int(task_id) != int(tuple_id):
                raise G2RegistryError("conflicting G2 task ids")
            task_id = int(tuple_id)
        else:
            name = str(task_ref)
            if ":" in name:
                name, encoded = name.rsplit(":", 1)
                if task_id is not None and int(task_id) != int(encoded):
                    raise G2RegistryError("conflicting G2 task ids")
                task_id = int(encoded)
        name = str(name).strip()
        name = TASK_NAME_ALIASES.get(name, name)
        if name in self._by_name:
            task = self._by_name[name]
            if task_id is not None and int(task_id) != task.task_id:
                raise G2RegistryError(
                    f"task {name!r} has id {task.task_id} on the {task.board} board, not {task_id}"
                )
            return task
        board = self._board(name)
        if task_id is None:
            raise G2RegistryError(f"a task id is required to select a task on the {board} board")
        try:
            return self._by_ref[(board, int(task_id))]
        except KeyError as exc:
            count = len(TASKS_BY_BOARD[board])
            raise G2RegistryError(
                f"the {board} board has task ids 0..{count - 1}, not {task_id}"
            ) from exc

    def by_task_name(self, task_name: str) -> G2TaskMetadata | None:
        """Metadata for the sub-task name the client reports (or its alias), or None when unlisted."""
        name = str(task_name).strip()
        return self._by_name.get(name) or self._by_name.get(TASK_NAME_ALIASES.get(name, ""))


__all__ = [
    "BOARDS",
    "BOARD_ALIASES",
    "DEFAULT_HORIZON",
    "FAMILY",
    "G2_GENIESIM_TASKS",
    "KNOWN_HORIZONS",
    "TASKS_BY_BOARD",
    "TASK_NAME_ALIASES",
    "G2RegistryError",
    "G2TaskMetadata",
    "G2TaskRegistry",
]
