# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. See LICENSE-FF.txt.
"""AgiBot G2 embodiment for the CaP harness (GenieSim 3.2 now, GDK later).

Layering, mirroring the other embodiments::

    CapApi                      embodiment-agnostic
      -> G2GenieSimAdapter      adapter.py         -- normalize, latch, account ticks
        -> GenieSimBridge       geniesim_bridge.py -- msgpack WebSocket policy server
          -> GenieSim CoRobotPolicy (client)        -- replays one chunk row per tick

``codec`` owns every unit, sign and frame conversion; ``config`` loads the
station profile; ``registry`` lists the 28 benchmark tasks; ``kinematics`` is
the URDF FK behind end-effector and camera poses; ``curobo_config`` converts
GenieSim's planner description for the cuRobo service. Imports of the bridge
and adapter are lazy so the package stays importable without ``websockets`` /
``msgpack``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cap_harness.g2.config import (
    G2StationConfig,
    load_g2_station_config,
    load_g2_station_config_from_path,
    station_config_path,
)
from cap_harness.g2.kinematics import G2Kinematics, UrdfModel
from cap_harness.g2.registry import (
    G2_GENIESIM_TASKS,
    G2RegistryError,
    G2TaskMetadata,
    G2TaskRegistry,
)

if TYPE_CHECKING:
    from cap_harness.g2.adapter import G2GenieSimAdapter
    from cap_harness.g2.geniesim_bridge import GenieSimBridge

_LAZY = {
    "G2GenieSimAdapter": ("cap_harness.g2.adapter", "G2GenieSimAdapter"),
    "GenieSimBridge": ("cap_harness.g2.geniesim_bridge", "GenieSimBridge"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    import importlib

    return getattr(importlib.import_module(module_name), attribute)


__all__ = [
    "G2_GENIESIM_TASKS",
    "G2GenieSimAdapter",
    "G2Kinematics",
    "G2RegistryError",
    "G2StationConfig",
    "G2TaskMetadata",
    "G2TaskRegistry",
    "GenieSimBridge",
    "UrdfModel",
    "load_g2_station_config",
    "load_g2_station_config_from_path",
    "station_config_path",
]
