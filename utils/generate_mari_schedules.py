#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026-present Inria
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate `dotbot/mari_schedules_generated.json` from a mari firmware checkout.

Run this whenever the mari pin this repo is validated against bumps, so the
standalone-install fallback (used when no mari checkout is findable) doesn't
silently drift from the firmware it's meant to mirror. Needs a mari checkout,
found the same way `dotbot.mari_schedules` finds one: `--mari-dir`,
`$MARI_FIRMWARE_DIR`, or a sibling `mari/` directory.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotbot import mari_schedules as ms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mari-dir", default=None, help="path to a mari checkout")
    parser.add_argument(
        "-o", "--out", default=None, help="output path (default: dotbot/mari_schedules_generated.json)"
    )
    args = parser.parse_args()

    resolved = ms.resolve_mari_dir(args.mari_dir)
    schedules = ms._load_from_firmware(str(resolved))
    slot_duration_ms = next(iter(schedules.values())).slot_duration_ms

    payload = {
        "slot_duration_ms": slot_duration_ms,
        "schedules": {
            name: {"id": s.id, "max_nodes": s.max_nodes, "cell_types": list(s.cell_types)}
            for name, s in sorted(schedules.items(), key=lambda item: item[1].max_nodes)
        },
    }

    out_path = Path(args.out) if args.out else ms.GENERATED_SNAPSHOT
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out_path} ({len(payload['schedules'])} schedules from {resolved}, "
        f"slot_duration_ms={slot_duration_ms})"
    )


if __name__ == "__main__":
    main()
