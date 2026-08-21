#!/usr/bin/env bash
# Historical name retained deliberately. A pilot cannot be controlled locally
# on the Pi: TC66C, SSH/SCP, T0/T1 and cooldown belong to the Mac controller.
set -euo pipefail

printf '%s\n' "[deprecated] Run ./scripts/pilot_5_modes_from_mac.sh from the Mac; this Pi-local controller is intentionally disabled." >&2
exit 2
