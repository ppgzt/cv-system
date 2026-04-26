"""Global in-memory frame buffer shared across all MAS agents.

Agents exchange numpy image arrays through this dict by key reference
instead of serialising them into FIPA-ACL payloads (which would require
base64 encoding and explode message size).

Keys are string frame identifiers; values are numpy ndarrays.
"""

from typing import Any

FRAME_BUFFER: dict[str, Any] = {}
