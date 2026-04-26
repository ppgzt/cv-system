"""Data Enhance Agent — applies image transformations via DataEnhanceAdapter.

Receives a FIPA INFORM from CaptureAgent containing only a frame key and
metadata.  Pulls the raw image from FRAME_BUFFER, enhances it in a
delegated thread (deferToThread), overwrites the same key in FRAME_BUFFER
with the enhanced version, and forwards the *unchanged* metadata payload
to FrameSelectionAgent.

The elapsed_time in the payload reflects the organic capture moment,
unaffected by processing latency.
"""

import json

from twisted.internet.threads import deferToThread

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.adapters.data_enhance_adapter import DataEnhanceAdapter
from mas.utils.globals import FRAME_BUFFER


class DataEnhanceAgent(Agent):
    """Data enhancement PADE agent — transformation pipeline node."""

    def __init__(
        self,
        aid,
        data_enhance_adapter: DataEnhanceAdapter,
        next_agent_aid: str,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.data_enhance_adapter = data_enhance_adapter
        self.next_agent_aid = next_agent_aid

    def _parse_payload(self, message) -> dict | None:
        try:
            payload = json.loads(message.content)
        except (TypeError, json.JSONDecodeError):
            display_message(self.aid.name, "[WARN] invalid JSON payload")
            return None
        if not payload.get("frame_id"):
            display_message(self.aid.name, "[WARN] missing frame_id")
            return None
        return payload

    def _on_enhance_done(self, _result, payload: dict):
        out = ACLMessage(ACLMessage.INFORM)
        out.set_ontology("frame-enhanced")
        out.add_receiver(AID(self.next_agent_aid))
        out.set_content(json.dumps(payload, ensure_ascii=True))
        self.send(out)
        display_message(
            self.aid.name,
            f"frame_id={payload['frame_id']} enhanced and forwarded.",
        )

    def _on_enhance_error(self, failure):
        display_message(
            self.aid.name,
            f"[ERROR] Enhancement failed: {failure.getErrorMessage()}",
        )

    def _schedule_enhance(self, payload: dict):
        frame_id = payload["frame_id"]
        img = FRAME_BUFFER.get(frame_id)
        if img is None:
            display_message(self.aid.name, f"[WARN] frame_id={frame_id} not in buffer")
            return

        def _enhance_and_store():
            enhanced = self.data_enhance_adapter.run(img)
            FRAME_BUFFER[frame_id] = enhanced
            return True

        d = deferToThread(_enhance_and_store)
        d.addCallback(self._on_enhance_done, payload)
        d.addErrback(self._on_enhance_error)

    def react(self, message):
        super().react(message)
        if message.performative != ACLMessage.INFORM:
            return
        if message.ontology != "frame-capture":
            return
        payload = self._parse_payload(message)
        if not payload:
            return
        self._schedule_enhance(payload)

    def on_start(self):
        super().on_start()
        display_message(self.aid.name, "DataEnhanceAgent started.")
