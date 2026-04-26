"""Capture Agent — biological simulator for periodic frame acquisition.

Models the temporal illusion of cattle passing through a scale one-by-one.
Each animal enters the camera FOV for `passage_time` seconds, then a gap
of `arrival_time` seconds elapses before the next animal appears.

The TimedBehaviour pulses at the FPS rate.  On each tick it computes the
organic elapsed time since the current animal entered the frame:

- PASSAGE mode  (elapsed <= passage_time):
    Capture a frame, store it in FRAME_BUFFER, and FIPA-INFORM the key
    (plus animal_id and elapsed_time) to the next agent.
- ARRIVAL mode  (passage_time < elapsed < passage_time + arrival_time):
    Idle — the animal is off-screen.  No capture, no message.
- RESTART       (elapsed >= passage_time + arrival_time):
    Advance to the next animal and reset the clock.  If all animals
    have been processed, stop the reactor.

Blocking calls (time.sleep, loops) are prohibited — the Twisted event
loop must never be blocked.
"""

import json
import threading
import time
import uuid

from twisted.internet import reactor

import mas  # noqa: F401

from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import TimedBehaviour
from pade.core.agent import Agent
from pade.misc.utility import display_message

from mas.adapters.capture_adapter import CaptureAdapter
from mas.utils.globals import FRAME_BUFFER


class CaptureBehaviour(TimedBehaviour):
    """Pulsing behaviour that drives the PASSAGE / ARRIVAL / RESTART cycle."""

    def __init__(
        self,
        agent: Agent,
        capture_adapter: CaptureAdapter,
        next_agent_aid: str,
        selection_agent_aid: str,
        passage_time: int,
        arrival_time: int,
        herd_size: int,
        interval_seconds: float,
    ):
        super().__init__(agent, interval_seconds)
        self.capture_adapter = capture_adapter
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.passage_time = passage_time
        self.arrival_time = arrival_time
        self.herd_size = herd_size

        self.current_animal_id = 1
        self.start_at = None
        self._finished = False
        self._lock = threading.Lock()
        
        self.captured_count = 0
        self.first_capture = None
        self.last_capture = None
        self.passage_signaled = False

    def on_time(self):
        super().on_time()
        
        if self._finished:
            return

        # --- WAIT for simulation ignition ---
        if not self.agent.simulation_started:
            return

        # Initialize timeline on the very first pulse after ignition
        if self.start_at is None:
            self.start_at = time.time()
            display_message(self.agent.aid.name, f"Timeline started for Animal {self.current_animal_id}")

        elapsed = time.time() - self.start_at
        cycle = self.passage_time + self.arrival_time

        # --- SIGNAL PASSAGE COMPLETE at exact arrival edge ---
        if elapsed > self.passage_time and not self.passage_signaled:
            if self.agent.simulation_started:
                msg = ACLMessage(ACLMessage.INFORM)
                msg.set_ontology("passage-complete")
                msg.add_receiver(AID(self.selection_agent_aid))  # Dynamically routed
                msg.set_content(json.dumps({
                    "animal_id": self.current_animal_id,
                    "total_frames": self.captured_count,
                    "first_capture": self.first_capture,
                    "last_capture": self.last_capture
                }))
                self.agent.send(msg)
            self.passage_signaled = True

        # --- RESTART: cycle finished, advance to next animal ---
        if elapsed >= cycle:
            self.current_animal_id += 1
            if self.current_animal_id > self.herd_size:
                display_message(self.agent.aid.name, f"[FINISH] Completed capturing {self.herd_size} animals.")
                self._finished = True
                return
            self.start_at = time.time()
            self.captured_count = 0
            self.first_capture = None
            self.last_capture = None
            self.passage_signaled = False
            
            display_message(
                self.agent.aid.name,
                f"[RESTART] Animal {self.current_animal_id}/{self.herd_size} entering scale.",
            )
            return

        # --- ARRIVAL mode: animal off-screen, idle ---
        if elapsed > self.passage_time:
            return

        # --- PASSAGE mode: animal is visible, capture and publish ---
                
        frame_id = str(uuid.uuid4())[:12]
        from datetime import datetime
        now_iso = datetime.now().isoformat()
        if self.captured_count == 0:
            self.first_capture = now_iso
        self.last_capture = now_iso
        self.captured_count += 1
        
        img = self.capture_adapter.get_frame()
        
        if img is None:
            display_message(self.agent.aid.name, f"[ERROR] capture_adapter.get_frame() returned None for animal {self.current_animal_id}!")
            return
            
        with self._lock:
            FRAME_BUFFER[frame_id] = img

        msg = ACLMessage(ACLMessage.INFORM)
        msg.set_ontology("frame-capture")
        msg.add_receiver(AID(self.next_agent_aid))
        msg.set_content(json.dumps({
            "frame_id": frame_id,
            "animal_id": self.current_animal_id,
            "frame_index": self.captured_count,
            "elapsed_time": round(elapsed, 4),
        }, ensure_ascii=True))
        
        self.agent.send(msg)


class CaptureAgent(Agent):
    """Capture PADE agent — biological simulator that publishes frame keys."""

    def __init__(
        self,
        aid,
        capture_adapter,
        next_agent_aid: str,
        selection_agent_aid: str,
        interval_seconds: float,
        herd_size: int,
        passage_time: int,
        arrival_time: int,
        wait_for_aids: list[str] = None,
        debug: bool = False,
    ):
        super().__init__(aid=aid, debug=debug)
        self.capture_adapter = capture_adapter
        self.next_agent_aid = next_agent_aid
        self.selection_agent_aid = selection_agent_aid
        self.interval_seconds = interval_seconds
        self.herd_size = herd_size
        self.passage_time = passage_time
        self.arrival_time = arrival_time
        self.wait_for_aids = set(wait_for_aids) if wait_for_aids else set()
        self.ready_agents = set()
        self.simulation_started = False

    def on_start(self):
        super().on_start()
        
        self.capture_behaviour = CaptureBehaviour(
            agent=self,
            capture_adapter=self.capture_adapter,
            next_agent_aid=self.next_agent_aid,
            selection_agent_aid=self.selection_agent_aid,
            passage_time=self.passage_time,
            arrival_time=self.arrival_time,
            herd_size=self.herd_size,
            interval_seconds=self.interval_seconds,
        )
        self.behaviours.append(self.capture_behaviour)
        
        if not self.wait_for_aids:
            self.simulation_started = True
            display_message(self.aid.name, f"CaptureAgent starting simulation — herd_size={self.herd_size}")
        else:
            display_message(self.aid.name, f"CaptureAgent waiting for agents: {self.wait_for_aids}")

    def _start_simulation(self):
        if self.simulation_started:
            return
        self.simulation_started = True
        
        display_message(
            self.aid.name,
            f"CaptureAgent IGNITED. Models loaded! Resuming capture."
        )

    def react(self, message):
        super().react(message)
        if message.ontology == "agent-ready":
            try:
                data = json.loads(message.content)
                agent_name = data.get("agent")
                self.ready_agents.add(agent_name)
                display_message(self.aid.name, f"Agent {agent_name} is READY.")
                
                if self.wait_for_aids.issubset(self.ready_agents):
                    display_message(self.aid.name, "All required agents are ready! Igniting simulation...")
                    self._start_simulation()
            except Exception as e:
                display_message(self.aid.name, f"[ERROR] Processing agent-ready: {e}")
