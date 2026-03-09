"""Pygame UI shell for the RCAF CFAST Trainer.

This task adds cognitive tests under a "Tests" submenu:
- Numerical Operations (mental arithmetic)
- Mathematics Reasoning (multiple-choice word problems)
- Airborne Numerical Test (HHMM timing with map/table UI)
- Vigilance (matrix scanning with routine/priority task switching)
- Cognitive Updating (multitask menu-driven system updates)
- Table Reading (cross-reference lookup tables)
- Sensory Motor Apparatus (joystick/pedal coordination tracking)
- Rapid Tracking (moving/stationary/obscured target tracking)
- Spatial Integration (multi-view 2-D to 3-D reconstruction)
- Auditory Capacity (multichannel auditory + psychomotor load)
- Situational Awareness (mixed verbal/numerical/pictorial updates)
- Trace Test 1 (3-D orientation change discrimination)
- Trace Test 2 (3-D movement memory and recall)

Deterministic timing/scoring/RNG/state lives in cfast_trainer/* (core modules).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import wave
from array import array
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import pygame

from .ant_drills import (
    AntDrillMode,
    build_ant_distance_scan_drill,
    build_ant_endurance_solve_drill,
    build_ant_fuel_burn_solve_drill,
    build_ant_info_grabber_drill,
    build_ant_mixed_tempo_set_drill,
    build_ant_payload_reference_drill,
    build_ant_route_time_solve_drill,
    build_ant_snap_facts_sprint_drill,
    build_ant_time_flip_drill,
)
from .ant_workouts import (
    AntWorkoutSession,
    AntWorkoutSnapshot,
    AntWorkoutStage,
    ant_workout_menu_entries,
    build_ant_workout_plan,
)
from .airborne_numerical import TEMPLATES_BY_NAME, AirborneScenario, build_airborne_numerical_test
from .angles_bearings_degrees import (
    AnglesBearingsDegreesPayload,
    AnglesBearingsQuestionKind,
    build_angles_bearings_degrees_test,
)
from .auditory_capacity import (
    AuditoryCapacityEngine,
    AuditoryCapacityGate,
    AuditoryCapacityPayload,
    build_auditory_capacity_test,
)
from .auditory_capacity_panda3d import (
    AuditoryCapacityPanda3DRenderer,
    panda3d_auditory_rendering_available,
)
from .clock import RealClock
from .cognitive_core import Phase, TestSnapshot
from .cognitive_updating import (
    CognitiveUpdatingPayload,
    CognitiveUpdatingRuntime,
    CognitiveUpdatingRuntimeSnapshot,
    build_cognitive_updating_test,
    decode_cognitive_updating_submission_raw,
)
from .colours_letters_numbers import (
    ColoursLettersNumbersPayload,
    build_colours_letters_numbers_test,
)
from .digit_recognition import DigitRecognitionPayload, build_digit_recognition_test
from .instrument_aircraft_cards import InstrumentAircraftCardSpriteBank
from .instrument_comprehension import (
    InstrumentComprehensionPayload,
    InstrumentComprehensionTrialKind,
    InstrumentState,
    airspeed_turn,
    altimeter_hand_turns,
    build_instrument_comprehension_test,
)
from .math_reasoning import MathReasoningPayload, build_math_reasoning_test
from .numerical_operations import build_numerical_operations_test
from .persistence import ResultsStore, TestSessionSummary
from .rapid_tracking import (
    RapidTrackingEngine,
    RapidTrackingPayload,
    build_rapid_tracking_test,
    rapid_tracking_target_cue,
    rapid_tracking_target_description,
    rapid_tracking_target_label,
)
from .rapid_tracking_panda3d import (
    RapidTrackingPanda3DRenderer,
    panda3d_rapid_tracking_rendering_available,
)
from .results import attempt_result_from_engine
from .sensory_motor_apparatus import (
    SensoryMotorApparatusEngine,
    SensoryMotorApparatusPayload,
    build_sensory_motor_apparatus_test,
)
from .situational_awareness import (
    SituationalAwarenessPayload,
    SituationalAwarenessQuestionKind,
    build_situational_awareness_test,
    cell_label_from_xy,
)
from .spatial_integration import (
    SpatialIntegrationPayload,
    SpatialIntegrationSceneView,
    SpatialIntegrationSection,
    SpatialIntegrationTrialStage,
    build_spatial_integration_test,
)
from .spatial_integration_panda3d import (
    SpatialIntegrationPanda3DRenderer,
    panda3d_spatial_integration_rendering_available,
)
from .system_logic import (
    SystemLogicDocument,
    SystemLogicFolder,
    SystemLogicPayload,
    build_system_logic_test,
)
from .table_reading import TableReadingPayload, TableReadingTable, build_table_reading_test
from .target_recognition import (
    TargetRecognitionPayload,
    TargetRecognitionSceneEntity,
    build_target_recognition_test,
)
from .trace_test_1 import (
    TraceTest1Attitude,
    TraceTest1Payload,
    TraceTest1TrialStage,
    build_trace_test_1_test,
    trace_test_1_scene_frames,
)
from .trace_test_1_panda3d import (
    TraceTest1Panda3DRenderer,
    panda3d_trace_test_1_rendering_available,
)
from .trace_test_2 import (
    TraceTest2AircraftTrack,
    TraceTest2Payload,
    TraceTest2Point3,
    TraceTest2TrialStage,
    build_trace_test_2_test,
    trace_test_2_track_position,
)
from .trace_test_2_panda3d import (
    TraceTest2Panda3DRenderer,
    panda3d_trace_test_2_rendering_available,
)
from .vigilance import VigilancePayload, VigilanceSymbolKind, build_vigilance_test
from .visual_search import VisualSearchPayload, VisualSearchTaskKind, build_visual_search_test

_SETTINGS_UNSET = object()
_AUDITORY_GUIDE_LANES = (-0.16, 0.10, -0.24, 0.18, 0.0, 0.26)
DIFFICULTY_SETTINGS_STORE_ENV = "CFAST_DIFFICULTY_SETTINGS_PATH"
DEFAULT_DIFFICULTY_LEVEL = 5
INTRO_LOADING_MIN_FRAMES = 4
TEST_DIFFICULTY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("numerical_operations", "Numerical Operations"),
    ("math_reasoning", "Mathematics Reasoning"),
    ("airborne_numerical", "Airborne Numerical Test"),
    ("ant_snap_facts_sprint", "Airborne Numerical: Snap Facts Sprint"),
    ("ant_time_flip", "Airborne Numerical: Time Flip"),
    ("ant_mixed_tempo_set", "Airborne Numerical: Mixed Tempo Set"),
    ("ant_route_time_solve", "Airborne Numerical: Route Time Solve"),
    ("ant_endurance_solve", "Airborne Numerical: Endurance Solve"),
    ("ant_fuel_burn_solve", "Airborne Numerical: Fuel Burn Solve"),
    ("ant_distance_scan", "Airborne Numerical: Distance Scan"),
    ("ant_payload_reference", "Airborne Numerical: Payload Reference"),
    ("ant_info_grabber", "Airborne Numerical: Info Grabber"),
    ("airborne_numerical_workout", "Airborne Numerical Workout"),
    ("digit_recognition", "Digit Recognition"),
    ("colours_letters_numbers", "Colours, Letters and Numbers"),
    ("angles_bearings_degrees", "Angles, Bearings and Degrees"),
    ("visual_search", "Visual Search"),
    ("instrument_comprehension", "Instrument Comprehension"),
    ("target_recognition", "Target Recognition"),
    ("system_logic", "System Logic"),
    ("table_reading", "Table Reading"),
    ("sensory_motor_apparatus", "Sensory Motor Apparatus"),
    ("auditory_capacity", "Auditory Capacity"),
    ("cognitive_updating", "Cognitive Updating"),
    ("situational_awareness", "Situational Awareness"),
    ("rapid_tracking", "Rapid Tracking"),
    ("spatial_integration", "Spatial Integration"),
    ("trace_test_1", "Trace Test 1"),
    ("trace_test_2", "Trace Test 2"),
    ("vigilance", "Vigilance"),
)


@dataclass(frozen=True)
class TestGuideBriefing:
    label: str
    assessment: str
    tasks: tuple[str, ...]
    timing: str
    prep: str
    controls: str
    app_flow: str


TEST_GUIDE_BRIEFS: dict[str, TestGuideBriefing] = {
    "numerical_operations": TestGuideBriefing(
        label="Numerical Operations",
        assessment="Reasoning test for rapid mental arithmetic under time pressure.",
        tasks=(
            "Solve addition, subtraction, multiplication, and division mentally.",
            "Work quickly and accurately without stopping on one item for too long.",
        ),
        timing="Guide time including instructions: about 2 minutes.",
        prep="Guide preparation: practice mental arithmetic.",
        controls="Type the answer digits and press Enter to submit.",
        app_flow="This trainer gives you a short practice block first, then a timed block.",
    ),
    "math_reasoning": TestGuideBriefing(
        label="Mathematics Reasoning",
        assessment="Reasoning test for solving written numerical problems.",
        tasks=(
            "Interpret short numerical word problems.",
            "Use time, speed, and distance reasoning to choose the best answer.",
        ),
        timing="Guide time including instructions: about 18 minutes.",
        prep="Guide preparation: practice time, speed, and distance calculations.",
        controls="Use A/S/D/F/G to choose, or 1-5 then Enter when needed.",
        app_flow="Practice confirms the response format before the timed block starts.",
    ),
    "airborne_numerical": TestGuideBriefing(
        label="Airborne Numerical",
        assessment="Reasoning test for airborne-style estimation under time pressure.",
        tasks=(
            "Estimate time, speed, distance, and fuel-consumption problems mentally.",
            "Use quick arithmetic without paper or a calculator.",
        ),
        timing="Guide time including instructions: about 35 minutes.",
        prep="Guide preparation: mental arithmetic plus time, speed, distance, and fuel work.",
        controls="Type the 4-digit answer. S, D, F, and A open the reference aids in this trainer.",
        app_flow="Use practice to learn the overlays and answer format before the timed block.",
    ),
    "ant_snap_facts_sprint": TestGuideBriefing(
        label="Airborne Numerical: Snap Facts Sprint",
        assessment="Timed Airborne Numerical drill for fast arithmetic retrieval under hard per-item caps.",
        tasks=(
            "Answer single-step arithmetic facts quickly without freezing on one item.",
            "Treat every timeout as a fixation miss and reset immediately on the next prompt.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: practice direct retrieval, not full re-computation.",
        controls="Type the answer digits and press Enter. Each item auto-advances when its cap expires.",
        app_flow="The mode sets cap pressure and feedback timing. Practice teaches the pace before the timed block.",
    ),
    "ant_time_flip": TestGuideBriefing(
        label="Airborne Numerical: Time Flip",
        assessment="Timed Airborne Numerical drill for fast time and rate-unit conversion under hard per-item caps.",
        tasks=(
            "Convert clean HHMM and minute values without freezing on the arithmetic.",
            "Flip between per-hour and per-minute rates quickly enough to preserve tempo.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: practice minute-hour conversion and clean 60-based rate changes.",
        controls="Type the answer digits and press Enter. Each item auto-advances when its cap expires.",
        app_flow="Practice teaches the conversion style first. The timed block then adapts around your recent accuracy and fixation.",
    ),
    "ant_mixed_tempo_set": TestGuideBriefing(
        label="Airborne Numerical: Mixed Tempo Set",
        assessment="Timed mixed Airborne Numerical drill that rotates retrieval, time/rate, route-time, endurance, fuel-burn, distance, and payload families under hard per-item caps.",
        tasks=(
            "Switch between arithmetic retrieval, time conversion, and airborne scenario prompts without warning.",
            "Keep moving when the family changes instead of spending extra time re-orienting.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: use the single-family drills first if family switches are still causing freezes.",
        controls="Type the answer digits and press Enter. Airborne items keep the same A/D/F overlays, and caps still auto-advance by family.",
        app_flow="This is the reusable mixed Airborne Numerical kernel for integration work. It carries the live caps, overlays, and scoring rules for every implemented Airborne Numerical family.",
    ),
    "ant_route_time_solve": TestGuideBriefing(
        label="Airborne Numerical: Route Time Solve",
        assessment="Timed Airborne Numerical drill for route-time and arrival/takeoff calculation using the live airborne scenario framework.",
        tasks=(
            "Read the airborne scenario, use route distances and parcel-speed reference, and answer with 4-digit HHMM.",
            "Keep moving when parcel speed comes from a chart instead of a clean table value.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: be comfortable with HHMM arithmetic and distance-speed-time transforms.",
        controls="Type 4 digits and press Enter. Hold A for distances and F for speed and parcel reference, matching the Airborne Numerical screen.",
        app_flow="This drill reuses the airborne scenario UI and scoring conventions from the Airborne Numerical Test.",
    ),
    "ant_endurance_solve": TestGuideBriefing(
        label="Airborne Numerical: Endurance Solve",
        assessment="Timed Airborne Numerical drill for empty-time and fuel-endurance solving using the live airborne scenario framework.",
        tasks=(
            "Convert start fuel and burn rate into endurance or empty time without losing the 4-digit Airborne Numerical format.",
            "Handle table-exact and chart-estimate fuel references using the same live Airborne Numerical tolerance rules.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: know the fuel-time relationship and be comfortable with HHMM output.",
        controls="Type 4 digits and press Enter. Hold D for speed and fuel reference, matching the Airborne Numerical screen.",
        app_flow="This drill reuses the airborne scenario UI and scoring conventions from the Airborne Numerical Test.",
    ),
    "ant_fuel_burn_solve": TestGuideBriefing(
        label="Airborne Numerical: Fuel Burn Solve",
        assessment="Timed Airborne Numerical drill for fuel-burn calculation using the live airborne scenario framework.",
        tasks=(
            "Use route time plus fuel-burn reference to compute fuel used under the 4-digit Airborne Numerical answer format.",
            "Handle table-exact and chart-estimate items using the same tolerance logic as the live Airborne Numerical test.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: know the route-time formula and basic fuel-burn transforms.",
        controls="Type 4 digits and press Enter. Hold A for distances, D for speed and fuel reference, and F for parcel speed reference.",
        app_flow="This drill reuses the airborne scenario UI and scoring conventions from the Airborne Numerical Test.",
    ),
    "ant_distance_scan": TestGuideBriefing(
        label="Airborne Numerical: Distance Scan",
        assessment="Timed Airborne Numerical drill for route scanning and distance summing using the live airborne scenario framework.",
        tasks=(
            "Trace the active route quickly, extract the leg distances, and total them once.",
            "Stay decisive when the route gets longer or the unit profile changes.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: practice one-pass route tracing and simple running totals.",
        controls="Type 4 digits and press Enter. Hold A for distances, matching the Airborne Numerical screen.",
        app_flow="This drill reuses the airborne scenario UI and scoring conventions from the Airborne Numerical Test.",
    ),
    "ant_payload_reference": TestGuideBriefing(
        label="Airborne Numerical: Payload Reference",
        assessment="Timed Airborne Numerical drill for parcel-weight and parcel-effect reference work using the live airborne scenario framework.",
        tasks=(
            "Read the parcel-speed reference cleanly and decide when the answer is exact versus estimated.",
            "Handle both parcel weight and parcel effect prompts without freezing on the reference page.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: be comfortable moving between the summary table and the parcel reference page.",
        controls="Type 4 digits and press Enter. Hold F for speed and parcel reference, matching the Airborne Numerical screen.",
        app_flow="This drill reuses the airborne scenario UI and scoring conventions from the Airborne Numerical Test.",
    ),
    "ant_info_grabber": TestGuideBriefing(
        label="Airborne Numerical: Info Grabber",
        assessment="Timed Airborne Numerical drill for rapid information extraction and retention using the live airborne scenario framework.",
        tasks=(
            "Grab one exact value from the airborne display or reference aids and keep it live through a short delay or interference step.",
            "Answer with exact typed digits; partial digit-position credit still contributes to the score ratio.",
        ),
        timing="Guide time depends on mode: Build 3 minutes, Tempo 2.5 minutes, Stress 3 minutes.",
        prep="Guide preparation: know the Airborne Numerical overlays first; this drill is about finding and holding the right value fast.",
        controls="Type 4 digits and press Enter. Use A, D, and F exactly as you would in Airborne Numerical while retaining the original target.",
        app_flow="This is the reusable Airborne Numerical lookup-and-retain drill kernel. Version one stays typed-response only and uses exact-hit accuracy plus partial digit credit.",
    ),
    "airborne_numerical_workout": TestGuideBriefing(
        label="Airborne Numerical Workout",
        assessment="Chained Airborne Numerical workout with typed reflection, warm-up blocks, tempo calculations, and full-question scenario sets.",
        tasks=(
            "Start with typed focus prompts, then warm up search, retention, arithmetic, unit conversion, and distance scanning under low pressure.",
            "Build into tempo calculation blocks, then finish with grouped full-question Airborne Numerical scenario sets under steady and pressure conditions.",
        ),
        timing="Workout drill time: 90 minutes, plus opening and closing reflection outside the timed blocks.",
        prep="Guide preparation: know the drill controls first; the workout is for chaining them under one structure.",
        controls="Use Left and Right to set workout or block difficulty, type reflections, and use digits plus Enter during blocks.",
        app_flow="Each block gets an untimed setup screen, and changing difficulty from workout settings restarts the workout from the beginning.",
    ),
    "digit_recognition": TestGuideBriefing(
        label="Digit Recognition",
        assessment="Short-term visual memory test.",
        tasks=(
            "Remember strings of digits of varying lengths.",
            "Answer questions about the digits you were shown.",
        ),
        timing="Guide time including instructions: about 4 minutes.",
        prep="Guide preparation: none required.",
        controls="Watch the display, then type the answer digits and press Enter.",
        app_flow="Practice shows the display and recall rhythm before the timed block.",
    ),
    "colours_letters_numbers": TestGuideBriefing(
        label="Colours, Letters and Numbers",
        assessment="Multitask test for shifting attention between different tasks.",
        tasks=(
            "Remember letter sequences while blank gaps interrupt the display.",
            "Solve simple mental arithmetic and react to colour-lane cues at the same time.",
        ),
        timing="Guide time including instructions: about 20 minutes.",
        prep="Guide preparation: practice mental arithmetic.",
        controls="Memory uses A/S/D/F/G or mouse, colours use Q/W/E/R, and math uses digits plus Enter.",
        app_flow="Practice lets you feel the three concurrent tasks before the timed block.",
    ),
    "angles_bearings_degrees": TestGuideBriefing(
        label="Angles, Bearings and Degrees",
        assessment="Spatial test for judging angles and bearings.",
        tasks=(
            "Estimate the angle between two lines.",
            "Estimate the bearing of an object from a reference point.",
        ),
        timing="Guide time including instructions: about 10 minutes.",
        prep="Guide preparation: get comfortable estimating common angles.",
        controls="A/S/D/F/G selects, Up/Down moves the highlight, and Enter submits.",
        app_flow="Practice introduces both angle and bearing item styles before timed work.",
    ),
    "visual_search": TestGuideBriefing(
        label="Visual Search",
        assessment="Scanning test for finding targets under time pressure.",
        tasks=(
            "Search for a target among letters or line-figure distractors.",
            "Work quickly and accurately while scanning the whole display.",
        ),
        timing="Guide time including instructions: about 4 minutes.",
        prep="Guide preparation: none required.",
        controls="Scan the board, type the matching block number, and press Enter.",
        app_flow="Practice lets you learn the board layout before the timed block unlocks.",
    ),
    "instrument_comprehension": TestGuideBriefing(
        label="Instrument Comprehension",
        assessment="Spatial visualization test using pictorial, numerical, and verbal information.",
        tasks=(
            "Inspect the instrument readings to visualize aircraft orientation.",
            "Match the instruments to the correct aircraft view or flight description.",
        ),
        timing="Guide time including instructions: about 26 minutes.",
        prep="Guide preparation: none required.",
        controls="Choose with A/S/D/F/G, then press Enter to submit.",
        app_flow="Practice introduces both parts before the scored block begins.",
    ),
    "target_recognition": TestGuideBriefing(
        label="Target Recognition",
        assessment="Multitask visual-search test for identifying several target types.",
        tasks=(
            "Scan for imagery, colour, symbol, code, and warning-sign targets.",
            "Prioritize the different streams and register as many valid targets as possible.",
        ),
        timing="Guide time including instructions: about 25 minutes.",
        prep="Guide preparation: none required.",
        controls="The live tasks are mostly mouse-driven in this trainer; click the matching targets and panels.",
        app_flow="Practice builds each category first, then the timed block combines them.",
    ),
    "system_logic": TestGuideBriefing(
        label="System Logic",
        assessment="Reasoning test for solving logic problems from multiple sources.",
        tasks=(
            "Search different folders and pages to find the relevant facts.",
            "Combine tables, graphs, diagrams, equations, and written statements to solve the problem.",
        ),
        timing="Guide time including instructions: about 38 minutes.",
        prep="Guide preparation: mental arithmetic helps.",
        controls="Left/Right changes folders, Up/Down moves, and digits plus Enter submit answers.",
        app_flow="Practice teaches the folder workflow so the timed block can stay fast.",
    ),
    "table_reading": TestGuideBriefing(
        label="Table Reading",
        assessment="Work-rate test for scanning and cross-referencing tables quickly.",
        tasks=(
            "Cross-reference row and column numbers to find values.",
            "Use multiple tables and reference cards to identify the correct answer.",
        ),
        timing="Guide time including instructions: about 11 minutes.",
        prep="Guide preparation: none required.",
        controls="Use the on-screen answer choices shown for the current table-reading task.",
        app_flow="Practice shows the lookup style before the timed block starts.",
    ),
    "sensory_motor_apparatus": TestGuideBriefing(
        label="Sensory Motor Apparatus",
        assessment="Eye-hand-foot coordination test.",
        tasks=(
            "Control horizontal and vertical motion together.",
            "Keep the moving dot as close to the center crosshair as possible.",
        ),
        timing="Guide time including instructions: about 9 minutes.",
        prep="Guide preparation: none required.",
        controls="Rudder or left-right input controls horizontal motion; joystick axis 1 or up-down controls vertical motion.",
        app_flow="Practice is for control familiarization before the timed tracking block.",
    ),
    "auditory_capacity": TestGuideBriefing(
        label="Auditory Capacity",
        assessment="Short-term memory test under multiple tasks and audio instructions.",
        tasks=(
            "Control the ball, react to colour and sound cues, and remember digit strings.",
            "Follow instructions presented over headphones while several tasks run together.",
        ),
        timing="Guide time including instructions: about 23 minutes.",
        prep="Guide preparation: none required.",
        controls="WASD/arrows or HOTAS fly the ball; Q/W/E/R handle colours; keypad 0-9 sets the ball number; digits plus Enter handle delayed recall.",
        app_flow="Practice teaches the audio-control rhythm before the timed block loads.",
    ),
    "cognitive_updating": TestGuideBriefing(
        label="Cognitive Updating",
        assessment="Multitask coordination test for a busy technical environment.",
        tasks=(
            "Monitor, control, set up, and adjust several systems against the clock.",
            "Use multifunction displays, pages, and menus while keeping information updated.",
        ),
        timing="Guide time including instructions: about 35 minutes.",
        prep="Guide preparation: none required.",
        controls="Use the page tabs, clickable controls, toggles, and keypad-style entries shown in the display.",
        app_flow="Practice is there to learn the page flow before the scored block begins.",
    ),
    "situational_awareness": TestGuideBriefing(
        label="Situational Awareness",
        assessment="Multitask test for building and updating a mental picture of a changing situation.",
        tasks=(
            "Combine verbal, numerical, pictorial, and coded information.",
            "Answer questions about current, past, and future movement or the best action to take.",
        ),
        timing="Guide time including instructions: about 30 minutes.",
        prep="Guide preparation: none required.",
        controls="Click the highlighted grid cells or options; A/S/D/F/G plus Enter also works.",
        app_flow="Practice lets you learn the map and query style before the timed block.",
    ),
    "rapid_tracking": TestGuideBriefing(
        label="Rapid Tracking",
        assessment="Eye-hand coordination test for tracking and targeting.",
        tasks=(
            "Track moving or stationary objects from a viewpoint that is moving continuously.",
            "Predict the movement of obscured or handed-off targets.",
        ),
        timing="Guide time including instructions: about 16 minutes.",
        prep="Guide preparation: none required.",
        controls="Rudder or left-right input steers horizontally, joystick axis 1 or up-down steers vertically, and trigger, Space, or left-click captures.",
        app_flow="Practice is for camera feel and capture timing before the timed run.",
    ),
    "spatial_integration": TestGuideBriefing(
        label="Spatial Integration",
        assessment="Spatial test for building a 3-D air-ground picture from 2-D views.",
        tasks=(
            "Interpret top, oblique, horizontal, and vertical viewpoints together.",
            "Track landmarks and aircraft positions across the three sections.",
        ),
        timing="Guide time including instructions: about 28 minutes.",
        prep="Guide preparation: none required.",
        controls="Use A/S/D/F/G to choose the answer shown on screen for each section.",
        app_flow="Each section gets practice before scored items, so use the practice blocks to learn the viewpoint changes.",
    ),
    "trace_test_1": TestGuideBriefing(
        label="Trace Test 1",
        assessment="Spatial test for orientation in three-dimensional space.",
        tasks=(
            "Perceive the changing orientation of a moving aircraft from another perspective.",
            "Identify the correct change in attitude from the answer options.",
        ),
        timing="Guide time including instructions: about 9 minutes.",
        prep="Guide preparation: none required.",
        controls="Use A/S/D/F to choose, Up/Down to move, and Enter to submit.",
        app_flow="Practice teaches the viewpoint logic before the timed block starts.",
    ),
    "trace_test_2": TestGuideBriefing(
        label="Trace Test 2",
        assessment="Memory test for movement in three-dimensional space.",
        tasks=(
            "Watch short dynamic scenarios with several aircraft.",
            "Recall movement, relative position, and sequence after the scene ends.",
        ),
        timing="Guide time including instructions: about 9 minutes.",
        prep="Guide preparation: none required.",
        controls="Watch the scene first, then answer with A/S/D/F and Enter.",
        app_flow="Practice lets you learn the scene-then-question rhythm before the timed block.",
    ),
    "vigilance": TestGuideBriefing(
        label="Vigilance",
        assessment="Scanning test for switching between routine and priority tasks.",
        tasks=(
            "Scan the matrix accurately for the active target position.",
            "Switch between routine and priority captures while keeping speed and accuracy up.",
        ),
        timing="Guide time including instructions: about 8 minutes.",
        prep="Guide preparation: none required.",
        controls="Click or focus the Row and Col fields, then type digits 1-9 to capture the target coordinates.",
        app_flow="Practice teaches the matrix entry rhythm before the timed block begins.",
    ),
}


def _auditory_guide_lane(index: int) -> float:
    idx = int(index) % len(_AUDITORY_GUIDE_LANES)
    return float(_AUDITORY_GUIDE_LANES[idx])


class Screen(Protocol):
    def handle_event(self, event: pygame.event.Event) -> None: ...
    def render(self, surface: pygame.Surface) -> None: ...


class CognitiveEngine(Protocol):
    def snapshot(self) -> TestSnapshot: ...
    def can_exit(self) -> bool: ...
    def start_practice(self) -> None: ...
    def start_scored(self) -> None: ...
    def submit_answer(self, raw: str) -> bool: ...
    def update(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MenuItem:
    label: str
    action: Callable[[], None]


@dataclass(slots=True)
class _TargetRecognitionSceneGlyph:
    glyph_id: int
    kind: str  # "entity" | "beacon" | "unknown"
    entity: TargetRecognitionSceneEntity | None
    nx: float
    ny: float
    scale: float
    heading: float
    alpha: float
    max_alpha: float
    matching_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AmbientTrack:
    path: Path | None
    base_volume: float
    base_weight: float
    late_weight: float
    sample_rate: int
    channels: int
    frame_count: int
    is_generated_noise: bool = False


class _OfflineTtsSpeaker:
    """Best-effort offline TTS via isolated subprocesses.

    The old in-process pyttsx3 path could hard-crash the app. This wrapper
    keeps TTS out-of-process while preserving spoken commands/chatter behavior.
    """

    _max_queue = 24
    _max_utterance_s = 12.0

    def __init__(
        self,
        *,
        rate_wpm: int = 150,
        volume: float = 0.95,
        backend_preference: tuple[str, ...] | None = None,
    ) -> None:
        self._enabled = False
        self._backends: list[str] = []
        self._backend: str | None = None
        self._pending: list[str] = []
        self._active_proc: subprocess.Popen[bytes] | None = None
        self._active_started_s = 0.0
        self._rate_wpm = max(90, min(220, int(rate_wpm)))
        self._volume = max(0.10, min(1.00, float(volume)))
        self._backend_preference = backend_preference

        if os.environ.get("CFAST_DISABLE_TTS", "0") == "1":
            return
        if os.environ.get("SDL_AUDIODRIVER", "").strip().lower() == "dummy":
            # Keep automated/headless runs silent and stable.
            return

        self._backends = self._resolve_backends()
        self._backend = self._backends[0] if self._backends else None
        self._enabled = self._backend is not None

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    @property
    def pending_count(self) -> int:
        active = 1 if self._active_proc is not None else 0
        return active + len(self._pending)

    def speak(self, text: str) -> None:
        if not self._enabled:
            return
        phrase = " ".join(str(text).strip().split())
        if phrase == "":
            return
        if self._pending and self._pending[-1] == phrase:
            return
        self._pending.append(phrase)
        if len(self._pending) > self._max_queue:
            del self._pending[: len(self._pending) - self._max_queue]

    def update(self) -> None:
        if not self._enabled:
            return

        proc = self._active_proc
        if proc is not None:
            if proc.poll() is None:
                if (time.monotonic() - self._active_started_s) > self._max_utterance_s:
                    self._terminate_process(proc)
                    self._active_proc = None
            else:
                self._active_proc = None

        if self._active_proc is not None:
            return
        if not self._pending:
            return

        while self._pending and self._enabled:
            next_text = self._pending[0]
            launched = self._launch_process(next_text)
            if launched is not None:
                del self._pending[0]
                self._active_proc = launched
                self._active_started_s = time.monotonic()
                return
            self._drop_current_backend()

        if not self._enabled:
            self._pending.clear()

    def stop(self) -> None:
        self._pending.clear()
        proc = self._active_proc
        self._active_proc = None
        if proc is not None:
            self._terminate_process(proc)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
        try:
            proc.terminate()
        except Exception:
            return
        try:
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _resolve_backends(self) -> list[str]:
        supported = ("pyttsx3-subprocess", "say", "powershell", "espeak")
        forced = os.environ.get("CFAST_TTS_BACKEND", "").strip().lower()
        if forced in supported and self._backend_available(forced):
            return [forced]

        candidates: list[str]
        if self._backend_preference is not None and len(self._backend_preference) > 0:
            candidates = [name for name in self._backend_preference if name in supported]
        else:
            candidates = []
            if sys.platform == "darwin":
                candidates.append("say")
            if os.name == "nt":
                candidates.append("powershell")
            candidates.extend(("pyttsx3-subprocess", "espeak"))

        seen: set[str] = set()
        resolved: list[str] = []
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            if self._backend_available(name):
                resolved.append(name)
        return resolved

    @staticmethod
    def _backend_available(name: str) -> bool:
        if name == "say":
            return (shutil.which("say") is not None) or Path("/usr/bin/say").exists()
        if name == "powershell":
            return (shutil.which("powershell") is not None) or (shutil.which("pwsh") is not None)
        if name == "pyttsx3-subprocess":
            return importlib.util.find_spec("pyttsx3") is not None
        if name == "espeak":
            return shutil.which("espeak") is not None
        return False

    def _drop_current_backend(self) -> None:
        backend = self._backend
        if backend is None:
            self._enabled = False
            return
        self._backends = [name for name in self._backends if name != backend]
        self._backend = self._backends[0] if self._backends else None
        self._enabled = self._backend is not None

    def _launch_process(self, text: str) -> subprocess.Popen[bytes] | None:
        backend = self._backend
        if backend is None:
            return None

        try:
            if backend == "pyttsx3-subprocess":
                script = (
                    "import sys\n"
                    "txt=' '.join(sys.argv[1:]).strip()\n"
                    "import pyttsx3\n"
                    "e=pyttsx3.init()\n"
                    f"e.setProperty('rate', {self._rate_wpm})\n"
                    f"e.setProperty('volume', {self._volume:.3f})\n"
                    "e.say(txt)\n"
                    "e.runAndWait()\n"
                )
                return subprocess.Popen(
                    [sys.executable, "-c", script, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if backend == "say":
                return subprocess.Popen(
                    [shutil.which("say") or "/usr/bin/say", "-r", str(self._rate_wpm), text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if backend == "powershell":
                ps_bin = shutil.which("powershell") or shutil.which("pwsh")
                if ps_bin is None:
                    return None
                script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Rate=1; "
                    f"$s.Volume={int(max(0, min(100, round(self._volume * 100))))}; "
                    "$txt=($args -join ' '); "
                    "$s.Speak($txt);"
                )
                return subprocess.Popen(
                    [ps_bin, "-NoProfile", "-NonInteractive", "-Command", script, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if backend == "espeak":
                return subprocess.Popen(
                    [
                        "espeak",
                        "-s",
                        str(self._rate_wpm),
                        "-a",
                        str(int(max(10, min(200, round(self._volume * 100))))),
                        text,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            return None
        return None


class _AuditoryCapacityAudioAdapter:
    """Pygame audio adapter for Auditory Capacity cues/background.

    This stays outside deterministic core logic. It reacts to payload state
    transitions and emits local sounds only.
    """

    _sample_rate = 22050
    _amp = 32767
    _ambient_asset_manifest: tuple[tuple[str, float, float, float], ...] = (
        ("bg_mature_distraction_loop.wav", 0.62, 5.40, 1.40),
        ("bg_noise_loop.wav", 0.56, 1.55, 0.10),
        ("bg_restaurant_loop.wav", 0.52, 1.20, 0.20),
        ("bg_book_reader_loop.wav", 0.50, 1.10, 0.30),
        ("bg_conversation_close_loop.wav", 0.52, 1.15, 0.35),
    )

    @classmethod
    def available_noise_sources(cls) -> tuple[str, ...]:
        return tuple(filename for filename, *_rest in cls._ambient_asset_manifest)

    @staticmethod
    def format_noise_source_label(key: str | None) -> str:
        if key is None or str(key).strip() == "":
            return "Auto"
        label = str(key).strip()
        if label.endswith(".wav"):
            label = label[:-4]
        if label.startswith("bg_"):
            label = label[3:]
        if label.endswith("_loop"):
            label = label[:-5]
        return label.replace("_", " ").title()

    def __init__(self) -> None:
        self._available = False
        self._callsign_cache: dict[str, pygame.mixer.Sound] = {}
        self._sequence_cache: dict[str, pygame.mixer.Sound] = {}

        self._noise_sound: pygame.mixer.Sound | None = None
        self._mixer_channels = 1
        self._ambient_tracks: dict[str, _AmbientTrack] = {}
        self._ambient_slots: list[str | None] = [None, None]
        self._ambient_active_sounds: list[pygame.mixer.Sound | None] = [None, None]
        self._ambient_recent_keys: list[str] = []
        self._ambient_target_layers = 0
        self._ambient_started_at_s = 0.0
        self._next_ambient_change_at_s = 0.0
        self._last_mature_clip_at_s = -10_000.0
        self._ambient_assets_dir = (
            Path(__file__).resolve().parents[1] / "assets" / "audio" / "auditory_capacity"
        )
        self._beep_sound: pygame.mixer.Sound | None = None
        self._color_sounds: dict[str, pygame.mixer.Sound] = {}

        self._bg_channel: pygame.mixer.Channel | None = None
        self._dist_channel: pygame.mixer.Channel | None = None
        self._cue_channel: pygame.mixer.Channel | None = None
        self._alert_channel: pygame.mixer.Channel | None = None
        self._fx_channel: pygame.mixer.Channel | None = None

        self._last_callsign_cue: str | None = None
        self._last_color_command: str | None = None
        self._last_beep_active = False
        self._last_sequence_display: str | None = None
        self._last_instruction_uid: int | None = None
        self._last_assigned_callsigns: tuple[str, ...] = ()
        self._active_noise_source: str | None = None
        self._tts_callsign = _OfflineTtsSpeaker(
            rate_wpm=168,
            volume=0.22,
            backend_preference=("pyttsx3-subprocess", "say", "espeak", "powershell"),
        )
        self._tts_commands = _OfflineTtsSpeaker(rate_wpm=172, volume=0.30)
        self._tts_story = _OfflineTtsSpeaker(
            rate_wpm=160,
            volume=0.22,
            backend_preference=("pyttsx3-subprocess", "say", "espeak", "powershell"),
        )
        self._voice_rng = random.Random(0xAC710)
        self._next_chatter_at_s = 0.0
        self._story_index = -1
        self._story_segments: tuple[str, ...] = (
            "Chapter one. The rain started before sunrise, and the streetlights stayed on.",
            "At the station, two conductors argued over a missing timetable "
            "and a locked office door.",
            "The clerk counted envelopes twice, then wrote a note to no one in particular.",
            "By noon the market was full, plates clattered, and voices bounced off the ceiling.",
            "A child asked for directions while someone nearby read a page from an old novel.",
            "He crossed the hall, paused at the stairs, "
            "and listened for footsteps that never came.",
            "At table seven, two strangers compared watch times and disagreed by three minutes.",
            "The server dropped a spoon; chairs scraped; a phone rang and went unanswered.",
            "A narrator read steadily, describing a ship that left port under low clouds.",
            "In the next paragraph, the captain checked a compass, then folded the map.",
            "Back at the terminal, an announcement repeated with static "
            "in the middle of each sentence.",
            "Someone laughed too loudly, then apologized, then started the same story again.",
            "The reader turned another page and continued without looking up from the text.",
            "Outside, traffic paused at the light and moved again before the rain eased.",
            "A mechanic tightened one bolt, loosened another, and wrote the change in a notebook.",
            "Two voices nearby switched topics from weather to schedules to unfinished errands.",
            "The chapter ended with a door closing and the hallway lights dimming.",
            "Then the next chapter began immediately, as if no one had left the room.",
            "He set the cup beside the radio and turned the dial until the static thinned out.",
            "The bulletin listed delays, route changes, and one cancellation with no explanation.",
            "A courier arrived late, signed the logbook, and left before anyone asked a question.",
            "Near the window, someone traced circles on the glass "
            "while counting under their breath.",
            "Three pages were missing from the binder, but the index still showed their numbers.",
            "On platform four, a whistle blew twice and then stopped halfway through a third note.",
            "The supervisor checked the wall clock, then checked a wristwatch, then frowned.",
            "A cart rolled by with empty crates that rattled louder than expected in the corridor.",
            "At the far desk, a light blinked in a pattern no one bothered to explain.",
            "The intercom clicked on, repeated a sentence, and cut off before the last word.",
            "By evening, the rain had slowed, "
            "but the sidewalks still reflected every passing sign.",
            "A driver unfolded a map on the hood and marked two streets with a blue pen.",
            "Inside the waiting room, a heater hummed "
            "while a newspaper page turned every few seconds.",
            "Someone asked for gate numbers twice, then nodded as if the answer was different.",
            "The recorder kept running after the interview ended, capturing chairs and footsteps.",
            "A technician replaced a battery, tightened a screw, and wrote the time on tape.",
            "In the warehouse, a forklift reversed, paused, and reversed again to clear a pallet.",
            "A printed checklist hung by the door with four boxes still empty at shift change.",
            "The final paragraph mentioned a key, a locker, and a code written on scrap paper.",
            "No one agreed on the exact minute the signal started, only that everyone heard it.",
            "The morning report arrived folded in half with coffee rings across the margin.",
            "A thin voice from the next room read names from a list and waited for replies.",
            "Two mechanics compared torque values and then checked the same wrench twice.",
            "By the loading bay, wind pushed a loose sign until it tapped against the wall.",
            "A student repeated the instructions quietly, "
            "then repeated them again without looking up.",
            "At the end of the chapter, the narrator said to continue and turned the page.",
        )

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=self._sample_rate, size=-16, channels=1, buffer=512)
            mix_state = pygame.mixer.get_init()
            if mix_state is not None:
                self._sample_rate = int(mix_state[0])
                self._mixer_channels = max(1, int(mix_state[2]))
            pygame.mixer.set_num_channels(max(8, int(pygame.mixer.get_num_channels())))

            self._noise_sound = self._build_noise_sound(
                duration_s=1.30,
                seed=0xAC01,
                gain=0.35,
            )
            self._ambient_tracks = self._load_ambient_tracks()
            self._beep_sound = self._build_tone_sound(1120.0, 0.11, gain=0.42)
            self._color_sounds = {
                "RED": self._build_tone_sound(380.0, 0.14, gain=0.30),
                "GREEN": self._build_tone_sound(510.0, 0.14, gain=0.30),
                "BLUE": self._build_tone_sound(640.0, 0.14, gain=0.30),
                "YELLOW": self._build_tone_sound(780.0, 0.14, gain=0.30),
            }

            self._bg_channel = pygame.mixer.Channel(0)
            self._dist_channel = pygame.mixer.Channel(1)
            self._cue_channel = pygame.mixer.Channel(2)
            self._alert_channel = pygame.mixer.Channel(3)
            self._fx_channel = pygame.mixer.Channel(4)

            self._available = True
        except Exception:
            self._available = False

    def sync(self, *, phase: Phase, payload: AuditoryCapacityPayload | None) -> None:
        if payload is None or phase not in (Phase.PRACTICE, Phase.SCORED):
            self.stop()
            return

        self._sync_assigned_callsigns(payload=payload)
        if self._available:
            self._sync_background_layers(payload=payload)
            self._sync_distortion_layer(payload=payload)
        self._sync_callsign_cue(payload=payload)
        self._sync_beep_cue(payload=payload)
        self._sync_color_cue(payload=payload)
        self._sync_sequence_cue(payload=payload)
        self._sync_instruction_cue(payload=payload)
        self._sync_voice_chatter(payload=payload)
        self._tts_callsign.update()
        self._tts_commands.update()
        self._tts_story.update()

    def stop(self) -> None:
        if not self._available:
            self._tts_callsign.stop()
            self._tts_commands.stop()
            self._tts_story.stop()
        else:
            channels = (
                self._bg_channel,
                self._dist_channel,
                self._cue_channel,
                self._alert_channel,
                self._fx_channel,
            )
            for channel in channels:
                if channel is not None:
                    try:
                        channel.stop()
                    except Exception:
                        pass
            self._tts_callsign.stop()
            self._tts_commands.stop()
            self._tts_story.stop()
        self._last_callsign_cue = None
        self._last_color_command = None
        self._last_beep_active = False
        self._last_sequence_display = None
        self._last_instruction_uid = None
        self._last_assigned_callsigns = ()
        self._next_chatter_at_s = 0.0
        self._story_index = -1
        self._ambient_slots = [None, None]
        self._ambient_active_sounds = [None, None]
        self._ambient_recent_keys = []
        self._ambient_target_layers = 0
        self._ambient_started_at_s = 0.0
        self._next_ambient_change_at_s = 0.0
        self._last_mature_clip_at_s = -10_000.0
        self._active_noise_source = None

    def debug_state(self, *, payload: AuditoryCapacityPayload | None = None) -> dict[str, object]:
        def _slot_label(key: str | None) -> str:
            if key is None or key.strip() == "":
                return "—"
            label = key
            if label.endswith(".wav"):
                label = label[:-4]
            if label.startswith("bg_"):
                label = label[3:]
            if label.endswith("_loop"):
                label = label[:-5]
            return label

        bg_busy = False
        dist_busy = False
        cue_busy = False
        alert_busy = False
        fx_busy = False
        try:
            bg_busy = bool(self._bg_channel.get_busy()) if self._bg_channel is not None else False
            dist_busy = (
                bool(self._dist_channel.get_busy()) if self._dist_channel is not None else False
            )
            cue_busy = (
                bool(self._cue_channel.get_busy()) if self._cue_channel is not None else False
            )
            alert_busy = (
                bool(self._alert_channel.get_busy()) if self._alert_channel is not None else False
            )
            fx_busy = bool(self._fx_channel.get_busy()) if self._fx_channel is not None else False
        except Exception:
            bg_busy = False
            dist_busy = False
            cue_busy = False
            alert_busy = False
            fx_busy = False

        ambient_slots = tuple(_slot_label(v) for v in self._ambient_slots)
        noise_level = None if payload is None else float(payload.background_noise_level)
        distortion_level = None if payload is None else float(payload.distortion_level)
        noise_source = None if payload is None else payload.background_noise_source

        return {
            "audio_available": bool(self._available),
            "ambient_layers_target": int(self._ambient_target_layers),
            "ambient_slots": ambient_slots,
            "noise_source": self.format_noise_source_label(
                noise_source if noise_source is not None else self._active_noise_source
            ),
            "ambient_recent": tuple(_slot_label(v) for v in self._ambient_recent_keys[-3:]),
            "channels_busy": {
                "bg": bg_busy,
                "dist": dist_busy,
                "cue": cue_busy,
                "alert": alert_busy,
                "fx": fx_busy,
            },
            "cue_state": {
                "callsign": self._last_callsign_cue,
                "color": self._last_color_command,
                "beep_active": bool(self._last_beep_active),
                "sequence": self._last_sequence_display,
            },
            "tts_pending": {
                "callsign": int(self._tts_callsign.pending_count),
                "commands": int(self._tts_commands.pending_count),
                "story": int(self._tts_story.pending_count),
            },
            "noise_level": noise_level,
            "distortion_level": distortion_level,
        }

    def _sync_background_layers(self, *, payload: AuditoryCapacityPayload) -> None:
        assert self._bg_channel is not None
        assert self._dist_channel is not None
        if not self._ambient_tracks:
            return

        now_s = float(pygame.time.get_ticks()) / 1000.0
        if self._ambient_started_at_s <= 0.0:
            self._ambient_started_at_s = now_s

        if self._next_ambient_change_at_s <= 0.0 or now_s >= self._next_ambient_change_at_s:
            self._replan_ambient_mix(now_s=now_s, payload=payload)

        noise_level = max(0.0, min(1.0, float(payload.background_noise_level)))
        elapsed_ratio = max(0.0, min(1.0, (now_s - self._ambient_started_at_s) / 240.0))

        channels = (self._bg_channel, self._dist_channel)
        for idx, channel in enumerate(channels):
            should_play = (idx < self._ambient_target_layers) or channel.get_busy()
            if not should_play:
                self._ambient_slots[idx] = None
                self._ambient_active_sounds[idx] = None
                channel.set_volume(0.0)
                if channel.get_busy():
                    channel.stop()
                continue

            active_key = self._ambient_slots[idx]
            needs_snippet = (active_key is None) or (not channel.get_busy())
            if needs_snippet:
                exclude = self._ambient_slots[1 - idx] if len(self._ambient_slots) > 1 else None
                next_key = self._pick_ambient_key(
                    elapsed_ratio=elapsed_ratio,
                    now_s=now_s,
                    exclude=exclude,
                    forced_source=payload.background_noise_source,
                )
                if next_key is None:
                    self._ambient_slots[idx] = None
                    self._ambient_active_sounds[idx] = None
                    channel.set_volume(0.0)
                    if channel.get_busy():
                        channel.stop()
                    continue

                snippet_sound = self._build_ambient_snippet_sound(
                    next_key, elapsed_ratio=elapsed_ratio
                )
                if snippet_sound is None:
                    self._ambient_slots[idx] = None
                    self._ambient_active_sounds[idx] = None
                    channel.set_volume(0.0)
                    if channel.get_busy():
                        channel.stop()
                    continue

                self._ambient_slots[idx] = next_key
                self._ambient_active_sounds[idx] = snippet_sound
                channel.play(snippet_sound)
                self._ambient_recent_keys.append(next_key)
                if len(self._ambient_recent_keys) > 6:
                    del self._ambient_recent_keys[:-6]
                if "mature" in next_key.lower():
                    self._last_mature_clip_at_s = now_s

            active_key = self._ambient_slots[idx]
            if active_key is None:
                self._ambient_active_sounds[idx] = None
                if channel.get_busy():
                    channel.stop()
                continue
            self._active_noise_source = active_key

            item = self._ambient_tracks.get(active_key)
            if item is None:
                self._ambient_slots[idx] = None
                self._ambient_active_sounds[idx] = None
                if channel.get_busy():
                    channel.stop()
                continue

            base_volume = float(item.base_volume)
            # Keep early-phase ambience noticeably quieter and ramp toward full intensity.
            intensity = 0.30 + (0.70 * noise_level)
            layer_gain = intensity
            volume = max(0.0, min(1.0, base_volume * layer_gain))
            channel.set_volume(volume)

    def _sync_distortion_layer(self, *, payload: AuditoryCapacityPayload) -> None:
        if self._fx_channel is None or self._noise_sound is None:
            return
        distortion_level = max(0.0, min(1.0, float(payload.distortion_level)))
        if distortion_level <= 0.001:
            if self._fx_channel.get_busy():
                self._fx_channel.stop()
            return
        if not self._fx_channel.get_busy():
            self._fx_channel.play(self._noise_sound, loops=-1)
        self._fx_channel.set_volume(0.05 + (0.28 * distortion_level))

    def _load_ambient_tracks(self) -> dict[str, _AmbientTrack]:
        tracks: dict[str, _AmbientTrack] = {}

        for filename, base_volume, base_weight, late_weight in self._ambient_asset_manifest:
            path = self._ambient_assets_dir / filename
            meta = self._probe_wav_metadata(path)
            if meta is None:
                continue
            sample_rate, channels, frame_count = meta
            tracks[filename] = _AmbientTrack(
                path=path,
                base_volume=float(base_volume),
                base_weight=float(base_weight),
                late_weight=float(late_weight),
                sample_rate=sample_rate,
                channels=channels,
                frame_count=frame_count,
                is_generated_noise=False,
            )

        if not tracks:
            tracks["generated_noise"] = _AmbientTrack(
                path=None,
                base_volume=0.22,
                base_weight=1.00,
                late_weight=0.00,
                sample_rate=self._sample_rate,
                channels=1,
                frame_count=max(1, int(self._sample_rate * 18)),
                is_generated_noise=True,
            )
        return tracks

    @staticmethod
    def _probe_wav_metadata(path: Path) -> tuple[int, int, int] | None:
        if not path.exists():
            return None
        try:
            with wave.open(str(path), "rb") as wav_file:
                channels = int(wav_file.getnchannels())
                sample_width = int(wav_file.getsampwidth())
                sample_rate = int(wav_file.getframerate())
                frame_count = int(wav_file.getnframes())
        except Exception:
            return None
        if sample_width != 2 or channels <= 0 or frame_count <= 0 or sample_rate <= 0:
            return None
        return sample_rate, channels, frame_count

    @staticmethod
    def _resample_pcm(*, samples: array[int], src_rate: int, dst_rate: int) -> array[int]:
        if src_rate <= 0 or dst_rate <= 0 or len(samples) == 0:
            return array("h")
        if src_rate == dst_rate:
            return array("h", samples)

        out_count = max(1, int(round((len(samples) * dst_rate) / float(src_rate))))
        out = array("h")
        max_src = len(samples) - 1
        for idx in range(out_count):
            src_idx = int((idx * src_rate) / float(dst_rate))
            if src_idx < 0:
                src_idx = 0
            if src_idx > max_src:
                src_idx = max_src
            out.append(int(samples[src_idx]))
        return out

    def _replan_ambient_mix(self, *, now_s: float, payload: AuditoryCapacityPayload) -> None:
        if not self._ambient_tracks:
            self._next_ambient_change_at_s = now_s + 5.0
            return

        elapsed_s = max(0.0, now_s - self._ambient_started_at_s)
        elapsed_ratio = max(0.0, min(1.0, elapsed_s / 240.0))
        noise_level = max(0.0, min(1.0, float(payload.background_noise_level)))
        forced_source = str(payload.background_noise_source or "").strip()
        if forced_source:
            self._ambient_target_layers = 0 if noise_level <= 0.001 else 1
            self._next_ambient_change_at_s = now_s + 12.0
            return

        # Keep ambience present most of the time, with brief breathers only.
        silence_prob = max(0.01, 0.08 - (0.03 * elapsed_ratio) - (0.03 * noise_level))
        two_prob = min(0.88, 0.20 + (0.26 * elapsed_ratio) + (0.42 * noise_level))
        one_prob = max(0.0, 1.0 - silence_prob - two_prob)

        roll = self._voice_rng.random()
        if roll < silence_prob:
            target_count = 0
        elif roll < (silence_prob + one_prob):
            target_count = 1
        else:
            target_count = 2
        self._ambient_target_layers = min(target_count, len(self._ambient_tracks), 2)

        if self._ambient_target_layers == 0:
            window_s = self._voice_rng.uniform(8.0, 22.0)
        elif self._ambient_target_layers == 1:
            window_s = self._voice_rng.uniform(70.0, 150.0) - (10.0 * noise_level)
        else:
            window_s = self._voice_rng.uniform(55.0, 120.0) - (8.0 * noise_level)
        self._next_ambient_change_at_s = now_s + max(6.0, float(window_s))

    def _pick_ambient_key(
        self,
        *,
        elapsed_ratio: float,
        now_s: float,
        exclude: str | None = None,
        forced_source: str | None = None,
    ) -> str | None:
        forced = str(forced_source or "").strip()
        if forced:
            if forced in self._ambient_tracks and forced != exclude:
                return forced
            if forced in self._ambient_tracks:
                return forced
        keys = [k for k in self._ambient_tracks if k != exclude]
        if self._ambient_recent_keys:
            filtered = [k for k in keys if k not in self._ambient_recent_keys]
            if filtered:
                keys = filtered
        if not keys:
            keys = list(self._ambient_tracks.keys())
        if not keys:
            return None

        total_weight = 0.0
        weighted: list[tuple[str, float]] = []
        for key in keys:
            track = self._ambient_tracks[key]
            base_weight = float(track.base_weight)
            late_weight = float(track.late_weight)
            weight = max(0.001, base_weight + (late_weight * elapsed_ratio))
            weighted.append((key, weight))
            total_weight += weight

        pick = self._voice_rng.uniform(0.0, total_weight)
        running = 0.0
        for key, weight in weighted:
            running += weight
            if running >= pick:
                return key
        return weighted[-1][0]

    def _build_ambient_snippet_sound(
        self,
        key: str,
        *,
        elapsed_ratio: float,
    ) -> pygame.mixer.Sound | None:
        track = self._ambient_tracks.get(key)
        if track is None:
            return None
        is_mature = "mature" in key.lower()
        if is_mature:
            lo_s = 22.0
            hi_s = 46.0 + (18.0 * elapsed_ratio)
        else:
            lo_s = 70.0
            hi_s = 150.0 + (40.0 * elapsed_ratio)

        length_s = self._voice_rng.uniform(lo_s, hi_s)
        target_take_samples = max(256, int(length_s * self._sample_rate))
        if track.is_generated_noise:
            seed = self._voice_rng.randint(0, 2_147_483_647)
            segment = self._render_noise_pcm(duration_s=length_s, seed=seed, gain=0.34)
        else:
            if track.path is None or track.frame_count <= 1:
                return None

            source_take_frames = max(64, int(length_s * track.sample_rate))
            if source_take_frames >= track.frame_count:
                source_take_frames = track.frame_count
                start_frame = 0
            else:
                start_frame = self._voice_rng.randint(0, track.frame_count - source_take_frames)

            source_pcm = self._read_wav_segment_pcm(
                path=track.path,
                channels=track.channels,
                start_frame=start_frame,
                frame_count=source_take_frames,
            )
            if source_pcm is None or len(source_pcm) <= 32:
                return None

            segment = source_pcm
            if track.sample_rate != self._sample_rate:
                segment = self._resample_pcm(
                    samples=segment,
                    src_rate=track.sample_rate,
                    dst_rate=self._sample_rate,
                )
            if len(segment) <= 32:
                return None
            if len(segment) < target_take_samples:
                segment = self._extend_segment_with_variation(
                    source=segment,
                    target_count=target_take_samples,
                    source_key=key,
                    elapsed_ratio=elapsed_ratio,
                )
            elif len(segment) > target_take_samples:
                segment = array("h", segment[:target_take_samples])

        self._apply_edge_fade(segment, fade_s=0.012)
        self._normalize_segment_level(segment, target_rms=5600.0, peak_ceiling=0.88)
        if len(segment) <= 0:
            return None
        try:
            return self._sound_from_pcm(segment)
        except Exception:
            return None

    def _extend_segment_with_variation(
        self,
        *,
        source: array[int],
        target_count: int,
        source_key: str,
        elapsed_ratio: float,
    ) -> array[int]:
        if len(source) <= 0 or target_count <= 0:
            return array("h")
        if len(source) >= target_count:
            return array("h", source[:target_count])

        out = array("h")
        source_n = len(source)
        min_chunk = max(128, int(self._sample_rate * 0.35))
        max_chunk = max(min_chunk + 64, int(self._sample_rate * 1.40))

        while len(out) < target_count:
            remaining = target_count - len(out)
            chunk_n = min(remaining, self._voice_rng.randint(min_chunk, max_chunk))

            donor = None
            if self._voice_rng.random() < 0.38:
                donor = self._sample_donor_chunk(
                    source_key=source_key,
                    target_samples=chunk_n,
                    elapsed_ratio=elapsed_ratio,
                )

            if donor is not None and len(donor) > 0:
                frag = donor
            else:
                if source_n <= chunk_n:
                    start = 0
                    frag = array("h", source)
                else:
                    start = self._voice_rng.randint(0, source_n - chunk_n)
                    frag = array("h", source[start : start + chunk_n])

            if self._voice_rng.random() < 0.20:
                frag.reverse()

            gain = self._voice_rng.uniform(0.90, 1.08)
            for idx, value in enumerate(frag):
                scaled = int(round(float(value) * gain))
                frag[idx] = max(-32768, min(32767, scaled))

            self._apply_edge_fade(frag, fade_s=0.006)
            out.extend(frag[:remaining])

        return out

    def _sample_donor_chunk(
        self,
        *,
        source_key: str,
        target_samples: int,
        elapsed_ratio: float,
    ) -> array[int] | None:
        candidate_keys = [k for k in self._ambient_tracks if k != source_key]
        if not candidate_keys:
            return None

        if self._ambient_recent_keys:
            filtered = [k for k in candidate_keys if k not in self._ambient_recent_keys]
            if filtered:
                candidate_keys = filtered

        key = self._voice_rng.choice(candidate_keys)
        track = self._ambient_tracks.get(key)
        if track is None:
            return None

        if track.is_generated_noise:
            seed = self._voice_rng.randint(0, 2_147_483_647)
            return self._render_noise_pcm(
                duration_s=max(0.10, target_samples / float(self._sample_rate)),
                seed=seed,
                gain=0.30,
            )

        if track.path is None or track.frame_count <= 1:
            return None

        desired_frames = max(
            32, int((target_samples / float(self._sample_rate)) * track.sample_rate)
        )
        desired_frames = min(desired_frames, track.frame_count)
        if desired_frames <= 0:
            return None

        if desired_frames >= track.frame_count:
            start_frame = 0
        else:
            start_frame = self._voice_rng.randint(0, track.frame_count - desired_frames)

        pcm = self._read_wav_segment_pcm(
            path=track.path,
            channels=track.channels,
            start_frame=start_frame,
            frame_count=desired_frames,
        )
        if pcm is None or len(pcm) <= 0:
            return None

        if track.sample_rate != self._sample_rate:
            pcm = self._resample_pcm(
                samples=pcm, src_rate=track.sample_rate, dst_rate=self._sample_rate
            )
        if len(pcm) <= 0:
            return None
        if len(pcm) > target_samples:
            return array("h", pcm[:target_samples])
        return pcm

    @staticmethod
    def _read_wav_segment_pcm(
        *,
        path: Path,
        channels: int,
        start_frame: int,
        frame_count: int,
    ) -> array[int] | None:
        try:
            with wave.open(str(path), "rb") as wav_file:
                wav_file.setpos(max(0, int(start_frame)))
                raw = wav_file.readframes(max(1, int(frame_count)))
        except Exception:
            return None

        if not raw:
            return None

        samples = array("h")
        samples.frombytes(raw)
        if len(samples) <= 0:
            return None

        if channels <= 1:
            return samples

        mono = array("h")
        step = max(1, int(channels))
        for idx in range(0, len(samples), step):
            acc = 0
            used = 0
            for c in range(step):
                pos = idx + c
                if pos >= len(samples):
                    break
                acc += int(samples[pos])
                used += 1
            if used > 0:
                mono.append(int(acc / used))
        return mono if len(mono) > 0 else None

    def _apply_edge_fade(self, samples: array[int], *, fade_s: float) -> None:
        if len(samples) <= 2:
            return
        fade_n = max(1, int(self._sample_rate * max(0.001, float(fade_s))))
        fade_n = min(fade_n, len(samples) // 2)
        if fade_n <= 0:
            return
        for idx in range(fade_n):
            gain = idx / float(fade_n)
            head = int(samples[idx] * gain)
            tail_idx = len(samples) - idx - 1
            tail = int(samples[tail_idx] * gain)
            samples[idx] = max(-32768, min(32767, head))
            samples[tail_idx] = max(-32768, min(32767, tail))

    @staticmethod
    def _normalize_segment_level(
        samples: array[int], *, target_rms: float, peak_ceiling: float
    ) -> None:
        if len(samples) <= 0:
            return
        peak = 0
        energy = 0.0
        for value in samples:
            v = abs(int(value))
            if v > peak:
                peak = v
            energy += float(value) * float(value)
        if peak <= 0:
            return

        rms = math.sqrt(energy / float(len(samples)))
        if rms <= 1e-6:
            return

        desired_gain = float(target_rms) / rms
        peak_limit = max(0.05, min(0.99, float(peak_ceiling))) * 32767.0
        max_gain = peak_limit / float(peak)
        gain = max(0.05, min(desired_gain, max_gain))

        for idx, value in enumerate(samples):
            out = int(round(float(value) * gain))
            samples[idx] = max(-32768, min(32767, out))

    def _sync_assigned_callsigns(self, *, payload: AuditoryCapacityPayload) -> None:
        assigned = tuple(str(v) for v in payload.assigned_callsigns)
        if assigned != self._last_assigned_callsigns and assigned:
            joined = ", ".join(assigned)
            self._tts_callsign.speak(f"Assigned call signs. {joined}.")
            self._last_assigned_callsigns = assigned

    def _sync_callsign_cue(self, *, payload: AuditoryCapacityPayload) -> None:
        cue = payload.callsign_cue
        if cue is None:
            self._last_callsign_cue = None
            return
        if cue == self._last_callsign_cue:
            return
        if self._available:
            sound = self._callsign_cache.get(cue)
            if sound is None:
                sound = self._build_callsign_sound(cue)
                self._callsign_cache[cue] = sound
            self._play_cue(sound, volume=0.18)
        elif payload.callsign_blocks_gate:
            self._tts_callsign.speak(f"Call sign {cue}. Hold current color gate.")
        else:
            self._tts_callsign.speak(f"Call sign {cue}.")
        self._last_callsign_cue = cue

    def _sync_beep_cue(self, *, payload: AuditoryCapacityPayload) -> None:
        active = bool(payload.beep_active)
        if active and not self._last_beep_active and self._beep_sound is not None:
            assert self._alert_channel is not None
            self._alert_channel.set_volume(0.58)
            self._alert_channel.play(self._beep_sound)
        self._last_beep_active = active

    def _sync_color_cue(self, *, payload: AuditoryCapacityPayload) -> None:
        command = payload.color_command
        if command is None:
            self._last_color_command = None
            return
        if command == self._last_color_command:
            return
        sound = self._color_sounds.get(command)
        if sound is not None:
            self._play_cue(sound, volume=0.28)
        self._tts_commands.speak(f"Change color. {command}.")
        self._last_color_command = command

    def _sync_sequence_cue(self, *, payload: AuditoryCapacityPayload) -> None:
        sequence = payload.sequence_display
        if sequence is None:
            self._last_sequence_display = None
            return
        if sequence == self._last_sequence_display:
            return
        if self._available:
            sound = self._sequence_cache.get(sequence)
            if sound is None:
                sound = self._build_sequence_sound(sequence)
                self._sequence_cache[sequence] = sound
            self._play_cue(sound, volume=0.34)
        digits = [ch for ch in sequence if ch.isdigit()]
        if digits:
            spoken = " ".join(digits)
            self._tts_commands.speak(f"Sequence. {spoken}.")
        self._last_sequence_display = sequence

    def _sync_instruction_cue(self, *, payload: AuditoryCapacityPayload) -> None:
        uid = payload.instruction_uid
        text = payload.instruction_text
        cmd = (payload.instruction_command_type or "").strip().lower()
        if uid is None or not text:
            self._last_instruction_uid = None
            return
        if uid == self._last_instruction_uid:
            return
        if cmd in {"change_colour", "digit_append"}:
            self._last_instruction_uid = uid
            return
        if self._tts_commands.pending_count > 1:
            self._last_instruction_uid = uid
            return
        self._tts_commands.speak(text)
        self._last_instruction_uid = uid

    def _sync_voice_chatter(self, *, payload: AuditoryCapacityPayload) -> None:
        if not self._tts_story.enabled:
            return
        if not self._story_segments:
            return
        if self._tts_story.pending_count > 2:
            return
        if self._story_index < 0:
            if len(self._story_segments) > 1:
                # Avoid always starting from the first sentence.
                self._story_index = self._voice_rng.randint(1, len(self._story_segments) - 1)
            else:
                self._story_index = 0

        now_s = float(pygame.time.get_ticks()) / 1000.0
        if self._next_chatter_at_s <= 0.0:
            self._next_chatter_at_s = now_s + 1.2
            return
        if now_s < self._next_chatter_at_s:
            return

        segment = str(self._story_segments[self._story_index % len(self._story_segments)])
        self._story_index += 1
        self._tts_story.speak(segment)

        words = max(1, len(segment.split()))
        nominal_read_time_s = words / 2.55
        noise_slowdown = 0.7 * float(payload.background_noise_level)
        jitter = self._voice_rng.uniform(-0.5, 0.8)
        self._next_chatter_at_s = now_s + max(4.0, nominal_read_time_s + noise_slowdown + jitter)

    def _play_cue(self, sound: pygame.mixer.Sound, *, volume: float) -> None:
        if not self._available:
            return
        assert self._cue_channel is not None
        self._cue_channel.set_volume(max(0.0, min(1.0, volume)))
        self._cue_channel.play(sound)

    def _build_callsign_sound(self, callsign: str) -> pygame.mixer.Sound:
        code = sum(ord(ch) for ch in callsign)
        freq_a = 290.0 + (code % 380)
        freq_b = 430.0 + ((code // 3) % 440)
        pcm = self._concat_pcm(
            (
                self._render_tone_pcm(freq_a, 0.10, gain=0.35),
                self._render_silence_pcm(0.030),
                self._render_tone_pcm(freq_b, 0.15, gain=0.33),
            )
        )
        return self._sound_from_pcm(pcm)

    def _build_sequence_sound(self, sequence: str) -> pygame.mixer.Sound:
        dtmf = {
            "0": 330.0,
            "1": 350.0,
            "2": 390.0,
            "3": 430.0,
            "4": 470.0,
            "5": 510.0,
            "6": 560.0,
            "7": 610.0,
            "8": 670.0,
            "9": 730.0,
        }
        parts: list[array[int]] = []
        digits = [ch for ch in str(sequence) if ch.isdigit()]
        for digit in digits:
            freq = dtmf.get(digit, 420.0)
            parts.append(self._render_tone_pcm(freq, 0.075, gain=0.32))
            parts.append(self._render_silence_pcm(0.020))
        if not parts:
            parts.append(self._render_tone_pcm(420.0, 0.08, gain=0.28))
        pcm = self._concat_pcm(tuple(parts))
        return self._sound_from_pcm(pcm)

    def _build_noise_sound(
        self,
        *,
        duration_s: float,
        seed: int,
        gain: float,
    ) -> pygame.mixer.Sound:
        pcm = self._render_noise_pcm(duration_s=duration_s, seed=seed, gain=gain)
        return self._sound_from_pcm(pcm)

    def _render_noise_pcm(self, *, duration_s: float, seed: int, gain: float) -> array[int]:
        rng = random.Random(int(seed))
        sample_count = max(1, int(self._sample_rate * duration_s))
        out = array("h")
        smooth = 0.0
        for _ in range(sample_count):
            raw = rng.uniform(-1.0, 1.0)
            smooth = (smooth * 0.86) + (raw * 0.14)
            sample = int(max(-1.0, min(1.0, smooth * gain)) * self._amp)
            out.append(sample)
        return out

    def _build_tone_sound(
        self, frequency_hz: float, duration_s: float, *, gain: float
    ) -> pygame.mixer.Sound:
        pcm = self._render_tone_pcm(frequency_hz, duration_s, gain=gain)
        return self._sound_from_pcm(pcm)

    def _render_tone_pcm(
        self, frequency_hz: float, duration_s: float, *, gain: float
    ) -> array[int]:
        sample_count = max(1, int(self._sample_rate * duration_s))
        fade_n = max(1, int(self._sample_rate * 0.008))
        out = array("h")
        for idx in range(sample_count):
            envelope = 1.0
            if idx < fade_n:
                envelope = idx / float(fade_n)
            tail = sample_count - idx - 1
            if tail < fade_n:
                envelope = min(envelope, tail / float(fade_n))
            phase = (2.0 * math.pi * float(frequency_hz) * idx) / float(self._sample_rate)
            sample = math.sin(phase) * gain * max(0.0, envelope)
            out.append(int(max(-1.0, min(1.0, sample)) * self._amp))
        return out

    def _render_silence_pcm(self, duration_s: float) -> array[int]:
        sample_count = max(1, int(self._sample_rate * duration_s))
        return array("h", [0] * sample_count)

    @staticmethod
    def _concat_pcm(parts: tuple[array[int], ...]) -> array[int]:
        out = array("h")
        for part in parts:
            out.extend(part)
        return out

    def _sound_from_pcm(self, mono_pcm: array[int]) -> pygame.mixer.Sound:
        pcm = self._to_mixer_channels(mono_pcm)
        return pygame.mixer.Sound(buffer=pcm.tobytes())

    def _to_mixer_channels(self, mono_pcm: array[int]) -> array[int]:
        channels = max(1, int(self._mixer_channels))
        if channels == 1:
            return mono_pcm

        interleaved = array("h")
        for sample in mono_pcm:
            val = int(sample)
            for _ in range(channels):
                interleaved.append(val)
        return interleaved


WINDOW_SIZE = (960, 540)
TARGET_FPS = 60


@dataclass(slots=True)
class _AuditoryGlScene:
    world: pygame.Rect
    payload: AuditoryCapacityPayload | None
    time_remaining_s: float | None
    time_fill_ratio: float | None


class _OpenGLAuditoryRenderer:
    """Optional OpenGL renderer for the Auditory Capacity tube scene.

    This keeps deterministic engine logic unchanged and only replaces the tube viewport
    draw path when OpenGL is available.
    """

    def __init__(self, *, window_size: tuple[int, int]) -> None:
        from OpenGL import GL as gl  # type: ignore[import-not-found]

        self._gl = gl
        self._win_w = max(1, int(window_size[0]))
        self._win_h = max(1, int(window_size[1]))

        tex = gl.glGenTextures(1)
        if isinstance(tex, (tuple, list)):
            tex = tex[0]
        self._ui_texture_id = int(tex)
        self._ui_tex_size: tuple[int, int] = (0, 0)
        self._vortex_texture_id = self._create_texture_id()
        self._vortex_tex_size: tuple[int, int] = (0, 0)
        self._vortex_rgba: bytes = b""
        self._init_vortex_texture(size=640)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        if hasattr(gl, "GL_LINE_SMOOTH_HINT") and hasattr(gl, "GL_NICEST"):
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        if hasattr(gl, "GL_MULTISAMPLE"):
            gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glLineWidth(1.0)

    def _create_texture_id(self) -> int:
        tex = self._gl.glGenTextures(1)
        if isinstance(tex, (tuple, list)):
            tex = tex[0]
        return int(tex)

    def _init_vortex_texture(self, *, size: int) -> None:
        gl = self._gl
        tex_size = max(64, int(size))
        self._vortex_rgba = self._build_vortex_rgba(size=tex_size)
        self._vortex_tex_size = (tex_size, tex_size)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._vortex_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            tex_size,
            tex_size,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self._vortex_rgba,
        )

    @staticmethod
    def _build_vortex_rgba(*, size: int) -> bytes:
        n = max(64, int(size))
        cx = (n - 1) * 0.5
        cy = (n - 1) * 0.5
        inv = 1.0 / max(1.0, float(n - 1))
        out = bytearray(n * n * 4)

        for y in range(n):
            ny = ((float(y) - cy) * 2.0) * inv
            for x in range(n):
                nx = ((float(x) - cx) * 2.0) * inv
                r = math.sqrt((nx * nx) + (ny * ny))
                a = math.atan2(ny, nx)
                # Deterministic swirl field.
                swirl = (
                    math.sin((r * 23.0) - (a * 5.6))
                    + math.cos((r * 31.0) + (a * 7.3))
                    + math.sin((nx * 17.0) - (ny * 13.0))
                )
                swirl_n = (swirl + 3.0) / 6.0
                falloff = max(0.0, min(1.0, 1.25 - (r * 0.95)))
                grain = math.sin((x * 0.173) + (y * 0.217)) * 0.09
                lum = max(0.0, min(1.0, (swirl_n * 0.64) + (falloff * 0.30) + grain))

                red = int(round(16 + (lum * 38)))
                green = int(round(36 + (lum * 84)))
                blue = int(round(68 + (lum * 146)))
                alpha = 255

                i = ((y * n) + x) * 4
                out[i] = max(0, min(255, red))
                out[i + 1] = max(0, min(255, green))
                out[i + 2] = max(0, min(255, blue))
                out[i + 3] = alpha

        return bytes(out)

    def resize(self, *, window_size: tuple[int, int]) -> None:
        self._win_w = max(1, int(window_size[0]))
        self._win_h = max(1, int(window_size[1]))

    def render_frame(
        self,
        *,
        ui_surface: pygame.Surface,
        scene: _AuditoryGlScene | None,
    ) -> None:
        gl = self._gl

        gl.glViewport(0, 0, self._win_w, self._win_h)
        gl.glClearColor(0.01, 0.02, 0.06, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if scene is not None:
            self._draw_auditory_scene(scene=scene)

        self._draw_ui_surface(ui_surface=ui_surface)

    def _draw_auditory_scene(self, *, scene: _AuditoryGlScene) -> None:
        gl = self._gl
        rect = scene.world
        payload = scene.payload
        time_fill_ratio = scene.time_fill_ratio

        vw = max(1, int(rect.w))
        vh = max(1, int(rect.h))
        vx = int(rect.x)
        vy = int(self._win_h - rect.bottom)

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(vx, vy, vw, vh)
        gl.glClearColor(0.03, 0.06, 0.15, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_SCISSOR_TEST)

        gl.glViewport(vx, vy, vw, vh)
        self._set_ortho_2d(width=vw, height=vh)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._vortex_texture_id)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(0.0, 0.0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(float(vw), 0.0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(float(vw), float(vh))
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(0.0, float(vh))
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

        cx = float(vw) * 0.50
        cy = float(vh) * 0.50
        max_r = min(vw, vh) * 0.54
        rings_2d = 20
        for idx in range(rings_2d):
            t = idx / float(max(1, rings_2d - 1))
            r = max_r * (0.10 + (0.90 * t))
            alpha = 0.12 * (1.0 - t)
            gl.glColor4f(0.10, 0.18, 0.38, alpha)
            gl.glBegin(gl.GL_LINE_LOOP)
            seg = 48
            for n in range(seg):
                a = (n / float(seg)) * math.tau
                gl.glVertex2f(cx + (math.cos(a) * r), cy + (math.sin(a) * r * 0.94))
            gl.glEnd()

        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glColor4f(0.04, 0.10, 0.28, 0.04)
        gl.glVertex2f(cx, cy)
        seg = 64
        for n in range(seg + 1):
            a = (n / float(seg)) * math.tau
            gl.glColor4f(0.01, 0.02, 0.08, 0.34)
            gl.glVertex2f(cx + (math.cos(a) * max_r), cy + (math.sin(a) * max_r * 0.92))
        gl.glEnd()

        self._set_perspective(
            fovy_deg=46.0,
            aspect=max(0.20, vw / float(vh)),
            z_near=0.10,
            z_far=80.0,
        )
        gl.glEnable(gl.GL_DEPTH_TEST)

        tube_rx = 0.88
        tube_ry = 0.54
        z_near = 2.0
        z_far = 26.0
        ring_steps = 28
        ring_count = 24
        ring_spacing = (z_far - z_near) / float(max(1, ring_count - 1))
        flow_speed_units_per_s = 1.45
        flow_offset = ((float(pygame.time.get_ticks()) / 1000.0) * flow_speed_units_per_s) % max(
            0.001, ring_spacing
        )

        ring_centers: list[tuple[float, float, float, float]] = []
        for idx in range(ring_count):
            t = idx / float(max(1, ring_count - 1))
            z = -(z_near + ((z_far - z_near) * t)) + flow_offset
            off_x, off_y = self._tube_offset_at_depth(z=z, z_near=z_near, z_far=z_far)
            ring_centers.append((z, off_x, off_y, t))
            lum = 0.76 - (0.54 * t)
            gl.glColor4f(0.10 * lum, 0.38 * lum, 0.90 * lum, 0.84)
            gl.glBegin(gl.GL_LINE_LOOP)
            for n in range(ring_steps):
                a = (n / float(ring_steps)) * math.tau
                gl.glVertex3f(
                    off_x + (math.cos(a) * tube_rx),
                    off_y + (math.sin(a) * tube_ry),
                    z,
                )
            gl.glEnd()

        wall_segments = 24
        for idx in range(len(ring_centers) - 1):
            z0, off_x0, off_y0, t0 = ring_centers[idx]
            z1, off_x1, off_y1, t1 = ring_centers[idx + 1]
            depth_mix = (t0 + t1) * 0.5
            alpha = 0.07 + (0.08 * (1.0 - depth_mix))
            gl.glBegin(gl.GL_QUAD_STRIP)
            for n in range(wall_segments + 1):
                a = (n / float(wall_segments)) * math.tau
                pulse = 0.62 + (0.38 * math.cos((a * 2.0) - (depth_mix * math.tau * 1.15)))
                gl.glColor4f(0.05 * pulse, 0.16 * pulse, 0.34 + (0.22 * pulse), alpha)
                gl.glVertex3f(
                    off_x0 + (math.cos(a) * tube_rx),
                    off_y0 + (math.sin(a) * tube_ry),
                    z0,
                )
                gl.glColor4f(
                    0.05 * pulse,
                    0.15 * pulse,
                    0.30 + (0.18 * pulse),
                    alpha * 0.94,
                )
                gl.glVertex3f(
                    off_x1 + (math.cos(a) * tube_rx),
                    off_y1 + (math.sin(a) * tube_ry),
                    z1,
                )
            gl.glEnd()

        for a in (0.0, math.pi * 0.5, math.pi, math.pi * 1.5):
            gl.glColor4f(0.16, 0.24, 0.46, 0.74)
            gl.glBegin(gl.GL_LINE_STRIP)
            for z, off_x, off_y, _ in ring_centers:
                gl.glVertex3f(
                    off_x + (math.cos(a) * tube_rx),
                    off_y + (math.sin(a) * tube_ry),
                    z,
                )
            gl.glEnd()

        base_cross_x, base_cross_y = self._tube_offset_at_depth(
            z=-3.1,
            z_near=z_near,
            z_far=z_far,
        )
        cross_x = 0.0
        cross_y = 0.0
        cross_z = -3.1

        if payload is not None:
            y_half_span = max(0.08, float(payload.tube_half_height))
            x_half_span = max(0.08, float(payload.tube_half_width))

            gates: list[tuple[float, AuditoryCapacityGate, float, float, float, int]] = []
            future_gates = [gate for gate in payload.gates if gate.x_norm >= -0.08]
            future_gates.sort(key=lambda gate: float(gate.x_norm))
            for visible_idx, gate in enumerate(future_gates[:6]):
                rel = gate.x_norm
                z = -(3.5 + (rel * 5.0))
                if z < -72.0 or z > -0.8:
                    continue
                gate_off_x, gate_off_y = self._tube_offset_at_depth(
                    z=z,
                    z_near=z_near,
                    z_far=z_far,
                )
                gy = max(-1.0, min(1.0, gate.y_norm / y_half_span)) * (tube_ry * 0.92)
                lane = _auditory_guide_lane(visible_idx)
                gx = gate_off_x + (lane * (0.34 + (0.18 * min(1.0, max(0.0, rel / 2.4)))))
                gy = gate_off_y + gy
                radius = 0.11 + (max(0.06, min(0.36, gate.aperture_norm)) * 1.10)
                if visible_idx < 2:
                    radius *= 1.10
                gates.append((z, gate, gx, gy, radius, visible_idx))

            for z, gate, gx, gy, radius, _visible_idx in sorted(gates, key=lambda g: g[0]):
                color = self._gate_color(gate.color)
                dz = 0.30
                gl.glColor4f(color[0] * 0.36, color[1] * 0.42, color[2] * 0.48, 0.24)
                self._draw_gate_shape_filled_3d(
                    shape=gate.shape,
                    x=gx,
                    y=gy,
                    z=z + (dz * 0.5),
                    radius=radius * 0.95,
                )
                gl.glColor4f(color[0], color[1], color[2], 0.95)
                self._draw_gate_shape_wire_3d(shape=gate.shape, x=gx, y=gy, z=z + dz, radius=radius)
                self._draw_gate_shape_wire_3d(shape=gate.shape, x=gx, y=gy, z=z, radius=radius)
                gl.glColor4f(
                    min(1.0, color[0] + 0.24),
                    min(1.0, color[1] + 0.24),
                    min(1.0, color[2] + 0.24),
                    0.72,
                )
                self._draw_gate_shape_wire_3d(
                    shape=gate.shape,
                    x=gx,
                    y=gy,
                    z=z + (dz * 0.38),
                    radius=radius * 0.92,
                )
                gl.glColor4f(color[0] * 0.78, color[1] * 0.78, color[2] * 0.78, 0.72)
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(gx - radius, gy, z)
                gl.glVertex3f(gx - radius, gy, z + dz)
                gl.glVertex3f(gx + radius, gy, z)
                gl.glVertex3f(gx + radius, gy, z + dz)
                gl.glVertex3f(gx, gy - radius, z)
                gl.glVertex3f(gx, gy - radius, z + dz)
                gl.glVertex3f(gx, gy + radius, z)
                gl.glVertex3f(gx, gy + radius, z + dz)
                gl.glEnd()

            bx = max(-1.0, min(1.0, payload.ball_x / x_half_span)) * (tube_rx * 0.90)
            by = max(-1.0, min(1.0, payload.ball_y / y_half_span)) * (tube_ry * 0.90)
            travel_t = (float(pygame.time.get_ticks()) / 1000.0) * flow_speed_units_per_s
            ball_z = -2.60 + (0.10 * math.sin(travel_t * 1.2))
            ball_r = 0.10
            ball_off_x, ball_off_y = self._tube_offset_at_depth(
                z=ball_z,
                z_near=z_near,
                z_far=z_far,
            )
            cross_z = ball_z + 0.002

            ball_contact = float(payload.ball_contact_ratio)
            if ball_contact >= 1.0:
                ball_edge = (0.95, 0.35, 0.38, 0.98)
            else:
                ball_edge = (0.92, 0.95, 1.0, 0.98)

            gl.glColor4f(0.56, 0.60, 0.70, 0.78)
            self._draw_filled_disc_3d(
                x=bx + (ball_r * 0.20),
                y=by - (ball_r * 0.25),
                z=ball_z - 0.05,
                radius=ball_r * 0.92,
            )
            gl.glColor4f(0.92, 0.95, 1.0, 0.98)
            self._draw_filled_disc_3d(x=bx, y=by, z=ball_z, radius=ball_r)
            gl.glColor4f(ball_edge[0], ball_edge[1], ball_edge[2], ball_edge[3])
            self._draw_disc_outline_3d(x=bx, y=by, z=ball_z, radius=ball_r)
            gl.glColor4f(1.0, 1.0, 1.0, 0.90)
            self._draw_filled_disc_3d(
                x=bx - (ball_r * 0.33),
                y=by + (ball_r * 0.32),
                z=ball_z + 0.001,
                radius=ball_r * 0.28,
            )

            if ball_contact >= 1.0:
                contact_strength = min(1.0, 0.35 + ((ball_contact - 1.0) * 4.2))
                local_x = max(-1.0, min(1.0, payload.ball_x / x_half_span))
                local_y = max(-1.0, min(1.0, payload.ball_y / y_half_span))
                local_norm = math.hypot(local_x, local_y)
                if local_norm > 1e-4:
                    hit_x = ball_off_x + ((local_x / local_norm) * tube_rx)
                    hit_y = ball_off_y + ((local_y / local_norm) * tube_ry)
                    hit_r = 0.12 + (0.05 * contact_strength)
                    gl.glColor4f(1.0, 0.40, 0.32, 0.62 + (0.18 * contact_strength))
                    self._draw_disc_outline_3d(
                        x=hit_x,
                        y=hit_y,
                        z=ball_z - 0.01,
                        radius=hit_r,
                    )
                    gl.glBegin(gl.GL_LINES)
                    gl.glVertex3f(hit_x, hit_y, ball_z - 0.02)
                    gl.glVertex3f(
                        hit_x - ((local_x / local_norm) * 0.28),
                        hit_y - ((local_y / local_norm) * 0.18),
                        ball_z - 0.02,
                    )
                    gl.glEnd()

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glColor4f(0.72, 0.18, 0.22, 0.84)
        gl.glBegin(gl.GL_LINES)
        gl.glVertex3f(cross_x, cross_y + (tube_ry * 0.96), cross_z)
        gl.glVertex3f(cross_x, cross_y - (tube_ry * 0.96), cross_z)
        gl.glVertex3f(cross_x - (tube_rx * 0.96), cross_y, cross_z)
        gl.glVertex3f(cross_x + (tube_rx * 0.96), cross_y, cross_z)
        gl.glEnd()

        self._set_ortho_2d(width=vw, height=vh)

        gl.glColor4f(0.08, 0.18, 0.56, 0.92)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(0.0, float(vh - 18))
        gl.glVertex2f(float(vw), float(vh - 18))
        gl.glVertex2f(float(vw), float(vh))
        gl.glVertex2f(0.0, float(vh))
        gl.glEnd()

        gl.glColor4f(0.46, 0.50, 0.62, 0.70)
        bar_w = 132.0
        bar_h = 15.0
        bar_x = float(vw) - bar_w - 16.0
        bar_y = 12.0
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(bar_x, bar_y)
        gl.glVertex2f(bar_x + bar_w, bar_y)
        gl.glVertex2f(bar_x + bar_w, bar_y + bar_h)
        gl.glVertex2f(bar_x, bar_y + bar_h)
        gl.glEnd()

        gl.glColor4f(0.16, 0.18, 0.22, 0.78)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(bar_x, bar_y)
        gl.glVertex2f(bar_x + bar_w, bar_y)
        gl.glVertex2f(bar_x + bar_w, bar_y + bar_h)
        gl.glVertex2f(bar_x, bar_y + bar_h)
        gl.glEnd()

        inner_x = bar_x + 4.0
        inner_y = bar_y + 4.0
        inner_w = bar_w - 8.0
        inner_h = bar_h - 8.0
        gl.glColor4f(0.12, 0.14, 0.18, 0.84)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(inner_x, inner_y)
        gl.glVertex2f(inner_x + inner_w, inner_y)
        gl.glVertex2f(inner_x + inner_w, inner_y + inner_h)
        gl.glVertex2f(inner_x, inner_y + inner_h)
        gl.glEnd()

        fill_ratio = 0.72 if time_fill_ratio is None else max(0.0, min(1.0, time_fill_ratio))
        fill_w = max(0.0, inner_w * fill_ratio)
        gl.glColor4f(0.76, 0.80, 0.86, 0.90)
        if fill_w > 0.0:
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex2f(inner_x, inner_y)
            gl.glVertex2f(inner_x + fill_w, inner_y)
            gl.glVertex2f(inner_x + fill_w, inner_y + inner_h)
            gl.glVertex2f(inner_x, inner_y + inner_h)
            gl.glEnd()
            gl.glColor4f(0.94, 0.96, 0.98, 0.70)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(inner_x + 1.0, inner_y + 1.0)
            gl.glVertex2f(inner_x + max(2.0, fill_w - 1.0), inner_y + 1.0)
            gl.glEnd()

        gl.glColor4f(0.16, 0.34, 0.84, 0.90)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(1.0, 1.0)
        gl.glVertex2f(float(vw - 1), 1.0)
        gl.glVertex2f(float(vw - 1), float(vh - 1))
        gl.glVertex2f(1.0, float(vh - 1))
        gl.glEnd()
        gl.glColor4f(0.38, 0.52, 0.88, 0.72)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex2f(4.0, 4.0)
        gl.glVertex2f(float(vw - 4), 4.0)
        gl.glVertex2f(float(vw - 4), float(vh - 4))
        gl.glVertex2f(4.0, float(vh - 4))
        gl.glEnd()

    def _draw_ui_surface(self, *, ui_surface: pygame.Surface) -> None:
        gl = self._gl
        w = max(1, int(ui_surface.get_width()))
        h = max(1, int(ui_surface.get_height()))

        pixels = pygame.image.tostring(ui_surface, "RGBA", True)

        gl.glViewport(0, 0, self._win_w, self._win_h)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._ui_texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        if self._ui_tex_size != (w, h):
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA,
                w,
                h,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                pixels,
            )
            self._ui_tex_size = (w, h)
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D,
                0,
                0,
                0,
                w,
                h,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                pixels,
            )

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0.0, float(self._win_w), 0.0, float(self._win_h), -1.0, 1.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(0.0, 0.0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(float(self._win_w), 0.0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(float(self._win_w), float(self._win_h))
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(0.0, float(self._win_h))
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

    def _set_ortho_2d(self, *, width: int, height: int) -> None:
        gl = self._gl
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0.0, float(width), 0.0, float(height), -1.0, 1.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _set_perspective(
        self, *, fovy_deg: float, aspect: float, z_near: float, z_far: float
    ) -> None:
        gl = self._gl
        fovy = float(fovy_deg)
        near = max(0.01, float(z_near))
        far = max(near + 0.10, float(z_far))
        top = near * math.tan(math.radians(fovy * 0.5))
        bottom = -top
        right = top * float(aspect)
        left = -right
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glFrustum(left, right, bottom, top, near, far)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    @staticmethod
    def _tube_offset_at_depth(*, z: float, z_near: float, z_far: float) -> tuple[float, float]:
        span = max(0.001, float(z_far) - float(z_near))
        depth = max(0.0, min(1.0, ((-float(z)) - float(z_near)) / span))
        curve = depth**1.18

        # Deterministic S-curves that are gentle near the player and stronger farther out.
        x_wave = (math.sin((depth * math.tau * 1.05) + 0.45) * 0.74) + (
            math.sin((depth * math.tau * 2.25) - 0.35) * 0.26
        )
        y_wave = (math.sin((depth * math.tau * 1.45) + 1.45) * 0.72) + (
            math.sin((depth * math.tau * 2.55) + 0.80) * 0.28
        )

        off_x = x_wave * (0.58 * curve)
        off_y = y_wave * (0.35 * curve)
        return off_x, off_y

    @staticmethod
    def _gate_color(name: str) -> tuple[float, float, float]:
        palette = {
            "RED": (0.92, 0.32, 0.34),
            "GREEN": (0.34, 0.88, 0.56),
            "BLUE": (0.36, 0.62, 0.94),
            "YELLOW": (0.94, 0.82, 0.34),
        }
        return palette.get(str(name).upper(), (0.86, 0.88, 0.92))

    def _draw_gate_shape_wire_3d(
        self, *, shape: str, x: float, y: float, z: float, radius: float
    ) -> None:
        gl = self._gl
        token = str(shape).upper()
        if token == "TRIANGLE":
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(x, y + radius, z)
            gl.glVertex3f(x - radius, y - radius, z)
            gl.glVertex3f(x + radius, y - radius, z)
            gl.glEnd()
            return
        if token == "SQUARE":
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(x - radius, y - radius, z)
            gl.glVertex3f(x + radius, y - radius, z)
            gl.glVertex3f(x + radius, y + radius, z)
            gl.glVertex3f(x - radius, y + radius, z)
            gl.glEnd()
            return

        self._draw_disc_outline_3d(x=x, y=y, z=z, radius=radius)

    def _draw_gate_shape_filled_3d(
        self, *, shape: str, x: float, y: float, z: float, radius: float
    ) -> None:
        gl = self._gl
        token = str(shape).upper()
        if token == "TRIANGLE":
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex3f(x, y + radius, z)
            gl.glVertex3f(x - radius, y - radius, z)
            gl.glVertex3f(x + radius, y - radius, z)
            gl.glEnd()
            return
        if token == "SQUARE":
            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(x - radius, y - radius, z)
            gl.glVertex3f(x + radius, y - radius, z)
            gl.glVertex3f(x + radius, y + radius, z)
            gl.glVertex3f(x - radius, y + radius, z)
            gl.glEnd()
            return
        self._draw_filled_disc_3d(x=x, y=y, z=z, radius=radius)

    def _draw_filled_disc_3d(self, *, x: float, y: float, z: float, radius: float) -> None:
        gl = self._gl
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex3f(x, y, z)
        segments = 28
        for i in range(segments + 1):
            a = (i / float(segments)) * math.tau
            gl.glVertex3f(x + (math.cos(a) * radius), y + (math.sin(a) * radius), z)
        gl.glEnd()

    def _draw_disc_outline_3d(self, *, x: float, y: float, z: float, radius: float) -> None:
        gl = self._gl
        gl.glBegin(gl.GL_LINE_LOOP)
        segments = 28
        for i in range(segments):
            a = (i / float(segments)) * math.tau
            gl.glVertex3f(x + (math.cos(a) * radius), y + (math.sin(a) * radius), z)
        gl.glEnd()


class App:
    def __init__(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        *,
        opengl_enabled: bool = False,
        results_store: ResultsStore | None = None,
        input_profiles_store: InputProfilesStore | None = None,
        difficulty_settings_store: DifficultySettingsStore | None = None,
        app_version: str = "dev",
    ) -> None:
        self._surface = surface
        self._font = font
        self._screens: list[Screen] = []
        self._running = True
        self._opengl_enabled = bool(opengl_enabled)
        self._auditory_gl_scene: _AuditoryGlScene | None = None
        self._results_store = results_store
        self._input_profiles_store = input_profiles_store
        self._difficulty_settings_store = difficulty_settings_store
        self._app_version = str(app_version).strip() or "dev"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def font(self) -> pygame.font.Font:
        return self._font

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    @property
    def opengl_enabled(self) -> bool:
        return self._opengl_enabled

    def set_surface(self, surface: pygame.Surface) -> None:
        self._surface = surface

    def set_opengl_enabled(self, enabled: bool) -> None:
        self._opengl_enabled = bool(enabled)
        if not self._opengl_enabled:
            self._auditory_gl_scene = None

    def push(self, screen: Screen) -> None:
        self._screens.append(screen)

    def replace_top(self, screen: Screen) -> None:
        if not self._screens:
            self.push(screen)
            return
        old = self._screens[-1]
        self._screens[-1] = screen
        close = getattr(old, "close", None)
        if callable(close):
            close()

    def pop(self) -> None:
        # Never pop the last/root screen; root handles its own quit/back behavior.
        if len(self._screens) > 1:
            screen = self._screens.pop()
            close = getattr(screen, "close", None)
            if callable(close):
                close()

    def pop_to_root(self) -> None:
        while len(self._screens) > 1:
            screen = self._screens.pop()
            close = getattr(screen, "close", None)
            if callable(close):
                close()

    def quit(self) -> None:
        self._running = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit()
            return
        if event.type == pygame.KEYDOWN:
            mod = int(getattr(event, "mod", 0))
            if event.key == pygame.K_q and (mod & (pygame.KMOD_CTRL | pygame.KMOD_META)):
                self.quit()
                return
        if not self._screens:
            return
        self._screens[-1].handle_event(event)

    def render(self) -> None:
        if not self._screens:
            return
        self._auditory_gl_scene = None
        self._screens[-1].render(self._surface)

    def queue_auditory_gl_scene(
        self,
        *,
        world: pygame.Rect,
        payload: AuditoryCapacityPayload | None,
        time_remaining_s: float | None,
        time_fill_ratio: float | None,
    ) -> None:
        if not self._opengl_enabled:
            return
        self._auditory_gl_scene = _AuditoryGlScene(
            world=pygame.Rect(world),
            payload=payload,
            time_remaining_s=time_remaining_s,
            time_fill_ratio=time_fill_ratio,
        )

    def consume_auditory_gl_scene(self) -> _AuditoryGlScene | None:
        return self._auditory_gl_scene

    def persist_attempt(
        self,
        *,
        engine: object,
        test_code: str,
        test_version: int = 1,
    ) -> list[str]:
        if self._results_store is None:
            return []
        try:
            result = attempt_result_from_engine(
                engine,
                test_code=test_code,
                test_version=test_version,
            )
            self._results_store.record_attempt(
                result=result,
                app_version=self._app_version,
                input_profile_id=self.active_input_profile_id(),
            )
            return self._build_persistence_summary_lines(test_code=test_code)
        except Exception:
            return ["Local save failed."]

    def active_input_profile_id(self) -> str | None:
        if self._input_profiles_store is None:
            return None
        try:
            return str(self._input_profiles_store.active_profile().profile_id)
        except Exception:
            return None

    def effective_difficulty_level(self, test_code: str | None) -> int:
        if self._difficulty_settings_store is None or not test_code:
            return DEFAULT_DIFFICULTY_LEVEL
        return self._difficulty_settings_store.effective_level(str(test_code))

    def effective_difficulty_ratio(self, test_code: str | None) -> float:
        level = self.effective_difficulty_level(test_code)
        return (max(1, min(10, int(level))) - 1) / 9.0

    def stored_test_difficulty_level(self, test_code: str | None) -> int:
        if self._difficulty_settings_store is None or not test_code:
            return DEFAULT_DIFFICULTY_LEVEL
        return self._difficulty_settings_store.test_level(str(test_code))

    def stored_global_difficulty_level(self) -> int:
        if self._difficulty_settings_store is None:
            return DEFAULT_DIFFICULTY_LEVEL
        return self._difficulty_settings_store.global_level()

    def difficulty_override_enabled(self) -> bool:
        if self._difficulty_settings_store is None:
            return False
        return self._difficulty_settings_store.global_override_enabled()

    def intro_difficulty_mode_label(self, test_code: str | None) -> str:
        if self._difficulty_settings_store is None or not test_code:
            return "Session"
        return self._difficulty_settings_store.intro_mode_label(str(test_code))

    def set_persistent_difficulty_level(self, *, test_code: str | None, level: int) -> int:
        clamped = max(1, min(10, int(level)))
        if self._difficulty_settings_store is None or not test_code:
            return clamped
        if self._difficulty_settings_store.global_override_enabled():
            self._difficulty_settings_store.set_global_level(clamped)
            return self._difficulty_settings_store.global_level()
        self._difficulty_settings_store.set_test_level(test_code=str(test_code), level=clamped)
        return self._difficulty_settings_store.test_level(str(test_code))

    def set_global_difficulty_override_enabled(self, enabled: bool) -> None:
        if self._difficulty_settings_store is None:
            return
        self._difficulty_settings_store.set_global_override_enabled(enabled)

    def set_global_difficulty_level(self, level: int) -> int:
        clamped = max(1, min(10, int(level)))
        if self._difficulty_settings_store is None:
            return clamped
        self._difficulty_settings_store.set_global_level(clamped)
        return self._difficulty_settings_store.global_level()

    def set_test_difficulty_level(self, *, test_code: str, level: int) -> int:
        clamped = max(1, min(10, int(level)))
        if self._difficulty_settings_store is None:
            return clamped
        self._difficulty_settings_store.set_test_level(test_code=test_code, level=clamped)
        return self._difficulty_settings_store.test_level(test_code)

    def _build_persistence_summary_lines(self, *, test_code: str) -> list[str]:
        if self._results_store is None:
            return []

        session_summary = self._results_store.session_summary()
        test_summary = self._results_store.test_session_summary(test_code)
        if session_summary is None:
            return ["Saved locally."]

        lines = [
            "Saved locally. "
            f"Session: {session_summary.attempt_count} attempt"
            f"{'' if session_summary.attempt_count == 1 else 's'} across "
            f"{session_summary.unique_tests} test"
            f"{'' if session_summary.unique_tests == 1 else 's'}."
        ]

        test_line = self._format_test_session_summary_line(test_summary)
        if test_line is not None:
            lines.append(test_line)
        return lines

    @staticmethod
    def _format_test_session_summary_line(summary: TestSessionSummary | None) -> str | None:
        if summary is None or summary.attempt_count <= 1:
            return None

        best_ratio = summary.best_score_ratio
        best_accuracy = summary.best_accuracy
        if best_ratio is not None and (best_accuracy is None or abs(best_ratio - best_accuracy) > 1e-6):
            label = "best score"
            value = int(round(best_ratio * 100.0))
        elif best_accuracy is not None:
            label = "best accuracy"
            value = int(round(best_accuracy * 100.0))
        else:
            return f"This test: {summary.attempt_count} attempts this session."

        return (
            f"This test: {summary.attempt_count} attempts this session, "
            f"{label} {value}%."
        )


class PlaceholderScreen:
    def __init__(self, app: App, title: str) -> None:
        self._app = app
        self._title = title

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        surface.fill((10, 10, 14))
        title = self._app.font.render(self._title, True, (235, 235, 245))
        hint = self._app.font.render("Placeholder. Press Esc to go back.", True, (180, 180, 190))
        surface.blit(title, (40, 40))
        surface.blit(hint, (40, 100))


class LoadingScreen:
    def __init__(
        self,
        app: App,
        *,
        title: str,
        detail: str,
        target_factory: Callable[[], Screen],
        minimum_frames: int = 1,
    ) -> None:
        self._app = app
        self._title = str(title)
        self._detail = str(detail)
        self._target_factory = target_factory
        self._minimum_frames = max(1, int(minimum_frames))
        self._frames_rendered = 0
        self._load_started = False
        self._error_message: str | None = None
        self._title_font = pygame.font.Font(None, 44)
        self._body_font = pygame.font.Font(None, 28)
        self._hint_font = pygame.font.Font(None, 22)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        bg = (3, 9, 78)
        panel_bg = (8, 18, 104)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)

        surface.fill(bg)

        panel = pygame.Rect(
            max(24, w // 10),
            max(24, h // 8),
            min(max(420, int(w * 0.72)), w - 48),
            min(max(220, int(h * 0.42)), h - 48),
        )
        panel.center = (w // 2, h // 2)
        pygame.draw.rect(surface, panel_bg, panel, border_radius=12)
        pygame.draw.rect(surface, border, panel, 2, border_radius=12)

        title = self._title_font.render(self._title, True, text_main)
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 26)))

        status = "Loading" if self._error_message is None else "Load Failed"
        status_surf = self._body_font.render(status, True, text_main)
        surface.blit(status_surf, status_surf.get_rect(midtop=(panel.centerx, panel.y + 88)))

        if self._error_message is None:
            dot_count = (pygame.time.get_ticks() // 220) % 4
            detail_text = f"{self._detail}{'.' * dot_count}"
        else:
            detail_text = self._error_message
        detail = self._body_font.render(detail_text, True, text_muted)
        surface.blit(detail, detail.get_rect(midtop=(panel.centerx, panel.y + 128)))

        hint_text = (
            "Esc/Backspace: Cancel"
            if self._error_message is None
            else "Esc/Backspace: Back"
        )
        hint = self._hint_font.render(hint_text, True, text_muted)
        surface.blit(hint, hint.get_rect(midbottom=(panel.centerx, panel.bottom - 18)))

        self._frames_rendered += 1
        if self._load_started or self._error_message is not None:
            return
        if self._frames_rendered <= self._minimum_frames:
            return

        self._load_started = True
        try:
            target = self._target_factory()
        except Exception as exc:
            self._error_message = f"Unable to open screen ({exc.__class__.__name__})"
            self._load_started = False
            return
        self._app.replace_top(target)


class AntWorkoutScreen:
    def __init__(
        self,
        app: App,
        *,
        session: AntWorkoutSession,
        test_code: str,
    ) -> None:
        self._app = app
        self._session = session
        self._test_code = str(test_code).strip()
        self._results_persisted = False
        self._results_persistence_lines: list[str] = []
        self._input = ""
        self._runtime_screen: object | None = None
        self._runtime_engine_id: int | None = None
        self._pause_menu_active = False
        self._pause_menu_mode = "menu"
        self._pause_menu_selected = 0
        self._pause_menu_hitboxes: dict[int, pygame.Rect] = {}
        self._pause_settings_selected = 0
        self._pause_settings_hitboxes: dict[int, pygame.Rect] = {}
        self._pause_settings_control_hitboxes: dict[tuple[int, str], pygame.Rect] = {}
        self._pause_staged_level: int | None = None

        self._title_font = pygame.font.Font(None, 44)
        self._subtitle_font = pygame.font.Font(None, 28)
        self._body_font = pygame.font.Font(None, 26)
        self._small_font = pygame.font.Font(None, 22)
        self._tiny_font = pygame.font.Font(None, 18)
        self._input_font = pygame.font.Font(None, 42)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            if self._pause_menu_active:
                self._handle_pause_event(event)
            return

        snap = self._session.snapshot()
        stage = snap.stage
        key = event.key

        if key == pygame.K_ESCAPE:
            self._pause_menu_active = not self._pause_menu_active
            if self._pause_menu_active:
                self._pause_menu_mode = "menu"
                self._pause_menu_selected = 0
                self._pause_settings_selected = 0
                self._pause_staged_level = self._app.effective_difficulty_level(self._test_code)
            return

        if self._pause_menu_active:
            self._handle_pause_event(event)
            return

        if stage is AntWorkoutStage.BLOCK:
            engine = self._session.current_engine()
            if engine is None or getattr(engine, "phase", None) is Phase.RESULTS:
                return
            self._ensure_runtime_screen()
            runtime = cast(CognitiveTestScreen | None, self._runtime_screen)
            if runtime is not None:
                runtime.handle_event(event)
            return

        if stage is AntWorkoutStage.INTRO:
            if key == pygame.K_BACKSPACE:
                self._app.pop()
                return
            if key == pygame.K_LEFT:
                self._session.adjust_starting_level(-1)
                return
            if key == pygame.K_RIGHT:
                self._session.adjust_starting_level(1)
                return
            if key == pygame.K_RETURN:
                self._session.activate()
            return

        if stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            if key == pygame.K_BACKSPACE:
                self._session.backspace_text()
                return
            if key == pygame.K_RETURN:
                self._session.activate()
                return
            raw = getattr(event, "unicode", "") or ""
            if raw:
                self._session.append_text(raw)
            return

        if stage is AntWorkoutStage.BLOCK_SETUP:
            if key == pygame.K_LEFT:
                self._session.adjust_block_level(-1)
                return
            if key == pygame.K_RIGHT:
                self._session.adjust_block_level(1)
                return
            if key == pygame.K_RETURN:
                self._session.activate()
            return

        if stage is AntWorkoutStage.RESULTS and key == pygame.K_RETURN:
            self._app.pop()
            return
        if stage is AntWorkoutStage.RESULTS and key == pygame.K_BACKSPACE:
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        snap = self._session.snapshot()
        self._persist_results_if_needed(snap)

        if snap.stage is AntWorkoutStage.BLOCK:
            self._ensure_runtime_screen()
            runtime = cast(CognitiveTestScreen | None, self._runtime_screen)
            if runtime is not None:
                runtime.render(surface)
            self._session.sync_runtime()
            latest = self._session.snapshot()
            if latest.stage is AntWorkoutStage.BLOCK:
                self._render_block_status_overlay(surface, latest)
            else:
                snap = latest
            if self._pause_menu_active:
                self._render_pause_overlay(surface)
            return

        self._runtime_screen = None
        self._runtime_engine_id = None

        w, h = surface.get_size()
        bg = (5, 10, 56)
        panel_fill = (8, 18, 104)
        panel_border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        text_faint = (140, 154, 182)
        highlight_fill = (24, 46, 144)
        highlight_border = (238, 245, 255)
        input_fill = (14, 22, 70)

        surface.fill(bg)

        panel = pygame.Rect(
            max(18, w // 16),
            max(16, h // 18),
            min(max(560, int(w * 0.82)), w - 36),
            min(max(420, int(h * 0.84)), h - 32),
        )
        panel.center = (w // 2, h // 2)
        pygame.draw.rect(surface, panel_fill, panel, border_radius=14)
        pygame.draw.rect(surface, panel_border, panel, 2, border_radius=14)

        title = self._title_font.render(snap.title, True, text_main)
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 18)))

        subtitle = self._subtitle_font.render(snap.subtitle, True, text_muted)
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 58)))

        info_top = panel.y + 98
        status_left = panel.x + 20
        status_lines = [
            f"Block {snap.block_index}/{snap.block_total}",
            f"Attempted {snap.attempted_total}",
            f"Correct {snap.correct_total}",
            f"Fixation {snap.fixation_rate * 100.0:.1f}%",
        ]
        if snap.stage is AntWorkoutStage.INTRO:
            status_lines.append(f"Workout level {snap.difficulty_level}/10")
        if snap.stage is AntWorkoutStage.BLOCK_SETUP and snap.block_override_level is not None:
            status_lines.append(f"Block level {snap.block_override_level}/10")
        if snap.block_time_remaining_s is not None:
            status_lines.append(f"Block time {self._format_time(snap.block_time_remaining_s)}")
        if snap.workout_time_remaining_s is not None:
            status_lines.append(f"Workout time {self._format_time(snap.workout_time_remaining_s)}")
        if snap.current_block_label:
            status_lines.append(snap.current_block_label)

        y = info_top
        for line in status_lines:
            surf = self._small_font.render(line, True, text_muted)
            surface.blit(surf, (status_left, y))
            y += self._small_font.get_linesize() + 2

        prompt_rect = pygame.Rect(
            panel.x + 20,
            info_top + 20,
            panel.w - 40,
            int(panel.h * 0.32),
        )
        self._draw_text_panel(
            surface,
            rect=prompt_rect,
            title="Prompt",
            lines=self._wrap_multiline_text(snap.prompt, self._body_font, prompt_rect.w - 20),
            title_font=self._small_font,
            body_font=self._body_font,
            fill=(11, 24, 116),
            border=(176, 196, 240),
            title_color=text_main,
            body_color=text_main,
        )

        options_rect = pygame.Rect(
            panel.x + 20,
            prompt_rect.bottom + 14,
            panel.w - 40,
            int(panel.h * 0.22),
        )
        if snap.options:
            self._draw_options(
                surface,
                rect=options_rect,
                snap=snap,
                fill=(10, 20, 90),
                border=(150, 170, 220),
                text_main=text_main,
                text_muted=text_muted,
                highlight_fill=highlight_fill,
                highlight_border=highlight_border,
            )
        elif snap.stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            self._draw_input_box(
                surface,
                rect=options_rect,
                fill=input_fill,
                border=(162, 182, 226),
                text_main=text_main,
                text_muted=text_muted,
                entry_text=snap.text_value,
                hint=f"Type and press Enter ({len(snap.text_value)}/{snap.text_max_length})",
            )
        elif snap.stage is AntWorkoutStage.BLOCK_SETUP:
            self._draw_difficulty_box(
                surface,
                rect=options_rect,
                fill=input_fill,
                border=(162, 182, 226),
                text_main=text_main,
                text_muted=text_muted,
                default_level=snap.block_default_level or snap.difficulty_level,
                block_level=snap.block_override_level or snap.difficulty_level,
            )
        else:
            self._draw_input_box(
                surface,
                rect=options_rect,
                fill=input_fill,
                border=(162, 182, 226),
                text_main=text_main,
                text_muted=text_muted,
                entry_text="",
                hint="",
            )

        notes_rect = pygame.Rect(
            panel.x + 20,
            options_rect.bottom + 14,
            panel.w - 40,
            panel.bottom - (options_rect.bottom + 60),
        )
        note_lines = list(snap.note_lines)
        if snap.stage is AntWorkoutStage.INTRO:
            note_lines.insert(
                1,
                f"Difficulty source: {self._app.intro_difficulty_mode_label(self._test_code)}",
            )
        if snap.stage is AntWorkoutStage.RESULTS and self._results_persistence_lines:
            note_lines.extend(self._results_persistence_lines)
        self._draw_text_panel(
            surface,
            rect=notes_rect,
            title="Notes",
            lines=self._wrap_note_lines(note_lines, notes_rect.w - 20),
            title_font=self._small_font,
            body_font=self._small_font,
            fill=(8, 16, 78),
            border=(128, 148, 204),
            title_color=text_main,
            body_color=text_muted,
        )

        footer = self._footer_text(snap.stage)
        footer_surf = self._tiny_font.render(footer, True, text_faint)
        surface.blit(footer_surf, footer_surf.get_rect(midbottom=(panel.centerx, panel.bottom - 12)))

        if self._pause_menu_active:
            self._render_pause_overlay(surface)

    def _ensure_runtime_screen(self) -> None:
        engine = self._session.current_engine()
        if engine is None:
            self._runtime_screen = None
            self._runtime_engine_id = None
            return
        engine_id = id(engine)
        if self._runtime_engine_id == engine_id and self._runtime_screen is not None:
            return
        self._runtime_engine_id = engine_id
        runtime = CognitiveTestScreen(
            self._app,
            engine_factory=lambda engine=engine: engine,
            test_code=None,
        )
        runtime._pause_menu_active = False
        self._runtime_screen = runtime

    def _persist_results_if_needed(self, snap: AntWorkoutSnapshot) -> None:
        if snap.stage is not AntWorkoutStage.RESULTS or self._results_persisted:
            return
        self._results_persisted = True
        self._results_persistence_lines = self._app.persist_attempt(
            engine=self._session,
            test_code=self._test_code,
        )

    @staticmethod
    def _format_time(value_s: float | None) -> str:
        if value_s is None:
            return "--:--"
        total = max(0, int(round(float(value_s))))
        minutes, seconds = divmod(total, 60)
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def _wrap_multiline_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        lines: list[str] = []
        for chunk in str(text).splitlines() or [""]:
            wrapped = AntWorkoutScreen._wrap_text(chunk, font, max_width)
            lines.extend(wrapped if wrapped else [""])
        return lines or [""]

    def _wrap_note_lines(self, lines: list[str], max_width: int) -> list[str]:
        wrapped: list[str] = []
        for raw in lines:
            wrapped.extend(self._wrap_text(str(raw), self._small_font, max_width))
        return wrapped or [""]

    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        words = str(text).split()
        if not words:
            return [""]
        out: list[str] = []
        current = ""
        for word in words:
            trial = word if current == "" else f"{current} {word}"
            if font.size(trial)[0] <= max_width:
                current = trial
                continue
            if current:
                out.append(current)
            current = word
        if current:
            out.append(current)
        return out or [""]

    def _draw_text_panel(
        self,
        surface: pygame.Surface,
        *,
        rect: pygame.Rect,
        title: str,
        lines: list[str],
        title_font: pygame.font.Font,
        body_font: pygame.font.Font,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        title_color: tuple[int, int, int],
        body_color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, border, rect, 1, border_radius=10)
        title_surf = title_font.render(title, True, title_color)
        surface.blit(title_surf, (rect.x + 10, rect.y + 8))
        y = rect.y + 32
        line_h = body_font.get_linesize() + 1
        for line in lines:
            if y + line_h > rect.bottom - 8:
                break
            text = body_font.render(line, True, body_color)
            surface.blit(text, (rect.x + 10, y))
            y += line_h

    def _draw_options(
        self,
        surface: pygame.Surface,
        *,
        rect: pygame.Rect,
        snap: AntWorkoutSnapshot,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        highlight_fill: tuple[int, int, int],
        highlight_border: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, border, rect, 1, border_radius=10)
        title = self._small_font.render("Selection", True, text_main)
        surface.blit(title, (rect.x + 10, rect.y + 8))

        row_h = max(28, min(34, (rect.h - 40) // max(1, len(snap.options))))
        y = rect.y + 34
        for idx, option in enumerate(snap.options):
            row = pygame.Rect(rect.x + 8, y, rect.w - 16, row_h)
            selected = idx == snap.selected_index
            pygame.draw.rect(
                surface,
                highlight_fill if selected else fill,
                row,
                border_radius=7,
            )
            pygame.draw.rect(
                surface,
                highlight_border if selected else border,
                row,
                1,
                border_radius=7,
            )
            color = text_main if selected else text_muted
            label = self._body_font.render(option, True, color)
            surface.blit(label, label.get_rect(midleft=(row.x + 10, row.centery)))
            y += row_h + 4

    def _draw_input_box(
        self,
        surface: pygame.Surface,
        *,
        rect: pygame.Rect,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        entry_text: str,
        hint: str,
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, border, rect, 1, border_radius=10)
        title = self._small_font.render("Answer", True, text_main)
        surface.blit(title, (rect.x + 10, rect.y + 8))
        entry = entry_text if entry_text else "_"
        entry_surf = self._input_font.render(entry, True, text_main)
        surface.blit(entry_surf, (rect.x + 14, rect.y + 42))
        if hint:
            hint_surf = self._small_font.render(hint, True, text_muted)
            surface.blit(hint_surf, (rect.x + 14, rect.bottom - 28))

    def _draw_difficulty_box(
        self,
        surface: pygame.Surface,
        *,
        rect: pygame.Rect,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
        default_level: int,
        block_level: int,
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, border, rect, 1, border_radius=10)
        title = self._small_font.render("Block Difficulty", True, text_main)
        surface.blit(title, (rect.x + 10, rect.y + 8))
        value = self._input_font.render(f"{block_level}/10", True, text_main)
        surface.blit(value, (rect.x + 14, rect.y + 42))
        hint = self._small_font.render(
            f"Workout default {default_level}/10. Left/Right changes this block only.",
            True,
            text_muted,
        )
        surface.blit(hint, (rect.x + 14, rect.bottom - 28))

    @staticmethod
    def _footer_text(stage: AntWorkoutStage) -> str:
        if stage is AntWorkoutStage.INTRO:
            return "Left/Right: Workout level  Enter: Start reflection  Esc: Pause  Backspace: Back"
        if stage in (AntWorkoutStage.PRE_REFLECTION, AntWorkoutStage.POST_REFLECTION):
            return "Type response  Enter: Confirm  Esc: Pause"
        if stage is AntWorkoutStage.BLOCK_SETUP:
            return "Left/Right: Block level  Enter: Start block  Esc: Pause"
        return "Enter: Back  Esc: Pause  Backspace: Back"

    def _render_block_status_overlay(self, surface: pygame.Surface, snap: AntWorkoutSnapshot) -> None:
        bar = pygame.Rect(18, 16, min(surface.get_width() - 36, 720), 62)
        bar.centerx = surface.get_width() // 2
        panel = pygame.Surface(bar.size, pygame.SRCALPHA)
        panel.fill((6, 12, 32, 176))
        surface.blit(panel, bar.topleft)
        pygame.draw.rect(surface, (226, 236, 255), bar, 1, border_radius=10)

        lines = (
            f"{snap.title}",
            (
                f"Block {snap.block_index}/{snap.block_total}: {snap.current_block_label}  "
                f"Level {snap.difficulty_level}/10  "
                f"Workout time {self._format_time(snap.workout_time_remaining_s)}"
            ),
        )
        surface.blit(self._small_font.render(lines[0], True, (238, 245, 255)), (bar.x + 12, bar.y + 10))
        surface.blit(self._tiny_font.render(lines[1], True, (188, 204, 228)), (bar.x + 12, bar.y + 34))

    def _handle_pause_event(self, event: pygame.event.Event) -> None:
        if self._pause_menu_mode == "settings":
            self._handle_pause_settings_event(event)
            return
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._pause_menu_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._pause_menu_selected = idx
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for idx, rect in self._pause_menu_hitboxes.items():
                if rect.collidepoint(pos):
                    self._pause_menu_selected = idx
                    self._activate_pause_selection()
                    return
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._pause_menu_active = False
            self._pause_menu_mode = "menu"
            return
        option_count = len(self._pause_menu_options())
        if event.key in (pygame.K_UP, pygame.K_w):
            self._pause_menu_selected = (self._pause_menu_selected - 1) % option_count
            return
        if event.key in (pygame.K_DOWN, pygame.K_s):
            self._pause_menu_selected = (self._pause_menu_selected + 1) % option_count
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate_pause_selection()

    def _activate_pause_selection(self) -> None:
        selected = self._pause_menu_selected % len(self._pause_menu_options())
        if selected == 0:
            self._pause_menu_active = False
            self._pause_menu_mode = "menu"
            return
        if selected == 1:
            self._pause_menu_mode = "settings"
            self._pause_settings_selected = 0
            return
        self._pause_menu_active = False
        self._pause_menu_mode = "menu"
        self._app.pop_to_root()

    @staticmethod
    def _pause_menu_options() -> tuple[str, ...]:
        return ("Resume", "Settings", "Main Menu")

    def _pause_settings_rows(self) -> list[tuple[str, str, str]]:
        level = self._pause_staged_level or self._app.effective_difficulty_level(self._test_code)
        return [
            ("difficulty", "Workout Difficulty", f"{level} / 10"),
            ("apply_restart", "Apply & Restart", "Restart from the beginning at this difficulty"),
            ("back", "Back", "Return to Pause Menu"),
        ]

    def _handle_pause_settings_event(self, event: pygame.event.Event) -> None:
        rows = self._pause_settings_rows()
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._pause_settings_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._pause_settings_selected = idx
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for (idx, action), rect in self._pause_settings_control_hitboxes.items():
                if rect.collidepoint(pos):
                    self._pause_settings_selected = idx
                    self._adjust_pause_setting(index=idx, direction=-1 if action == "dec" else 1)
                    return
            for idx, rect in self._pause_settings_hitboxes.items():
                if not rect.collidepoint(pos):
                    continue
                self._pause_settings_selected = idx
                key = rows[idx][0]
                if key == "apply_restart":
                    self._apply_pause_restart()
                elif key == "back":
                    self._pause_menu_mode = "menu"
                return
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._pause_menu_mode = "menu"
            self._pause_settings_selected = 0
            return
        row_count = len(rows)
        if event.key in (pygame.K_UP, pygame.K_w):
            self._pause_settings_selected = (self._pause_settings_selected - 1) % row_count
            return
        if event.key in (pygame.K_DOWN, pygame.K_s):
            self._pause_settings_selected = (self._pause_settings_selected + 1) % row_count
            return
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self._adjust_pause_setting(index=self._pause_settings_selected, direction=-1)
            return
        if event.key in (pygame.K_RIGHT, pygame.K_d):
            self._adjust_pause_setting(index=self._pause_settings_selected, direction=1)
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            key = rows[self._pause_settings_selected][0]
            if key == "apply_restart":
                self._apply_pause_restart()
            elif key == "back":
                self._pause_menu_mode = "menu"

    def _adjust_pause_setting(self, *, index: int, direction: int) -> None:
        rows = self._pause_settings_rows()
        key = rows[index % len(rows)][0]
        if key != "difficulty":
            return
        current = self._pause_staged_level or self._app.effective_difficulty_level(self._test_code)
        self._pause_staged_level = max(1, min(10, current + int(direction)))

    def _apply_pause_restart(self) -> None:
        level = self._pause_staged_level or self._app.effective_difficulty_level(self._test_code)
        self._app.set_persistent_difficulty_level(test_code=self._test_code, level=level)
        self._pause_menu_active = False
        self._pause_menu_mode = "menu"
        self._pause_staged_level = None
        self._app.replace_top(
            AntWorkoutScreen(
                self._app,
                session=AntWorkoutSession(
                    clock=self._session._clock,
                    seed=self._session.seed,
                    plan=build_ant_workout_plan(self._test_code),
                    starting_level=self._app.effective_difficulty_level(self._test_code),
                ),
                test_code=self._test_code,
            )
        )

    def _render_pause_overlay(self, surface: pygame.Surface) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((4, 8, 18, 182))
        surface.blit(dim, (0, 0))
        if self._pause_menu_mode == "settings":
            self._render_pause_settings_overlay(surface)
            return

        panel = pygame.Rect((surface.get_width() - 420) // 2, (surface.get_height() - 280) // 2, 420, 280)
        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=10)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=10)
        title = self._app.font.render("Paused", True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 16)))
        subtitle = self._tiny_font.render("Resume / Settings / Main Menu", True, (188, 204, 228))
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 56)))

        options = self._pause_menu_options()
        row_h = 42
        gap = 12
        y = panel.y + 96
        self._pause_menu_hitboxes = {}
        for idx, label in enumerate(options):
            row = pygame.Rect(panel.x + 28, y, panel.w - 56, row_h)
            self._pause_menu_hitboxes[idx] = row.copy()
            selected = idx == (self._pause_menu_selected % len(options))
            pygame.draw.rect(
                surface,
                (244, 248, 255) if selected else (9, 20, 106),
                row,
                border_radius=6,
            )
            pygame.draw.rect(
                surface,
                (120, 142, 196) if selected else (62, 84, 152),
                row,
                2 if selected else 1,
                border_radius=6,
            )
            text = self._small_font.render(
                label,
                True,
                (14, 26, 74) if selected else (238, 245, 255),
            )
            surface.blit(text, text.get_rect(center=row.center))
            y += row_h + gap

    def _render_pause_settings_overlay(self, surface: pygame.Surface) -> None:
        rows = self._pause_settings_rows()
        panel = pygame.Rect((surface.get_width() - 640) // 2, (surface.get_height() - 320) // 2, 640, 320)
        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=10)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=10)
        title = self._app.font.render("Workout Settings", True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 16)))
        subtitle = self._tiny_font.render(
            "Left/Right adjusts staged difficulty. Apply restarts the workout.",
            True,
            (188, 204, 228),
        )
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 56)))

        self._pause_settings_hitboxes = {}
        self._pause_settings_control_hitboxes = {}
        y = panel.y + 92
        selected_index = self._pause_settings_selected % max(1, len(rows))
        for idx, (key, label, value) in enumerate(rows):
            row = pygame.Rect(panel.x + 24, y, panel.w - 48, 48)
            self._pause_settings_hitboxes[idx] = row.copy()
            selected = idx == selected_index
            pygame.draw.rect(surface, (242, 246, 255) if selected else (9, 20, 106), row, border_radius=6)
            pygame.draw.rect(surface, (120, 142, 196) if selected else (62, 84, 152), row, 2 if selected else 1, border_radius=6)
            label_surf = self._small_font.render(label, True, (14, 26, 74) if selected else (238, 245, 255))
            surface.blit(label_surf, (row.x + 14, row.y + 8))
            if key == "difficulty":
                value_box = pygame.Rect(row.right - 192, row.y + 7, 178, row.h - 14)
                dec_box = pygame.Rect(value_box.x, value_box.y, 34, value_box.h)
                inc_box = pygame.Rect(value_box.right - 34, value_box.y, 34, value_box.h)
                value_mid = pygame.Rect(dec_box.right + 6, value_box.y, value_box.w - 80, value_box.h)
                self._pause_settings_control_hitboxes[(idx, "dec")] = dec_box.copy()
                self._pause_settings_control_hitboxes[(idx, "inc")] = inc_box.copy()
                for box, glyph in ((dec_box, "<"), (inc_box, ">")):
                    pygame.draw.rect(surface, (20, 32, 92), box, border_radius=5)
                    pygame.draw.rect(surface, (110, 130, 184), box, 1, border_radius=5)
                    surface.blit(self._small_font.render(glyph, True, (238, 245, 255)), self._small_font.render(glyph, True, (238, 245, 255)).get_rect(center=box.center))
                pygame.draw.rect(surface, (14, 26, 78), value_mid, border_radius=5)
                pygame.draw.rect(surface, (92, 112, 168), value_mid, 1, border_radius=5)
                value_surf = self._tiny_font.render(value, True, (238, 245, 255))
                surface.blit(value_surf, value_surf.get_rect(center=value_mid.center))
            else:
                value_surf = self._tiny_font.render(value, True, (46, 62, 112) if selected else (198, 212, 242))
                surface.blit(value_surf, value_surf.get_rect(midright=(row.right - 16, row.centery)))
            y += 58


INPUT_PROFILE_STORE_ENV = "CFAST_INPUT_PROFILES_PATH"


@dataclass(slots=True)
class AxisCalibrationSettings:
    min_raw: float = -1.0
    max_raw: float = 1.0
    deadzone: float = 0.05
    invert: bool = False
    curve: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_raw": float(self.min_raw),
            "max_raw": float(self.max_raw),
            "deadzone": float(self.deadzone),
            "invert": bool(self.invert),
            "curve": float(self.curve),
        }

    @classmethod
    def from_dict(cls, data: object) -> AxisCalibrationSettings:
        if not isinstance(data, dict):
            return cls()
        min_raw = _clamp(_as_float(data.get("min_raw"), -1.0), -1.0, 0.0)
        max_raw = _clamp(_as_float(data.get("max_raw"), 1.0), 0.0, 1.0)
        if min_raw >= -0.001:
            min_raw = -1.0
        if max_raw <= 0.001:
            max_raw = 1.0
        return cls(
            min_raw=min_raw,
            max_raw=max_raw,
            deadzone=_clamp(_as_float(data.get("deadzone"), 0.05), 0.0, 0.45),
            invert=bool(data.get("invert", False)),
            curve=_clamp(_as_float(data.get("curve"), 1.0), 0.4, 2.6),
        )


@dataclass(slots=True)
class InputProfile:
    profile_id: str
    name: str
    axis_calibrations: dict[str, AxisCalibrationSettings]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "name": self.name,
            "axis_calibrations": {
                axis_key: settings.to_dict()
                for axis_key, settings in self.axis_calibrations.items()
            },
        }


@dataclass(slots=True)
class DifficultySettingsState:
    global_override_enabled: bool = False
    global_level: int = DEFAULT_DIFFICULTY_LEVEL
    per_test_levels: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_override_enabled": bool(self.global_override_enabled),
            "global_level": int(self.global_level),
            "per_test_levels": {
                str(test_code): int(level)
                for test_code, level in self.per_test_levels.items()
            },
        }


class DifficultySettingsStore:
    _version = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state = DifficultySettingsState()
        self._load()

    @classmethod
    def default_path(cls) -> Path:
        explicit = os.environ.get(DIFFICULTY_SETTINGS_STORE_ENV)
        if explicit:
            return Path(explicit).expanduser()
        return Path.home() / ".rcaf_cfast_difficulty_settings.json"

    @staticmethod
    def _clamp_level(value: object) -> int:
        try:
            level = int(value)
        except Exception:
            level = DEFAULT_DIFFICULTY_LEVEL
        return max(1, min(10, level))

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        self._state.global_override_enabled = bool(payload.get("global_override_enabled", False))
        self._state.global_level = self._clamp_level(payload.get("global_level", DEFAULT_DIFFICULTY_LEVEL))

        raw_levels = payload.get("per_test_levels")
        if isinstance(raw_levels, dict):
            cleaned: dict[str, int] = {}
            for test_code, level in raw_levels.items():
                cleaned[str(test_code)] = self._clamp_level(level)
            self._state.per_test_levels = cleaned

    def save(self) -> None:
        payload = {
            "version": self._version,
            **self._state.to_dict(),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._path)
        except Exception:
            return

    def global_override_enabled(self) -> bool:
        return bool(self._state.global_override_enabled)

    def set_global_override_enabled(self, enabled: bool) -> None:
        self._state.global_override_enabled = bool(enabled)
        self.save()

    def global_level(self) -> int:
        return int(self._state.global_level)

    def set_global_level(self, level: int) -> None:
        self._state.global_level = self._clamp_level(level)
        self.save()

    def test_level(self, test_code: str) -> int:
        return int(
            self._state.per_test_levels.get(str(test_code), DEFAULT_DIFFICULTY_LEVEL)
        )

    def set_test_level(self, *, test_code: str, level: int) -> None:
        self._state.per_test_levels[str(test_code)] = self._clamp_level(level)
        self.save()

    def effective_level(self, test_code: str) -> int:
        if self.global_override_enabled():
            return self.global_level()
        return self.test_level(test_code)

    def effective_ratio(self, test_code: str) -> float:
        level = self.effective_level(test_code)
        return (max(1, min(10, level)) - 1) / 9.0

    def intro_mode_label(self, test_code: str) -> str:
        _ = test_code
        return "Global Override" if self.global_override_enabled() else "This Test"


class InputProfilesStore:
    _version = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._profiles: list[InputProfile] = [InputProfile("default", "Default", {})]
        self._active_profile_id = "default"
        self._load()

    @classmethod
    def default_path(cls) -> Path:
        explicit = os.environ.get(INPUT_PROFILE_STORE_ENV)
        if explicit:
            return Path(explicit).expanduser()
        return Path.home() / ".rcaf_cfast_input_profiles.json"

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list):
            return

        loaded_profiles: list[InputProfile] = []
        for item in raw_profiles:
            if not isinstance(item, dict):
                continue
            profile_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            if profile_id == "":
                continue
            if name == "":
                name = profile_id
            raw_axes = item.get("axis_calibrations")
            axis_calibrations: dict[str, AxisCalibrationSettings] = {}
            if isinstance(raw_axes, dict):
                for axis_key, settings in raw_axes.items():
                    axis_calibrations[str(axis_key)] = AxisCalibrationSettings.from_dict(settings)
            loaded_profiles.append(
                InputProfile(
                    profile_id=profile_id,
                    name=name,
                    axis_calibrations=axis_calibrations,
                )
            )

        if loaded_profiles:
            self._profiles = loaded_profiles
        raw_active = str(payload.get("active_profile_id", "")).strip()
        if raw_active != "" and any(p.profile_id == raw_active for p in self._profiles):
            self._active_profile_id = raw_active
        else:
            self._active_profile_id = self._profiles[0].profile_id

    def save(self) -> None:
        payload = {
            "version": self._version,
            "active_profile_id": self._active_profile_id,
            "profiles": [profile.to_dict() for profile in self._profiles],
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(f"{self._path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self._path)
        except Exception:
            return

    def profiles(self) -> list[InputProfile]:
        return list(self._profiles)

    def active_profile(self) -> InputProfile:
        for profile in self._profiles:
            if profile.profile_id == self._active_profile_id:
                return profile
        self._active_profile_id = self._profiles[0].profile_id
        return self._profiles[0]

    def set_active_profile(self, profile_id: str) -> None:
        if any(profile.profile_id == profile_id for profile in self._profiles):
            self._active_profile_id = profile_id
            self.save()

    def create_profile(self, *, name: str, copy_from: InputProfile | None) -> InputProfile:
        candidate = str(name).strip()
        if candidate == "":
            candidate = "Profile"
        profile_id = self._next_profile_id(candidate)
        copied: dict[str, AxisCalibrationSettings] = {}
        if copy_from is not None:
            copied = {
                axis_key: AxisCalibrationSettings.from_dict(settings.to_dict())
                for axis_key, settings in copy_from.axis_calibrations.items()
            }
        profile = InputProfile(profile_id=profile_id, name=candidate, axis_calibrations=copied)
        self._profiles.append(profile)
        self._active_profile_id = profile.profile_id
        self.save()
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        if len(self._profiles) <= 1:
            return False
        idx = next((i for i, p in enumerate(self._profiles) if p.profile_id == profile_id), -1)
        if idx < 0:
            return False
        del self._profiles[idx]
        if not any(p.profile_id == self._active_profile_id for p in self._profiles):
            self._active_profile_id = self._profiles[max(0, idx - 1)].profile_id
        self.save()
        return True

    def rename_profile(self, profile_id: str, new_name: str) -> bool:
        cleaned = str(new_name).strip()
        if cleaned == "":
            return False
        for profile in self._profiles:
            if profile.profile_id == profile_id:
                profile.name = cleaned
                self.save()
                return True
        return False

    def get_axis_calibration(
        self,
        *,
        profile_id: str,
        axis_key: str,
    ) -> AxisCalibrationSettings:
        for profile in self._profiles:
            if profile.profile_id != profile_id:
                continue
            found = profile.axis_calibrations.get(axis_key)
            if found is None:
                found = AxisCalibrationSettings()
                profile.axis_calibrations[axis_key] = found
            return found
        return AxisCalibrationSettings()

    def set_axis_calibration(
        self,
        *,
        profile_id: str,
        axis_key: str,
        settings: AxisCalibrationSettings,
    ) -> None:
        for profile in self._profiles:
            if profile.profile_id == profile_id:
                profile.axis_calibrations[axis_key] = AxisCalibrationSettings.from_dict(
                    settings.to_dict()
                )
                self.save()
                return

    def _next_profile_id(self, base_name: str) -> str:
        base = "".join(ch.lower() if ch.isalnum() else "_" for ch in base_name).strip("_")
        if base == "":
            base = "profile"
        candidate = base
        suffix = 2
        existing = {profile.profile_id for profile in self._profiles}
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate


def _as_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    if value <= lo:
        return lo
    if value >= hi:
        return hi
    return float(value)


def _iter_connected_joysticks() -> list[pygame.joystick.Joystick]:
    joysticks: list[pygame.joystick.Joystick] = []
    try:
        count = int(pygame.joystick.get_count())
    except Exception:
        return joysticks
    for idx in range(count):
        try:
            js = pygame.joystick.Joystick(idx)
            if not js.get_init():
                js.init()
            joysticks.append(js)
        except Exception:
            continue
    return joysticks


def _joystick_guid(joystick: pygame.joystick.Joystick) -> str:
    getter = getattr(joystick, "get_guid", None)
    if callable(getter):
        try:
            value = str(getter()).strip()
            if value != "":
                return value
        except Exception:
            pass
    return ""


def _joystick_key(joystick: pygame.joystick.Joystick) -> str:
    name = str(joystick.get_name())
    guid = _joystick_guid(joystick)
    return f"{name}|{guid}" if guid != "" else name


def _axis_key(joystick: pygame.joystick.Joystick, axis_index: int) -> str:
    return f"{_joystick_key(joystick)}::axis{int(axis_index)}"


def _axis_raw_value(joystick: pygame.joystick.Joystick, axis_index: int) -> float:
    try:
        return _clamp(float(joystick.get_axis(axis_index)), -1.0, 1.0)
    except Exception:
        return 0.0


def _apply_axis_calibration(raw: float, settings: AxisCalibrationSettings) -> float:
    value = _clamp(float(raw), -1.0, 1.0)
    if settings.invert:
        value = -value

    pos_span = max(0.001, float(settings.max_raw))
    neg_span = max(0.001, abs(float(settings.min_raw)))
    normalized = value / (pos_span if value >= 0.0 else neg_span)
    normalized = _clamp(normalized, -1.0, 1.0)

    deadzone = _clamp(float(settings.deadzone), 0.0, 0.45)
    magnitude = abs(normalized)
    if magnitude <= deadzone:
        normalized = 0.0
    else:
        scaled = (magnitude - deadzone) / max(0.001, 1.0 - deadzone)
        normalized = math.copysign(scaled, normalized)

    curve = _clamp(float(settings.curve), 0.4, 2.6)
    curved = math.copysign(abs(normalized) ** curve, normalized)
    return _clamp(curved, -1.0, 1.0)


def _draw_axis_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    *,
    fill_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
) -> None:
    pygame.draw.rect(surface, (7, 14, 95), rect)
    pygame.draw.rect(surface, line_color, rect, 1)
    center_x = rect.x + rect.w // 2
    pygame.draw.line(surface, line_color, (center_x, rect.y + 2), (center_x, rect.bottom - 2), 1)

    value = _clamp(value, -1.0, 1.0)
    if abs(value) < 0.001:
        return
    if value > 0.0:
        width = int((rect.w // 2 - 2) * value)
        fill = pygame.Rect(center_x, rect.y + 2, max(1, width), rect.h - 4)
    else:
        width = int((rect.w // 2 - 2) * abs(value))
        fill = pygame.Rect(center_x - width, rect.y + 2, max(1, width), rect.h - 4)
    pygame.draw.rect(surface, fill_color, fill)


class AxisCalibrationScreen:
    def __init__(self, app: App, *, profiles: InputProfilesStore) -> None:
        self._app = app
        self._profiles = profiles
        self._title_font = pygame.font.Font(None, 40)
        self._small_font = pygame.font.Font(None, 25)
        self._tiny_font = pygame.font.Font(None, 20)
        self._device_index = 0
        self._axis_index = 0
        self._capturing = False
        self._message = ""

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        key = int(event.key)
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()
            return

        joysticks = _iter_connected_joysticks()
        if not joysticks:
            self._message = "No joystick detected."
            return
        self._device_index %= len(joysticks)
        joystick = joysticks[self._device_index]
        axis_count = max(1, int(joystick.get_numaxes()))
        self._axis_index %= axis_count

        if key == pygame.K_LEFT:
            self._device_index = (self._device_index - 1) % len(joysticks)
            self._axis_index = 0
            self._capturing = False
            return
        if key == pygame.K_RIGHT:
            self._device_index = (self._device_index + 1) % len(joysticks)
            self._axis_index = 0
            self._capturing = False
            return
        if key == pygame.K_UP:
            self._axis_index = (self._axis_index - 1) % axis_count
            self._capturing = False
            return
        if key == pygame.K_DOWN:
            self._axis_index = (self._axis_index + 1) % axis_count
            self._capturing = False
            return

        axis_key = _axis_key(joystick, self._axis_index)
        profile = self._profiles.active_profile()
        settings = self._profiles.get_axis_calibration(
            profile_id=profile.profile_id, axis_key=axis_key
        )

        if key == pygame.K_SPACE:
            if not self._capturing:
                current = _axis_raw_value(joystick, self._axis_index)
                settings.min_raw = min(-0.001, current)
                settings.max_raw = max(0.001, current)
                self._capturing = True
                self._message = (
                    "Capture started. Move axis through full range, then press Space again."
                )
            else:
                self._capturing = False
                self._profiles.set_axis_calibration(
                    profile_id=profile.profile_id,
                    axis_key=axis_key,
                    settings=settings,
                )
                self._message = "Capture complete. Calibration saved."
            return

        if key == pygame.K_i:
            settings.invert = not settings.invert
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=settings,
            )
            self._message = "Invert toggled."
            return

        if key == pygame.K_r:
            self._capturing = False
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=AxisCalibrationSettings(),
            )
            self._message = "Axis calibration reset."
            return

        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            settings.deadzone = _clamp(settings.deadzone - 0.01, 0.0, 0.45)
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=settings,
            )
            return
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            settings.deadzone = _clamp(settings.deadzone + 0.01, 0.0, 0.45)
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=settings,
            )
            return
        if key == pygame.K_LEFTBRACKET:
            settings.curve = _clamp(settings.curve - 0.05, 0.4, 2.6)
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=settings,
            )
            return
        if key == pygame.K_RIGHTBRACKET:
            settings.curve = _clamp(settings.curve + 0.05, 0.4, 2.6)
            self._profiles.set_axis_calibration(
                profile_id=profile.profile_id,
                axis_key=axis_key,
                settings=settings,
            )
            return

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill((3, 9, 78))
        frame = pygame.Rect(12, 12, w - 24, h - 24)
        pygame.draw.rect(surface, (8, 18, 104), frame)
        pygame.draw.rect(surface, (226, 236, 255), frame, 2)

        profile = self._profiles.active_profile()
        title = self._title_font.render("Axis Calibration", True, (238, 245, 255))
        surface.blit(title, (30, 26))
        subtitle = self._small_font.render(f"Profile: {profile.name}", True, (188, 204, 228))
        surface.blit(subtitle, (32, 66))

        joysticks = _iter_connected_joysticks()
        if not joysticks:
            msg = self._small_font.render(
                "No joystick detected. Connect device and reopen this screen.",
                True,
                (238, 245, 255),
            )
            surface.blit(msg, (32, 120))
            footer = self._tiny_font.render("Esc/Backspace: Back", True, (188, 204, 228))
            surface.blit(footer, (32, frame.bottom - 30))
            return

        self._device_index %= len(joysticks)
        joystick = joysticks[self._device_index]
        axis_count = max(1, int(joystick.get_numaxes()))
        self._axis_index %= axis_count
        axis_key = _axis_key(joystick, self._axis_index)
        settings = self._profiles.get_axis_calibration(
            profile_id=profile.profile_id, axis_key=axis_key
        )
        raw = _axis_raw_value(joystick, self._axis_index)
        calibrated = _apply_axis_calibration(raw, settings)

        if self._capturing:
            settings.min_raw = min(settings.min_raw, raw)
            settings.max_raw = max(settings.max_raw, raw)

        device_text = self._small_font.render(
            f"Device {self._device_index + 1}/{len(joysticks)}: {joystick.get_name()}",
            True,
            (238, 245, 255),
        )
        surface.blit(device_text, (32, 112))

        axis_text = self._small_font.render(
            f"Axis {self._axis_index + 1}/{axis_count}",
            True,
            (238, 245, 255),
        )
        surface.blit(axis_text, (32, 145))

        raw_rect = pygame.Rect(32, 178, frame.w - 64, 24)
        cal_rect = pygame.Rect(32, 214, frame.w - 64, 24)
        _draw_axis_bar(
            surface,
            raw_rect,
            raw,
            fill_color=(116, 208, 255),
            line_color=(98, 130, 190),
        )
        _draw_axis_bar(
            surface,
            cal_rect,
            calibrated,
            fill_color=(145, 232, 178),
            line_color=(98, 130, 190),
        )
        raw_label = self._tiny_font.render(f"Raw: {raw:+.3f}", True, (188, 204, 228))
        cal_label = self._tiny_font.render(f"Calibrated: {calibrated:+.3f}", True, (188, 204, 228))
        surface.blit(raw_label, (raw_rect.right - raw_label.get_width(), raw_rect.y - 18))
        surface.blit(cal_label, (cal_rect.right - cal_label.get_width(), cal_rect.y - 18))

        rows = [
            f"min_raw: {settings.min_raw:+.3f}",
            f"max_raw: {settings.max_raw:+.3f}",
            f"deadzone: {settings.deadzone:.2f}",
            f"invert: {'ON' if settings.invert else 'OFF'}",
            f"curve: {settings.curve:.2f}",
            f"capture: {'ACTIVE' if self._capturing else 'idle'}",
        ]
        y = cal_rect.bottom + 20
        for row in rows:
            text = self._small_font.render(row, True, (238, 245, 255))
            surface.blit(text, (32, y))
            y += 28

        if self._message != "":
            note = self._tiny_font.render(self._message, True, (188, 204, 228))
            surface.blit(note, (32, min(frame.bottom - 54, y + 8)))

        footer = self._tiny_font.render(
            "Left/Right: Device  Up/Down: Axis  Space: Capture  I: Invert  "
            "+/-: Deadzone  [/]: Curve  R: Reset  Esc: Back",
            True,
            (188, 204, 228),
        )
        surface.blit(footer, (32, frame.bottom - 28))


class AxisVisualizerScreen:
    def __init__(self, app: App, *, profiles: InputProfilesStore) -> None:
        self._app = app
        self._profiles = profiles
        self._title_font = pygame.font.Font(None, 40)
        self._small_font = pygame.font.Font(None, 24)
        self._tiny_font = pygame.font.Font(None, 19)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill((3, 9, 78))
        frame = pygame.Rect(12, 12, w - 24, h - 24)
        pygame.draw.rect(surface, (8, 18, 104), frame)
        pygame.draw.rect(surface, (226, 236, 255), frame, 2)

        profile = self._profiles.active_profile()
        title = self._title_font.render("Axis Visualizer", True, (238, 245, 255))
        surface.blit(title, (30, 26))
        subtitle = self._small_font.render(f"Profile: {profile.name}", True, (188, 204, 228))
        surface.blit(subtitle, (32, 66))

        joysticks = _iter_connected_joysticks()
        if not joysticks:
            msg = self._small_font.render("No joystick detected.", True, (238, 245, 255))
            surface.blit(msg, (32, 120))
            footer = self._tiny_font.render("Esc/Backspace: Back", True, (188, 204, 228))
            surface.blit(footer, (32, frame.bottom - 30))
            return

        max_visible_rows = max(1, (frame.h - 170) // 28)
        y = 108
        drawn_rows = 0
        for joystick_index, joystick in enumerate(joysticks, start=1):
            if drawn_rows >= max_visible_rows:
                break
            axis_count = max(0, int(joystick.get_numaxes()))
            button_count = max(0, int(joystick.get_numbuttons()))
            pressed_buttons = sum(1 for i in range(button_count) if bool(joystick.get_button(i)))
            header = self._small_font.render(
                f"[{joystick_index}] {joystick.get_name()}  axes:{axis_count}  "
                f"buttons:{pressed_buttons}/{button_count}",
                True,
                (238, 245, 255),
            )
            surface.blit(header, (32, y))
            y += 26
            drawn_rows += 1

            for axis_idx in range(axis_count):
                if drawn_rows >= max_visible_rows:
                    break
                raw = _axis_raw_value(joystick, axis_idx)
                axis_key = _axis_key(joystick, axis_idx)
                settings = self._profiles.get_axis_calibration(
                    profile_id=profile.profile_id,
                    axis_key=axis_key,
                )
                calibrated = _apply_axis_calibration(raw, settings)

                label = self._tiny_font.render(f"A{axis_idx:02d}", True, (188, 204, 228))
                surface.blit(label, (36, y + 6))

                raw_rect = pygame.Rect(76, y + 2, (frame.w - 130) // 2, 16)
                cal_rect = pygame.Rect(raw_rect.right + 10, y + 2, (frame.w - 130) // 2, 16)
                _draw_axis_bar(
                    surface,
                    raw_rect,
                    raw,
                    fill_color=(116, 208, 255),
                    line_color=(98, 130, 190),
                )
                _draw_axis_bar(
                    surface,
                    cal_rect,
                    calibrated,
                    fill_color=(145, 232, 178),
                    line_color=(98, 130, 190),
                )
                raw_text = self._tiny_font.render(f"{raw:+.2f}", True, (188, 204, 228))
                cal_text = self._tiny_font.render(f"{calibrated:+.2f}", True, (188, 204, 228))
                surface.blit(
                    raw_text, raw_text.get_rect(midtop=(raw_rect.centerx, raw_rect.bottom + 2))
                )
                surface.blit(
                    cal_text, cal_text.get_rect(midtop=(cal_rect.centerx, cal_rect.bottom + 2))
                )

                y += 28
                drawn_rows += 1

        footer = self._tiny_font.render(
            "Live raw + calibrated axis values. Esc/Backspace: Back", True, (188, 204, 228)
        )
        surface.blit(footer, (32, frame.bottom - 28))


class InputProfilesScreen:
    def __init__(self, app: App, *, profiles: InputProfilesStore) -> None:
        self._app = app
        self._profiles = profiles
        self._title_font = pygame.font.Font(None, 40)
        self._small_font = pygame.font.Font(None, 28)
        self._tiny_font = pygame.font.Font(None, 21)
        self._selected_index = 0
        self._renaming = False
        self._rename_buffer = ""
        self._message = ""
        self._row_hitboxes: dict[int, pygame.Rect] = {}

    def handle_event(self, event: pygame.event.Event) -> None:
        profiles = self._profiles.profiles()
        if not profiles:
            return
        self._selected_index %= len(profiles)
        selected = profiles[self._selected_index]

        if not self._renaming and event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._row_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._selected_index = idx
                        break
            return

        if not self._renaming and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for idx, rect in self._row_hitboxes.items():
                if rect.collidepoint(pos):
                    self._selected_index = idx
                    profile = profiles[idx]
                    self._profiles.set_active_profile(profile.profile_id)
                    self._message = f"Active profile set: {profile.name}"
                    return
            return

        if event.type != pygame.KEYDOWN:
            return
        key = int(event.key)
        selected = profiles[self._selected_index]

        if self._renaming:
            if key == pygame.K_ESCAPE:
                self._renaming = False
                self._rename_buffer = ""
                return
            if key == pygame.K_BACKSPACE:
                self._rename_buffer = self._rename_buffer[:-1]
                return
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self._profiles.rename_profile(selected.profile_id, self._rename_buffer):
                    self._message = "Profile renamed."
                self._renaming = False
                self._rename_buffer = ""
                return
            ch = event.unicode
            if ch and ch.isprintable() and len(self._rename_buffer) < 24:
                self._rename_buffer += ch
            return

        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()
            return
        if key == pygame.K_UP:
            self._selected_index = (self._selected_index - 1) % len(profiles)
            return
        if key == pygame.K_DOWN:
            self._selected_index = (self._selected_index + 1) % len(profiles)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._profiles.set_active_profile(selected.profile_id)
            self._message = f"Active profile set: {selected.name}"
            return
        if key == pygame.K_n:
            name = f"Profile {len(profiles) + 1}"
            created = self._profiles.create_profile(
                name=name, copy_from=self._profiles.active_profile()
            )
            new_profiles = self._profiles.profiles()
            self._selected_index = next(
                i
                for i, profile in enumerate(new_profiles)
                if profile.profile_id == created.profile_id
            )
            self._message = f"Created {created.name}."
            return
        if key == pygame.K_c:
            created = self._profiles.create_profile(
                name=f"{selected.name} Copy",
                copy_from=selected,
            )
            new_profiles = self._profiles.profiles()
            self._selected_index = next(
                i
                for i, profile in enumerate(new_profiles)
                if profile.profile_id == created.profile_id
            )
            self._message = f"Copied {selected.name}."
            return
        if key == pygame.K_d:
            deleted = self._profiles.delete_profile(selected.profile_id)
            profiles = self._profiles.profiles()
            if deleted:
                self._selected_index = min(self._selected_index, len(profiles) - 1)
                self._message = "Profile deleted."
            else:
                self._message = "Cannot delete the last profile."
            return
        if key == pygame.K_r:
            self._renaming = True
            self._rename_buffer = selected.name

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        surface.fill((3, 9, 78))
        frame = pygame.Rect(12, 12, w - 24, h - 24)
        pygame.draw.rect(surface, (8, 18, 104), frame)
        pygame.draw.rect(surface, (226, 236, 255), frame, 2)

        title = self._title_font.render("Input Profiles", True, (238, 245, 255))
        surface.blit(title, (30, 26))

        profiles = self._profiles.profiles()
        if not profiles:
            note = self._small_font.render("No profiles available.", True, (238, 245, 255))
            surface.blit(note, (32, 120))
            return

        self._selected_index %= len(profiles)
        active_id = self._profiles.active_profile().profile_id

        list_rect = pygame.Rect(32, 100, frame.w - 64, frame.h - 170)
        pygame.draw.rect(surface, (6, 13, 92), list_rect)
        pygame.draw.rect(surface, (78, 102, 170), list_rect, 1)

        self._row_hitboxes = {}
        y = list_rect.y + 10
        row_h = 34
        for idx, profile in enumerate(profiles):
            row = pygame.Rect(list_rect.x + 10, y, list_rect.w - 20, row_h)
            self._row_hitboxes[idx] = row.copy()
            selected = idx == self._selected_index
            is_active = profile.profile_id == active_id
            if selected:
                pygame.draw.rect(surface, (244, 248, 255), row)
                text_color = (16, 32, 88)
            else:
                pygame.draw.rect(surface, (9, 20, 106), row)
                text_color = (238, 245, 255)
            pygame.draw.rect(surface, (62, 84, 152), row, 1)
            marker = "*" if is_active else " "
            label = f"{marker} {profile.name}"
            text = self._small_font.render(label, True, text_color)
            surface.blit(text, (row.x + 10, row.y + (row.h - text.get_height()) // 2))
            y += row_h + 6
            if y + row_h > list_rect.bottom:
                break

        if self._renaming:
            editor = pygame.Rect(32, frame.bottom - 98, frame.w - 64, 34)
            pygame.draw.rect(surface, (9, 20, 106), editor)
            pygame.draw.rect(surface, (120, 144, 198), editor, 2)
            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            entry = self._small_font.render(self._rename_buffer + caret, True, (238, 245, 255))
            surface.blit(entry, (editor.x + 10, editor.y + 6))
            help_text = "Renaming: type name, Enter to save, Esc to cancel"
        else:
            help_text = (
                "Enter: Set Active  N: New  C: Copy  R: Rename  "
                "D: Delete  Up/Down: Select  Esc: Back"
            )

        hint = self._tiny_font.render(help_text, True, (188, 204, 228))
        surface.blit(hint, (32, frame.bottom - 52))
        if self._message != "":
            msg = self._tiny_font.render(self._message, True, (188, 204, 228))
            surface.blit(msg, (32, frame.bottom - 30))


class MenuScreen:
    def __init__(
        self, app: App, title: str, items: list[MenuItem], *, is_root: bool = False
    ) -> None:
        self._app = app
        self._title = title
        self._items = items
        self._selected = 0
        self._is_root = is_root
        # HOTAS devices can emit stale startup button events; debounce briefly.
        self._joy_input_unlock_ms = pygame.time.get_ticks() + 900
        self._title_font = pygame.font.Font(None, 42)
        self._item_font = pygame.font.Font(None, 32)
        self._hint_font = pygame.font.Font(None, 22)
        self._item_hitboxes: dict[int, pygame.Rect] = {}
        self._scroll_top = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
            return

        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._item_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._selected = idx
                        break
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for idx, rect in self._item_hitboxes.items():
                if rect.collidepoint(pos):
                    self._selected = idx
                    self._activate()
                    return
            return

        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                self._move(-1)
            elif event.y < 0:
                self._move(1)
            return

        if event.type == pygame.JOYHATMOTION:
            if pygame.time.get_ticks() < self._joy_input_unlock_ms:
                return
            # D-pad / hat navigation (works on many sticks).
            _, y = event.value
            if y == 1:
                self._move(-1)
            elif y == -1:
                self._move(1)
            return

        if event.type == pygame.JOYBUTTONDOWN:
            if pygame.time.get_ticks() < self._joy_input_unlock_ms:
                return
            # Common mapping: 0 = select, 1 = back/cancel.
            if event.button == 0:
                self._activate()
            elif event.button == 1:
                self._back()

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_UP, pygame.K_w):
            self._move(-1)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self._move(1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate()
        elif key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._back()

    def _move(self, delta: int) -> None:
        if not self._items:
            return
        self._selected = (self._selected + delta) % len(self._items)

    def _activate(self) -> None:
        if not self._items:
            return
        self._items[self._selected].action()

    def _back(self) -> None:
        if self._is_root:
            self._app.quit()
        else:
            self._app.pop()

    def _fit_label(self, font: pygame.font.Font, label: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(label)[0] <= max_width:
            return label
        clipped = label
        while clipped and font.size(f"{clipped}...")[0] > max_width:
            clipped = clipped[:-1]
        return f"{clipped}..." if clipped else "..."

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        bg = (3, 9, 78)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (186, 200, 224)
        active_bg = (244, 248, 255)
        active_text = (14, 26, 74)

        surface.fill(bg)

        frame_margin = max(10, min(26, w // 34))
        frame = pygame.Rect(
            frame_margin,
            frame_margin,
            max(260, w - frame_margin * 2),
            max(220, h - frame_margin * 2),
        )
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(34, min(52, h // 8))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface,
            border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )

        tag = self._hint_font.render("MENU", True, text_muted)
        surface.blit(tag, (header.x + 12, header.y + (header.h - tag.get_height()) // 2))

        title = self._title_font.render(self._title, True, text_main)
        surface.blit(title, title.get_rect(center=(frame.centerx, header.centery)))

        content_top = header.bottom + max(16, h // 30)
        content_bottom = frame.bottom - max(44, h // 12)
        list_rect = pygame.Rect(
            frame.x + max(14, w // 44),
            content_top,
            frame.w - max(28, w // 22),
            max(120, content_bottom - content_top),
        )
        pygame.draw.rect(surface, (6, 13, 92), list_rect)
        pygame.draw.rect(surface, (78, 102, 170), list_rect, 1)

        item_count = len(self._items)
        if item_count <= 0:
            self._item_hitboxes = {}
            footer = "Enter/Space: Select  |  Esc/Backspace: Back  |  D-pad + Button0/1"
            foot = self._hint_font.render(footer, True, text_muted)
            surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))
            return

        self._selected %= item_count

        gap = max(3, min(8, list_rect.h // 40))
        min_row_h = max(18, self._item_font.get_height() + 4)
        fit_row_h = (list_rect.h - gap * (item_count + 1)) // item_count

        if fit_row_h >= min_row_h:
            # Everything fits: no scrolling required.
            row_h = min(44, fit_row_h)
            visible_count = item_count
            self._scroll_top = 0
            total_h = row_h * visible_count + gap * (visible_count - 1)
            y = list_rect.y + max(gap, (list_rect.h - total_h) // 2)
        else:
            # Long menu: keep rows readable and scroll the visible window.
            row_h = max(min_row_h, min(40, list_rect.h // 8))
            visible_count = max(1, (list_rect.h - gap) // (row_h + gap))
            max_scroll_top = max(0, item_count - visible_count)
            if self._scroll_top > max_scroll_top:
                self._scroll_top = max_scroll_top
            if self._selected < self._scroll_top:
                self._scroll_top = self._selected
            elif self._selected >= self._scroll_top + visible_count:
                self._scroll_top = self._selected - visible_count + 1
            y = list_rect.y + gap

        self._item_hitboxes = {}
        start = int(self._scroll_top)
        end = min(item_count, start + visible_count)
        for idx in range(start, end):
            item = self._items[idx]
            row = pygame.Rect(list_rect.x + 12, y, list_rect.w - 24, row_h)
            self._item_hitboxes[idx] = row.copy()
            selected = idx == self._selected
            if selected:
                pygame.draw.rect(surface, active_bg, row)
                pygame.draw.rect(surface, (120, 142, 196), row, 2)
            else:
                pygame.draw.rect(surface, (9, 20, 106), row)
                pygame.draw.rect(surface, (62, 84, 152), row, 1)

            color = active_text if selected else text_main
            label = self._fit_label(self._item_font, item.label, row.w - 20)
            text = self._item_font.render(label, True, color)
            surface.blit(text, (row.x + 10, row.y + (row.h - text.get_height()) // 2))
            y += row_h + gap

        max_scroll_top = max(0, item_count - visible_count)
        if max_scroll_top > 0:
            up_color = text_main if self._scroll_top > 0 else (98, 118, 166)
            down_color = text_main if self._scroll_top < max_scroll_top else (98, 118, 166)
            up = self._hint_font.render("^", True, up_color)
            down = self._hint_font.render("v", True, down_color)
            surface.blit(up, up.get_rect(topright=(list_rect.right - 8, list_rect.y + 4)))
            surface.blit(down, down.get_rect(bottomright=(list_rect.right - 8, list_rect.bottom - 4)))

        footer = "Enter/Space: Select  |  Esc/Backspace: Back  |  D-pad + Button0/1"
        foot = self._hint_font.render(footer, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))


class DifficultySettingsScreen:
    def __init__(self, app: App) -> None:
        self._app = app
        self._selected = 0
        self._scroll_top = 0
        self._title_font = pygame.font.Font(None, 42)
        self._item_font = pygame.font.Font(None, 30)
        self._hint_font = pygame.font.Font(None, 22)
        self._row_hitboxes: dict[int, pygame.Rect] = {}
        self._control_hitboxes: dict[tuple[int, str], pygame.Rect] = {}

    def _rows(self) -> list[tuple[str, str, str]]:
        rows = [
            (
                "override",
                "Global Difficulty Override",
                "ON" if self._app.difficulty_override_enabled() else "OFF",
            ),
            (
                "global_level",
                "Global Difficulty",
                f"{self._app.stored_global_difficulty_level()} / 10",
            ),
        ]
        for test_code, label in TEST_DIFFICULTY_OPTIONS:
            rows.append(
                (
                    f"test:{test_code}",
                    label,
                    f"{self._app.stored_test_difficulty_level(test_code)} / 10",
                )
            )
        rows.append(("back", "Back", "Return to Main Menu"))
        return rows

    def handle_event(self, event: pygame.event.Event) -> None:
        rows = self._rows()
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._row_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._selected = idx
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for (idx, action), rect in self._control_hitboxes.items():
                if rect.collidepoint(pos):
                    self._selected = idx
                    self._adjust_row(rows[idx][0], -1 if action == "dec" else 1)
                    return
            for idx, rect in self._row_hitboxes.items():
                if rect.collidepoint(pos):
                    self._selected = idx
                    if rows[idx][0] == "back":
                        self._app.pop()
                    return
            return
        if event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                self._selected = max(0, self._selected - 1)
            elif event.y < 0:
                self._selected = min(max(0, len(rows) - 1), self._selected + 1)
            return
        if event.type != pygame.KEYDOWN:
            return

        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._app.pop()
            return
        if key in (pygame.K_UP, pygame.K_w):
            self._selected = (self._selected - 1) % max(1, len(rows))
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self._selected = (self._selected + 1) % max(1, len(rows))
            return
        if key in (pygame.K_LEFT, pygame.K_a):
            self._adjust_row(rows[self._selected][0], -1)
            return
        if key in (pygame.K_RIGHT, pygame.K_d):
            self._adjust_row(rows[self._selected][0], 1)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if rows[self._selected][0] == "back":
                self._app.pop()
            elif rows[self._selected][0] == "override":
                self._adjust_row("override", 1)

    def _adjust_row(self, key: str, direction: int) -> None:
        if key == "override":
            self._app.set_global_difficulty_override_enabled(
                not self._app.difficulty_override_enabled()
            )
            return
        if key == "global_level":
            self._app.set_global_difficulty_level(
                self._app.stored_global_difficulty_level() + int(direction)
            )
            return
        if key.startswith("test:"):
            test_code = key.split(":", 1)[1]
            self._app.set_test_difficulty_level(
                test_code=test_code,
                level=self._app.stored_test_difficulty_level(test_code) + int(direction),
            )

    def render(self, surface: pygame.Surface) -> None:
        w, h = surface.get_size()
        bg = (4, 10, 72)
        panel_bg = (8, 18, 104)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (186, 200, 224)
        active_bg = (244, 248, 255)
        active_text = (14, 26, 74)

        surface.fill(bg)
        panel = pygame.Rect(max(12, w // 28), max(12, h // 24), w - max(24, w // 14), h - max(24, h // 12))
        pygame.draw.rect(surface, panel_bg, panel)
        pygame.draw.rect(surface, border, panel, 2)

        title = self._title_font.render("Settings", True, text_main)
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 14)))
        subtitle = self._hint_font.render(
            "Global override applies everywhere. Per-test values are stored underneath.",
            True,
            text_muted,
        )
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 54)))

        rows = self._rows()
        self._selected %= max(1, len(rows))

        list_rect = pygame.Rect(panel.x + 18, panel.y + 86, panel.w - 36, panel.h - 128)
        pygame.draw.rect(surface, (6, 13, 92), list_rect)
        pygame.draw.rect(surface, (78, 102, 170), list_rect, 1)

        gap = 8
        row_h = 42
        visible_count = max(1, (list_rect.h - gap) // (row_h + gap))
        max_scroll_top = max(0, len(rows) - visible_count)
        if self._scroll_top > max_scroll_top:
            self._scroll_top = max_scroll_top
        if self._selected < self._scroll_top:
            self._scroll_top = self._selected
        elif self._selected >= self._scroll_top + visible_count:
            self._scroll_top = self._selected - visible_count + 1

        self._row_hitboxes = {}
        self._control_hitboxes = {}
        y = list_rect.y + gap
        start = self._scroll_top
        end = min(len(rows), start + visible_count)
        for idx in range(start, end):
            key, label, value = rows[idx]
            row = pygame.Rect(list_rect.x + 10, y, list_rect.w - 20, row_h)
            self._row_hitboxes[idx] = row.copy()
            selected = idx == self._selected
            fill = active_bg if selected else (9, 20, 106)
            edge = (120, 142, 196) if selected else (62, 84, 152)
            pygame.draw.rect(surface, fill, row, border_radius=6)
            pygame.draw.rect(surface, edge, row, 2 if selected else 1, border_radius=6)
            label_color = active_text if selected else text_main
            value_color = (46, 62, 112) if selected else text_muted

            label_text = MenuScreen._fit_label(self, self._item_font, label, row.w - 240)
            label_surf = self._item_font.render(label_text, True, label_color)
            surface.blit(label_surf, (row.x + 12, row.y + (row.h - label_surf.get_height()) // 2))

            if key == "back":
                value_surf = self._hint_font.render(value, True, value_color)
                surface.blit(value_surf, value_surf.get_rect(midright=(row.right - 14, row.centery)))
            else:
                value_box = pygame.Rect(row.right - 176, row.y + 6, 164, row.h - 12)
                dec_box = pygame.Rect(value_box.x, value_box.y, 32, value_box.h)
                inc_box = pygame.Rect(value_box.right - 32, value_box.y, 32, value_box.h)
                mid_box = pygame.Rect(dec_box.right + 6, value_box.y, value_box.w - 76, value_box.h)
                self._control_hitboxes[(idx, "dec")] = dec_box.copy()
                self._control_hitboxes[(idx, "inc")] = inc_box.copy()
                for box, glyph in ((dec_box, "<"), (inc_box, ">")):
                    pygame.draw.rect(surface, (20, 32, 92), box, border_radius=5)
                    pygame.draw.rect(surface, (110, 130, 184), box, 1, border_radius=5)
                    glyph_surf = self._item_font.render(glyph, True, text_main)
                    surface.blit(glyph_surf, glyph_surf.get_rect(center=box.center))
                pygame.draw.rect(surface, (14, 26, 78), mid_box, border_radius=5)
                pygame.draw.rect(surface, (92, 112, 168), mid_box, 1, border_radius=5)
                value_surf = self._hint_font.render(value, True, text_main)
                surface.blit(value_surf, value_surf.get_rect(center=mid_box.center))

            y += row_h + gap

        footer = self._hint_font.render(
            "Up/Down: Select  Left/Right: Adjust  Enter: Toggle/Back  Esc: Main Menu",
            True,
            text_muted,
        )
        surface.blit(footer, footer.get_rect(midbottom=(panel.centerx, panel.bottom - 12)))


class CognitiveTestScreen:
    def __init__(
        self,
        app: App,
        *,
        engine_factory: Callable[[], CognitiveEngine],
        test_code: str | None = None,
        test_version: int = 1,
    ) -> None:
        self._app = app
        self._engine_factory = engine_factory
        self._engine: CognitiveEngine = engine_factory()
        self._test_code = None if test_code is None else str(test_code).strip() or None
        self._test_version = int(test_version)
        self._input = ""
        self._math_choice = 1
        self._results_persisted = False
        self._results_persistence_lines: list[str] = []
        self._input_prompt_key: tuple[str, str] | None = None

        self._small_font = pygame.font.Font(None, 24)
        self._tiny_font = pygame.font.Font(None, 18)
        self._big_font = pygame.font.Font(None, 72)
        self._mid_font = pygame.font.Font(None, 52)
        self._num_header_font = pygame.font.Font(None, 28)
        self._num_prompt_fonts = [
            pygame.font.Font(None, 112),
            pygame.font.Font(None, 96),
            pygame.font.Font(None, 84),
            pygame.font.Font(None, 72),
        ]
        self._num_input_font = pygame.font.Font(None, 58)

        # Airborne-specific UI state (hold-to-show overlays).
        self._air_overlay: str | None = None  # "intro" | "fuel" | "parcel"
        self._air_show_distances = False

        # Cached procedural sprites for Instrument Comprehension dials.
        self._instrument_sprite_cache: dict[tuple[object, ...], pygame.Surface] = {}
        self._instrument_card_bank = InstrumentAircraftCardSpriteBank()

        # CLN mouse-selection hitboxes (code -> rect), refreshed during render.
        self._cln_option_hitboxes: dict[int, pygame.Rect] = {}

        # Situational Awareness interaction + optional TTS callouts.
        self._sa_option_hitboxes: dict[int, pygame.Rect] = {}
        self._sa_grid_hitboxes: dict[str, pygame.Rect] = {}
        self._sa_last_announced_token: tuple[object, ...] | None = None
        self._sa_tts = _OfflineTtsSpeaker(
            rate_wpm=192,
            volume=0.58,
            backend_preference=("pyttsx3-subprocess", "say", "espeak", "powershell"),
        )

        # Vigilance row/column capture controls.
        self._vigilance_row_input = ""
        self._vigilance_col_input = ""
        self._vigilance_focus = "row"  # "row" | "col"
        self._vigilance_row_hitbox: pygame.Rect | None = None
        self._vigilance_col_hitbox: pygame.Rect | None = None

        # Target-recognition panel interaction + animation state.
        self._tr_selection_payload_id: int | None = None
        self._tr_selected_panels: set[str] = set()
        self._tr_selector_hitboxes: dict[str, pygame.Rect] = {}
        self._tr_light_payload_id: int | None = None
        self._tr_light_rng: random.Random | None = None
        self._tr_light_next_change_ms = 0
        self._tr_light_current_pattern: tuple[str, str, str] = ("G", "G", "G")
        self._tr_light_target_pattern_live: tuple[str, str, str] = ("G", "G", "G")
        self._tr_light_points = 0
        self._tr_light_hits = 0
        self._tr_light_early_presses = 0
        self._tr_light_button_hitbox: pygame.Rect | None = None

        self._tr_scan_payload_id: int | None = None
        self._tr_scan_rng: random.Random | None = None
        self._tr_scan_token_pool: tuple[str, ...] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_next_change_ms = 0
        self._tr_scan_current_pattern: tuple[str, str, str, str] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_target_pattern_live: tuple[str, str, str, str] = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_points = 0
        self._tr_scan_hits = 0
        self._tr_scan_early_presses = 0
        self._tr_scan_button_hitbox: pygame.Rect | None = None
        self._tr_scan_reveal_index = 3
        self._tr_scan_next_step_ms = 0
        self._tr_scan_passes_left = 0

        self._tr_system_payload_id: int | None = None
        self._tr_system_rng: random.Random | None = None
        self._tr_system_columns: list[list[str]] = [[], [], []]
        self._tr_system_row_offset = 0
        self._tr_system_row_frac = 0.0
        self._tr_system_target_code = "----"
        self._tr_system_step_interval_ms = 1700
        self._tr_system_last_step_ms = 0
        self._tr_system_points = 0
        self._tr_system_hits = 0
        self._tr_system_string_hitboxes: list[tuple[pygame.Rect, str]] = []
        self._tr_scene_payload_id: int | None = None
        self._tr_scene_rng: random.Random | None = None
        self._tr_scene_glyphs: dict[int, _TargetRecognitionSceneGlyph] = {}
        self._tr_scene_glyph_order: list[int] = []
        self._tr_scene_symbol_hitboxes: list[tuple[pygame.Rect, int]] = []
        self._tr_scene_next_glyph_id = 1
        self._tr_scene_target_queue: list[str] = []
        self._tr_scene_active_targets: list[str] = []
        self._tr_scene_target_cap = 5
        self._tr_scene_next_target_add_ms = 0
        self._tr_scene_points = 0
        self._tr_scene_hits = 0
        self._tr_scene_misses = 0
        self._tr_scene_beacon_hits = 0
        self._tr_scene_unknown_hits = 0
        self._tr_scene_anim_frame = 0
        self._tr_scene_last_update_ms = 0
        self._tr_scene_base_cache: pygame.Surface | None = None
        self._tr_scene_base_cache_size: tuple[int, int] = (0, 0)
        self._tr_scene_base_cache_seed = 0

        # System Logic panel navigation state.
        self._system_logic_payload_id: int | None = None
        self._system_logic_folder_index = 0
        self._system_logic_doc_index = 0
        self._cognitive_updating_payload_id: int | None = None
        self._cognitive_updating_runtime: CognitiveUpdatingRuntime | None = None
        self._cognitive_updating_upper_tab_index = 2
        self._cognitive_updating_lower_tab_index = 1
        self._cognitive_updating_hitboxes: dict[str, pygame.Rect] = {}
        self._cognitive_updating_phase_start_ms = 0
        self._cognitive_updating_parcel_values = ["", "", ""]
        self._cognitive_updating_active_parcel_field = 0
        self._cognitive_updating_comms_input = ""
        self._cognitive_updating_speed_knots: int | None = None
        self._cognitive_updating_pump_on: bool | None = None
        self._cognitive_updating_active_tank: int | None = None
        self._cognitive_updating_alpha_armed = False
        self._cognitive_updating_bravo_armed = False
        self._cognitive_updating_air_sensor_armed = False
        self._cognitive_updating_ground_sensor_armed = False
        self._cognitive_updating_dispenser_lit: int | None = None
        self._auditory_audio: _AuditoryCapacityAudioAdapter | None = None
        self._auditory_audio_debug: dict[str, object] = {}
        self._auditory_panda_renderer: AuditoryCapacityPanda3DRenderer | None = None
        self._auditory_panda_failed = False
        self._rapid_tracking_panda_renderer: RapidTrackingPanda3DRenderer | None = None
        self._rapid_tracking_panda_failed = False
        self._spatial_integration_panda_renderer: SpatialIntegrationPanda3DRenderer | None = None
        self._spatial_integration_panda_failed = False
        self._trace_test_1_panda_renderer: TraceTest1Panda3DRenderer | None = None
        self._trace_test_1_panda_failed = False
        self._trace_test_2_panda_renderer: TraceTest2Panda3DRenderer | None = None
        self._trace_test_2_panda_failed = False
        self._auditory_testing_menu = os.environ.get(
            "CFAST_AUDITORY_TESTING_MENU", "1"
        ).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self._pause_menu_active = False
        self._pause_menu_selected = 0
        self._pause_menu_mode = "menu"
        self._pause_menu_hitboxes: dict[int, pygame.Rect] = {}
        self._pause_settings_selected = 0
        self._pause_settings_hitboxes: dict[int, pygame.Rect] = {}
        self._pause_settings_control_hitboxes: dict[tuple[int, str], pygame.Rect] = {}
        self._intro_difficulty_control_hitboxes: dict[str, pygame.Rect] = {}
        self._staged_difficulty_level: int | None = None
        self._intro_loading_token: str | None = None
        self._intro_loading_frames_rendered = 0
        self._intro_loading_ready = True

        self._target_recognition_reset_practice_breakdown()
        self._target_recognition_reset_scene_subtask()
        self._target_recognition_reset_light_subtask()
        self._target_recognition_reset_scan_subtask()
        self._target_recognition_reset_system_subtask()
        self._sync_intro_loading_state(self._engine.snapshot().phase)

    @staticmethod
    def _intro_loading_phase_token(phase: Phase) -> str | None:
        if phase is Phase.INSTRUCTIONS:
            return "practice"
        if phase is Phase.PRACTICE_DONE:
            return "scored"
        return None

    def _sync_intro_loading_state(self, phase: Phase) -> None:
        token = self._intro_loading_phase_token(phase)
        if token == self._intro_loading_token:
            return
        self._intro_loading_token = token
        self._intro_loading_frames_rendered = 0
        self._intro_loading_ready = token is None

    def _intro_loading_complete(self, phase: Phase) -> bool:
        token = self._intro_loading_phase_token(phase)
        if token is None:
            return True
        return self._intro_loading_token == token and self._intro_loading_ready

    def _advance_intro_loading(
        self,
        *,
        surface_size: tuple[int, int],
        snap: TestSnapshot,
    ) -> None:
        token = self._intro_loading_phase_token(snap.phase)
        if token is None or self._intro_loading_ready:
            return

        self._prime_intro_stage_assets(surface_size=surface_size, snap=snap)
        self._intro_loading_frames_rendered += 1
        if self._intro_loading_frames_rendered >= INTRO_LOADING_MIN_FRAMES:
            self._intro_loading_ready = True

    def _prime_intro_stage_assets(
        self,
        *,
        surface_size: tuple[int, int],
        snap: TestSnapshot,
    ) -> None:
        payload = snap.payload
        try:
            if snap.title == "Rapid Tracking":
                self._get_rapid_tracking_panda_renderer(size=surface_size)
                return
            if snap.title == "Spatial Integration":
                self._get_spatial_integration_panda_renderer(size=surface_size)
                return
            if snap.title == "Trace Test 1":
                self._get_trace_test_1_panda_renderer(size=surface_size)
                return
            if snap.title == "Trace Test 2":
                renderer = self._get_trace_test_2_panda_renderer(size=surface_size)
                if renderer is not None:
                    renderer.render(
                        payload=payload if isinstance(payload, TraceTest2Payload) else None
                    )
                return
            if snap.title == "Auditory Capacity":
                renderer = self._get_auditory_panda_renderer(size=surface_size)
                if renderer is not None:
                    renderer.render(
                        payload=payload if isinstance(payload, AuditoryCapacityPayload) else None
                    )
                return
        except Exception:
            return

    def _render_intro_loading_indicator(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        if self._intro_loading_complete(snap.phase):
            return

        dot_count = (pygame.time.get_ticks() // 220) % 4
        label = f"Loading{'.' * dot_count}"
        text = self._tiny_font.render(label, True, (238, 245, 255))
        pill = text.get_rect()
        pill.inflate_ip(18, 10)
        pill.bottomright = (surface.get_width() - 18, surface.get_height() - 18)
        pygame.draw.rect(surface, (8, 18, 104), pill, border_radius=8)
        pygame.draw.rect(surface, (226, 236, 255), pill, 1, border_radius=8)
        surface.blit(text, text.get_rect(center=pill.center))

    @staticmethod
    def _wrap_text_lines(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        if max_width <= 0:
            return [""]
        words = str(text).split()
        if not words:
            return [""]
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if current == "" else f"{current} {word}"
            if font.size(trial)[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    def _draw_intro_section(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        title: str,
        lines: tuple[str, ...],
        title_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=10)
        pygame.draw.rect(surface, border, rect, 1, border_radius=10)
        title_surf = self._small_font.render(title, True, title_color)
        surface.blit(title_surf, (rect.x + 12, rect.y + 10))

        max_width = rect.w - 24
        y = rect.y + 36
        line_h = self._tiny_font.get_linesize() + 2
        max_y = rect.bottom - 10
        for raw in lines:
            wrapped = self._wrap_text_lines(raw, self._tiny_font, max_width)
            for line in wrapped:
                if y + line_h > max_y:
                    return
                surface.blit(self._tiny_font.render(line, True, text_color), (rect.x + 12, y))
                y += line_h
            y += 2

    def _render_test_intro_overlay(self, surface: pygame.Surface, snap: TestSnapshot) -> None:
        if snap.phase not in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            return
        if self._test_code is None:
            return

        briefing = TEST_GUIDE_BRIEFS.get(self._test_code)
        if briefing is None:
            return

        loading = not self._intro_loading_complete(snap.phase)
        is_practice_stage = snap.phase is Phase.INSTRUCTIONS

        w, h = surface.get_size()
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((4, 8, 26, 186))
        surface.blit(dim, (0, 0))

        panel = pygame.Rect(
            max(18, w // 18),
            max(18, h // 18),
            w - max(36, w // 9),
            h - max(36, h // 9),
        )
        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=14)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=14)

        title = self._app.font.render(briefing.label, True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 16)))

        if loading:
            stage_text = "Loading Practice Block" if is_practice_stage else "Loading Timed Block"
            status_text = (
                "Assets and controls are still loading. Enter stays locked until this finishes."
            )
        elif is_practice_stage:
            stage_text = "Instructions"
            status_text = "Practice is ready. Review the guide notes and start when you are ready."
        else:
            stage_text = "Practice Complete"
            status_text = "Review the summary and controls. Enter will start the timed block."

        stage = self._small_font.render(stage_text, True, (188, 204, 228))
        surface.blit(stage, stage.get_rect(midtop=(panel.centerx, panel.y + 56)))
        status = self._tiny_font.render(status_text, True, (188, 204, 228))
        surface.blit(status, status.get_rect(midtop=(panel.centerx, panel.y + 84)))

        body = pygame.Rect(panel.x + 14, panel.y + 112, panel.w - 28, panel.h - 164)
        gap = 12
        left_w = (body.w - gap) // 2
        right_w = body.w - left_w - gap
        top_h = max(92, int(body.h * 0.27))
        mid_h = max(134, int(body.h * 0.39))
        bottom_h = body.h - top_h - mid_h - gap * 2

        assessment_rect = pygame.Rect(body.x, body.y, left_w, top_h)
        guide_rect = pygame.Rect(body.x + left_w + gap, body.y, right_w, top_h)
        tasks_rect = pygame.Rect(body.x, assessment_rect.bottom + gap, left_w, mid_h)
        controls_rect = pygame.Rect(body.x + left_w + gap, guide_rect.bottom + gap, right_w, mid_h)
        note_rect = pygame.Rect(body.x, tasks_rect.bottom + gap, body.w, bottom_h)

        self._draw_intro_section(
            surface,
            assessment_rect,
            title="What This Assesses",
            lines=(briefing.assessment,),
            title_color=(238, 245, 255),
            text_color=(188, 204, 228),
            fill=(10, 20, 92),
            border=(78, 102, 170),
        )
        self._draw_intro_section(
            surface,
            guide_rect,
            title="Guide Notes",
            lines=(briefing.timing, briefing.prep),
            title_color=(238, 245, 255),
            text_color=(188, 204, 228),
            fill=(10, 20, 92),
            border=(78, 102, 170),
        )
        self._draw_intro_section(
            surface,
            tasks_rect,
            title="You Will Need To",
            lines=tuple(f"- {line}" for line in briefing.tasks),
            title_color=(238, 245, 255),
            text_color=(188, 204, 228),
            fill=(10, 20, 92),
            border=(78, 102, 170),
        )

        flow_line = (
            "Next: timed block loads first, then Enter starts it."
            if snap.phase is Phase.PRACTICE_DONE
            else briefing.app_flow
        )
        controls_lines = (briefing.controls, flow_line)
        if loading:
            controls_lines += ("Use this loading window to review the controls before start.",)

        self._draw_intro_section(
            surface,
            controls_rect,
            title="Controls In This Trainer",
            lines=controls_lines,
            title_color=(238, 245, 255),
            text_color=(188, 204, 228),
            fill=(10, 20, 92),
            border=(78, 102, 170),
        )

        prompt_text = " ".join(str(snap.prompt).split())
        if prompt_text == "":
            prompt_text = (
                "Practice uses the same response style as the timed block. Focus on learning the task."
                if is_practice_stage
                else "Practice is complete. The next screen state will be the scored block."
            )
        note_title = "Trainer Note" if is_practice_stage else "Practice Summary"
        self._draw_intro_section(
            surface,
            note_rect,
            title=note_title,
            lines=(prompt_text,),
            title_color=(238, 245, 255),
            text_color=(188, 204, 228),
            fill=(14, 24, 76),
            border=(92, 112, 168),
        )

    def _show_intro_loading_screen(self, phase: Phase) -> None:
        token = self._intro_loading_phase_token(phase)
        if token is None:
            self._sync_intro_loading_state(phase)
            return
        if self._engine.snapshot().phase is not phase:
            self._force_engine_phase(phase)
        self._intro_loading_token = token
        self._intro_loading_frames_rendered = 0
        self._intro_loading_ready = False

    def _restart_activity(self, *, auto_start_practice: bool = False) -> None:
        self._stop_auditory_audio()
        self._stop_situational_awareness_audio()
        replacement = CognitiveTestScreen(
            self._app,
            engine_factory=self._engine_factory,
            test_code=self._test_code,
            test_version=self._test_version,
        )
        if auto_start_practice:
            replacement._engine.start_practice()
            replacement._input = ""
            replacement._math_choice = 1
        self._app.replace_top(replacement)

    def _persist_staged_difficulty_level(self) -> int:
        level = self._staged_difficulty_level
        if level is None:
            return self._current_engine_difficulty_level()
        if self._test_code is not None:
            persisted = self._app.set_persistent_difficulty_level(
                test_code=self._test_code,
                level=level,
            )
        else:
            persisted = level
        self._staged_difficulty_level = None
        return int(persisted)

    def _has_staged_difficulty_change(self) -> bool:
        level = self._staged_difficulty_level
        if level is None:
            return False
        if self._test_code is not None:
            return int(level) != int(self._app.effective_difficulty_level(self._test_code))
        return int(level) != int(self._current_engine_difficulty_level())

    def _current_engine_difficulty_level(self) -> int:
        raw = getattr(self._engine, "_difficulty", None)
        if not isinstance(raw, int | float):
            return DEFAULT_DIFFICULTY_LEVEL
        normalized = max(0.0, min(1.0, float(raw)))
        return max(1, min(10, int(round(normalized * 9.0)) + 1))

    def handle_event(self, event: pygame.event.Event) -> None:
        snap = self._engine.snapshot()
        self._sync_intro_loading_state(snap.phase)
        p = snap.payload
        scenario = p if isinstance(p, AirborneScenario) else None
        math_payload: MathReasoningPayload | None = (
            p if isinstance(p, MathReasoningPayload) else None
        )
        sensory_payload: SensoryMotorApparatusPayload | None = (
            p if isinstance(p, SensoryMotorApparatusPayload) else None
        )
        rapid_tracking_payload: RapidTrackingPayload | None = (
            p if isinstance(p, RapidTrackingPayload) else None
        )
        spatial_payload: SpatialIntegrationPayload | None = (
            p if isinstance(p, SpatialIntegrationPayload) else None
        )
        trace_test_1_payload: TraceTest1Payload | None = (
            p if isinstance(p, TraceTest1Payload) else None
        )
        trace_test_2_payload: TraceTest2Payload | None = (
            p if isinstance(p, TraceTest2Payload) else None
        )
        table_payload: TableReadingPayload | None = (
            p if isinstance(p, TableReadingPayload) else None
        )
        situational_awareness_payload: SituationalAwarenessPayload | None = (
            p if isinstance(p, SituationalAwarenessPayload) else None
        )
        angles_payload: AnglesBearingsDegreesPayload | None = (
            p if isinstance(p, AnglesBearingsDegreesPayload) else None
        )
        vs: VisualSearchPayload | None = p if isinstance(p, VisualSearchPayload) else None
        vigilance_payload: VigilancePayload | None = p if isinstance(p, VigilancePayload) else None
        cln_payload: ColoursLettersNumbersPayload | None = (
            p if isinstance(p, ColoursLettersNumbersPayload) else None
        )
        system_logic_payload: SystemLogicPayload | None = (
            p if isinstance(p, SystemLogicPayload) else None
        )
        ic: InstrumentComprehensionPayload | None = (
            p if isinstance(p, InstrumentComprehensionPayload) else None
        )
        cognitive_updating_payload: CognitiveUpdatingPayload | None = (
            p if isinstance(p, CognitiveUpdatingPayload) else None
        )
        tr_payload: TargetRecognitionPayload | None = (
            p if isinstance(p, TargetRecognitionPayload) else None
        )
        auditory_payload: AuditoryCapacityPayload | None = (
            p if isinstance(p, AuditoryCapacityPayload) else None
        )

        # Emergency exit: allow a hard escape from any state (including SCORED).
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F12:
                self._stop_auditory_audio()
                self._stop_situational_awareness_audio()
                self._app.pop()
                return
            if event.key == pygame.K_ESCAPE and (event.mod & pygame.KMOD_SHIFT):
                self._stop_auditory_audio()
                self._stop_situational_awareness_audio()
                self._app.pop()
                return
            if snap.title == "Auditory Capacity" and event.key == pygame.K_F9:
                self._auditory_testing_menu = not self._auditory_testing_menu
                return
            if snap.title == "Spatial Integration":
                if event.key == pygame.K_F10:
                    accepted = self._engine.submit_answer("__skip_practice__")
                    if accepted:
                        self._input = ""
                        self._math_choice = 1
                    return
                if event.key == pygame.K_F11:
                    accepted = self._engine.submit_answer("__skip_section__")
                    if accepted:
                        self._input = ""
                        self._math_choice = 1
                    return
                if event.key == pygame.K_F8:
                    accepted = self._engine.submit_answer("__skip_all__")
                    if accepted:
                        self._input = ""
                        self._math_choice = 1
                    return
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE) and snap.phase in (
                Phase.INSTRUCTIONS,
                Phase.PRACTICE,
                Phase.PRACTICE_DONE,
                Phase.SCORED,
                Phase.RESULTS,
            ):
                self._pause_menu_active = not self._pause_menu_active
                if self._pause_menu_active:
                    self._pause_menu_selected = 0
                    self._pause_menu_mode = "menu"
                    self._pause_settings_selected = 0
                    self._staged_difficulty_level = self._current_engine_difficulty_level()
                    self._stop_auditory_audio()
                    self._stop_situational_awareness_audio()
                else:
                    self._staged_difficulty_level = None
                return

        if self._pause_menu_active:
            self._handle_pause_menu_event(event)
            return

        if self._handle_intro_difficulty_event(event, phase=snap.phase):
            return

        dr: DigitRecognitionPayload | None = None
        if p is not None and hasattr(p, "display_digits") and hasattr(p, "accepting_input"):
            dr = cast(DigitRecognitionPayload, p)
        if dr is not None and not dr.accepting_input:
            return
        # Airborne: hold-to-show overlays.
        if scenario is not None:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self._air_overlay = "intro"
                elif event.key == pygame.K_d:
                    self._air_overlay = "fuel"
                elif event.key == pygame.K_f:
                    self._air_overlay = "parcel"
                elif event.key == pygame.K_a:
                    self._air_show_distances = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    self._air_show_distances = False
                elif event.key == pygame.K_s and self._air_overlay == "intro":
                    self._air_overlay = None
                elif event.key == pygame.K_d and self._air_overlay == "fuel":
                    self._air_overlay = None
                elif event.key == pygame.K_f and self._air_overlay == "parcel":
                    self._air_overlay = None

        if (
            event.type == pygame.JOYBUTTONDOWN
            and rapid_tracking_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
            and int(getattr(event, "button", -1)) in (0, 1)
        ):
            self._engine.submit_answer("CAPTURE")
            return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and rapid_tracking_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
            and getattr(event, "button", 0) == 1
        ):
            self._engine.submit_answer("CAPTURE")
            return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and tr_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
        ):
            self._target_recognition_sync_selection(tr_payload)
            self._target_recognition_sync_light_stream(tr_payload)
            self._target_recognition_sync_scan_stream(tr_payload)
            self._target_recognition_sync_system_stream(tr_payload)
            pos = getattr(event, "pos", None)
            if pos is not None:
                expected = self._target_recognition_expected_panels(tr_payload)

                for hit_rect, glyph_id in reversed(self._tr_scene_symbol_hitboxes):
                    if not hit_rect.collidepoint(pos):
                        continue
                    scene_success = self._target_recognition_handle_scene_press(
                        tr_payload,
                        glyph_id=glyph_id,
                    )
                    if "scene" in expected:
                        if scene_success:
                            self._tr_selected_panels.add("scene")
                        if scene_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                if (
                    self._tr_light_button_hitbox is not None
                    and self._tr_light_button_hitbox.collidepoint(pos)
                ):
                    light_success = self._target_recognition_handle_light_press(tr_payload)
                    if "light" in expected:
                        if light_success:
                            self._tr_selected_panels.add("light")
                        else:
                            self._tr_selected_panels.discard("light")
                        if light_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                if (
                    self._tr_scan_button_hitbox is not None
                    and self._tr_scan_button_hitbox.collidepoint(pos)
                ):
                    scan_success = self._target_recognition_handle_scan_press(tr_payload)
                    if "scan" in expected:
                        if scan_success:
                            self._tr_selected_panels.add("scan")
                        else:
                            self._tr_selected_panels.discard("scan")
                        if scan_success and self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                    return

                for hit_rect, hit_code in self._tr_system_string_hitboxes:
                    if hit_rect.collidepoint(pos):
                        system_success = self._target_recognition_handle_system_press(
                            tr_payload,
                            clicked_code=hit_code,
                        )
                        if "system" in expected:
                            if system_success:
                                self._tr_selected_panels.add("system")
                            else:
                                self._tr_selected_panels.discard("system")
                            if system_success and self._tr_selected_panels == expected:
                                selected_snapshot = set(self._tr_selected_panels)
                                accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                                if accepted:
                                    if snap.phase is Phase.PRACTICE:
                                        self._target_recognition_record_practice_trial(
                                            selected=selected_snapshot,
                                            expected=expected,
                                        )
                                    self._tr_selected_panels.clear()
                        return

                for panel, rect in self._tr_selector_hitboxes.items():
                    if rect.collidepoint(pos):
                        if panel == "scene":
                            if bool(tr_payload.scene_has_target):
                                self._tr_selected_panels.add("scene")
                            else:
                                self._tr_selected_panels.discard("scene")
                        else:
                            if panel in self._tr_selected_panels:
                                self._tr_selected_panels.remove(panel)
                            else:
                                self._tr_selected_panels.add(panel)

                        if self._tr_selected_panels == expected:
                            selected_snapshot = set(self._tr_selected_panels)
                            accepted = self._engine.submit_answer(str(len(selected_snapshot)))
                            if accepted:
                                if snap.phase is Phase.PRACTICE:
                                    self._target_recognition_record_practice_trial(
                                        selected=selected_snapshot,
                                        expected=expected,
                                    )
                                self._tr_selected_panels.clear()
                        return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and cln_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
            and cln_payload.options_active
        ):
            pos = getattr(event, "pos", None)
            if pos is not None:
                for code, rect in self._cln_option_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._engine.submit_answer(f"MEM:{code}")
                        return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and situational_awareness_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
        ):
            pos = getattr(event, "pos", None)
            if pos is not None:
                if (
                    situational_awareness_payload.kind
                    is SituationalAwarenessQuestionKind.POSITION_PROJECTION
                ):
                    for cell_label, rect in self._sa_grid_hitboxes.items():
                        if not rect.collidepoint(pos):
                            continue
                        selected = next(
                            (
                                option
                                for option in situational_awareness_payload.options
                                if option.cell_label == cell_label
                            ),
                            None,
                        )
                        if selected is None:
                            return
                        self._math_choice = int(selected.code)
                        self._input = str(selected.code)
                        accepted = self._engine.submit_answer(self._input)
                        if accepted:
                            self._input = ""
                            self._math_choice = 1
                        return

                for code, rect in self._sa_option_hitboxes.items():
                    if not rect.collidepoint(pos):
                        continue
                    self._math_choice = int(code)
                    self._input = str(code)
                    accepted = self._engine.submit_answer(self._input)
                    if accepted:
                        self._input = ""
                        self._math_choice = 1
                    return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and vigilance_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
        ):
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            if self._vigilance_row_hitbox is not None and self._vigilance_row_hitbox.collidepoint(pos):
                self._vigilance_focus = "row"
                return
            if self._vigilance_col_hitbox is not None and self._vigilance_col_hitbox.collidepoint(pos):
                self._vigilance_focus = "col"
                return

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and getattr(event, "button", 0) == 1
            and cognitive_updating_payload is not None
            and snap.phase in (Phase.PRACTICE, Phase.SCORED)
        ):
            self._cognitive_updating_sync_payload(cognitive_updating_payload)
            runtime = self._cognitive_updating_runtime
            pos = getattr(event, "pos", None)
            if pos is not None and runtime is not None:
                for code, rect in reversed(list(self._cognitive_updating_hitboxes.items())):
                    if not rect.collidepoint(pos):
                        continue

                    scope = ""
                    action = code
                    if "|" in code:
                        scope, action = code.split("|", 1)

                    if action.startswith("tab:"):
                        try:
                            idx = int(action.split(":", 1)[1])
                        except ValueError:
                            return
                        if scope == "upper":
                            self._cognitive_updating_upper_tab_index = idx % 6
                        elif scope == "lower":
                            self._cognitive_updating_lower_tab_index = idx % 6
                        return
                    if action == "camera_alpha":
                        runtime.toggle_camera("alpha")
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "camera_bravo":
                        runtime.toggle_camera("bravo")
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "sensor_air":
                        runtime.toggle_sensor("air")
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "sensor_ground":
                        runtime.toggle_sensor("ground")
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "knots_dec":
                        runtime.adjust_knots(-1)
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "knots_inc":
                        runtime.adjust_knots(1)
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "pump_on":
                        runtime.set_pump(True)
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "pump_off":
                        runtime.set_pump(False)
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "objective_activate":
                        runtime.activate_dispenser()
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action == "comms_submit":
                        raw_submission = runtime.build_submission_raw()
                        if raw_submission == "":
                            return
                        accepted = self._engine.submit_answer(raw_submission)
                        if accepted:
                            runtime.clear_comms()
                            self._cognitive_updating_refresh_runtime_state()
                            self._input = ""
                        return
                    if action.startswith("tank:"):
                        try:
                            runtime.set_active_tank(int(action.split(":", 1)[1]))
                            self._cognitive_updating_refresh_runtime_state()
                        except ValueError:
                            pass
                        return
                    if action.startswith("parcel_field:"):
                        try:
                            runtime.set_parcel_field(int(action.split(":", 1)[1]))
                            self._cognitive_updating_refresh_runtime_state()
                        except ValueError:
                            pass
                        return
                    if action.startswith("objective_digit:"):
                        digit = action.split(":", 1)[1]
                        runtime.append_parcel_digit(digit)
                        self._cognitive_updating_refresh_runtime_state()
                        return
                    if action.startswith("comms_digit:"):
                        digit = action.split(":", 1)[1]
                        runtime.append_comms_digit(digit)
                        self._cognitive_updating_refresh_runtime_state()
                        return

        if event.type != pygame.KEYDOWN:
            return

        key = event.key

        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE) and not self._intro_loading_complete(
                snap.phase
            ):
                return
            if snap.phase is Phase.INSTRUCTIONS:
                if self._has_staged_difficulty_change():
                    self._persist_staged_difficulty_level()
                    self._restart_activity(auto_start_practice=True)
                    return
                if snap.title == "Target Recognition":
                    self._target_recognition_reset_practice_breakdown()
                    self._target_recognition_reset_scene_subtask()
                    self._target_recognition_reset_light_subtask()
                    self._target_recognition_reset_scan_subtask()
                    self._target_recognition_reset_system_subtask()
                if snap.title == "Vigilance":
                    self._vigilance_clear_inputs()
                self._engine.start_practice()
                self._input = ""
                self._math_choice = 1
                return
            if snap.phase is Phase.PRACTICE_DONE:
                if self._has_staged_difficulty_change():
                    self._persist_staged_difficulty_level()
                    self._restart_activity(auto_start_practice=False)
                    return
                self._engine.start_scored()
                self._input = ""
                self._math_choice = 1
                if snap.title == "Vigilance":
                    self._vigilance_clear_inputs()
                return
            if snap.phase is Phase.RESULTS:
                self._stop_auditory_audio()
                self._stop_situational_awareness_audio()
                self._app.pop()
                return
            if vigilance_payload is not None:
                self._vigilance_submit_inputs()
                return
            if tr_payload is not None:
                # Target Recognition answers are mouse-only in PRACTICE/SCORED.
                return
            if auditory_payload is not None:
                if auditory_payload.sequence_response_open:
                    accepted = self._engine.submit_answer(f"SEQ:{self._input}")
                    if accepted:
                        self._input = ""
                return
            if cognitive_updating_payload is not None:
                self._cognitive_updating_sync_payload(cognitive_updating_payload)
                runtime = self._cognitive_updating_runtime
                if runtime is None:
                    return
                raw_submission = runtime.build_submission_raw()
                if raw_submission == "":
                    return
                accepted = self._engine.submit_answer(raw_submission)
                if accepted:
                    runtime.clear_comms()
                    self._cognitive_updating_refresh_runtime_state()
                    self._input = ""
                return
            if dr is not None and not dr.accepting_input:
                return
            if (
                spatial_payload is not None
                and spatial_payload.trial_stage is not SpatialIntegrationTrialStage.QUESTION
            ):
                return
            if math_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if table_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if situational_awareness_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if angles_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if spatial_payload is not None and self._input == "":
                self._input = str(self._math_choice)
            if (
                trace_test_2_payload is not None
                and trace_test_2_payload.trial_stage is TraceTest2TrialStage.QUESTION
                and self._input == ""
            ):
                self._input = str(self._math_choice)
            if ic is not None and self._input == "":
                self._input = str(self._math_choice)

            raw_submission = self._input
            if scenario is not None:
                if raw_submission == "":
                    return
                answer_digits = max(1, int(getattr(scenario, "answer_digits", 4)))
                raw_submission = raw_submission.zfill(answer_digits)

            accepted = self._engine.submit_answer(raw_submission)
            if accepted:
                self._input = ""
                self._math_choice = 1
                if cognitive_updating_payload is not None:
                    self._cognitive_updating_comms_input = ""
            return

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            return

        if dr is not None and not dr.accepting_input:
            return

        if cln_payload is not None:
            memory_key = {
                pygame.K_a: 1,
                pygame.K_s: 2,
                pygame.K_d: 3,
                pygame.K_f: 4,
                pygame.K_g: 5,
            }.get(key)
            if memory_key is not None and cln_payload.options_active:
                self._engine.submit_answer(f"MEM:{memory_key}")
                return

            color_key = {
                pygame.K_q: "Q",
                pygame.K_w: "W",
                pygame.K_e: "E",
                pygame.K_r: "R",
            }.get(key)
            if color_key is not None:
                self._engine.submit_answer(f"CLR:{color_key}")
                return

        if auditory_payload is not None:
            color_key = {
                pygame.K_q: "BLUE",
                pygame.K_w: "GREEN",
                pygame.K_e: "YELLOW",
                pygame.K_r: "RED",
            }.get(key)
            if color_key is not None:
                self._engine.submit_answer(f"COL:{color_key}")
                return

            number_key = {
                pygame.K_KP0: 0,
                pygame.K_KP1: 1,
                pygame.K_KP2: 2,
                pygame.K_KP3: 3,
                pygame.K_KP4: 4,
                pygame.K_KP5: 5,
                pygame.K_KP6: 6,
                pygame.K_KP7: 7,
                pygame.K_KP8: 8,
                pygame.K_KP9: 9,
            }.get(key)
            if number_key is not None:
                self._engine.submit_answer(f"NUM:{number_key}")
                return

            if key == pygame.K_BACKSPACE:
                self._input = self._input[:-1]
                return

            ch = event.unicode
            if ch and ch.isdigit() and len(self._input) < 16:
                self._input += ch
            return

        if cognitive_updating_payload is not None:
            self._cognitive_updating_sync_payload(cognitive_updating_payload)
            runtime = self._cognitive_updating_runtime

            if key == pygame.K_TAB:
                delta = -1 if (event.mod & pygame.KMOD_SHIFT) else 1
                self._cognitive_updating_shift_upper_tab(delta)
                return
            if key in (pygame.K_q, pygame.K_LEFT):
                self._cognitive_updating_shift_upper_tab(-1)
                return
            if key in (pygame.K_e, pygame.K_RIGHT):
                self._cognitive_updating_shift_upper_tab(1)
                return
            if key in (pygame.K_a, pygame.K_UP):
                self._cognitive_updating_shift_lower_tab(-1)
                return
            if key in (pygame.K_d, pygame.K_DOWN):
                self._cognitive_updating_shift_lower_tab(1)
                return
            if key == pygame.K_BACKSPACE:
                # No backspace editing for Cognitive Updating.
                return

            if runtime is None:
                return

            nav_active = (self._cognitive_updating_upper_tab_index == 3) or (
                self._cognitive_updating_lower_tab_index == 3
            )
            controls_active = (self._cognitive_updating_upper_tab_index == 2) or (
                self._cognitive_updating_lower_tab_index == 2
            )

            if key in (pygame.K_MINUS, pygame.K_KP_MINUS) and nav_active:
                runtime.adjust_knots(-1)
                self._cognitive_updating_refresh_runtime_state()
                return
            if key in (pygame.K_EQUALS, pygame.K_KP_PLUS) and nav_active:
                runtime.adjust_knots(1)
                self._cognitive_updating_refresh_runtime_state()
                return

            ch = event.unicode
            if ch and ch.isdigit():
                if controls_active:
                    runtime.append_comms_digit(ch)
                    self._cognitive_updating_refresh_runtime_state()
                    return
                runtime.append_parcel_digit(ch)
                self._cognitive_updating_refresh_runtime_state()
                return
            if ch:
                return

        if system_logic_payload is not None:
            self._system_logic_sync_payload(system_logic_payload)

            if key == pygame.K_TAB:
                delta = -1 if (event.mod & pygame.KMOD_SHIFT) else 1
                self._system_logic_shift_folder(system_logic_payload, delta)
                return
            if key == pygame.K_LEFT:
                self._system_logic_shift_folder(system_logic_payload, -1)
                return
            if key == pygame.K_RIGHT:
                self._system_logic_shift_folder(system_logic_payload, 1)
                return
            if key == pygame.K_UP:
                self._system_logic_shift_document(system_logic_payload, -1)
                return
            if key == pygame.K_DOWN:
                self._system_logic_shift_document(system_logic_payload, 1)
                return

        if math_payload is not None:
            option_count = max(1, len(math_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if table_payload is not None:
            option_count = max(1, len(table_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if situational_awareness_payload is not None:
            option_count = max(1, len(situational_awareness_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if angles_payload is not None:
            option_count = max(1, len(angles_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if spatial_payload is not None:
            if spatial_payload.trial_stage is not SpatialIntegrationTrialStage.QUESTION:
                return
            option_count = max(1, len(spatial_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if trace_test_1_payload is not None:
            if trace_test_1_payload.trial_stage is not TraceTest1TrialStage.QUESTION:
                return
            command = {
                pygame.K_LEFT: "LEFT",
                pygame.K_RIGHT: "RIGHT",
                pygame.K_UP: "UP",
                pygame.K_DOWN: "DOWN",
            }.get(key)
            if command is None:
                return
            accepted = self._engine.submit_answer(command)
            if accepted:
                self._input = ""
                self._math_choice = 1
            return

        if trace_test_2_payload is not None:
            if trace_test_2_payload.trial_stage is not TraceTest2TrialStage.QUESTION:
                return
            option_count = max(1, len(trace_test_2_payload.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if ic is not None:
            option_count = max(1, len(ic.options))
            if key == pygame.K_UP:
                self._math_choice = (
                    option_count if self._math_choice <= 1 else self._math_choice - 1
                )
                self._input = str(self._math_choice)
                return
            if key == pygame.K_DOWN:
                self._math_choice = (
                    1 if self._math_choice >= option_count else self._math_choice + 1
                )
                self._input = str(self._math_choice)
                return

            if self._apply_multiple_choice_key(key=key, option_count=option_count):
                return
            return

        if vigilance_payload is not None:
            if key in (pygame.K_LEFT, pygame.K_UP):
                self._vigilance_focus = "row"
                return
            if key in (pygame.K_RIGHT, pygame.K_DOWN):
                self._vigilance_focus = "col"
                return
            if key == pygame.K_TAB:
                self._vigilance_focus = "col" if self._vigilance_focus == "row" else "row"
                return
            if key == pygame.K_BACKSPACE:
                if self._vigilance_focus == "row":
                    self._vigilance_row_input = self._vigilance_row_input[:-1]
                else:
                    self._vigilance_col_input = self._vigilance_col_input[:-1]
                return

            ch = event.unicode
            if ch and ch.isdigit() and ch != "0":
                if self._vigilance_focus == "row":
                    self._vigilance_row_input = ch
                    if self._vigilance_col_input == "":
                        self._vigilance_focus = "col"
                else:
                    self._vigilance_col_input = ch
                if self._vigilance_row_input != "" and self._vigilance_col_input != "":
                    self._vigilance_submit_inputs()
                return
            return

        if sensory_payload is not None:
            return

        if rapid_tracking_payload is not None:
            if key == pygame.K_SPACE:
                self._engine.submit_answer("CAPTURE")
            return

        if tr_payload is not None:
            return

        if key == pygame.K_BACKSPACE:
            # Airborne test: no backspace editing.
            if scenario is None:
                self._input = self._input[:-1]
            return

        ch = event.unicode
        if scenario is not None:
            answer_digits = max(1, int(getattr(scenario, "answer_digits", 4)))
            if ch and ch.isdigit() and len(self._input) < answer_digits:
                self._input += ch
            return

        if vs is not None:
            if ch and ch.isdigit() and len(self._input) < 2:
                self._input += ch
            return

        if ch and (ch.isdigit() or (ch == "-" and self._input == "")):
            self._input += ch

    def _handle_pause_menu_event(self, event: pygame.event.Event) -> None:
        if self._pause_menu_mode == "settings":
            self._handle_pause_settings_event(event)
            return
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._pause_menu_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._pause_menu_selected = idx
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for idx, rect in self._pause_menu_hitboxes.items():
                if rect.collidepoint(pos):
                    self._pause_menu_selected = idx
                    self._activate_pause_menu_selection()
                    return
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._pause_menu_active = False
            self._pause_menu_mode = "menu"
            self._staged_difficulty_level = None
            return
        option_count = len(self._pause_menu_options())
        if key in (pygame.K_UP, pygame.K_w):
            self._pause_menu_selected = (self._pause_menu_selected - 1) % option_count
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self._pause_menu_selected = (self._pause_menu_selected + 1) % option_count
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._activate_pause_menu_selection()

    def _activate_pause_menu_selection(self) -> None:
        selected = self._pause_menu_selected % len(self._pause_menu_options())
        if selected == 0:
            self._pause_menu_active = False
            self._pause_menu_mode = "menu"
            self._staged_difficulty_level = None
            return
        if selected == 1:
            self._pause_menu_mode = "settings"
            self._pause_settings_selected = 0
            return
        self._pause_menu_active = False
        self._pause_menu_mode = "menu"
        self._staged_difficulty_level = None
        self._stop_auditory_audio()
        self._stop_situational_awareness_audio()
        self._app.pop_to_root()

    @staticmethod
    def _pause_menu_options() -> tuple[str, ...]:
        return ("Resume", "Settings", "Main Menu")

    def _handle_pause_settings_event(self, event: pygame.event.Event) -> None:
        rows = self._pause_settings_rows()
        if not rows:
            self._pause_menu_mode = "menu"
            self._pause_settings_selected = 0
            return
        if event.type == pygame.MOUSEMOTION:
            pos = getattr(event, "pos", None)
            if pos is not None:
                for idx, rect in self._pause_settings_hitboxes.items():
                    if rect.collidepoint(pos):
                        self._pause_settings_selected = idx
                        break
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            for (idx, action), rect in self._pause_settings_control_hitboxes.items():
                if rect.collidepoint(pos):
                    self._pause_settings_selected = idx
                    self._adjust_pause_setting(index=idx, direction=-1 if action == "dec" else 1)
                    return
            for idx, rect in self._pause_settings_hitboxes.items():
                if rect.collidepoint(pos):
                    self._pause_settings_selected = idx
                    selected_key = rows[idx][0]
                    if selected_key == "apply_restart":
                        self._apply_pause_restart()
                    if selected_key == "back":
                        self._pause_menu_mode = "menu"
                    return
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._pause_menu_mode = "menu"
            self._pause_settings_selected = 0
            return
        row_count = len(rows)
        if key in (pygame.K_UP, pygame.K_w):
            self._pause_settings_selected = (self._pause_settings_selected - 1) % row_count
            return
        if key in (pygame.K_DOWN, pygame.K_s):
            self._pause_settings_selected = (self._pause_settings_selected + 1) % row_count
            return
        if key in (pygame.K_LEFT, pygame.K_a):
            self._adjust_pause_setting(index=self._pause_settings_selected, direction=-1)
            return
        if key in (pygame.K_RIGHT, pygame.K_d):
            self._adjust_pause_setting(index=self._pause_settings_selected, direction=1)
            return
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            selected_key = rows[self._pause_settings_selected][0]
            if selected_key == "apply_restart":
                self._apply_pause_restart()
            if selected_key == "back":
                self._pause_menu_mode = "menu"

    def _pause_settings_rows(self) -> list[tuple[str, str, str]]:
        rows = [
            (
                "difficulty",
                "Difficulty",
                f"{self._get_pause_difficulty_level()} / 10",
            )
        ]
        if self._supports_auditory_pause_settings():
            rows.extend(
                [
                    (
                        "auditory_noise",
                        "Noise Level",
                        f"{self._get_auditory_noise_level_step()} / 10",
                    ),
                    (
                        "auditory_distortion",
                        "Distortion",
                        f"{self._get_auditory_distortion_level_step()} / 10",
                    ),
                    (
                        "auditory_source",
                        "Noise Source",
                        _AuditoryCapacityAudioAdapter.format_noise_source_label(
                            self._get_auditory_noise_source_override()
                        ),
                    ),
                ]
            )
        rows.append(
            (
                "apply_restart",
                "Apply & Restart",
                "Restart from the beginning with this difficulty",
            )
        )
        rows.append(("back", "Back", "Return to Pause Menu"))
        return rows

    def _adjust_pause_setting(self, *, index: int, direction: int) -> None:
        rows = self._pause_settings_rows()
        if not rows:
            return
        key = rows[index % len(rows)][0]
        if key == "difficulty":
            level = self._get_pause_difficulty_level()
            self._set_pause_difficulty_level(level + direction)
            return
        if key == "auditory_noise":
            step = self._get_auditory_noise_level_step()
            self._apply_auditory_pause_settings(noise_step=step + direction)
            return
        if key == "auditory_distortion":
            step = self._get_auditory_distortion_level_step()
            self._apply_auditory_pause_settings(distortion_step=step + direction)
            return
        if key == "auditory_source":
            options = self._auditory_noise_source_options()
            if not options:
                return
            current = self._get_auditory_noise_source_override()
            try:
                current_index = options.index(current)
            except ValueError:
                current_index = 0
            next_index = (current_index + direction) % len(options)
            self._apply_auditory_pause_settings(noise_source=options[next_index])
            return
        if key == "apply_restart":
            self._apply_pause_restart()
            return
        if key == "back":
            self._pause_menu_mode = "menu"

    def _get_pause_difficulty_level(self) -> int:
        if self._staged_difficulty_level is not None:
            return int(self._staged_difficulty_level)
        return self._current_engine_difficulty_level()

    def _set_pause_difficulty_level(self, level: int) -> None:
        self._staged_difficulty_level = max(1, min(10, int(level)))

    def _apply_pause_restart(self) -> None:
        self._persist_staged_difficulty_level()
        self._pause_menu_active = False
        self._pause_menu_mode = "menu"
        self._restart_activity(auto_start_practice=False)

    def _get_intro_difficulty_level(self) -> int:
        if self._staged_difficulty_level is not None:
            return int(self._staged_difficulty_level)
        if self._test_code is not None:
            return self._app.effective_difficulty_level(self._test_code)
        return self._current_engine_difficulty_level()

    def _set_intro_difficulty_level(self, level: int) -> None:
        self._staged_difficulty_level = max(1, min(10, int(level)))

    def _handle_intro_difficulty_event(
        self,
        event: pygame.event.Event,
        *,
        phase: Phase,
    ) -> bool:
        if phase not in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            return False
        if event.type == pygame.MOUSEMOTION:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 0) == 1:
            pos = getattr(event, "pos", None)
            if pos is None:
                return False
            if (rect := self._intro_difficulty_control_hitboxes.get("dec")) and rect.collidepoint(pos):
                self._set_intro_difficulty_level(self._get_intro_difficulty_level() - 1)
                return True
            if (rect := self._intro_difficulty_control_hitboxes.get("inc")) and rect.collidepoint(pos):
                self._set_intro_difficulty_level(self._get_intro_difficulty_level() + 1)
                return True
            return False
        if event.type != pygame.KEYDOWN:
            return False
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self._set_intro_difficulty_level(self._get_intro_difficulty_level() - 1)
            return True
        if event.key in (pygame.K_RIGHT, pygame.K_d):
            self._set_intro_difficulty_level(self._get_intro_difficulty_level() + 1)
            return True
        return False

    def _supports_auditory_pause_settings(self) -> bool:
        return hasattr(self._engine, "set_audio_overrides") or (
            self._engine.snapshot().title == "Auditory Capacity"
        )

    @staticmethod
    def _step_to_ratio(step: int) -> float:
        clamped = max(0, min(10, int(step)))
        return clamped / 10.0

    def _auditory_noise_source_options(self) -> tuple[str | None, ...]:
        return (None,) + _AuditoryCapacityAudioAdapter.available_noise_sources()

    def _get_auditory_noise_level_step(self) -> int:
        override = getattr(self._engine, "_noise_level_override", None)
        if isinstance(override, int | float):
            return max(0, min(10, int(round(float(override) * 10.0))))
        snap = self._engine.snapshot()
        payload = snap.payload
        if isinstance(payload, AuditoryCapacityPayload):
            return max(0, min(10, int(round(float(payload.background_noise_level) * 10.0))))
        return 0

    def _get_auditory_distortion_level_step(self) -> int:
        override = getattr(self._engine, "_distortion_level_override", None)
        if isinstance(override, int | float):
            return max(0, min(10, int(round(float(override) * 10.0))))
        snap = self._engine.snapshot()
        payload = snap.payload
        if isinstance(payload, AuditoryCapacityPayload):
            return max(0, min(10, int(round(float(payload.distortion_level) * 10.0))))
        return 0

    def _get_auditory_noise_source_override(self) -> str | None:
        override = getattr(self._engine, "_noise_source_override", None)
        if isinstance(override, str):
            stripped = override.strip()
            return stripped or None
        return None

    def _apply_auditory_pause_settings(
        self,
        *,
        noise_step: int | None = None,
        distortion_step: int | None = None,
        noise_source: str | None | object = _SETTINGS_UNSET,
    ) -> None:
        if not hasattr(self._engine, "set_audio_overrides"):
            return
        current_noise = getattr(self._engine, "_noise_level_override", None)
        current_distortion = getattr(self._engine, "_distortion_level_override", None)
        current_source = getattr(self._engine, "_noise_source_override", None)
        next_noise = current_noise
        next_distortion = current_distortion
        next_source = current_source
        if noise_step is not None:
            next_noise = self._step_to_ratio(noise_step)
        if distortion_step is not None:
            next_distortion = self._step_to_ratio(distortion_step)
        if noise_source is not _SETTINGS_UNSET:
            next_source = noise_source
        self._engine.set_audio_overrides(
            noise_level=next_noise,
            distortion_level=next_distortion,
            noise_source=cast(str | None, next_source),
        )

    def _debug_skip_forward(self) -> None:
        self._input = ""
        self._math_choice = 1
        snap = self._engine.snapshot()
        phase = snap.phase
        if phase is Phase.RESULTS:
            return

        if phase is Phase.INSTRUCTIONS:
            self._engine.start_scored()
            self._engine.update()
            if self._engine.snapshot().phase is Phase.INSTRUCTIONS:
                self._engine.start_practice()
                if self._engine.snapshot().phase is Phase.PRACTICE:
                    self._debug_skip_forward()
            return

        if phase is Phase.PRACTICE:
            for token in ("__skip_practice__", "skip_practice"):
                if self._engine.submit_answer(token):
                    self._show_intro_loading_screen(Phase.PRACTICE_DONE)
                    return
            self._show_intro_loading_screen(Phase.PRACTICE_DONE)
            return

        if phase is Phase.PRACTICE_DONE:
            self._show_intro_loading_screen(Phase.PRACTICE_DONE)
            return

        for token in (
            "__skip_section__",
            "__skip_all__",
            "skip_section",
            "skip_all",
        ):
            if self._engine.submit_answer(token):
                return

        self._force_engine_phase(Phase.RESULTS)

    def _force_engine_phase(self, phase: Phase) -> None:
        if not hasattr(self._engine, "_phase"):
            return
        self._engine._phase = phase
        for attr in ("_current", "_presented_at_s"):
            if hasattr(self._engine, attr):
                setattr(self._engine, attr, None)
        if hasattr(self._engine, "_pending_done_action"):
            if phase is Phase.PRACTICE_DONE:
                self._engine._pending_done_action = "start_scored"
            elif phase is Phase.RESULTS:
                self._engine._pending_done_action = None

    def render(self, surface: pygame.Surface) -> None:
        if self._pause_menu_active:
            snap = self._engine.snapshot()
        else:
            if isinstance(self._engine, (SensoryMotorApparatusEngine, RapidTrackingEngine)):
                control_x, control_y = self._read_sensory_motor_control()
                self._engine.set_control(horizontal=control_x, vertical=control_y)
            elif isinstance(self._engine, AuditoryCapacityEngine):
                control_x, control_y = self._read_sensory_motor_control()
                self._engine.set_control(horizontal=control_x, vertical=control_y)

            # Update engine and take a fresh snapshot.
            self._engine.update()
            snap = self._engine.snapshot()

            # If the timer expires mid-entry, auto-submit what was typed so far.
            if (
                snap.phase is Phase.SCORED
                and snap.time_remaining_s is not None
                and snap.time_remaining_s <= 0.0
                and self._input.strip() != ""
                and not isinstance(snap.payload, TargetRecognitionPayload)
            ):
                if isinstance(snap.payload, CognitiveUpdatingPayload):
                    self._cognitive_updating_sync_payload(snap.payload)
                    runtime = self._cognitive_updating_runtime
                    raw_submission = runtime.build_submission_raw() if runtime is not None else ""
                    if raw_submission != "":
                        self._engine.submit_answer(raw_submission)
                else:
                    self._engine.submit_answer(self._input)
                self._input = ""
                self._engine.update()
                snap = self._engine.snapshot()

        self._sync_intro_loading_state(snap.phase)

        if self._pause_menu_active and snap.phase not in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.PRACTICE_DONE,
            Phase.SCORED,
        ):
            self._pause_menu_active = False
            self._pause_menu_mode = "menu"

        self._persist_results_if_needed(snap)

        prompt_key: tuple[str, str] | None = None
        if snap.phase in (Phase.PRACTICE, Phase.SCORED) and (
            snap.payload is None or isinstance(snap.payload, AirborneScenario)
        ):
            prompt_key = (snap.phase.value, str(snap.prompt))
        if prompt_key != self._input_prompt_key:
            if self._input_prompt_key is not None and prompt_key is not None:
                self._input = ""
            self._input_prompt_key = prompt_key

        # Identify payloads.
        p = snap.payload
        scenario: AirborneScenario | None = p if isinstance(p, AirborneScenario) else None
        abd: AnglesBearingsDegreesPayload | None = (
            p if isinstance(p, AnglesBearingsDegreesPayload) else None
        )
        ic: InstrumentComprehensionPayload | None = (
            p if isinstance(p, InstrumentComprehensionPayload) else None
        )
        tr: TargetRecognitionPayload | None = p if isinstance(p, TargetRecognitionPayload) else None
        vs: VisualSearchPayload | None = p if isinstance(p, VisualSearchPayload) else None
        vigilance_payload: VigilancePayload | None = p if isinstance(p, VigilancePayload) else None
        mr: MathReasoningPayload | None = p if isinstance(p, MathReasoningPayload) else None
        sensory_payload: SensoryMotorApparatusPayload | None = (
            p if isinstance(p, SensoryMotorApparatusPayload) else None
        )
        rapid_tracking_payload: RapidTrackingPayload | None = (
            p if isinstance(p, RapidTrackingPayload) else None
        )
        table_payload: TableReadingPayload | None = (
            p if isinstance(p, TableReadingPayload) else None
        )
        spatial_payload: SpatialIntegrationPayload | None = (
            p if isinstance(p, SpatialIntegrationPayload) else None
        )
        trace_test_1_payload: TraceTest1Payload | None = (
            p if isinstance(p, TraceTest1Payload) else None
        )
        trace_test_2_payload: TraceTest2Payload | None = (
            p if isinstance(p, TraceTest2Payload) else None
        )
        sa_payload: SituationalAwarenessPayload | None = (
            p if isinstance(p, SituationalAwarenessPayload) else None
        )
        sl: SystemLogicPayload | None = p if isinstance(p, SystemLogicPayload) else None
        cu: CognitiveUpdatingPayload | None = p if isinstance(p, CognitiveUpdatingPayload) else None
        cln: ColoursLettersNumbersPayload | None = (
            p if isinstance(p, ColoursLettersNumbersPayload) else None
        )
        ac: AuditoryCapacityPayload | None = p if isinstance(p, AuditoryCapacityPayload) else None
        if self._pause_menu_active:
            self._stop_auditory_audio()
            self._stop_situational_awareness_audio()
        else:
            self._sync_auditory_audio(phase=snap.phase, payload=ac)
            self._sync_situational_awareness_audio(phase=snap.phase, payload=sa_payload)
        dr: DigitRecognitionPayload | None = None
        if p is not None:
            if isinstance(p, DigitRecognitionPayload):
                dr = p
            elif hasattr(p, "display_digits") and hasattr(p, "accepting_input"):
                dr = cast(DigitRecognitionPayload, p)

        is_numerical_ops = snap.title == "Numerical Operations"
        is_math_reasoning = snap.title == "Mathematics Reasoning"
        is_angles_bearings = snap.title == "Angles, Bearings and Degrees"
        is_visual_search = snap.title == "Visual Search"
        is_vigilance = snap.title == "Vigilance"
        is_digit_recognition = snap.title == "Digit Recognition"
        is_colours_letters_numbers = snap.title == "Colours, Letters and Numbers"
        is_instrument_comprehension = snap.title == "Instrument Comprehension"
        is_target_recognition = snap.title == "Target Recognition"
        is_system_logic = snap.title == "System Logic"
        is_cognitive_updating = snap.title == "Cognitive Updating"
        is_sensory_motor_apparatus = snap.title == "Sensory Motor Apparatus"
        is_rapid_tracking = snap.title == "Rapid Tracking"
        is_spatial_integration = snap.title == "Spatial Integration"
        is_trace_test_1 = snap.title == "Trace Test 1"
        is_trace_test_2 = snap.title == "Trace Test 2"
        is_table_reading = snap.title == "Table Reading"
        is_auditory_capacity = snap.title == "Auditory Capacity"
        is_situational_awareness = snap.title == "Situational Awareness"
        if is_numerical_ops and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            self._render_numerical_operations_question(surface, snap)
        elif is_math_reasoning:
            self._render_math_reasoning(surface, snap, mr)
        elif is_sensory_motor_apparatus:
            self._render_sensory_motor_apparatus_screen(surface, snap, sensory_payload)
        elif is_rapid_tracking:
            self._render_rapid_tracking_screen(surface, snap, rapid_tracking_payload)
        elif is_spatial_integration:
            self._render_spatial_integration_screen(surface, snap, spatial_payload)
        elif is_trace_test_1:
            self._render_trace_test_1_screen(surface, snap, trace_test_1_payload)
        elif is_trace_test_2:
            self._render_trace_test_2_screen(surface, snap, trace_test_2_payload)
        elif is_auditory_capacity:
            self._render_auditory_capacity_screen(surface, snap, ac)
        elif is_situational_awareness:
            self._render_situational_awareness_screen(surface, snap, sa_payload)
        elif is_table_reading:
            self._render_table_reading_screen(surface, snap, table_payload)
        elif is_system_logic:
            self._render_system_logic_screen(surface, snap, sl)
        elif is_cognitive_updating:
            self._render_cognitive_updating_screen(surface, snap, cu)
        elif is_angles_bearings:
            self._render_angles_bearings_screen(surface, snap, abd)
        elif is_colours_letters_numbers:
            self._render_colours_letters_numbers_screen(surface, snap, cln)
        elif is_digit_recognition:
            self._render_digit_recognition_screen(surface, snap, dr)
        elif is_instrument_comprehension:
            self._render_instrument_comprehension_screen(surface, snap, ic)
        elif is_target_recognition:
            self._render_target_recognition_screen(surface, snap, tr)
        elif is_vigilance:
            self._render_vigilance_screen(surface, snap, vigilance_payload)
        elif is_visual_search:
            self._render_visual_search_question(surface, snap, vs)
        else:
            surface.fill((10, 10, 14))

            title = self._app.font.render(snap.title, True, (235, 235, 245))
            surface.blit(title, (40, 30))

            y_info = 80
            if snap.time_remaining_s is not None:
                rem = int(round(snap.time_remaining_s))
                mm = rem // 60
                ss = rem % 60
                timer = self._small_font.render(
                    f"Time remaining: {mm:02d}:{ss:02d}", True, (200, 200, 210)
                )
                surface.blit(timer, (40, y_info))
                y_info += 28

            stats = self._small_font.render(
                f"Scored: {snap.correct_scored}/{snap.attempted_scored}", True, (180, 180, 190)
            )
            surface.blit(stats, (40, y_info))

            if scenario is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
                self._render_airborne_question(surface, snap, scenario)
            else:
                if snap.phase in (Phase.PRACTICE, Phase.SCORED):
                    prompt_rect = pygame.Rect(
                        max(36, surface.get_width() // 10),
                        max(90, surface.get_height() // 5),
                        surface.get_width() - max(72, surface.get_width() // 5),
                        max(120, min(200, surface.get_height() // 3)),
                    )
                    self._render_centered_prompt_panel(
                        surface,
                        prompt_rect,
                        str(snap.prompt),
                        fill=(18, 24, 72),
                        border=(102, 118, 178),
                        text_color=(235, 235, 245),
                        preferred_size=max(28, min(46, surface.get_height() // 14)),
                        min_size=max(18, min(28, surface.get_height() // 22)),
                    )
                else:
                    prompt_lines = str(snap.prompt).split("\n")
                    y = 140
                    for line in prompt_lines[:10]:
                        txt = self._small_font.render(line, True, (235, 235, 245))
                        surface.blit(txt, (40, y))
                        y += 26

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            if is_numerical_ops:
                self._render_numerical_operations_answer_box(surface, snap)
            elif is_math_reasoning:
                pass
            elif is_sensory_motor_apparatus:
                pass
            elif is_rapid_tracking:
                pass
            elif is_spatial_integration:
                pass
            elif is_trace_test_1:
                pass
            elif is_auditory_capacity:
                self._render_auditory_capacity_answer_box(surface, snap, ac)
            elif is_situational_awareness:
                pass
            elif is_table_reading:
                pass
            elif is_system_logic:
                pass
            elif is_cognitive_updating:
                pass
            elif is_angles_bearings:
                pass
            elif is_colours_letters_numbers:
                pass
            elif is_digit_recognition:
                self._render_digit_recognition_answer_box(surface, snap, dr)
            elif is_instrument_comprehension:
                pass
            elif is_target_recognition:
                pass
            elif vs is not None or vigilance_payload is not None:
                pass
            elif scenario is None and (dr is None or dr.accepting_input):
                box_w = max(300, min(520, int(surface.get_width() * 0.48)))
                box_h = max(56, min(72, int(surface.get_height() * 0.10)))
                box = pygame.Rect(
                    (surface.get_width() - box_w) // 2,
                    int(surface.get_height() * 0.68),
                    box_w,
                    box_h,
                )
                self._render_centered_input_box(
                    surface,
                    box,
                    label="Answer",
                    hint=snap.input_hint,
                    entry_text=self._input,
                    fill=(30, 30, 40),
                    border=(90, 90, 110),
                    label_color=(235, 235, 245),
                    input_color=(235, 235, 245),
                    hint_color=(140, 140, 150),
                    label_font=self._small_font,
                    input_font=self._app.font,
                    hint_font=self._small_font,
                )

        if not self._pause_menu_active:
            self._render_feedback_banner(surface, snap)
            self._render_test_intro_overlay(surface, snap)
            self._render_intro_difficulty_overlay(surface, snap)
            self._render_intro_loading_indicator(surface, snap)

        if not self._engine.can_exit() and snap.phase is Phase.SCORED:
            lock = self._small_font.render("Test in progress: cannot exit.", True, (140, 140, 150))
            surface.blit(lock, (460, surface.get_height() - 60))

        if snap.phase is Phase.RESULTS and not self._pause_menu_active:
            self._render_standard_results_overlay(surface, snap)

        if self._pause_menu_active:
            self._render_pause_overlay(surface)
        else:
            self._advance_intro_loading(surface_size=surface.get_size(), snap=snap)

    def _persist_results_if_needed(self, snap: TestSnapshot) -> None:
        if snap.phase is not Phase.RESULTS or self._results_persisted:
            return
        self._results_persisted = True
        if self._test_code is None:
            return
        self._results_persistence_lines = self._app.persist_attempt(
            engine=self._engine,
            test_code=self._test_code,
            test_version=self._test_version,
        )

    def _render_standard_results_overlay(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        w, h = surface.get_size()

        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((4, 10, 30, 190))
        surface.blit(dim, (0, 0))

        panel_w = min(max(420, int(w * 0.82)), 1020)
        panel_h = min(max(290, int(h * 0.76)), h - 26)
        panel = pygame.Rect((w - panel_w) // 2, (h - panel_h) // 2, panel_w, panel_h)

        pygame.draw.rect(surface, (6, 18, 108), panel, border_radius=14)
        pygame.draw.rect(surface, (230, 238, 255), panel, 2, border_radius=14)

        title_size = max(28, min(64, h // 13))
        subtitle_size = max(24, min(52, h // 16))
        hint_size = max(18, min(30, h // 34))

        title_font = pygame.font.Font(None, title_size)
        subtitle_font = pygame.font.Font(None, subtitle_size)
        hint_font = pygame.font.Font(None, hint_size)

        title = title_font.render(snap.title, True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 18)))

        subtitle = subtitle_font.render("Results", True, (188, 204, 228))
        subtitle_top = panel.y + 18 + title.get_height() + 4
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, subtitle_top)))

        prompt_lines = [
            line.strip() for line in str(snap.prompt).splitlines() if line.strip() != ""
        ]
        if prompt_lines and prompt_lines[0].casefold() == "results":
            prompt_lines = prompt_lines[1:]

        score_line = f"Scored {snap.correct_scored}/{snap.attempted_scored}"
        content_lines = [score_line] + prompt_lines
        if not prompt_lines:
            content_lines = [score_line]
        if self._results_persistence_lines:
            content_lines.extend([""] + self._results_persistence_lines)

        body_top = panel.y + 18 + title.get_height() + subtitle.get_height() + 22
        body_bottom = panel.bottom - hint_font.get_linesize() - 24
        body_rect = pygame.Rect(
            panel.x + 26,
            body_top,
            panel.w - 52,
            max(40, body_bottom - body_top),
        )

        body_font, wrapped_lines = self._fit_results_text(
            lines=content_lines,
            max_width=body_rect.w,
            max_height=body_rect.h,
            preferred_size=max(20, min(34, h // 22)),
            min_size=max(14, min(22, h // 34)),
        )
        line_h = body_font.get_linesize() + 4
        start_y = body_rect.y + max(0, (body_rect.h - (len(wrapped_lines) * line_h)) // 2)
        y = start_y
        for line in wrapped_lines:
            text = body_font.render(line, True, (238, 245, 255))
            surface.blit(text, text.get_rect(midtop=(panel.centerx, y)))
            y += line_h

        hint = hint_font.render("Enter: Return to Tests", True, (188, 204, 228))
        surface.blit(hint, hint.get_rect(midbottom=(panel.centerx, panel.bottom - 12)))

    def _fit_results_text(
        self,
        *,
        lines: list[str],
        max_width: int,
        max_height: int,
        preferred_size: int,
        min_size: int,
    ) -> tuple[pygame.font.Font, list[str]]:
        if max_width <= 0 or max_height <= 0:
            fallback = pygame.font.Font(None, max(14, min_size))
            return fallback, [""]

        size = max(preferred_size, min_size)
        while size >= min_size:
            font = pygame.font.Font(None, size)
            wrapped = self._wrap_centered_lines(lines=lines, font=font, max_width=max_width)
            line_h = font.get_linesize() + 4
            if len(wrapped) * line_h <= max_height:
                return font, wrapped
            size -= 1

        font = pygame.font.Font(None, max(14, min_size))
        wrapped = self._wrap_centered_lines(lines=lines, font=font, max_width=max_width)
        line_h = font.get_linesize() + 4
        max_lines = max(1, max_height // max(1, line_h))
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            if wrapped:
                tail = wrapped[-1]
                while tail and font.size(f"{tail}...")[0] > max_width:
                    tail = tail[:-1]
                wrapped[-1] = f"{tail}..." if tail else "..."
        return font, wrapped

    def _wrap_centered_lines(
        self,
        *,
        lines: list[str],
        font: pygame.font.Font,
        max_width: int,
    ) -> list[str]:
        out: list[str] = []
        for raw in lines:
            text = raw.strip()
            if text == "":
                out.append("")
                continue

            words = text.split()
            current = ""
            for word in words:
                trial = word if current == "" else f"{current} {word}"
                if font.size(trial)[0] <= max_width:
                    current = trial
                else:
                    if current:
                        out.append(current)
                    current = word
            if current:
                out.append(current)

        return out if out else [""]

    def _render_feedback_banner(self, surface: pygame.Surface, snap: TestSnapshot) -> None:
        feedback = getattr(snap, "practice_feedback", None)
        if snap.phase not in (Phase.PRACTICE, Phase.SCORED) or not feedback:
            return
        text = str(feedback).strip()
        if text == "":
            return

        max_width = min(surface.get_width() - 40, 760)
        lines = self._wrap_text_lines(text, self._tiny_font, max_width - 24)
        line_h = self._tiny_font.get_linesize() + 2
        height = 26 + len(lines) * line_h
        panel = pygame.Rect(0, 0, max_width, height)
        panel.centerx = surface.get_width() // 2
        panel.y = 20

        tint = pygame.Surface(panel.size, pygame.SRCALPHA)
        tint.fill((6, 18, 58, 212))
        surface.blit(tint, panel.topleft)
        pygame.draw.rect(surface, (226, 236, 255), panel, 1, border_radius=10)
        y = panel.y + 12
        for line in lines:
            surf = self._tiny_font.render(line, True, (238, 245, 255))
            surface.blit(surf, surf.get_rect(midtop=(panel.centerx, y)))
            y += line_h

    def _render_centered_prompt_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        text: str,
        *,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        text_color: tuple[int, int, int],
        preferred_size: int,
        min_size: int,
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=14)
        pygame.draw.rect(surface, border, rect, 2, border_radius=14)

        lines = [line.strip() for line in str(text).splitlines() if line.strip() != ""]
        if not lines:
            lines = [""]

        body = rect.inflate(-20, -18)
        font, wrapped_lines = self._fit_results_text(
            lines=lines,
            max_width=body.w,
            max_height=body.h,
            preferred_size=preferred_size,
            min_size=min_size,
        )
        line_h = font.get_linesize() + 4
        y = body.y + max(0, (body.h - (len(wrapped_lines) * line_h)) // 2)
        for line in wrapped_lines:
            surf = font.render(line, True, text_color)
            surface.blit(surf, surf.get_rect(midtop=(rect.centerx, y)))
            y += line_h

    def _render_centered_input_box(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        label: str,
        hint: str,
        entry_text: str,
        fill: tuple[int, int, int],
        border: tuple[int, int, int],
        label_color: tuple[int, int, int],
        input_color: tuple[int, int, int],
        hint_color: tuple[int, int, int],
        label_font: pygame.font.Font,
        input_font: pygame.font.Font,
        hint_font: pygame.font.Font,
    ) -> None:
        pygame.draw.rect(surface, fill, rect, border_radius=12)
        pygame.draw.rect(surface, border, rect, 2, border_radius=12)

        label_surf = label_font.render(label, True, label_color)
        surface.blit(label_surf, label_surf.get_rect(midbottom=(rect.centerx, rect.y - 8)))

        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry_surf = input_font.render(entry_text + caret, True, input_color)
        if entry_surf.get_width() > rect.w - 24:
            entry_surf = self._small_font.render(entry_text + caret, True, input_color)
        surface.blit(entry_surf, entry_surf.get_rect(center=rect.center))

        hint_surf = hint_font.render(hint, True, hint_color)
        surface.blit(hint_surf, hint_surf.get_rect(midtop=(rect.centerx, rect.bottom + 10)))

    def _render_pause_overlay(self, surface: pygame.Surface) -> None:
        dim = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        dim.fill((4, 8, 18, 182))
        surface.blit(dim, (0, 0))

        if self._pause_menu_mode == "settings":
            self._render_pause_settings_overlay(surface)
            return

        panel_w = min(480, max(320, surface.get_width() - 64))
        panel_h = 332
        panel = pygame.Rect(
            (surface.get_width() - panel_w) // 2,
            (surface.get_height() - panel_h) // 2,
            panel_w,
            panel_h,
        )

        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=10)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=10)

        title = self._app.font.render("Paused", True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 16)))

        subtitle = self._tiny_font.render(
            "Resume / Settings / Main Menu",
            True,
            (188, 204, 228),
        )
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 56)))

        options = self._pause_menu_options()
        row_h = 42
        gap = 12
        total_h = (row_h * len(options)) + (gap * (len(options) - 1))
        y = panel.y + 88 + max(0, (panel.h - 178 - total_h) // 2)
        self._pause_menu_hitboxes = {}
        for idx, label in enumerate(options):
            row = pygame.Rect(panel.x + 28, y, panel.w - 56, row_h)
            self._pause_menu_hitboxes[idx] = row.copy()
            selected = idx == (self._pause_menu_selected % len(options))
            if selected:
                pygame.draw.rect(surface, (244, 248, 255), row, border_radius=6)
                pygame.draw.rect(surface, (120, 142, 196), row, 2, border_radius=6)
                color = (14, 26, 74)
            else:
                pygame.draw.rect(surface, (9, 20, 106), row, border_radius=6)
                pygame.draw.rect(surface, (62, 84, 152), row, 1, border_radius=6)
                color = (238, 245, 255)
            text = self._small_font.render(label, True, color)
            surface.blit(text, text.get_rect(center=row.center))
            y += row_h + gap

        help_text = self._tiny_font.render(
            "Up/Down: Select  Enter: Confirm  Esc/Backspace: Back",
            True,
            (188, 204, 228),
        )
        surface.blit(help_text, help_text.get_rect(midbottom=(panel.centerx, panel.bottom - 14)))

    def _render_pause_settings_overlay(self, surface: pygame.Surface) -> None:
        panel_w = min(680, max(380, surface.get_width() - 64))
        rows = self._pause_settings_rows()
        row_h = 48
        gap = 10
        panel_h = min(
            surface.get_height() - 32,
            max(320, 146 + (len(rows) * row_h) + (max(0, len(rows) - 1) * gap)),
        )
        panel = pygame.Rect(
            (surface.get_width() - panel_w) // 2,
            (surface.get_height() - panel_h) // 2,
            panel_w,
            panel_h,
        )

        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=10)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=10)

        title = self._app.font.render("Test Settings", True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(panel.centerx, panel.y + 16)))

        subtitle = self._tiny_font.render(
            "Left/Right or A/D: Stage difficulty  Enter: Apply row  Esc: Back to Pause Menu",
            True,
            (188, 204, 228),
        )
        surface.blit(subtitle, subtitle.get_rect(midtop=(panel.centerx, panel.y + 56)))

        self._pause_settings_hitboxes = {}
        self._pause_settings_control_hitboxes = {}
        y = panel.y + 92
        selected_index = self._pause_settings_selected % max(1, len(rows))
        for idx, (key, label, value) in enumerate(rows):
            row = pygame.Rect(panel.x + 24, y, panel.w - 48, row_h)
            self._pause_settings_hitboxes[idx] = row.copy()
            selected = idx == selected_index
            fill = (242, 246, 255) if selected else (9, 20, 106)
            border = (120, 142, 196) if selected else (62, 84, 152)
            text_color = (14, 26, 74) if selected else (238, 245, 255)
            subtext_color = (46, 62, 112) if selected else (198, 212, 242)
            pygame.draw.rect(surface, fill, row, border_radius=6)
            pygame.draw.rect(surface, border, row, 2 if selected else 1, border_radius=6)

            label_surf = self._small_font.render(label, True, text_color)
            surface.blit(label_surf, (row.x + 14, row.y + 8))

            if key in {"back", "apply_restart"}:
                value_surf = self._tiny_font.render(value, True, subtext_color)
                surface.blit(
                    value_surf,
                    value_surf.get_rect(midright=(row.right - 16, row.centery)),
                )
            else:
                value_box = pygame.Rect(row.right - 192, row.y + 7, 178, row.h - 14)
                dec_box = pygame.Rect(value_box.x, value_box.y, 34, value_box.h)
                inc_box = pygame.Rect(value_box.right - 34, value_box.y, 34, value_box.h)
                value_mid = pygame.Rect(
                    dec_box.right + 6,
                    value_box.y,
                    value_box.w - 80,
                    value_box.h,
                )
                self._pause_settings_control_hitboxes[(idx, "dec")] = dec_box.copy()
                self._pause_settings_control_hitboxes[(idx, "inc")] = inc_box.copy()
                for box, glyph in ((dec_box, "<"), (inc_box, ">")):
                    pygame.draw.rect(surface, (20, 32, 92), box, border_radius=5)
                    pygame.draw.rect(surface, (110, 130, 184), box, 1, border_radius=5)
                    glyph_surf = self._small_font.render(glyph, True, (238, 245, 255))
                    surface.blit(glyph_surf, glyph_surf.get_rect(center=box.center))
                pygame.draw.rect(surface, (14, 26, 78), value_mid, border_radius=5)
                pygame.draw.rect(surface, (92, 112, 168), value_mid, 1, border_radius=5)
                value_surf = self._tiny_font.render(value, True, (238, 245, 255))
                surface.blit(value_surf, value_surf.get_rect(center=value_mid.center))

            y += row_h + gap

    def _render_intro_difficulty_overlay(self, surface: pygame.Surface, snap: TestSnapshot) -> None:
        if snap.phase not in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            self._intro_difficulty_control_hitboxes = {}
            return

        level = self._get_intro_difficulty_level()
        mode_label = self._app.intro_difficulty_mode_label(self._test_code)
        stage_label = "Practice Difficulty" if snap.phase is Phase.INSTRUCTIONS else "Timed Test Difficulty"

        panel = pygame.Rect(0, 0, 284, 88)
        panel.left = 24
        panel.bottom = surface.get_height() - 24
        pygame.draw.rect(surface, (8, 18, 104), panel, border_radius=10)
        pygame.draw.rect(surface, (226, 236, 255), panel, 2, border_radius=10)

        title = self._small_font.render(stage_label, True, (238, 245, 255))
        surface.blit(title, (panel.x + 12, panel.y + 10))
        mode = self._tiny_font.render(mode_label, True, (188, 204, 228))
        surface.blit(mode, (panel.x + 12, panel.y + 34))

        value_box = pygame.Rect(panel.right - 122, panel.y + 14, 102, 30)
        dec_box = pygame.Rect(value_box.x, value_box.y, 26, value_box.h)
        inc_box = pygame.Rect(value_box.right - 26, value_box.y, 26, value_box.h)
        mid_box = pygame.Rect(dec_box.right + 4, value_box.y, value_box.w - 60, value_box.h)
        self._intro_difficulty_control_hitboxes = {
            "dec": dec_box.copy(),
            "inc": inc_box.copy(),
        }

        for box, glyph in ((dec_box, "<"), (inc_box, ">")):
            pygame.draw.rect(surface, (18, 30, 92), box, border_radius=5)
            pygame.draw.rect(surface, (110, 130, 184), box, 1, border_radius=5)
            glyph_surf = self._small_font.render(glyph, True, (238, 245, 255))
            surface.blit(glyph_surf, glyph_surf.get_rect(center=box.center))
        pygame.draw.rect(surface, (14, 26, 78), mid_box, border_radius=5)
        pygame.draw.rect(surface, (92, 112, 168), mid_box, 1, border_radius=5)
        level_surf = self._small_font.render(f"{level}/10", True, (238, 245, 255))
        surface.blit(level_surf, level_surf.get_rect(center=mid_box.center))

        hint = self._tiny_font.render(
            "Left/Right stages this difficulty. Enter applies it when you start.",
            True,
            (188, 204, 228),
        )
        surface.blit(hint, (panel.x + 12, panel.bottom - 22))

    def _vigilance_clear_inputs(self) -> None:
        self._vigilance_row_input = ""
        self._vigilance_col_input = ""
        self._vigilance_focus = "row"

    def _vigilance_submit_inputs(self) -> bool:
        row = self._vigilance_row_input.strip()
        col = self._vigilance_col_input.strip()
        if row == "" or col == "":
            self._vigilance_focus = "row" if row == "" else "col"
            return False
        accepted = self._engine.submit_answer(f"{row},{col}")
        if accepted:
            self._vigilance_clear_inputs()
            self._input = ""
        return bool(accepted)

    @staticmethod
    def _choice_from_key(key: int) -> int | None:
        mapping = {
            pygame.K_a: 1,
            pygame.K_s: 2,
            pygame.K_d: 3,
            pygame.K_f: 4,
            pygame.K_g: 5,
            pygame.K_1: 1,
            pygame.K_2: 2,
            pygame.K_3: 3,
            pygame.K_4: 4,
            pygame.K_5: 5,
            pygame.K_KP1: 1,
            pygame.K_KP2: 2,
            pygame.K_KP3: 3,
            pygame.K_KP4: 4,
            pygame.K_KP5: 5,
        }
        return mapping.get(key)

    def _apply_multiple_choice_key(self, *, key: int, option_count: int) -> bool:
        """Handle choice hotkeys for 1..5 options.

        A/S/D/F/G both select and submit immediately.
        1..5 keys still prefill selection and wait for Enter.
        """

        choice = self._choice_from_key(key)
        if choice is None or not (1 <= choice <= max(1, int(option_count))):
            return False

        self._math_choice = choice
        self._input = str(choice)

        if key in (pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_f, pygame.K_g):
            accepted = self._engine.submit_answer(self._input)
            if accepted:
                self._input = ""
                self._math_choice = 1
        return True

    @staticmethod
    def _choice_key_label(code: int) -> str:
        labels = ("A", "S", "D", "F", "G")
        idx = int(code) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return str(code)

    def _choice_input_label(self, raw: str) -> str:
        token = str(raw).strip()
        if token.isdigit():
            return self._choice_key_label(int(token))
        return token.upper()

    @staticmethod
    def _fit_label(font: pygame.font.Font, label: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.size(label)[0] <= max_width:
            return label
        clipped = label
        while clipped and font.size(f"{clipped}...")[0] > max_width:
            clipped = clipped[:-1]
        return f"{clipped}..." if clipped else "..."

    def _sync_auditory_audio(
        self,
        *,
        phase: Phase,
        payload: AuditoryCapacityPayload | None,
    ) -> None:
        if payload is None or phase not in (Phase.PRACTICE, Phase.SCORED):
            self._stop_auditory_audio()
            return
        if self._auditory_audio is None:
            self._auditory_audio = _AuditoryCapacityAudioAdapter()
        self._auditory_audio.sync(phase=phase, payload=payload)
        self._auditory_audio_debug = self._auditory_audio.debug_state(payload=payload)

    def _stop_auditory_audio(self) -> None:
        if self._auditory_audio is None:
            self._auditory_audio_debug = {}
            return
        self._auditory_audio.stop()
        self._auditory_audio = None
        self._auditory_audio_debug = {}

    def _should_use_auditory_panda_renderer(self) -> bool:
        if self._auditory_panda_failed:
            return False
        pref = os.environ.get("CFAST_AUDITORY_RENDERER", "panda").strip().lower()
        if pref in {"pygame", "2d", "off"}:
            return False
        return panda3d_auditory_rendering_available()

    def _dispose_auditory_panda_renderer(self) -> None:
        if self._auditory_panda_renderer is not None:
            try:
                self._auditory_panda_renderer.close()
            except Exception:
                pass
        self._auditory_panda_renderer = None

    def _get_auditory_panda_renderer(
        self,
        *,
        size: tuple[int, int],
    ) -> AuditoryCapacityPanda3DRenderer | None:
        if not self._should_use_auditory_panda_renderer():
            self._dispose_auditory_panda_renderer()
            return None
        if self._auditory_panda_renderer is not None and self._auditory_panda_renderer.size != size:
            self._dispose_auditory_panda_renderer()
        if self._auditory_panda_renderer is None:
            try:
                self._auditory_panda_renderer = AuditoryCapacityPanda3DRenderer(size=size)
            except Exception:
                self._auditory_panda_failed = True
                self._dispose_auditory_panda_renderer()
                return None
        return self._auditory_panda_renderer

    def _should_use_rapid_tracking_panda_renderer(self) -> bool:
        if self._rapid_tracking_panda_failed:
            return False
        pref = os.environ.get("CFAST_RAPID_TRACKING_RENDERER", "panda").strip().lower()
        if pref in {"pygame", "2d", "off"}:
            return False
        return panda3d_rapid_tracking_rendering_available()

    def _should_use_spatial_integration_panda_renderer(self) -> bool:
        if self._spatial_integration_panda_failed:
            return False
        pref = os.environ.get("CFAST_SPATIAL_INTEGRATION_RENDERER", "panda").strip().lower()
        if pref in {"pygame", "2d", "off"}:
            return False
        return panda3d_spatial_integration_rendering_available()

    def _dispose_rapid_tracking_panda_renderer(self) -> None:
        if self._rapid_tracking_panda_renderer is not None:
            try:
                self._rapid_tracking_panda_renderer.close()
            except Exception:
                pass
        self._rapid_tracking_panda_renderer = None

    def _dispose_spatial_integration_panda_renderer(self) -> None:
        if self._spatial_integration_panda_renderer is not None:
            try:
                self._spatial_integration_panda_renderer.close()
            except Exception:
                pass
        self._spatial_integration_panda_renderer = None

    def _get_rapid_tracking_panda_renderer(
        self,
        *,
        size: tuple[int, int],
    ) -> RapidTrackingPanda3DRenderer | None:
        if not self._should_use_rapid_tracking_panda_renderer():
            self._dispose_rapid_tracking_panda_renderer()
            return None
        if (
            self._rapid_tracking_panda_renderer is not None
            and self._rapid_tracking_panda_renderer.size != size
        ):
            self._dispose_rapid_tracking_panda_renderer()
        if self._rapid_tracking_panda_renderer is None:
            try:
                self._rapid_tracking_panda_renderer = RapidTrackingPanda3DRenderer(size=size)
            except Exception:
                self._rapid_tracking_panda_failed = True
                self._dispose_rapid_tracking_panda_renderer()
                return None
        return self._rapid_tracking_panda_renderer

    def _get_spatial_integration_panda_renderer(
        self,
        *,
        size: tuple[int, int],
    ) -> SpatialIntegrationPanda3DRenderer | None:
        if not self._should_use_spatial_integration_panda_renderer():
            self._dispose_spatial_integration_panda_renderer()
            return None
        if (
            self._spatial_integration_panda_renderer is not None
            and self._spatial_integration_panda_renderer.size != size
        ):
            self._dispose_spatial_integration_panda_renderer()
        if self._spatial_integration_panda_renderer is None:
            try:
                self._spatial_integration_panda_renderer = SpatialIntegrationPanda3DRenderer(size=size)
            except Exception:
                self._spatial_integration_panda_failed = True
                self._dispose_spatial_integration_panda_renderer()
                return None
        return self._spatial_integration_panda_renderer

    def _should_use_trace_test_1_panda_renderer(self) -> bool:
        if self._trace_test_1_panda_failed:
            return False
        pref = os.environ.get("CFAST_TRACE_TEST_1_RENDERER", "panda").strip().lower()
        if pref in {"pygame", "2d", "off"}:
            return False
        return panda3d_trace_test_1_rendering_available()

    def _dispose_trace_test_1_panda_renderer(self) -> None:
        if self._trace_test_1_panda_renderer is not None:
            try:
                self._trace_test_1_panda_renderer.close()
            except Exception:
                pass
        self._trace_test_1_panda_renderer = None

    def _get_trace_test_1_panda_renderer(
        self,
        *,
        size: tuple[int, int],
    ) -> TraceTest1Panda3DRenderer | None:
        if not self._should_use_trace_test_1_panda_renderer():
            self._dispose_trace_test_1_panda_renderer()
            return None
        if (
            self._trace_test_1_panda_renderer is not None
            and self._trace_test_1_panda_renderer.size != size
        ):
            self._dispose_trace_test_1_panda_renderer()
        if self._trace_test_1_panda_renderer is None:
            try:
                self._trace_test_1_panda_renderer = TraceTest1Panda3DRenderer(size=size)
            except Exception:
                self._trace_test_1_panda_failed = True
                self._dispose_trace_test_1_panda_renderer()
                return None
        return self._trace_test_1_panda_renderer

    def _should_use_trace_test_2_panda_renderer(self) -> bool:
        if self._trace_test_2_panda_failed:
            return False
        pref = os.environ.get("CFAST_TRACE_TEST_2_RENDERER", "panda").strip().lower()
        if pref in {"pygame", "2d", "off"}:
            return False
        return panda3d_trace_test_2_rendering_available()

    def _dispose_trace_test_2_panda_renderer(self) -> None:
        if self._trace_test_2_panda_renderer is not None:
            try:
                self._trace_test_2_panda_renderer.close()
            except Exception:
                pass
        self._trace_test_2_panda_renderer = None

    def _get_trace_test_2_panda_renderer(
        self,
        *,
        size: tuple[int, int],
    ) -> TraceTest2Panda3DRenderer | None:
        if not self._should_use_trace_test_2_panda_renderer():
            self._dispose_trace_test_2_panda_renderer()
            return None
        if (
            self._trace_test_2_panda_renderer is not None
            and self._trace_test_2_panda_renderer.size != size
        ):
            self._dispose_trace_test_2_panda_renderer()
        if self._trace_test_2_panda_renderer is None:
            try:
                self._trace_test_2_panda_renderer = TraceTest2Panda3DRenderer(size=size)
            except Exception:
                self._trace_test_2_panda_failed = True
                self._dispose_trace_test_2_panda_renderer()
                return None
        return self._trace_test_2_panda_renderer

    def close(self) -> None:
        self._persist_results_if_needed(self._engine.snapshot())
        self._stop_auditory_audio()
        self._stop_situational_awareness_audio()
        self._dispose_auditory_panda_renderer()
        self._dispose_rapid_tracking_panda_renderer()
        self._dispose_spatial_integration_panda_renderer()
        self._dispose_trace_test_1_panda_renderer()
        self._dispose_trace_test_2_panda_renderer()

    def _sync_situational_awareness_audio(
        self,
        *,
        phase: Phase,
        payload: SituationalAwarenessPayload | None,
    ) -> None:
        self._sa_tts.update()
        if payload is None or phase not in (Phase.PRACTICE, Phase.SCORED):
            if phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
                self._sa_tts.stop()
            self._sa_last_announced_token = None
            return

        token = self._situational_awareness_audio_token(payload)
        if token != self._sa_last_announced_token:
            self._queue_situational_awareness_announcements(payload)
            self._sa_last_announced_token = token

    def _stop_situational_awareness_audio(self) -> None:
        self._sa_tts.stop()
        self._sa_last_announced_token = None

    def _queue_situational_awareness_announcements(
        self, payload: SituationalAwarenessPayload
    ) -> None:
        lines = self._situational_awareness_announcements(payload)
        if not lines:
            return
        pending = int(self._sa_tts.pending_count)
        if pending > 6:
            return
        if pending > 2:
            self._sa_tts.speak(lines[0])
            return
        for line in lines:
            self._sa_tts.speak(line)

    @staticmethod
    def _situational_awareness_audio_token(
        payload: SituationalAwarenessPayload,
    ) -> tuple[object, ...]:
        contacts = tuple(
            (
                contact.callsign,
                int(contact.x),
                int(contact.y),
                contact.heading,
                int(contact.speed_cells_per_min),
                contact.fuel_state,
            )
            for contact in payload.contacts
        )
        return (
            payload.kind,
            payload.stem,
            int(payload.horizon_min),
            int(payload.correct_code),
            contacts,
        )

    @staticmethod
    def _situational_awareness_announcements(
        payload: SituationalAwarenessPayload,
    ) -> tuple[str, ...]:
        concise = ", ".join(
            f"{contact.callsign} {contact.heading}{contact.speed_cells_per_min}"
            for contact in payload.contacts[:3]
        )
        verbose = ", ".join(
            f"{contact.callsign} at {chr(ord('A') + int(contact.y))}{int(contact.x)} "
            f"heading {contact.heading} speed {contact.speed_cells_per_min}"
            for contact in payload.contacts[:3]
        )
        horizon = int(payload.horizon_min)

        if payload.kind is SituationalAwarenessQuestionKind.POSITION_PROJECTION:
            query = payload.query_callsign or "lead contact"
            return (
                f"Tactical. {concise}. Predict {query} plus {horizon}.",
                (
                    f"Situation update. {verbose}. "
                    f"Determine expected grid position of {query} in {horizon} minutes."
                ),
                "Immediate response. Commit answer now.",
            )
        if payload.kind is SituationalAwarenessQuestionKind.CONFLICT_PREDICTION:
            return (
                f"Traffic alert. {concise}. Find conflict pair plus {horizon}.",
                (
                    f"Situation update. {verbose}. "
                    f"Determine which two contacts share the same block in {horizon} minutes."
                ),
                "High tempo. Identify and answer now.",
            )
        return (
            "Priority fuel advisory. Select immediate deconfliction.",
            (
                f"Situation update. {verbose}. "
                "A minimum-fuel aircraft has priority. Select the best immediate action."
            ),
            "Urgent decision. Execute now.",
        )

    def _read_sensory_motor_control(self) -> tuple[float, float]:
        keys = pygame.key.get_pressed()

        arrow_horizontal = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (
            1.0 if keys[pygame.K_LEFT] else 0.0
        )
        ad_horizontal = (1.0 if keys[pygame.K_d] else 0.0) - (
            1.0 if keys[pygame.K_a] else 0.0
        )
        key_horizontal = max(-1.0, min(1.0, arrow_horizontal + ad_horizontal))
        key_vertical = (1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0) - (
            1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0
        )

        horizontal = 0.0
        vertical = 0.0
        joysticks = _iter_connected_joysticks()
        if joysticks:
            primary = joysticks[0]
            # Requested mapping:
            # - Horizontal: rudder axis (axis 3 / rudder function)
            # - Vertical: joystick axis 1
            if primary.get_numaxes() > 1:
                vertical = max(-1.0, min(1.0, float(primary.get_axis(1))))
            elif primary.get_numaxes() > 0:
                vertical = max(-1.0, min(1.0, float(primary.get_axis(0))))

            rudder_value: float | None = None

            # Prefer a separate rudder pedal device by name.
            for device in joysticks:
                name = str(device.get_name()).lower()
                if "rudder" not in name and "pedal" not in name:
                    continue
                axis_count = int(device.get_numaxes())
                if axis_count <= 0:
                    continue
                # Prefer axis 3 for rudder first, then alternate common pedal axes.
                preferred_indices = (3, 2, 0, 1)
                for idx in preferred_indices:
                    if idx < axis_count:
                        rudder_value = max(-1.0, min(1.0, float(device.get_axis(idx))))
                        break
                if rudder_value is not None:
                    break

            # Fallback to primary stick rudder-related axes.
            if rudder_value is None:
                axis_count = int(primary.get_numaxes())
                for idx in (3, 4, 5, 2, 1, 0):
                    if idx < axis_count:
                        rudder_value = max(-1.0, min(1.0, float(primary.get_axis(idx))))
                        break

            horizontal = 0.0 if rudder_value is None else rudder_value

        out_horizontal = key_horizontal if abs(key_horizontal) > 0.001 else horizontal
        out_vertical = vertical if joysticks else key_vertical
        return (
            max(-1.0, min(1.0, out_horizontal)),
            max(-1.0, min(1.0, out_vertical)),
        )

    def _render_sensory_motor_apparatus_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: SensoryMotorApparatusPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 10, 74)
        panel_bg = (7, 16, 99)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        accent = (255, 52, 70)

        surface.fill(bg)
        frame = pygame.Rect(10, 10, w - 20, h - 20)
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, 42)
        pygame.draw.rect(surface, (16, 28, 118), header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        title = self._small_font.render(f"Sensory Motor Apparatus - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 12, header.y + 8))

        stats = self._tiny_font.render(
            f"Windows {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        body = pygame.Rect(
            frame.x + 8,
            header.bottom + 8,
            frame.w - 16,
            frame.bottom - header.bottom - 44,
        )
        track_margin = max(4, min(10, min(body.w, body.h) // 60))
        track = body.inflate(-track_margin * 2, -track_margin * 2)

        pygame.draw.rect(surface, (0, 0, 0), track)
        pygame.draw.rect(surface, (96, 108, 140), track, 1)

        cx = track.centerx
        cy = track.centery
        line_len = max(18, min(track.w, track.h) // 14)
        pygame.draw.line(surface, accent, (cx - line_len, cy), (cx + line_len, cy), 2)
        pygame.draw.line(surface, accent, (cx, cy - line_len), (cx, cy + line_len), 2)

        if payload is not None:
            x_scale = (track.w // 2) - 12
            y_scale = (track.h // 2) - 12
            dot_x = int(round(cx + (payload.dot_x * x_scale)))
            dot_y = int(round(cy + (payload.dot_y * y_scale)))
            radius = max(4, min(9, track.w // 34))
            pygame.draw.circle(surface, accent, (dot_x, dot_y), radius)

        if payload is not None:
            metrics_bg = pygame.Rect(track.x + 8, track.y + 8, track.w - 16, 24)
            pygame.draw.rect(surface, (8, 18, 104), metrics_bg)
            pygame.draw.rect(surface, (78, 102, 170), metrics_bg, 1)
            on_target_pct = payload.on_target_ratio * 100.0
            left_text = self._tiny_font.render(
                f"Err {payload.mean_error:.3f}  RMS {payload.rms_error:.3f}  "
                f"On {on_target_pct:.1f}%",
                True,
                text_main,
            )
            right_text = self._tiny_font.render(
                f"Ctrl {payload.control_x:+.2f},{payload.control_y:+.2f}  "
                f"Drift {payload.disturbance_x:+.2f},{payload.disturbance_y:+.2f}",
                True,
                text_muted,
            )
            surface.blit(left_text, (metrics_bg.x + 8, metrics_bg.y + 4))
            surface.blit(
                right_text,
                right_text.get_rect(midright=(metrics_bg.right - 8, metrics_bg.centery)),
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            prompt_bg = pygame.Rect(track.x + 10, track.bottom - 92, track.w - 20, 82)
            pygame.draw.rect(surface, (8, 18, 104), prompt_bg)
            pygame.draw.rect(surface, (78, 102, 170), prompt_bg, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                prompt_bg.inflate(-10, -8),
                color=text_muted,
                font=self._tiny_font,
                max_lines=6,
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        elif snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = (
                "Rudder axis controls left/right; Joystick axis 1 controls up/down "
                "(WASD fallback, arrows L/R inverted)."
            )
        else:
            footer = "Enter: Return to Tests"
        foot = self._tiny_font.render(footer, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))

    def _render_rapid_tracking_panda_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        payload: RapidTrackingPayload | None,
        renderer: RapidTrackingPanda3DRenderer,
        active_phase: bool,
    ) -> bool:
        try:
            view = renderer.render(payload=payload)
        except Exception:
            self._rapid_tracking_panda_failed = True
            self._dispose_rapid_tracking_panda_renderer()
            return False

        if view.get_size() != world.size:
            view = pygame.transform.smoothscale(view, world.size)
        surface.blit(view, world.topleft)

        vignette = pygame.Surface(world.size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (4, 10, 8, 42), vignette.get_rect(), 14)
        surface.blit(vignette, world.topleft)

        if active_phase:
            edge_alpha = 24 + int(round(12.0 * (0.5 + (0.5 * math.sin(pygame.time.get_ticks() / 160.0)))))
            left_band = pygame.Surface((16, world.h), pygame.SRCALPHA)
            right_band = pygame.Surface((16, world.h), pygame.SRCALPHA)
            left_band.fill((68, 194, 156, edge_alpha))
            right_band.fill((196, 130, 74, edge_alpha))
            surface.blit(left_band, (world.x, world.y))
            surface.blit(right_band, (world.right - 16, world.y))

        return True

    def _render_trace_test_1_panda_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        reference: TraceTest1Attitude,
        candidate: TraceTest1Attitude,
        correct_code: int,
        viewpoint_bearing_deg: int,
        scene_turn_index: int,
        animate: bool,
        motion_progress: float | None = None,
    ) -> bool:
        renderer = self._get_trace_test_1_panda_renderer(size=world.size)
        if renderer is None:
            return False
        try:
            view = renderer.render(
                reference=reference,
                candidate=candidate,
                correct_code=correct_code,
                viewpoint_bearing_deg=viewpoint_bearing_deg,
                scene_turn_index=scene_turn_index,
                animate=animate,
                progress=motion_progress,
            )
        except Exception:
            self._trace_test_1_panda_failed = True
            self._dispose_trace_test_1_panda_renderer()
            return False

        if view.get_size() != world.size:
            view = pygame.transform.smoothscale(view, world.size)
        surface.blit(view, world.topleft)

        vignette = pygame.Surface(world.size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (8, 14, 24, 34), vignette.get_rect(), 12)
        surface.blit(vignette, world.topleft)
        return True

    def _render_trace_test_2_panda_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        payload: TraceTest2Payload | None,
    ) -> bool:
        renderer = self._get_trace_test_2_panda_renderer(size=world.size)
        if renderer is None:
            return False
        try:
            view = renderer.render(payload=payload)
        except Exception:
            self._trace_test_2_panda_failed = True
            self._dispose_trace_test_2_panda_renderer()
            return False

        if view.get_size() != world.size:
            view = pygame.transform.smoothscale(view, world.size)
        surface.blit(view, world.topleft)

        vignette = pygame.Surface(world.size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (6, 10, 20, 32), vignette.get_rect(), 12)
        surface.blit(vignette, world.topleft)
        return True

    def _render_spatial_integration_panda_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        payload: SpatialIntegrationPayload | None,
    ) -> bool:
        renderer = self._get_spatial_integration_panda_renderer(size=world.size)
        if renderer is None:
            return False
        try:
            view = renderer.render(payload=payload)
        except Exception:
            self._spatial_integration_panda_failed = True
            self._dispose_spatial_integration_panda_renderer()
            return False

        if view.get_size() != world.size:
            view = pygame.transform.smoothscale(view, world.size)
        surface.blit(view, world.topleft)

        vignette = pygame.Surface(world.size, pygame.SRCALPHA)
        pygame.draw.rect(vignette, (8, 14, 24, 30), vignette.get_rect(), 12)
        surface.blit(vignette, world.topleft)
        return True

    def _render_rapid_tracking_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: RapidTrackingPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (6, 24, 20)
        panel_bg = (12, 44, 34)
        border = (180, 228, 204)
        text_main = (229, 248, 237)
        text_muted = (168, 210, 190)

        surface.fill(bg)
        frame = pygame.Rect(10, 10, w - 20, h - 20)
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, 42)
        pygame.draw.rect(surface, (16, 66, 52), header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        title = self._small_font.render(f"Rapid Tracking - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 12, header.y + 8))

        stats = self._tiny_font.render(
            f"Windows {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        body = pygame.Rect(
            frame.x + 8,
            header.bottom + 8,
            frame.w - 16,
            frame.bottom - header.bottom - 44,
        )
        track_margin = max(4, min(10, min(body.w, body.h) // 60))
        track = body.inflate(-track_margin * 2, -track_margin * 2)

        pygame.draw.rect(surface, (4, 10, 8), track)
        pygame.draw.rect(surface, (84, 122, 109), track, 1)

        cam_x = float(payload.camera_x) if payload is not None else 0.0
        cam_y = float(payload.camera_y) if payload is not None else 0.0
        scene_progress = float(payload.scene_progress) if payload is not None else 0.0
        target_kind = str(payload.target_kind) if payload is not None else ""
        target_rel_x_for_rig = float(payload.target_rel_x) if payload is not None else 0.0
        target_rel_y_for_rig = float(payload.target_rel_y) if payload is not None else 0.0
        capture_zoom_for_rig = float(payload.capture_zoom) if payload is not None else 0.0
        assist_strength_for_rig = float(payload.camera_assist_strength) if payload is not None else 0.0
        turbulence_strength_for_rig = float(payload.turbulence_strength) if payload is not None else 0.65
        panda_renderer = self._get_rapid_tracking_panda_renderer(size=(track.w, track.h))
        using_panda = False
        panda_overlay = None

        rig = RapidTrackingPanda3DRenderer._camera_rig_state(
            elapsed_s=pygame.time.get_ticks() / 1000.0,
            progress=scene_progress,
            cam_x=cam_x,
            cam_y=cam_y,
            zoom=capture_zoom_for_rig,
            target_kind=target_kind,
            target_rel_x=target_rel_x_for_rig,
            target_rel_y=target_rel_y_for_rig,
            assist_strength=assist_strength_for_rig,
            turbulence_strength=turbulence_strength_for_rig,
        )
        altitude_span = max(1.0, 24.0 - 5.8)
        descent_ratio = max(0.0, min(1.0, (24.0 - rig.altitude_agl) / altitude_span))
        horizon_y = track.centery - int(round(cam_y * track.h * 0.08))
        horizon_y += int(round((descent_ratio - 0.35) * track.h * 0.14))
        horizon_y = max(track.y + 24, min(track.bottom - 28, horizon_y))
        heading_bias = math.sin(math.radians(rig.carrier_heading_deg)) * (0.06 + (0.10 * (1.0 - rig.orbit_weight)))
        vanish_x = track.centerx - int(round(cam_x * track.w * 0.12))
        vanish_x += int(round(heading_bias * track.w))
        vanish_x = max(track.x + 34, min(track.right - 34, vanish_x))
        anim_s = pygame.time.get_ticks() / 1000.0

        if panda_renderer is not None:
            using_panda = self._render_rapid_tracking_panda_view(
                surface=surface,
                world=track,
                payload=payload,
                renderer=panda_renderer,
                active_phase=snap.phase in (Phase.PRACTICE, Phase.SCORED),
            )
            if using_panda:
                panda_overlay = panda_renderer.target_overlay_state()

        if not using_panda:
            old_clip = surface.get_clip()
            surface.set_clip(track)
            try:
                sky_h = max(1, horizon_y - track.y)
                for row in range(track.y, horizon_y):
                    t = (row - track.y) / float(sky_h)
                    r = int(8 + (22 * t))
                    g = int(30 + (58 * t))
                    b = int(56 + (46 * t))
                    pygame.draw.line(surface, (r, g, b), (track.x, row), (track.right, row))

                # Layered mountain silhouettes for stronger depth cues.
                width = track.w
                for layer in range(3):
                    base_y = horizon_y + 2 + (layer * 14)
                    amp = 24 + (layer * 12)
                    color = (
                        9 + (layer * 9),
                        34 + (layer * 15),
                        34 + (layer * 12),
                    )
                    freq = 0.010 + (layer * 0.0028)
                    drift = (cam_x * (20 - (layer * 4))) + (anim_s * (3.0 - (layer * 0.8)))
                    pts: list[tuple[int, int]] = [(track.x, base_y)]
                    step = max(20, width // 14)
                    for gx in range(track.x - step, track.right + step * 2, step):
                        nx = gx + drift
                        ridge = math.sin(nx * freq) + (
                            0.65 * math.sin(nx * freq * 1.9 + (layer * 0.7))
                        )
                        py = base_y - int(round(((ridge + 1.5) * 0.5) * amp))
                        pts.append((gx, py))
                    pts.append((track.right, base_y))
                    pygame.draw.polygon(surface, color, pts)

                # Horizon tree spikes as fine visual clutter.
                for idx in range(28):
                    x_ratio = idx / 27.0
                    tx = track.x + int(round(x_ratio * width))
                    tx -= int(round((cam_x * 10.0) + (anim_s * 2.0)))
                    if tx < track.x - 6 or tx > track.right + 6:
                        continue
                    hgt = 5 + (idx % 5) * 2
                    top = horizon_y + 2 - hgt
                    pygame.draw.line(surface, (18, 68, 52), (tx, horizon_y + 2), (tx, top), 1)
                    pygame.draw.line(surface, (24, 88, 66), (tx - 2, top + 3), (tx, top), 1)
                    pygame.draw.line(surface, (24, 88, 66), (tx + 2, top + 3), (tx, top), 1)

                # Cloud bands to increase sky motion distraction.
                for idx in range(9):
                    phase = (anim_s * (6.0 + (idx % 3))) + (idx * 31.0) - (cam_x * 42.0)
                    cloud_x = track.x + int(round(phase % (width + 160))) - 80
                    cloud_y = track.y + 18 + (idx % 4) * 16 + int(
                        round(math.sin(anim_s + idx) * 3.0)
                    )
                    if cloud_y >= horizon_y - 10:
                        continue
                    shade = 84 + ((idx * 13) % 44)
                    for blob in range(3):
                        ox = (blob - 1) * 16
                        oy = -abs(blob - 1) * 3
                        rect = pygame.Rect(cloud_x + ox - 15, cloud_y + oy - 8, 30, 16)
                        pygame.draw.ellipse(surface, (shade, shade + 10, shade + 16), rect)

                # Urban/built-up band near foothills.
                for idx in range(22):
                    scroll = (cam_x * 44.0) + (anim_s * 7.8) + (idx * 41.0)
                    bx = track.x + int(round(scroll % (width + 220))) - 110
                    b_w = 14 + (idx % 5) * 7
                    b_h = 22 + (idx % 7) * 8
                    base_y = horizon_y + 18 + (idx % 3) * 4
                    rect = pygame.Rect(bx, base_y - b_h, b_w, b_h)
                    tone = 26 + (idx % 4) * 8
                    col = (tone, 50 + (idx % 5) * 6, 46 + (idx % 4) * 6)
                    pygame.draw.rect(surface, col, rect)
                    pygame.draw.rect(surface, (14, 28, 26), rect, 1)
                    if b_w >= 18 and b_h >= 26:
                        win_col = (118, 146, 126) if idx % 2 == 0 else (162, 130, 82)
                        for wy in range(rect.y + 4, rect.bottom - 4, 6):
                            for wx in range(rect.x + 4, rect.right - 3, 6):
                                if ((wx + wy + idx) % 3) == 0:
                                    continue
                                pygame.draw.rect(surface, win_col, pygame.Rect(wx, wy, 2, 3))

                # Airborne distractions: glints and crossing traffic markers.
                for idx in range(11):
                    speed = 0.65 + (idx % 5) * 0.22
                    phase = (anim_s * speed * 36.0) + (idx * 57.0) + (cam_x * 28.0)
                    px = track.x + int(round(phase % (width + 220))) - 110
                    py = track.y + 26 + ((idx * 19) % max(30, horizon_y - track.y - 10))
                    blink = 0.45 + 0.55 * math.sin(anim_s * (3.3 + (idx * 0.17)) + idx)
                    alpha = int(round(60 + (130 * max(0.0, blink))))
                    core = (alpha, min(255, alpha + 40), min(255, alpha + 70))
                    if idx % 3 == 0:
                        pygame.draw.circle(surface, core, (px, py), 2)
                        pygame.draw.circle(surface, (24, 92, 76), (px, py), 5, 1)
                    else:
                        pygame.draw.line(surface, core, (px - 3, py), (px + 3, py), 1)
                        pygame.draw.line(surface, core, (px, py - 3), (px, py + 3), 1)

                ground_h = max(1, track.bottom - horizon_y)
                for row in range(horizon_y, track.bottom):
                    t = (row - horizon_y) / float(ground_h)
                    r = int(18 - (10 * t))
                    g = int(58 - (28 * t))
                    b = int(46 - (20 * t))
                    pygame.draw.line(surface, (r, g, b), (track.x, row), (track.right, row))

                # Closer low-rise buildings in foreground for stronger motion parallax.
                for idx in range(14):
                    scroll = (cam_x * 86.0) + (anim_s * 13.0) + (idx * 73.0)
                    bx = track.x + int(round(scroll % (width + 260))) - 130
                    b_w = 26 + (idx % 4) * 14
                    b_h = 30 + (idx % 5) * 11
                    ground_y = track.bottom - 26 - ((idx % 2) * 6)
                    rect = pygame.Rect(bx, ground_y - b_h, b_w, b_h)
                    col = (22 + (idx % 3) * 7, 44 + (idx % 4) * 8, 38 + (idx % 3) * 6)
                    pygame.draw.rect(surface, col, rect)
                    pygame.draw.rect(surface, (12, 22, 20), rect, 1)
                    door = pygame.Rect(rect.centerx - 3, rect.bottom - 10, 6, 10)
                    pygame.draw.rect(surface, (16, 22, 20), door)
                    win_col = (132, 166, 146) if idx % 3 == 0 else (172, 138, 92)
                    for wy in range(rect.y + 5, rect.bottom - 12, 7):
                        for wx in range(rect.x + 5, rect.right - 4, 8):
                            if ((idx + wx + wy) % 4) == 0:
                                continue
                            pygame.draw.rect(surface, win_col, pygame.Rect(wx, wy, 3, 3))

                # Ground distractions: lateral moving lights along lower field.
                for idx in range(14):
                    sweep = (anim_s * (18.0 + (idx % 4) * 3.2)) + (idx * 41.0) + (cam_x * 52.0)
                    lx = track.x + int(round(sweep % (width + 90))) - 45
                    ly = track.bottom - 12 - ((idx % 4) * 5)
                    color_a = (150, 210, 168) if idx % 2 == 0 else (188, 150, 90)
                    pygame.draw.line(surface, color_a, (lx - 2, ly), (lx + 2, ly), 1)

                depth_lines = 12
                for idx in range(depth_lines):
                    t = (idx + 1) / float(depth_lines + 1)
                    y = horizon_y + int(round((t**1.55) * (track.bottom - horizon_y)))
                    shade = int(58 + (82 * (1.0 - t)))
                    pygame.draw.line(surface, (22, shade, 64), (track.x, y), (track.right, y), 1)

                for lane in range(-5, 6):
                    lane_t = lane / 5.0
                    lane_shift = int(round(cam_x * 22.0))
                    bottom_x = track.centerx + int(round(lane_t * track.w * 0.62)) - lane_shift
                    top_x = vanish_x + int(round(lane_t * 24.0))
                    lane_color = (18, 72 + int(24 * (1.0 - abs(lane_t))), 62)
                    pygame.draw.line(
                        surface, lane_color, (bottom_x, track.bottom), (top_x, horizon_y), 1
                    )

                pygame.draw.line(
                    surface,
                    (172, 214, 196),
                    (track.x, horizon_y),
                    (track.right, horizon_y),
                    1,
                )

                # Peripheral flashing bands to simulate cockpit distraction load.
                if snap.phase in (Phase.PRACTICE, Phase.SCORED):
                    pulse = 0.5 + (0.5 * math.sin(anim_s * 7.4))
                    band_alpha = int(round(20 + (42 * pulse)))
                    left_band = pygame.Surface((18, track.h), pygame.SRCALPHA)
                    left_band.fill((60, 190, 150, band_alpha))
                    right_band = pygame.Surface((18, track.h), pygame.SRCALPHA)
                    right_band.fill((190, 120, 72, band_alpha))
                    surface.blit(left_band, (track.x, track.y))
                    surface.blit(right_band, (track.right - 18, track.y))
            finally:
                surface.set_clip(old_clip)

        cx = track.centerx
        cy = track.centery
        x_scale = (track.w // 2) - 12
        y_scale = int(((track.h // 2) - 12) * 0.90)
        box_half_w = max(
            18,
            int(
                round(
                    x_scale
                    * (
                        float(payload.capture_box_half_width)
                        if payload is not None
                        else 0.085
                    )
                )
            ),
        )
        box_half_h = max(
            14,
            int(
                round(
                    y_scale
                    * (
                        float(payload.capture_box_half_height)
                        if payload is not None
                        else 0.075
                    )
                )
            ),
        )

        def draw_target_sprite(
            *,
            kind: str,
            variant: str = "",
            center: tuple[int, int],
            size: int,
            base_color: tuple[int, int, int],
            accent_color: tuple[int, int, int] | None = None,
        ) -> None:
            tx, ty = center
            k = str(kind).strip().lower()
            v = str(variant).strip().lower()
            outline = (18, 28, 22)
            accent = accent_color or (
                min(255, base_color[0] + 32),
                max(0, base_color[1] - 18),
                max(0, base_color[2] - 18),
            )

            if k == "soldier":
                body_w = max(6, size * 2)
                body_h = max(10, size * 3)
                torso = pygame.Rect(tx - (body_w // 2), ty - (body_h // 2), body_w, body_h)
                head_r = max(2, size // 2)
                vest = pygame.Rect(
                    torso.x + max(1, torso.w // 4),
                    torso.y + max(1, torso.h // 6),
                    max(2, torso.w // 2),
                    max(4, torso.h // 2),
                )
                arm_y = torso.y + max(2, torso.h // 4)
                leg_y = torso.bottom - 1
                pygame.draw.circle(surface, (214, 196, 170), (tx, torso.y - head_r + 1), head_r)
                pygame.draw.rect(surface, base_color, torso, border_radius=3)
                pygame.draw.rect(surface, accent, vest, border_radius=2)
                pygame.draw.line(
                    surface,
                    base_color,
                    (torso.left + 1, arm_y),
                    (torso.left - max(2, size // 2), arm_y + max(2, size // 2)),
                    2,
                )
                pygame.draw.line(
                    surface,
                    base_color,
                    (torso.right - 1, arm_y),
                    (torso.right + max(2, size // 2), arm_y + max(2, size // 2)),
                    2,
                )
                pygame.draw.line(
                    surface,
                    base_color,
                    (tx - max(1, size // 3), leg_y),
                    (tx - max(2, size // 2), leg_y + max(3, size)),
                    2,
                )
                pygame.draw.line(
                    surface,
                    base_color,
                    (tx + max(1, size // 3), leg_y),
                    (tx + max(2, size // 2), leg_y + max(3, size)),
                    2,
                )
                pygame.draw.rect(surface, outline, torso, 1, border_radius=3)
                pygame.draw.circle(surface, outline, (tx, torso.y - head_r + 1), head_r, 1)
                return

            if k == "building":
                if v == "tower":
                    shaft_w = max(8, size * 2)
                    shaft_h = max(18, size * 5)
                    shaft = pygame.Rect(tx - (shaft_w // 2), ty - shaft_h, shaft_w, shaft_h)
                    cab = pygame.Rect(
                        tx - max(10, size * 2),
                        shaft.y - max(7, size * 2),
                        max(20, size * 4),
                        max(8, size * 2),
                    )
                    pygame.draw.rect(surface, base_color, shaft, border_radius=3)
                    pygame.draw.rect(surface, (78, 112, 128), cab, border_radius=3)
                    pygame.draw.rect(surface, outline, shaft, 1, border_radius=3)
                    pygame.draw.rect(surface, outline, cab, 1, border_radius=3)
                else:
                    body_w = max(24, size * 6)
                    body_h = max(16, size * 4)
                    body = pygame.Rect(tx - (body_w // 2), ty - (body_h // 2), body_w, body_h)
                    roof = (
                        (body.left + 4, body.top),
                        (body.centerx, body.top - max(4, size)),
                        (body.right - 4, body.top),
                    )
                    door_w = max(8, body.w // (2 if v == "garage" else 4))
                    door = pygame.Rect(body.centerx - (door_w // 2), body.bottom - max(10, size * 2), door_w, max(10, size * 2))
                    pygame.draw.rect(surface, base_color, body, border_radius=4)
                    pygame.draw.polygon(surface, accent, roof)
                    pygame.draw.rect(surface, (22, 28, 26), door, border_radius=2)
                    pygame.draw.rect(surface, outline, body, 1, border_radius=4)
                    pygame.draw.polygon(surface, outline, roof, 1)
                return

            if k == "truck":
                body_w = max(18, size * 4)
                body_h = max(10, size * 2)
                bed = pygame.Rect(tx - (body_w // 2), ty - (body_h // 2), int(body_w * 0.58), body_h)
                cab = pygame.Rect(bed.right - 2, ty - max(6, size), max(10, int(body_w * 0.38)), max(12, body_h + 2))
                for wheel_x in (bed.left + 5, bed.right - 3, cab.left + 3, cab.right - 4):
                    pygame.draw.circle(surface, (20, 24, 22), (wheel_x, ty + body_h // 2), max(2, size // 2))
                pygame.draw.rect(surface, base_color, bed, border_radius=3)
                pygame.draw.rect(surface, accent, cab, border_radius=3)
                pygame.draw.rect(surface, outline, bed, 1, border_radius=3)
                pygame.draw.rect(surface, outline, cab, 1, border_radius=3)
                return

            if k == "tank":
                hull_w = max(20, size * 4)
                hull_h = max(12, size * 2)
                hull = pygame.Rect(tx - (hull_w // 2), ty - (hull_h // 2), hull_w, hull_h)
                turret = pygame.Rect(
                    tx - max(6, size),
                    ty - max(5, size // 2),
                    max(12, size * 2),
                    max(10, size),
                )
                barrel_len = max(10, size * 2)
                track_h = max(3, size // 2)
                pygame.draw.rect(surface, (26, 30, 28), hull.inflate(0, track_h + 2), border_radius=4)
                pygame.draw.rect(surface, base_color, hull, border_radius=4)
                pygame.draw.rect(surface, accent, turret, border_radius=5)
                pygame.draw.line(
                    surface,
                    accent,
                    (turret.centerx, turret.centery),
                    (turret.centerx + barrel_len, turret.centery),
                    3,
                )
                pygame.draw.rect(surface, outline, hull, 1, border_radius=4)
                pygame.draw.rect(surface, outline, turret, 1, border_radius=5)
                return

            if k == "helicopter":
                body = pygame.Rect(tx - max(10, size * 2), ty - max(4, size), max(20, size * 4), max(10, size * 2))
                cockpit = pygame.Rect(body.x + max(2, size // 2), body.y + 1, max(6, size + 2), max(8, size * 2 - 2))
                tail_x = body.right + max(7, size * 2)
                pygame.draw.ellipse(surface, base_color, body)
                pygame.draw.ellipse(surface, (180, 214, 226), cockpit)
                pygame.draw.line(surface, base_color, (body.right - 2, ty - 1), (tail_x, ty - 1), 3)
                pygame.draw.line(surface, base_color, (tail_x, ty - 4), (tail_x, ty + 3), 2)
                pygame.draw.line(surface, accent, (tx, body.y - 2), (tx, body.y - max(8, size * 2)), 2)
                pygame.draw.line(surface, accent, (tx - max(10, size * 2), body.y - max(5, size)), (tx + max(10, size * 2), body.y - max(5, size)), 2)
                pygame.draw.line(surface, outline, (body.left + 3, body.bottom), (body.left + 10, body.bottom + 4), 2)
                pygame.draw.line(surface, outline, (body.right - 3, body.bottom), (body.right - 10, body.bottom + 4), 2)
                pygame.draw.ellipse(surface, outline, body, 1)
                return

            # Fighter-style fixed-wing silhouette.
            fus_len = max(14, size * 4)
            nose = (tx, ty - fus_len)
            wing_y = ty - max(1, size // 3)
            wing_span = max(11, size * 4)
            wing_back_y = ty + max(2, int(round(size * 0.9)))
            tail_y = ty + max(5, int(round(size * 1.7)))
            tail_span = max(5, int(round(size * 1.5)))
            jet = (
                nose,
                (tx + max(2, size // 2), ty - max(3, size)),
                (tx + wing_span, wing_y),
                (tx + max(3, int(round(size * 1.1))), wing_back_y),
                (tx + tail_span, tail_y),
                (tx, ty + max(8, int(round(size * 2.4)))),
                (tx - tail_span, tail_y),
                (tx - max(3, int(round(size * 1.1))), wing_back_y),
                (tx - wing_span, wing_y),
                (tx - max(2, size // 2), ty - max(3, size)),
            )
            pygame.draw.polygon(surface, base_color, jet)
            pygame.draw.polygon(
                surface,
                (230, 238, 244),
                (
                    (tx, ty - max(7, int(round(size * 1.8)))),
                    (tx + max(2, size // 3), ty - max(4, size)),
                    (tx, ty - max(2, int(round(size * 0.2)))),
                    (tx - max(2, size // 3), ty - max(4, size)),
                ),
            )
            pygame.draw.line(
                surface,
                outline,
                (tx, ty - fus_len),
                (tx, ty + max(8, int(round(size * 2.3)))),
                1,
            )

        def terrain_ridge_for(rel_x: float) -> float:
            major = 0.14 * math.sin((rel_x * 2.5) + (cam_x * 0.62) + 0.35)
            mid = 0.09 * math.sin((rel_x * 4.7) - (cam_x * 0.44) + 1.15)
            fine = 0.05 * math.sin((rel_x * 8.4) + (cam_x * 0.18) - 0.55)
            return 0.12 + major + mid + fine

        def project_rel_point(rel_x: float, rel_y: float) -> tuple[int, int]:
            return (
                int(round(cx + (rel_x * x_scale))),
                int(round(cy + (rel_y * y_scale))),
            )

        def clamped_track_point(target_px: int, target_py: int, *, inset: int = 18) -> tuple[int, int]:
            inner = track.inflate(-inset * 2, -inset * 2)
            return (
                max(inner.left, min(inner.right, target_px)),
                max(inner.top, min(inner.bottom, target_py)),
            )

        def draw_edge_pointer(
            *,
            target_px: int,
            target_py: int,
            label: str,
        ) -> None:
            edge_x, edge_y = clamped_track_point(target_px, target_py, inset=18)
            dx = float(target_px - cx)
            dy = float(target_py - cy)
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                dx = 0.0
                dy = -1.0
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                return
            nx = dx / length
            ny = dy / length
            tip = (edge_x, edge_y)
            base = (
                int(round(edge_x - (nx * 16.0))),
                int(round(edge_y - (ny * 16.0))),
            )
            left = (
                int(round(base[0] + (-ny * 8.0))),
                int(round(base[1] + (nx * 8.0))),
            )
            right = (
                int(round(base[0] - (-ny * 8.0))),
                int(round(base[1] - (nx * 8.0))),
            )
            pygame.draw.line(surface, (224, 74, 74), (cx, cy), tip, 1)
            pygame.draw.polygon(surface, (236, 82, 82), (tip, left, right))
            pygame.draw.polygon(surface, (255, 228, 228), (tip, left, right), 1)
            pointer_text = self._tiny_font.render(label, True, (244, 214, 214))
            surface.blit(pointer_text, pointer_text.get_rect(midbottom=(edge_x, edge_y - 10)))

        def draw_live_target_feed(
            *,
            preview_rect: pygame.Rect,
            source: pygame.Surface,
            focus_px: int,
            focus_py: int,
            label: str,
            detail: str,
            cue: str,
            on_screen: bool,
            visible: bool,
            switch_preview_s: float,
        ) -> None:
            alert = switch_preview_s > 0.0
            border_color = (188, 88, 88) if alert else (126, 186, 164)
            accent_color = (244, 204, 204) if alert else (198, 224, 212)
            pygame.draw.rect(surface, (12, 44, 36), preview_rect)
            pygame.draw.rect(surface, border_color, preview_rect, 1)

            title = self._tiny_font.render("LIVE TARGET FEED", True, accent_color)
            surface.blit(title, (preview_rect.x + 8, preview_rect.y + 6))

            state_text = "ON SCREEN" if on_screen and visible else ("OBSCURED" if on_screen else "OFFSCREEN")
            state_color = (246, 210, 210) if alert else (152, 202, 184)
            state = self._tiny_font.render(state_text, True, state_color)
            surface.blit(
                state,
                state.get_rect(topright=(preview_rect.right - 8, preview_rect.y + 6)),
            )

            feed_rect = pygame.Rect(
                preview_rect.x + 8,
                preview_rect.y + 24,
                preview_rect.w - 16,
                max(44, preview_rect.h - 58),
            )
            pygame.draw.rect(surface, (6, 18, 16), feed_rect, border_radius=4)
            pygame.draw.rect(surface, (72, 106, 96), feed_rect, 1, border_radius=4)

            if on_screen:
                crop_w = max(42, min(track.w // 5, 84))
                crop_h = max(36, min(track.h // 5, 72))
                crop = pygame.Rect(0, 0, crop_w, crop_h)
                crop.center = (focus_px - track.x, focus_py - track.y)
                bounds = pygame.Rect(0, 0, track.w, track.h)
                crop.clamp_ip(bounds)
                feed = source.subsurface(crop).copy()
                if feed.get_size() != feed_rect.size:
                    feed = pygame.transform.smoothscale(feed, feed_rect.size)
                surface.blit(feed, feed_rect.topleft)
                scan_alpha = pygame.Surface(feed_rect.size, pygame.SRCALPHA)
                for row in range(0, feed_rect.h, 4):
                    pygame.draw.line(
                        scan_alpha,
                        (0, 0, 0, 26),
                        (0, row),
                        (feed_rect.w, row),
                        1,
                    )
                surface.blit(scan_alpha, feed_rect.topleft)
                feed_cx = feed_rect.centerx
                feed_cy = feed_rect.centery
                pygame.draw.line(surface, (236, 82, 82), (feed_cx - 14, feed_cy), (feed_cx + 14, feed_cy), 1)
                pygame.draw.line(surface, (236, 82, 82), (feed_cx, feed_cy - 14), (feed_cx, feed_cy + 14), 1)
                pygame.draw.circle(surface, (236, 82, 82), (feed_cx, feed_cy), 4)
            else:
                for idx in range(feed_rect.y, feed_rect.bottom, 4):
                    shade = 18 + ((idx - feed_rect.y) % 12)
                    pygame.draw.line(surface, (shade, 34, 34), (feed_rect.x, idx), (feed_rect.right, idx))
                offscreen = self._tiny_font.render("TARGET OUT OF FRAME", True, (232, 208, 208))
                surface.blit(offscreen, offscreen.get_rect(center=feed_rect.center))

            label_surf = self._small_font.render(label, True, (232, 246, 238))
            surface.blit(label_surf, (preview_rect.x + 8, preview_rect.bottom - 29))
            detail_text = detail
            while detail_text and self._tiny_font.size(detail_text)[0] > (preview_rect.w - 86):
                detail_text = detail_text[:-1]
            if detail_text != detail and len(detail_text) >= 4:
                detail_text = f"{detail_text[:-3]}..."
            detail_surf = self._tiny_font.render(detail_text, True, (168, 208, 192))
            surface.blit(detail_surf, (preview_rect.x + 8, preview_rect.bottom - 14))
            cue_text = cue
            while cue_text and self._tiny_font.size(cue_text)[0] > (preview_rect.w - 110):
                cue_text = cue_text[:-1]
            if cue_text != cue and len(cue_text) >= 4:
                cue_text = f"{cue_text[:-3]}..."
            cue_surf = self._tiny_font.render(cue_text, True, (132, 188, 170))
            surface.blit(cue_surf, cue_surf.get_rect(bottomright=(preview_rect.right - 8, preview_rect.bottom - 14)))

        if not using_panda:
            old_clip = surface.get_clip()
            surface.set_clip(track)
            try:
                compound_center_x = vanish_x - int(round(cam_x * 12.0))
                compound_center_y = horizon_y + 58 + int(round((descent_ratio - 0.35) * 18.0))
                apron = pygame.Rect(
                    compound_center_x - max(110, track.w // 7),
                    compound_center_y - 12,
                    max(220, track.w // 3),
                    max(86, track.h // 5),
                )
                pygame.draw.rect(surface, (22, 52, 44), apron, border_radius=12)
                pygame.draw.rect(surface, (74, 112, 100), apron, 1, border_radius=12)

                road_col = (34, 76, 62)
                pygame.draw.rect(
                    surface,
                    road_col,
                    pygame.Rect(apron.centerx - 12, apron.y - 8, 24, apron.h + 22),
                )
                pygame.draw.rect(
                    surface,
                    road_col,
                    pygame.Rect(apron.x - 16, apron.centery - 10, apron.w + 32, 20),
                )
                for lane_y in (apron.centery - 10, apron.centery + 10):
                    pygame.draw.line(
                        surface,
                        (140, 178, 160),
                        (apron.x - 10, lane_y),
                        (apron.right + 10, lane_y),
                        1,
                    )
                for lane_y in range(apron.y + 12, apron.bottom, 18):
                    pygame.draw.line(
                        surface,
                        (124, 166, 148),
                        (apron.centerx, lane_y),
                        (apron.centerx, min(apron.bottom, lane_y + 8)),
                        1,
                    )

                compound_distractors = (
                    ("building", "hangar", -86, -10, 8, (90, 122, 116), (116, 150, 144)),
                    ("building", "garage", -24, -14, 7, (102, 136, 128), (124, 160, 152)),
                    ("building", "tower", 54, -6, 6, (108, 144, 138), (130, 170, 164)),
                    ("truck", "", -70, 20, 5, (132, 132, 82), (172, 148, 96)),
                    ("truck", "", 34, 24, 5, (126, 128, 82), (168, 146, 94)),
                    ("tank", "", 72, 6, 5, (104, 112, 72), (154, 144, 90)),
                    ("soldier", "", -36, 30, 4, (98, 124, 94), (126, 138, 118)),
                    ("soldier", "", -10, 34, 4, (98, 124, 94), (126, 138, 118)),
                    ("soldier", "", 18, 36, 4, (98, 124, 94), (126, 138, 118)),
                    ("soldier", "", 54, 30, 4, (98, 124, 94), (126, 138, 118)),
                    ("building", "hangar", -54, 44, 6, (90, 122, 116), (116, 150, 144)),
                    ("building", "garage", 6, 48, 6, (100, 134, 126), (124, 160, 150)),
                )
                for kind, variant, ox, oy, size, fill, accent in compound_distractors:
                    draw_target_sprite(
                        kind=kind,
                        variant=variant,
                        center=(compound_center_x + ox, compound_center_y + oy),
                        size=size,
                        base_color=fill,
                        accent_color=accent,
                    )
            finally:
                surface.set_clip(old_clip)

        capture_box = pygame.Rect(cx - box_half_w, cy - box_half_h, box_half_w * 2, box_half_h * 2)
        capture_ready = bool(payload.target_in_capture_box) if payload is not None else False
        capture_zoom = float(payload.capture_zoom) if payload is not None else 0.0
        capture_color = (146, 198, 182)
        if capture_ready:
            capture_color = (142, 246, 188)
        if payload is not None and payload.capture_flash_s > 0.0:
            capture_color = (236, 248, 240) if payload.capture_feedback != "MISS" else (246, 142, 132)

        cross_len = max(10, min(track.w, track.h) // 30)
        pygame.draw.rect(surface, capture_color, capture_box, 2, border_radius=5)
        pygame.draw.line(surface, capture_color, (cx - cross_len, cy), (cx + cross_len, cy), 1)
        pygame.draw.line(surface, capture_color, (cx, cy - cross_len), (cx, cy + cross_len), 1)
        pygame.draw.circle(surface, capture_color, (cx, cy), 5, 1)

        if capture_zoom > 0.0:
            zoom_ring = max(capture_box.w, capture_box.h) // 2 + int(round(8.0 * capture_zoom))
            pygame.draw.circle(surface, (236, 248, 240), (cx, cy), zoom_ring, 1)

        if payload is not None:
            target_rel_x = max(-1.3, min(1.3, payload.target_rel_x))
            target_rel_y = max(-1.3, min(1.3, payload.target_rel_y))
            depth = max(0.0, min(1.0, (target_rel_y + 1.3) / 2.6))
            target_kind = str(payload.target_kind).strip().lower()
            target_variant = str(payload.target_variant).strip().lower()
            target_label = rapid_tracking_target_label(
                kind=target_kind,
                variant=target_variant,
            )
            target_radius = max(5, int(round(5 + (depth * 8))))
            if target_kind == "building":
                target_radius = max(target_radius, 8)
            if using_panda and panda_overlay is not None:
                overlay_size = panda_renderer.size if panda_renderer is not None else track.size
                sx = track.x + int(round((float(panda_overlay.screen_x) / max(1, overlay_size[0])) * track.w))
                sy = track.y + int(round((float(panda_overlay.screen_y) / max(1, overlay_size[1])) * track.h))
                target_x = sx
                target_y = sy
                target_on_screen = bool(panda_overlay.on_screen and panda_overlay.in_front)
            else:
                target_x = int(round(cx + (target_rel_x * x_scale)))
                target_y = int(round(cy + (target_rel_y * y_scale)))
                target_on_screen = track.inflate(-10, -10).collidepoint(target_x, target_y)
            preview_source: pygame.Surface | None = None

            if not using_panda:
                # Mixed ground/air decoys to keep the scan busy during handoffs.
                for idx in range(30):
                    branch = idx % 5
                    if branch == 0:
                        decoy_kind = "soldier"
                        speed = 0.05 + (idx % 5) * 0.010
                        world_y = 0.42 + ((idx % 5) * 0.06)
                        world_y += math.sin((anim_s * 1.2) + (idx * 0.6)) * 0.018
                    elif branch == 1:
                        decoy_kind = "truck"
                        speed = 0.18 + (idx % 4) * 0.022
                        world_y = 0.56 + ((idx % 4) * 0.05)
                    elif branch == 2:
                        decoy_kind = "tank"
                        speed = 0.14 + (idx % 4) * 0.016
                        world_y = 0.60 + ((idx % 3) * 0.05)
                    elif branch == 3:
                        decoy_kind = "helicopter"
                        speed = 0.22 + (idx % 5) * 0.032
                        world_y = -0.18 + ((idx % 4) * 0.10)
                        world_y += math.sin((anim_s * 1.4) + (idx * 0.7)) * 0.038
                    else:
                        decoy_kind = "jet"
                        speed = 0.28 + (idx % 6) * 0.040
                        world_y = -0.78 + ((idx % 6) * 0.12)
                        world_y += math.sin((anim_s * 1.8) + (idx * 0.9)) * 0.048

                    if branch == 2:
                        phase = (anim_s * speed * 1.2) + (idx * 0.47)
                        world_x = (-1.10 + ((idx % 4) * 0.72))
                    elif branch == 1:
                        phase = (anim_s * speed * 1.6) + (idx * 0.64)
                        world_x = ((phase % 6.8) - 3.4)
                        if idx % 2 == 1:
                            world_x = -world_x
                    elif branch == 0:
                        phase = (anim_s * speed * 1.1) + (idx * 0.54)
                        world_x = ((phase % 5.8) - 2.9)
                        if idx % 3 == 1:
                            world_x = -world_x
                    else:
                        phase = (anim_s * speed * 2.6) + (idx * 0.82)
                        world_x = ((phase + (idx * 0.36)) % 8.6) - 4.3
                        if idx % 3 == 1:
                            world_x = -world_x
                        world_x += math.sin((anim_s * 0.41) + idx) * 0.14

                    if decoy_kind == "tank":
                        spin_phase = (anim_s * (0.8 + (idx % 3) * 0.2)) + (idx * 0.35)
                        world_x += math.sin(spin_phase) * 0.02
                    elif decoy_kind == "soldier":
                        world_y += math.sin((anim_s * 2.2) + (idx * 0.8)) * 0.012
                    elif decoy_kind == "truck":
                        world_y += 0.0

                    ground_contact = decoy_kind in {"soldier", "truck", "tank"}
                    rel_x = world_x - (cam_x * (0.82 if ground_contact else 1.0))
                    rel_y = world_y - (cam_y * (0.56 if ground_contact else 0.72))

                    if rel_x < -1.55 or rel_x > 1.55 or rel_y < -1.45 or rel_y > 1.45:
                        continue

                    ridge_y = terrain_ridge_for(rel_x)
                    if ground_contact:
                        decoy_occluded = rel_y >= (ridge_y - 0.01)
                    elif decoy_kind == "helicopter":
                        decoy_occluded = rel_y >= (ridge_y + 0.04)
                    else:
                        decoy_occluded = rel_y >= (ridge_y + 0.11)

                    decoy_x = int(round(cx + (rel_x * x_scale)))
                    decoy_y = int(round(cy + (rel_y * y_scale)))
                    decoy_depth = max(0.0, min(1.0, (rel_y + 1.3) / 2.6))
                    base_size = 3 + (decoy_depth * 4)
                    if decoy_kind == "building":
                        base_size += 2.0
                    decoy_size = max(3, int(round(base_size)))
                    if decoy_kind == "soldier":
                        decoy_color = (88, 124, 96)
                    elif decoy_kind == "truck":
                        decoy_color = (154, 144, 92)
                    elif decoy_kind == "tank":
                        decoy_color = (120, 126, 76)
                    elif decoy_kind == "helicopter":
                        decoy_color = (102, 168, 132)
                    else:
                        decoy_color = (112, 170, 206)

                    if not decoy_occluded:
                        draw_target_sprite(
                            kind=decoy_kind,
                            center=(decoy_x, decoy_y),
                            size=decoy_size,
                            base_color=decoy_color,
                        )
                        if decoy_kind in {"truck", "jet", "tank"} and idx % 2 == 0:
                            box_w = max(14, int(round(15 + (decoy_depth * 14))))
                            box_h = max(10, int(round(11 + (decoy_depth * 10))))
                            box = pygame.Rect(
                                decoy_x - (box_w // 2), decoy_y - (box_h // 2), box_w, box_h
                            )
                            box_col = (210, 120, 86) if idx % 4 == 0 else (124, 160, 214)
                            pygame.draw.rect(surface, box_col, box, 1)
                    elif idx % 3 == 0:
                        occ_r = max(6, int(round(6 + (decoy_depth * 7))))
                        pygame.draw.circle(surface, (10, 18, 16), (decoy_x, decoy_y), occ_r)
                        pygame.draw.circle(surface, (74, 94, 88), (decoy_x, decoy_y), occ_r, 1)

                if payload.target_visible:
                    if target_kind == "soldier":
                        target_color = (226, 108, 96)
                        accent_color = (240, 214, 106)
                    elif target_kind == "building":
                        target_color = (164, 202, 188)
                        accent_color = (118, 146, 164)
                    elif target_kind == "truck":
                        target_color = (226, 200, 96)
                        accent_color = (166, 124, 72)
                    elif target_kind == "helicopter":
                        target_color = (132, 214, 176)
                        accent_color = (206, 238, 224)
                    else:
                        target_color = (120, 202, 255)
                        accent_color = (238, 246, 252)
                    glow_radius = target_radius + max(2, int(round(target_radius * 0.9)))
                    if target_kind == "soldier":
                        entourage_offsets = (
                            (-target_radius * 2, 0),
                            (target_radius * 2, target_radius // 3),
                            (-target_radius, target_radius * 2),
                            (target_radius, target_radius * 2),
                        )
                        for ox, oy in entourage_offsets:
                            draw_target_sprite(
                                kind="soldier",
                                center=(target_x + ox, target_y + oy),
                                size=max(4, target_radius - 2),
                                base_color=(92, 120, 94),
                                accent_color=(120, 132, 118),
                            )
                    pygame.draw.circle(
                        surface,
                        (target_color[0] // 3, target_color[1] // 3, target_color[2] // 3),
                        (target_x, target_y),
                        glow_radius,
                    )
                    draw_target_sprite(
                        kind=target_kind,
                        variant=target_variant,
                        center=(target_x, target_y),
                        size=target_radius,
                        base_color=target_color,
                        accent_color=accent_color,
                    )
                    if target_kind in {"jet", "helicopter"}:
                        pygame.draw.circle(
                            surface,
                            (255, 255, 255),
                            (target_x, target_y - max(2, target_radius // 2)),
                            max(1, target_radius // 3),
                        )
                    if payload.target_is_moving:
                        trail_len = max(
                            8, int(round((0.45 + (depth * 0.75)) * min(x_scale, y_scale) * 0.22))
                        )
                        vel_end = (
                            int(round(target_x + (payload.target_vx * trail_len * 4.0))),
                            int(round(target_y + (payload.target_vy * trail_len * 4.0))),
                        )
                        pygame.draw.line(surface, target_color, (target_x, target_y), vel_end, 2)
                    elif target_kind == "building":
                        pygame.draw.circle(
                            surface,
                            (236, 248, 240),
                            (target_x, target_y),
                            glow_radius + 3,
                            1,
                        )
                else:
                    mask_radius = max(20, int(round((track.w * 0.065) + (depth * 14))))
                    pygame.draw.circle(surface, (10, 16, 14), (target_x, target_y), mask_radius)
                    pygame.draw.circle(
                        surface, (128, 148, 138), (target_x, target_y), mask_radius, 1
                    )
                    obscured_txt = self._tiny_font.render("OBSCURED", True, (198, 216, 206))
                    surface.blit(
                        obscured_txt, obscured_txt.get_rect(midbottom=(target_x, target_y - 4))
                    )
            elif not payload.target_visible:
                obscured_txt = self._tiny_font.render("OBSCURED", True, (198, 216, 206))
                surface.blit(
                    obscured_txt, obscured_txt.get_rect(midbottom=(target_x, target_y - 4))
                )

            preview_source = surface.subsurface(track).copy()

            marker_color = (236, 82, 82) if payload.target_visible else (188, 118, 118)
            if target_on_screen:
                pygame.draw.line(surface, marker_color, (cx, cy), (target_x, target_y), 2)
                pygame.draw.circle(surface, marker_color, (target_x, target_y), max(4, target_radius // 2))
                pygame.draw.circle(
                    surface,
                    (255, 228, 228),
                    (target_x, target_y),
                    max(7, target_radius + 2),
                    1,
                )
            else:
                draw_edge_pointer(
                    target_px=target_x,
                    target_py=target_y,
                    label=target_label,
                )

            preview = pygame.Rect(track.right - 214, track.y + 42, 206, 122)
            draw_live_target_feed(
                preview_rect=preview,
                source=preview_source,
                focus_px=target_x,
                focus_py=target_y,
                label=target_label,
                detail=rapid_tracking_target_description(
                    kind=target_kind,
                    variant=target_variant,
                ),
                cue=rapid_tracking_target_cue(
                    kind=target_kind,
                    variant=target_variant,
                    handoff_mode=payload.target_handoff_mode,
                ),
                on_screen=target_on_screen,
                visible=bool(payload.target_visible),
                switch_preview_s=float(payload.target_switch_preview_s),
            )

            if payload.hud_visible:
                cross_color = (236, 248, 240)
                line_len = max(16, min(track.w, track.h) // 18)
                pygame.draw.line(surface, cross_color, (cx - line_len, cy), (cx + line_len, cy), 2)
                pygame.draw.line(surface, cross_color, (cx, cy - line_len), (cx, cy + line_len), 2)
                pygame.draw.circle(surface, cross_color, (cx, cy), max(8, line_len // 2), 2)

                box_w = max(28, int(round(30 + (depth * 26))))
                box_h = max(18, int(round(22 + (depth * 20))))
                track_box = pygame.Rect(
                    target_x - (box_w // 2), target_y - (box_h // 2), box_w, box_h
                )
                box_color = (150, 248, 188) if payload.target_visible else (148, 174, 162)
                pygame.draw.rect(surface, box_color, track_box, 2)
                # Corner ticks for a classic tracking-box look.
                tick = max(4, min(8, box_w // 8))
                pygame.draw.line(
                    surface,
                    box_color,
                    (track_box.left, track_box.top),
                    (track_box.left + tick, track_box.top),
                    2,
                )
                pygame.draw.line(
                    surface,
                    box_color,
                    (track_box.right - tick, track_box.top),
                    (track_box.right, track_box.top),
                    2,
                )
                pygame.draw.line(
                    surface,
                    box_color,
                    (track_box.left, track_box.bottom),
                    (track_box.left + tick, track_box.bottom),
                    2,
                )
                pygame.draw.line(
                    surface,
                    box_color,
                    (track_box.right - tick, track_box.bottom),
                    (track_box.right, track_box.bottom),
                    2,
                )

            if payload.capture_feedback != "" and payload.capture_flash_s > 0.0:
                flash_label = "CAM LOCK" if payload.capture_feedback != "MISS" else "MISS"
                if payload.capture_feedback not in {"", "MISS"}:
                    flash_label = f"{flash_label} {payload.capture_feedback}"
                flash_color = (236, 248, 240) if payload.capture_feedback != "MISS" else (244, 150, 142)
                flash = self._small_font.render(flash_label, True, flash_color)
                surface.blit(flash, flash.get_rect(midbottom=(cx, track.y + 52)))

            metrics_bg = pygame.Rect(track.x + 8, track.y + 8, track.w - 16, 30)
            pygame.draw.rect(surface, (14, 52, 42), metrics_bg)
            pygame.draw.rect(surface, (86, 130, 114), metrics_bg, 1)

            if target_kind == "building":
                status = "STATIC"
            else:
                status = "MOVING" if payload.target_is_moving else "STEADY"
            vis = "VISIBLE" if payload.target_visible else "OBSCURED"
            capture_acc = payload.capture_accuracy * 100.0
            left = self._tiny_font.render(
                (
                    f"Err {payload.mean_error:.3f}  RMS {payload.rms_error:.3f}  "
                    f"On {payload.on_target_ratio * 100.0:.1f}%  "
                    f"Cam {payload.capture_points}pts"
                ),
                True,
                text_main,
            )
            right = self._tiny_font.render(
                (
                    f"{target_label} {status} | {vis} | {payload.target_handoff_mode.upper()} | "
                    f"Lock {payload.lock_progress * 100.0:.0f}% | "
                    f"Shots {payload.capture_hits}/{payload.capture_attempts} "
                    f"({capture_acc:.0f}%)"
                ),
                True,
                text_muted,
            )
            surface.blit(left, (metrics_bg.x + 8, metrics_bg.y + 4))
            surface.blit(
                right, right.get_rect(midright=(metrics_bg.right - 8, metrics_bg.centery + 7))
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            prompt_bg = pygame.Rect(track.x + 10, track.bottom - 96, track.w - 20, 86)
            pygame.draw.rect(surface, (14, 52, 42), prompt_bg)
            pygame.draw.rect(surface, (86, 130, 114), prompt_bg, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                prompt_bg.inflate(-10, -8),
                color=text_muted,
                font=self._tiny_font,
                max_lines=6,
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        elif snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = (
                "Rudder axis controls left/right; Joystick axis 1 controls up/down. "
                "Trigger/Space/LMB captures when the target enters the center camera box."
            )
        else:
            footer = "Enter: Return to Tests"
        foot = self._tiny_font.render(footer, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))

    def _render_auditory_capacity_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: AuditoryCapacityPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (7, 10, 26)
        panel_bg = (12, 18, 42)
        border = (92, 106, 150)
        text_main = (232, 238, 252)
        text_muted = (172, 182, 216)

        def build_bg() -> pygame.Surface:
            layer = pygame.Surface((w, h), pygame.SRCALPHA)
            hh = max(1, h - 1)
            for y in range(h):
                t = y / float(hh)
                c = self._auditory_mix_rgb((12, 18, 44), bg, mix=t)
                pygame.draw.line(layer, (c[0], c[1], c[2], 255), (0, y), (w, y))
            return layer

        surface.blit(self._get_instrument_sprite(("auditory_panel_bg_v2", w, h), build_bg), (0, 0))

        margin = max(10, min(20, w // 42))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        self._draw_auditory_glass_panel(
            surface,
            frame,
            top_color=(18, 26, 62),
            bottom_color=panel_bg,
            border_color=border,
            border_radius=7,
            gloss_alpha=44,
        )
        pygame.draw.rect(surface, (56, 72, 122), frame.inflate(-6, -6), 1, border_radius=6)

        header_h = 46
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        self._draw_auditory_glass_panel(
            surface,
            header,
            top_color=(26, 40, 86),
            bottom_color=(16, 26, 58),
            border_color=(104, 122, 174),
            border_radius=6,
            gloss_alpha=52,
        )
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        intro_loading = snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE) and not self._intro_loading_complete(
            snap.phase
        )

        title = self._small_font.render(f"Auditory Capacity - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 12, header.y + 10))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        body = pygame.Rect(frame.x + 8, header.bottom + 8, frame.w - 16, frame.h - header_h - 18)
        is_active_play = snap.phase in (Phase.PRACTICE, Phase.SCORED)
        time_fill_ratio = self._auditory_time_fill_ratio(snap)
        if is_active_play:
            tube_panel = pygame.Rect(body.x, body.y, body.w, body.h)
            info_panel: pygame.Rect | None = None
        else:
            gap = max(8, min(14, body.w // 52))
            tube_panel_w = int(body.w * 0.74)
            tube_panel = pygame.Rect(body.x, body.y, tube_panel_w, body.h)
            info_panel = pygame.Rect(
                tube_panel.right + gap, body.y, body.w - tube_panel_w - gap, body.h
            )

        self._draw_auditory_glass_panel(
            surface,
            tube_panel,
            top_color=(12, 20, 48),
            bottom_color=(8, 14, 34),
            border_color=(88, 108, 168),
            border_radius=6,
            gloss_alpha=36,
        )
        if info_panel is not None:
            self._draw_auditory_glass_panel(
                surface,
                info_panel,
                top_color=(14, 22, 54),
                bottom_color=(8, 14, 34),
                border_color=(88, 108, 168),
                border_radius=6,
                gloss_alpha=30,
            )

        world = tube_panel.inflate(-18, -28)
        panda_renderer = self._get_auditory_panda_renderer(size=(world.w, world.h))
        if panda_renderer is not None:
            self._render_auditory_capacity_panda_view(
                surface=surface,
                world=world,
                payload=payload,
                time_remaining_s=snap.time_remaining_s,
                time_fill_ratio=time_fill_ratio,
                renderer=panda_renderer,
            )
        else:
            fallback_panel = world.inflate(-max(18, world.w // 10), -max(24, world.h // 6))
            self._draw_auditory_glass_panel(
                surface,
                fallback_panel,
                top_color=(18, 26, 62),
                bottom_color=(10, 16, 42),
                border_color=(110, 126, 178),
                border_radius=8,
                gloss_alpha=34,
            )
            title_text = self._small_font.render("3D Renderer Required", True, text_main)
            surface.blit(
                title_text,
                title_text.get_rect(midtop=(fallback_panel.centerx, fallback_panel.y + 16)),
            )
            lines = [
                "Auditory Capacity now uses the Panda3D scene only.",
                "The 2D fallback path is disabled for this test.",
                "Install or re-enable Panda3D rendering to continue.",
            ]
            text_rect = fallback_panel.inflate(-18, -46)
            self._draw_wrapped_text(
                surface,
                "\n".join(lines),
                text_rect,
                color=text_muted,
                font=self._tiny_font,
                max_lines=6,
            )
            pygame.draw.rect(surface, (20, 42, 140), world, 2)
            pygame.draw.rect(surface, (78, 104, 178), world.inflate(-4, -4), 1)

        if info_panel is not None:
            info_lines: list[str] = []
            if intro_loading:
                stage_label = (
                    "practice block"
                    if snap.phase is Phase.INSTRUCTIONS
                    else "timed block"
                )
                info_lines.extend(
                    [
                        "Loading auditory scene",
                        "",
                        f"Preparing {stage_label}.",
                        "Warming audio routing and display assets.",
                        "Centering the tube view and gate cues.",
                        "Enter stays locked until loading completes.",
                    ]
                )
            elif payload is None:
                info_lines.extend(str(snap.prompt).split("\n"))
            else:
                assigned = (
                    ", ".join(payload.assigned_callsigns) if payload.assigned_callsigns else "—"
                )
                info_lines.append(f"Your call signs: {assigned}")
                info_lines.append(
                    f"Ball state: {payload.ball_color} / {payload.ball_number}"
                )
                info_lines.append(f"Instruction: {payload.instruction_text or '—'}")

                rule_bits: list[str] = []
                if payload.forbidden_gate_color is not None:
                    rule_bits.append(f"avoid {payload.forbidden_gate_color} colour")
                if payload.forbidden_gate_shape is not None:
                    rule_bits.append(f"avoid {payload.forbidden_gate_shape} shape")
                info_lines.append(
                    f"Gate rule: {', '.join(rule_bits) if rule_bits else 'none active'}"
                )

                if payload.color_command is not None:
                    info_lines.append(f"Colour command: {payload.color_command}")
                elif payload.number_command is not None:
                    info_lines.append(f"Number command: {payload.number_command}")
                else:
                    info_lines.append("State command: —")

                if payload.sequence_display is not None:
                    info_lines.append(f"Memory digit: {payload.sequence_display}")
                elif payload.sequence_response_open:
                    info_lines.append("Recall: type digits and press Enter")
                else:
                    info_lines.append(
                        f"Memory buffer: {payload.digit_buffer_length} digits"
                    )

                info_lines.append("")
                noise_pct = int(round(payload.background_noise_level * 100))
                info_lines.append(
                    f"Gates hit/miss: {payload.gate_hits}/{payload.gate_misses}  "
                    f"Forbidden hits: {payload.forbidden_gate_hits}"
                )
                info_lines.append(
                    f"Cmd ok/missed: {payload.correct_command_executions}/{payload.missed_valid_commands}  "
                    f"False responses: {payload.false_responses_to_distractors}"
                )
                info_lines.append(
                    f"Boundary violations: {payload.collisions}  Recall accuracy: "
                    f"{int(round(payload.digit_recall_accuracy * 100))}%"
                )
                info_lines.append(f"Noise: {noise_pct}%  Distortion: {int(round(payload.distortion_level * 100))}%")
                info_lines.append(
                    "Noise source: "
                    f"{_AuditoryCapacityAudioAdapter.format_noise_source_label(payload.background_noise_source)}"
                )

            info_rect = info_panel.inflate(-10, -10)
            y = info_rect.y
            for line in info_lines:
                rendered = self._tiny_font.render(
                    line, True, text_main if line != "" else text_muted
                )
                surface.blit(rendered, (info_rect.x, y))
                y += 18
                if y > info_rect.bottom - 18:
                    break

        if intro_loading:
            overlay = pygame.Rect(
                tube_panel.x + max(22, tube_panel.w // 9),
                tube_panel.y + max(24, tube_panel.h // 5),
                tube_panel.w - max(44, tube_panel.w // 5),
                min(120, tube_panel.h // 3),
            )
            self._draw_auditory_glass_panel(
                surface,
                overlay,
                top_color=(18, 26, 62),
                bottom_color=(10, 16, 42),
                border_color=(110, 126, 178),
                border_radius=8,
                gloss_alpha=34,
            )
            title_text = self._small_font.render("Loading", True, text_main)
            surface.blit(title_text, title_text.get_rect(midtop=(overlay.centerx, overlay.y + 16)))
            dot_count = (pygame.time.get_ticks() // 220) % 4
            status_text = self._tiny_font.render(
                f"Preparing audio cues and display{'.' * dot_count}",
                True,
                text_muted,
            )
            surface.blit(
                status_text,
                status_text.get_rect(midtop=(overlay.centerx, overlay.y + 48)),
            )
            hint_text = self._tiny_font.render(
                "Start is disabled until loading completes.",
                True,
                text_muted,
            )
            surface.blit(
                hint_text,
                hint_text.get_rect(midtop=(overlay.centerx, overlay.y + 72)),
            )

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            prompt_rect = pygame.Rect(
                tube_panel.x + 10,
                tube_panel.bottom - 96,
                tube_panel.w - 20,
                86,
            )
            self._draw_auditory_glass_panel(
                surface,
                prompt_rect,
                top_color=(16, 26, 62),
                bottom_color=(10, 16, 42),
                border_color=(90, 106, 152),
                border_radius=6,
                gloss_alpha=32,
            )
            prompt_text = str(snap.prompt)
            if intro_loading:
                if snap.phase is Phase.INSTRUCTIONS:
                    prompt_text = (
                        "Loading practice block. Review the controls while the auditory scene "
                        "finishes preparing."
                    )
                else:
                    prompt_text = (
                        "Loading timed block. Enter will work once the next auditory run is ready."
                    )
            self._draw_wrapped_text(
                surface,
                prompt_text,
                prompt_rect.inflate(-8, -8),
                color=text_muted,
                font=self._tiny_font,
                max_lines=6,
            )

        if self._auditory_testing_menu and not intro_loading:
            self._render_auditory_testing_menu(
                surface=surface,
                snap=snap,
                payload=payload,
                frame=frame,
                header=header,
            )

        footer_text = (
            "Q/W/E/R: colour  Keypad 0-9: number  Digits+Enter: recall  |  "
            "WASD/arrows or HOTAS to fly"
        )
        if snap.phase is Phase.RESULTS:
            footer_text = "Enter: Return to Tests"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer_text = (
                "Loading...  |  Esc/Backspace: Back"
                if intro_loading
                else "Enter: Continue  |  Esc/Backspace: Back"
            )
        if self._auditory_testing_menu:
            footer_text += "  |  F9: hide testing menu"
        else:
            footer_text += "  |  F9: show testing menu"

        foot = self._tiny_font.render(footer_text, True, text_muted)
        surface.blit(foot, foot.get_rect(midbottom=(frame.centerx, frame.bottom - 10)))

    def _render_auditory_capacity_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: AuditoryCapacityPayload | None,
    ) -> None:
        if payload is None or not payload.sequence_response_open:
            return
        if snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            return

        box = pygame.Rect(34, surface.get_height() - 112, 380, 44)
        pygame.draw.rect(surface, (20, 24, 46), box)
        pygame.draw.rect(surface, (112, 126, 166), box, 2)

        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry = self._app.font.render(self._input + caret, True, (236, 242, 255))
        surface.blit(entry, (box.x + 10, box.y + 8))

        hint = self._small_font.render("Recall digits then press Enter", True, (156, 170, 204))
        surface.blit(hint, (34, surface.get_height() - 58))

    def _render_auditory_capacity_panda_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        payload: AuditoryCapacityPayload | None,
        time_remaining_s: float | None,
        time_fill_ratio: float | None,
        renderer: AuditoryCapacityPanda3DRenderer,
    ) -> None:
        view = renderer.render(payload=payload)
        if view.get_size() != world.size:
            view = pygame.transform.smoothscale(view, world.size)
        surface.blit(view, world.topleft)

        top_strip = pygame.Rect(world.x + 1, world.y + 1, world.w - 2, 18)
        self._draw_auditory_glass_panel(
            surface,
            top_strip,
            top_color=(22, 56, 172),
            bottom_color=(10, 32, 120),
            border_color=(84, 112, 188),
            border_radius=0,
            gloss_alpha=28,
        )
        strip_text = self._tiny_font.render("Auditory Capacity Test", True, (226, 236, 255))
        surface.blit(strip_text, strip_text.get_rect(center=top_strip.center))

        if time_remaining_s is not None:
            rem = max(0, int(round(time_remaining_s)))
            timer = pygame.Rect(world.right - 148, world.y + 14, 136, 46)
            self._draw_auditory_glass_panel(
                surface,
                timer,
                top_color=(122, 136, 160),
                bottom_color=(68, 80, 106),
                border_color=(148, 164, 194),
                border_radius=9,
                gloss_alpha=56,
            )
            t_label = self._small_font.render("Seconds", True, (236, 242, 255))
            t_val = self._small_font.render(str(rem), True, (236, 242, 255))
            surface.blit(t_label, t_label.get_rect(center=(timer.centerx, timer.y + 15)))
            surface.blit(t_val, t_val.get_rect(center=(timer.centerx, timer.y + 33)))

        bar = pygame.Rect(world.right - 144, world.bottom - 26, 126, 14)
        self._draw_auditory_glass_panel(
            surface,
            bar,
            top_color=(116, 122, 136),
            bottom_color=(64, 72, 90),
            border_color=(146, 152, 170),
            border_radius=7,
            gloss_alpha=28,
        )
        fill_ratio = 0.72 if time_fill_ratio is None else max(0.0, min(1.0, time_fill_ratio))
        fill = pygame.Rect(
            bar.x + 5,
            bar.y + 4,
            int(round((bar.w - 10) * fill_ratio)),
            bar.h - 8,
        )
        if fill.w > 0:
            self._draw_auditory_glass_panel(
                surface,
                fill,
                top_color=(206, 212, 220),
                bottom_color=(150, 158, 172),
                border_color=(224, 228, 236),
                border_radius=4,
                gloss_alpha=20,
            )

        pygame.draw.rect(surface, (20, 42, 140), world, 2)
        pygame.draw.rect(surface, (78, 104, 178), world.inflate(-4, -4), 1)

    @staticmethod
    def _auditory_color_rgb(color_name: str, strength: float) -> tuple[int, int, int]:
        base = {
            "RED": (232, 72, 86),
            "GREEN": (84, 214, 136),
            "BLUE": (96, 154, 244),
            "YELLOW": (238, 206, 84),
        }.get(color_name.upper(), (210, 214, 226))
        s = max(0.0, min(1.0, float(strength)))
        mix = 1.0 - s
        return (
            int(round(base[0] * s + 255.0 * mix)),
            int(round(base[1] * s + 255.0 * mix)),
            int(round(base[2] * s + 255.0 * mix)),
        )

    @staticmethod
    def _draw_auditory_gate_shape(
        surface: pygame.Surface,
        *,
        shape: str,
        center: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
        depth: float = 1.0,
    ) -> None:
        cx, cy = center
        half = max(8, int(round(size * 1.42)))
        d = max(0.0, min(1.0, float(depth)))
        stroke = max(1, int(round(1.0 + (d * 1.2))))
        glow_alpha = max(16, min(96, int(round(18 + (d * 42)))))
        glow = (
            max(0, min(255, int(round((color[0] * 0.48) + 32)))),
            max(0, min(255, int(round((color[1] * 0.48) + 32)))),
            max(0, min(255, int(round((color[2] * 0.48) + 32)))),
            glow_alpha,
        )
        highlight = (
            max(0, min(255, int(round((color[0] * 0.82) + 28)))),
            max(0, min(255, int(round((color[1] * 0.82) + 28)))),
            max(0, min(255, int(round((color[2] * 0.82) + 28)))),
        )

        token = str(shape).upper()
        points: list[tuple[int, int]]
        if token == "CIRCLE":
            points = []
            steps = 56
            for idx in range(steps):
                angle = (idx / float(steps)) * math.tau
                points.append(
                    (
                        int(round(cx + (math.cos(angle) * half))),
                        int(round(cy + (math.sin(angle) * half))),
                    )
                )
        elif token == "TRIANGLE":
            top_y = cy - int(round(half * 1.34))
            base_y = cy + int(round(half * 1.04))
            span_x = max(6, int(round(half * 0.56)))
            points = [(cx, top_y), (cx - span_x, base_y), (cx + span_x, base_y)]
        else:
            rect_half = int(round(half * 1.10))
            points = [
                (cx - rect_half, cy - rect_half),
                (cx + rect_half, cy - rect_half),
                (cx + rect_half, cy + rect_half),
                (cx - rect_half, cy + rect_half),
            ]

        pad = max(4, stroke + 3)
        min_x = min(point[0] for point in points) - pad
        min_y = min(point[1] for point in points) - pad
        max_x = max(point[0] for point in points) + pad
        max_y = max(point[1] for point in points) + pad
        halo = pygame.Surface((max(1, max_x - min_x + 1), max(1, max_y - min_y + 1)), pygame.SRCALPHA)
        local_points = [(x - min_x, y - min_y) for x, y in points]
        pygame.draw.aalines(halo, glow, True, local_points)
        if stroke > 1:
            pygame.draw.lines(halo, glow, True, local_points, stroke + 1)
        surface.blit(halo, (min_x, min_y))

        if stroke > 1:
            pygame.draw.lines(surface, color, True, points, stroke)
        else:
            pygame.draw.aalines(surface, color, True, points)
        pygame.draw.aalines(surface, highlight, True, points)

    def _auditory_time_fill_ratio(self, snap: TestSnapshot) -> float | None:
        if snap.time_remaining_s is None:
            return None
        cfg = getattr(self._engine, "_cfg", None)
        if cfg is None:
            return None
        duration: float | None = None
        if snap.phase is Phase.PRACTICE:
            raw = getattr(cfg, "practice_duration_s", None)
            if isinstance(raw, int | float) and raw > 0:
                duration = float(raw)
        elif snap.phase is Phase.SCORED:
            raw = getattr(cfg, "scored_duration_s", None)
            if isinstance(raw, int | float) and raw > 0:
                duration = float(raw)
        if duration is None or duration <= 0.0:
            return None
        return max(0.0, min(1.0, float(snap.time_remaining_s) / duration))

    def _render_auditory_capacity_tube_chase_view(
        self,
        *,
        surface: pygame.Surface,
        world: pygame.Rect,
        payload: AuditoryCapacityPayload | None,
        time_remaining_s: float | None,
        time_fill_ratio: float | None,
    ) -> None:
        def build_vortex() -> pygame.Surface:
            tex_size = max(196, min(1024, max(world.w, world.h)))
            rgba = _OpenGLAuditoryRenderer._build_vortex_rgba(size=tex_size)
            tex = pygame.image.frombuffer(bytearray(rgba), (tex_size, tex_size), "RGBA").copy()
            bg = pygame.transform.smoothscale(tex, (world.w, world.h))

            cx0 = world.w // 2
            cy0 = world.h // 2
            max_rx = world.w * 0.50
            max_ry = world.h * 0.50

            ring_surface = pygame.Surface((world.w, world.h), pygame.SRCALPHA)
            ring_count = 34
            for i in range(ring_count):
                t = i / float(max(1, ring_count - 1))
                rx = max(6, int(round(max_rx * (0.16 + (0.84 * t)))))
                ry = max(6, int(round(max_ry * (0.16 + (0.84 * t)))))
                ring = pygame.Rect(0, 0, rx * 2, ry * 2)
                ring.center = (cx0, cy0)
                alpha = int(round(56 * (1.0 - t)))
                pygame.draw.ellipse(
                    ring_surface,
                    (36, 72, 168, alpha),
                    ring,
                    1,
                )

            arm_count = 20
            steps = 140
            for arm in range(arm_count):
                phase = (arm / float(arm_count)) * math.tau
                prev: tuple[int, int] | None = None
                for step in range(steps):
                    s = step / float(max(1, steps - 1))
                    theta = phase + (s * math.tau * 2.8)
                    radius = s**1.07
                    px = int(round(cx0 + math.cos(theta) * max_rx * radius))
                    py = int(round(cy0 + math.sin(theta) * max_ry * radius))
                    if prev is not None:
                        alpha = int(round(48 * (1.0 - s)))
                        pygame.draw.aaline(
                            ring_surface,
                            (78, 120, 220, alpha),
                            prev,
                            (px, py),
                        )
                    prev = (px, py)

            vignette = pygame.Surface((world.w, world.h), pygame.SRCALPHA)
            for i in range(44):
                t = i / 43.0
                rx = int(round((world.w * 0.52) * (1.0 - (t * 0.92))))
                ry = int(round((world.h * 0.52) * (1.0 - (t * 0.92))))
                if rx <= 1 or ry <= 1:
                    continue
                alpha = int(round(5 + (72 * (t**1.35))))
                pygame.draw.ellipse(
                    vignette,
                    (4, 10, 28, alpha),
                    pygame.Rect(cx0 - rx, cy0 - ry, rx * 2, ry * 2),
                    1,
                )

            rng = random.Random((world.w * 9241) + (world.h * 7519) + 133)
            specks = max(1400, (world.w * world.h) // 170)
            stars = pygame.Surface((world.w, world.h), pygame.SRCALPHA)
            for _ in range(specks):
                x = rng.randrange(0, world.w)
                y = rng.randrange(0, world.h)
                dx = (x - cx0) / max(1.0, max_rx)
                dy = (y - cy0) / max(1.0, max_ry)
                dist = min(1.0, math.sqrt((dx * dx) + (dy * dy)))
                alpha = int(round(20 + (86 * dist)))
                lum = int(round(122 + (122 * rng.random())))
                stars.set_at((x, y), (lum // 5, lum // 2, lum, alpha))

            bg.blit(ring_surface, (0, 0))
            bg.blit(stars, (0, 0))
            bg.blit(vignette, (0, 0))
            return bg

        key = ("auditory_vortex_reference_v2", world.w, world.h)
        surface.blit(self._get_instrument_sprite(key, build_vortex), world.topleft)

        top_strip = pygame.Rect(world.x + 1, world.y + 1, world.w - 2, 18)
        self._draw_auditory_glass_panel(
            surface,
            top_strip,
            top_color=(22, 56, 172),
            bottom_color=(10, 32, 120),
            border_color=(84, 112, 188),
            border_radius=0,
            gloss_alpha=28,
        )
        strip_text = self._tiny_font.render("Auditory Capacity Test", True, (226, 236, 255))
        surface.blit(strip_text, strip_text.get_rect(center=top_strip.center))

        cx = world.centerx
        cy = world.centery
        cross = (136, 32, 38)

        if time_remaining_s is not None:
            rem = max(0, int(round(time_remaining_s)))
            timer = pygame.Rect(world.right - 148, world.y + 14, 136, 46)
            self._draw_auditory_glass_panel(
                surface,
                timer,
                top_color=(122, 136, 160),
                bottom_color=(68, 80, 106),
                border_color=(148, 164, 194),
                border_radius=9,
                gloss_alpha=56,
            )
            t_label = self._small_font.render("Seconds", True, (236, 242, 255))
            t_val = self._small_font.render(str(rem), True, (236, 242, 255))
            surface.blit(t_label, t_label.get_rect(center=(timer.centerx, timer.y + 15)))
            surface.blit(t_val, t_val.get_rect(center=(timer.centerx, timer.y + 33)))

        if payload is not None:
            y_half_span = max(0.08, float(payload.tube_half_height))
            x_half_span = max(0.08, float(payload.tube_half_width))

            gates: list[tuple[float, AuditoryCapacityGate, float, float, int]] = []
            far_depth = 2.20
            future_gates = [gate for gate in payload.gates if gate.x_norm >= -0.18]
            future_gates.sort(key=lambda gate: float(gate.x_norm))
            for visible_idx, gate in enumerate(future_gates[:8]):
                rel = gate.x_norm
                depth = max(0.0, min(1.0, rel / far_depth))
                approach = 1.0 - depth
                gy = max(-1.0, min(1.0, gate.y_norm / y_half_span))
                lane = _auditory_guide_lane(visible_idx)
                gates.append((approach, gate, lane, gy, visible_idx))

            for approach, gate, lane, gy, visible_idx in sorted(gates, key=lambda g: g[0]):
                lane_scale = (world.w * 0.06) + (world.w * 0.13 * approach)
                rise_scale = (world.h * 0.06) + (world.h * 0.14 * approach)
                sx = cx + int(round(lane * lane_scale))
                sy = cy + int(round(gy * rise_scale))
                size = max(7, int(round(min(world.w, world.h) * (0.026 + (0.092 * approach)))))
                if visible_idx < 2:
                    size = int(round(size * 1.18))
                gate_color = self._auditory_color_rgb(gate.color, 1.0)
                draw_color = self._auditory_mix_rgb(
                    (24, 34, 64), gate_color, mix=0.30 + (0.70 * approach)
                )
                self._draw_auditory_gate_shape(
                    surface,
                    shape=gate.shape,
                    center=(sx, sy),
                    size=size,
                    color=draw_color,
                    depth=approach,
                )

            bx = max(-1.0, min(1.0, payload.ball_x / x_half_span))
            by = max(-1.0, min(1.0, payload.ball_y / y_half_span))
            ball_x = cx + int(round(bx * (world.w * 0.16)))
            ball_y = cy + int(round(by * (world.h * 0.16)))
            ball_x = max(world.left + 9, min(world.right - 9, ball_x))
            ball_y = max(world.top + 9, min(world.bottom - 9, ball_y))
            ball_radius = max(5, min(13, world.h // 23))
            danger = float(payload.ball_contact_ratio) >= 1.0
            self._draw_auditory_ball_3d(
                surface,
                center=(ball_x, ball_y),
                radius=ball_radius,
                danger=danger,
            )

            bar = pygame.Rect(world.right - 144, world.bottom - 26, 126, 14)
            self._draw_auditory_glass_panel(
                surface,
                bar,
                top_color=(116, 122, 136),
                bottom_color=(64, 72, 90),
                border_color=(146, 152, 170),
                border_radius=7,
                gloss_alpha=28,
            )
            fill_ratio = 0.72 if time_fill_ratio is None else max(0.0, min(1.0, time_fill_ratio))
            fill = pygame.Rect(
                bar.x + 5,
                bar.y + 4,
                int(round((bar.w - 10) * fill_ratio)),
                bar.h - 8,
            )
            self._draw_auditory_glass_panel(
                surface,
                fill,
                top_color=(206, 212, 220),
                bottom_color=(150, 158, 172),
                border_color=(224, 228, 236),
                border_radius=4,
                gloss_alpha=20,
            )

        self._draw_auditory_crosshair(
            surface,
            world=world,
            center=(cx, cy),
            color=cross,
        )
        pygame.draw.rect(surface, (20, 42, 140), world, 2)
        pygame.draw.rect(surface, (78, 104, 178), world.inflate(-4, -4), 1)

    @staticmethod
    def _auditory_mix_rgb(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
        *,
        mix: float,
    ) -> tuple[int, int, int]:
        m = max(0.0, min(1.0, float(mix)))
        return (
            int(round((a[0] * (1.0 - m)) + (b[0] * m))),
            int(round((a[1] * (1.0 - m)) + (b[1] * m))),
            int(round((a[2] * (1.0 - m)) + (b[2] * m))),
        )

    @staticmethod
    def _draw_auditory_glass_panel(
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        top_color: tuple[int, int, int],
        bottom_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        border_radius: int = 0,
        gloss_alpha: int = 0,
    ) -> None:
        if rect.w <= 0 or rect.h <= 0:
            return

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        h = max(1, rect.h - 1)
        for y in range(rect.h):
            t = y / float(h)
            color = (
                int(round((top_color[0] * (1.0 - t)) + (bottom_color[0] * t))),
                int(round((top_color[1] * (1.0 - t)) + (bottom_color[1] * t))),
                int(round((top_color[2] * (1.0 - t)) + (bottom_color[2] * t))),
                255,
            )
            pygame.draw.line(panel, color, (0, y), (rect.w, y))

        if border_radius > 0:
            mask = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                mask, (255, 255, 255, 255), mask.get_rect(), border_radius=border_radius
            )
            panel.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        if gloss_alpha > 0:
            gloss_h = max(2, rect.h // 2)
            for y in range(gloss_h):
                alpha = int(round(gloss_alpha * (1.0 - (y / float(max(1, gloss_h - 1))))))
                pygame.draw.line(
                    panel,
                    (255, 255, 255, alpha),
                    (2, y + 1),
                    (max(2, rect.w - 3), y + 1),
                )

        surface.blit(panel, rect.topleft)
        pygame.draw.rect(surface, border_color, rect, 1, border_radius=border_radius)

    @staticmethod
    def _draw_auditory_crosshair(
        surface: pygame.Surface,
        *,
        world: pygame.Rect,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        glow = (
            max(0, min(255, color[0] + 26)),
            max(0, min(255, color[1] + 8)),
            max(0, min(255, color[2] + 8)),
            56,
        )
        glow_layer = pygame.Surface((world.w, world.h), pygame.SRCALPHA)
        rel_x = cx - world.x
        rel_y = cy - world.y
        pygame.draw.line(glow_layer, glow, (rel_x, 0), (rel_x, world.h - 1), 3)
        pygame.draw.line(glow_layer, glow, (0, rel_y), (world.w - 1, rel_y), 3)
        surface.blit(glow_layer, world.topleft)

        pygame.draw.aaline(surface, color, (cx, world.top + 1), (cx, world.bottom - 1))
        pygame.draw.aaline(surface, color, (world.left + 1, cy), (world.right - 1, cy))

    @staticmethod
    def _draw_auditory_ball_3d(
        surface: pygame.Surface,
        *,
        center: tuple[int, int],
        radius: int,
        danger: bool,
    ) -> None:
        cx, cy = center
        r = max(2, int(radius))
        shadow_color = (92, 108, 146)
        pygame.draw.circle(
            surface,
            shadow_color,
            (cx + max(1, r // 3), cy + max(1, r // 3)),
            max(2, r - 1),
        )

        lo = (150, 166, 204)
        hi = (242, 248, 255)
        for i in range(r, 0, -1):
            t = i / float(max(1, r))
            mix = t**0.72
            col = (
                int(round((lo[0] * (1.0 - mix)) + (hi[0] * mix))),
                int(round((lo[1] * (1.0 - mix)) + (hi[1] * mix))),
                int(round((lo[2] * (1.0 - mix)) + (hi[2] * mix))),
            )
            pygame.draw.circle(surface, col, (cx, cy), i)

        edge = (248, 252, 255)
        if danger:
            edge = (244, 82, 92)
        pygame.draw.circle(surface, edge, (cx, cy), r, 2)

        pygame.draw.circle(
            surface,
            (255, 255, 255),
            (cx - max(1, r // 3), cy - max(1, r // 3)),
            max(1, r // 4),
        )

    @staticmethod
    def _fmt_debug_seconds(value: float | None) -> str:
        if value is None:
            return "—"
        v = max(0.0, float(value))
        return f"{v:.1f}s"

    def _render_auditory_testing_menu(
        self,
        *,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: AuditoryCapacityPayload | None,
        frame: pygame.Rect,
        header: pygame.Rect,
    ) -> None:
        lines: list[str] = []
        lines.append(f"Phase: {snap.phase.value}")

        if payload is None:
            lines.append("Waiting for runtime payload...")
        else:
            ball_color = str(payload.ball_color).upper()
            assigned = ", ".join(payload.assigned_callsigns) if payload.assigned_callsigns else "—"
            lines.append(f"Your call signs: {assigned}")
            lines.append(f"Ball state: {ball_color} / {payload.ball_number}")
            lines.append(f"Instruction: {payload.instruction_text or '—'}")
            lines.append(
                f"Colour cue: {payload.color_command or '—'}    "
                f"Number cue: {payload.number_command if payload.number_command is not None else '—'}"
            )
            lines.append(
                f"Forbidden colour: {payload.forbidden_gate_color or '—'}    "
                f"Forbidden shape: {payload.forbidden_gate_shape or '—'}"
            )

            if payload.sequence_display is not None:
                lines.append(f"Digit cue: {payload.sequence_display}")
            elif payload.sequence_response_open:
                lines.append(
                    f"Recall open: YES    target len: {payload.recall_target_length or 0}    "
                    f"typed: {self._input or '—'}"
                )
            else:
                lines.append(f"Memory buffer length: {payload.digit_buffer_length}")

            nearest_gate: AuditoryCapacityGate | None = None
            nearest_delta = 1_000_000.0
            for gate in payload.gates:
                if gate.x_norm < payload.ball_x:
                    continue
                delta = float(gate.x_norm - payload.ball_x)
                if delta < nearest_delta:
                    nearest_delta = delta
                    nearest_gate = gate

            if nearest_gate is None and payload.gates:
                nearest_gate = min(
                    payload.gates, key=lambda g: abs(float(g.x_norm - payload.ball_x))
                )
                nearest_delta = abs(float(nearest_gate.x_norm - payload.ball_x))

            if nearest_gate is not None:
                gate_color = str(nearest_gate.color).upper()
                gate_shape = str(nearest_gate.shape).upper()
                should_pass_gate = True
                if payload.forbidden_gate_color is not None and gate_color == payload.forbidden_gate_color:
                    should_pass_gate = False
                if payload.forbidden_gate_shape is not None and gate_shape == payload.forbidden_gate_shape:
                    should_pass_gate = False
                action = "PASS" if should_pass_gate else "AVOID"
                lines.append(
                    f"Next gate: {gate_color}/{gate_shape}  "
                    f"dx={nearest_delta:.2f}  Action: {action}"
                )
            else:
                lines.append("Next gate: —")

            lines.append(
                "Cue timers left:"
                f" CMD {self._fmt_debug_seconds(payload.command_time_left_s)}"
                f"  SHOW {self._fmt_debug_seconds(payload.sequence_show_time_left_s)}"
                f"  RESP {self._fmt_debug_seconds(payload.sequence_response_time_left_s)}"
            )
            lines.append(
                "Next events in:"
                f" GATE {self._fmt_debug_seconds(payload.next_gate_in_s)}"
                f"  CMD {self._fmt_debug_seconds(payload.next_command_in_s)}"
            )
            lines.append(
                f"Cmd ok/missed: {payload.correct_command_executions}/{payload.missed_valid_commands}  "
                f"False responses: {payload.false_responses_to_distractors}  "
                f"Recall acc: {int(round(payload.digit_recall_accuracy * 100))}%"
            )

            audio = self._auditory_audio_debug
            if audio:
                available = bool(audio.get("audio_available", False))
                layers = int(audio.get("ambient_layers_target", 0))
                lines.append(
                    f"Audio available: {'YES' if available else 'NO'}  "
                    f"Ambient layers target: {layers}"
                )

                slots_raw = audio.get("ambient_slots")
                if isinstance(slots_raw, tuple | list):
                    slot_a = str(slots_raw[0]) if len(slots_raw) > 0 else "—"
                    slot_b = str(slots_raw[1]) if len(slots_raw) > 1 else "—"
                else:
                    slot_a = "—"
                    slot_b = "—"
                lines.append(f"Ambient tracks: L1={slot_a}  L2={slot_b}")
                source_label = str(audio.get("noise_source", "Auto"))
                lines.append(f"Noise source: {source_label}")

                busy_raw = audio.get("channels_busy")
                if isinstance(busy_raw, dict):
                    busy_text = (
                        f"bg={int(bool(busy_raw.get('bg', False)))} "
                        f"dist={int(bool(busy_raw.get('dist', False)))} "
                        f"cue={int(bool(busy_raw.get('cue', False)))} "
                        f"alert={int(bool(busy_raw.get('alert', False)))} "
                        f"fx={int(bool(busy_raw.get('fx', False)))}"
                    )
                    lines.append(f"Channels busy: {busy_text}")

                tts_raw = audio.get("tts_pending")
                if isinstance(tts_raw, dict):
                    lines.append(
                        "TTS queue:"
                        f" call={int(tts_raw.get('callsign', 0))}"
                        f" cmd={int(tts_raw.get('commands', 0))}"
                        f" story={int(tts_raw.get('story', 0))}"
                    )

                noise_level = audio.get("noise_level")
                distortion_level = audio.get("distortion_level")
                if isinstance(noise_level, int | float) and isinstance(
                    distortion_level, int | float
                ):
                    lines.append(
                        f"Mix levels: noise={int(round(float(noise_level) * 100))}%"
                        f" distortion={int(round(float(distortion_level) * 100))}%"
                    )

        panel_w = min(520, max(360, frame.w // 2))
        panel_x = frame.right - panel_w - 10
        panel_y = header.bottom + 8
        desired_h = 48 + (len(lines) * 16)
        panel_h = min(frame.bottom - panel_y - 10, max(180, desired_h))
        panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        self._draw_auditory_glass_panel(
            surface,
            panel,
            top_color=(14, 26, 62),
            bottom_color=(8, 16, 40),
            border_color=(114, 136, 198),
            border_radius=8,
            gloss_alpha=42,
        )
        title = self._small_font.render("Auditory Testing Menu", True, (236, 242, 255))
        surface.blit(title, (panel.x + 10, panel.y + 8))

        y = panel.y + 36
        for line in lines:
            txt = self._tiny_font.render(line, True, (214, 226, 252))
            surface.blit(txt, (panel.x + 10, y))
            y += 16
            if y > panel.bottom - 16:
                break

    def _render_math_reasoning(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: MathReasoningPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 12, 84)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        active_bg = (244, 248, 255)
        active_text = (16, 32, 88)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        surface.blit(
            self._tiny_font.render(phase_label, True, text_muted),
            (header.x + 12, header.y + (header.h - self._tiny_font.get_height()) // 2),
        )

        title = self._small_font.render("Mathematics Reasoning", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 145, header.centery)))

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 12, header.centery))
        surface.blit(stats, stats_rect)
        stats_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(
            stats_label, stats_label.get_rect(midright=(stats_rect.left - 6, header.centery))
        )

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, (6, 13, 92), content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        if payload is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            domain_tag = self._tiny_font.render(payload.domain.upper(), True, text_muted)
            surface.blit(domain_tag, (content.x + 12, content.y + 10))

            stem_rect = pygame.Rect(
                content.x + 12, content.y + 30, content.w - 24, max(72, content.h // 3)
            )
            self._draw_wrapped_text(
                surface,
                payload.stem,
                stem_rect,
                color=text_main,
                font=self._small_font,
                max_lines=4,
            )

            selected = self._math_choice
            if self._input.isdigit():
                selected = int(self._input)

            top = stem_rect.bottom + 10
            gap = 8
            rows = max(1, len(payload.options))
            row_h = max(42, min(58, (content.bottom - top - gap * (rows + 1)) // rows))
            y = top + gap

            for option in payload.options:
                row = pygame.Rect(content.x + 12, y, content.w - 24, row_h)
                is_selected = option.code == selected
                if is_selected:
                    pygame.draw.rect(surface, active_bg, row)
                    pygame.draw.rect(surface, (124, 148, 202), row, 2)
                else:
                    pygame.draw.rect(surface, (9, 20, 106), row)
                    pygame.draw.rect(surface, (62, 84, 152), row, 1)

                text_color = active_text if is_selected else text_main
                label = self._small_font.render(
                    f"{self._choice_key_label(option.code)}  {option.text}",
                    True,
                    text_color,
                )
                surface.blit(label, (row.x + 12, row.y + (row.h - label.get_height()) // 2))
                y += row_h + gap
        else:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-24, -24),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "A/S/D/F/G: Select  |  Up/Down: Move  |  Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(
            footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12))
        )

    def _render_table_reading_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: TableReadingPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 12, 84)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        active_bg = (244, 248, 255)
        active_text = (16, 32, 88)
        row_bg = (9, 20, 106)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        surface.blit(
            self._tiny_font.render(phase_label, True, text_muted),
            (header.x + 12, header.y + (header.h - self._tiny_font.get_height()) // 2),
        )

        title = self._small_font.render("Table Reading", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 130, header.centery)))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, (6, 13, 92), content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        is_question_phase = snap.phase in (Phase.PRACTICE, Phase.SCORED)
        if payload is None or not is_question_phase:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-24, -24),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
        else:
            left_w = max(280, int(content.w * 0.56))
            left = pygame.Rect(
                content.x + 10, content.y + 10, min(left_w, content.w - 220), content.h - 20
            )
            right = pygame.Rect(
                left.right + 10, content.y + 10, content.right - left.right - 20, content.h - 20
            )

            if payload.secondary_table is None:
                self._draw_table_reading_table(
                    surface,
                    left,
                    payload.primary_table,
                )
            else:
                upper_h = max(120, (left.h - 10) // 2)
                top = pygame.Rect(left.x, left.y, left.w, upper_h)
                bottom = pygame.Rect(left.x, top.bottom + 10, left.w, left.bottom - top.bottom - 10)

                self._draw_table_reading_table(
                    surface,
                    top,
                    payload.primary_table,
                )
                self._draw_table_reading_table(
                    surface,
                    bottom,
                    payload.secondary_table,
                )

            pygame.draw.rect(surface, panel_bg, right)
            pygame.draw.rect(surface, (62, 84, 152), right, 1)

            if payload.part.value == "part_one_cross_reference":
                part_label = "Part 1 - Cross-reference"
            else:
                part_label = "Part 2 - Multi-table"
            part = self._tiny_font.render(part_label, True, text_muted)
            surface.blit(part, (right.x + 10, right.y + 8))

            stem_rect = pygame.Rect(right.x + 10, right.y + 26, right.w - 20, max(74, right.h // 3))
            self._draw_wrapped_text(
                surface,
                payload.stem,
                stem_rect,
                color=text_main,
                font=self._small_font,
                max_lines=4,
            )

            selected = self._math_choice
            if self._input.isdigit():
                selected = int(self._input)

            options_top = stem_rect.bottom + 8
            row_count = max(1, len(payload.options))
            row_gap = 6
            row_h = max(
                32, min(44, (right.bottom - options_top - row_gap * (row_count + 1)) // row_count)
            )
            y = options_top + row_gap

            for option in payload.options:
                row = pygame.Rect(right.x + 10, y, right.w - 20, row_h)
                is_selected = option.code == selected
                pygame.draw.rect(surface, active_bg if is_selected else row_bg, row)
                pygame.draw.rect(surface, (62, 84, 152), row, 1)
                color = active_text if is_selected else text_main
                label = f"{self._choice_key_label(option.code)}  {option.value}"
                text = self._small_font.render(label, True, color)
                surface.blit(text, (row.x + 10, row.y + (row.h - text.get_height()) // 2))
                y += row_h + row_gap

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "A/S/D/F/G: Select  |  Up/Down: Move  |  Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(
            footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12))
        )

    def _draw_table_reading_table(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        table: TableReadingTable,
    ) -> None:
        panel_bg = (8, 18, 104)
        line = (78, 102, 170)
        header_bg = (18, 30, 118)
        header_text = (212, 223, 244)
        text_main = (238, 245, 255)

        pygame.draw.rect(surface, panel_bg, rect)
        pygame.draw.rect(surface, line, rect, 1)

        title = self._tiny_font.render(table.title, True, header_text)
        surface.blit(title, (rect.x + 8, rect.y + 6))

        grid = pygame.Rect(rect.x + 6, rect.y + 24, rect.w - 12, rect.h - 30)
        rows = len(table.row_labels) + 1
        cols = len(table.column_labels) + 1
        if rows <= 0 or cols <= 0:
            return

        cell_w = max(18, grid.w // cols)
        cell_h = max(16, grid.h // rows)
        draw_w = cell_w * cols
        draw_h = cell_h * rows
        start_x = grid.x + (grid.w - draw_w) // 2
        start_y = grid.y + (grid.h - draw_h) // 2

        corner_label = f"{table.row_header}/{table.column_header}"
        for row_idx in range(rows):
            for col_idx in range(cols):
                cell = pygame.Rect(
                    start_x + (col_idx * cell_w),
                    start_y + (row_idx * cell_h),
                    cell_w,
                    cell_h,
                )

                is_header = row_idx == 0 or col_idx == 0

                fill = header_bg if is_header else panel_bg
                text_color = header_text if is_header else text_main

                pygame.draw.rect(surface, fill, cell)
                pygame.draw.rect(surface, line, cell, 1)

                if row_idx == 0 and col_idx == 0:
                    value = corner_label
                    font = self._tiny_font
                elif row_idx == 0:
                    value = table.column_labels[col_idx - 1]
                    font = self._small_font
                elif col_idx == 0:
                    value = table.row_labels[row_idx - 1]
                    font = self._small_font
                else:
                    value = str(table.values[row_idx - 1][col_idx - 1])
                    font = self._tiny_font

                clipped = self._fit_label(font, value, cell.w - 4)
                text = font.render(clipped, True, text_color)
                surface.blit(text, text.get_rect(center=cell.center))

    def _system_logic_sync_payload(self, payload: SystemLogicPayload) -> None:
        payload_id = id(payload)
        if payload_id == self._system_logic_payload_id:
            return
        self._system_logic_payload_id = payload_id
        self._system_logic_folder_index = 0
        self._system_logic_doc_index = 0

    def _system_logic_shift_folder(self, payload: SystemLogicPayload, delta: int) -> None:
        if not payload.folders:
            return
        self._system_logic_folder_index = (self._system_logic_folder_index + delta) % len(
            payload.folders
        )
        self._system_logic_doc_index = 0

    def _system_logic_shift_document(self, payload: SystemLogicPayload, delta: int) -> None:
        if not payload.folders:
            return
        folder = payload.folders[self._system_logic_folder_index % len(payload.folders)]
        if not folder.documents:
            self._system_logic_doc_index = 0
            return
        self._system_logic_doc_index = (self._system_logic_doc_index + delta) % len(
            folder.documents
        )

    def _system_logic_active_folder(self, payload: SystemLogicPayload) -> SystemLogicFolder | None:
        if not payload.folders:
            return None
        self._system_logic_folder_index %= len(payload.folders)
        return payload.folders[self._system_logic_folder_index]

    def _system_logic_active_document(
        self, payload: SystemLogicPayload
    ) -> SystemLogicDocument | None:
        active_folder = self._system_logic_active_folder(payload)
        if active_folder is None:
            return None
        documents = active_folder.documents
        if not documents:
            return None
        self._system_logic_doc_index %= len(documents)
        return documents[self._system_logic_doc_index]

    def _cognitive_updating_sync_payload(self, payload: CognitiveUpdatingPayload) -> None:
        payload_id = id(payload)
        if payload_id == self._cognitive_updating_payload_id:
            self._cognitive_updating_refresh_runtime_state()
            return
        self._cognitive_updating_payload_id = payload_id
        self._cognitive_updating_upper_tab_index = 2
        self._cognitive_updating_lower_tab_index = 1
        clock = getattr(self._engine, "clock", None)
        if not hasattr(clock, "now"):
            clock = RealClock()
        self._cognitive_updating_runtime = CognitiveUpdatingRuntime(payload=payload, clock=clock)
        self._cognitive_updating_refresh_runtime_state()
        self._cognitive_updating_hitboxes.clear()

    def _cognitive_updating_shift_upper_tab(self, delta: int) -> None:
        self._cognitive_updating_upper_tab_index = (
            self._cognitive_updating_upper_tab_index + delta
        ) % 6

    def _cognitive_updating_shift_lower_tab(self, delta: int) -> None:
        self._cognitive_updating_lower_tab_index = (
            self._cognitive_updating_lower_tab_index + delta
        ) % 6

    def _cognitive_updating_refresh_runtime_state(self) -> CognitiveUpdatingRuntimeSnapshot | None:
        runtime = self._cognitive_updating_runtime
        if runtime is None:
            return None
        snap = runtime.snapshot()
        self._cognitive_updating_phase_start_ms = pygame.time.get_ticks() - (snap.elapsed_s * 1000)
        self._cognitive_updating_parcel_values = list(snap.parcel_values)
        self._cognitive_updating_active_parcel_field = snap.active_parcel_field
        self._cognitive_updating_comms_input = snap.comms_input
        self._cognitive_updating_speed_knots = snap.current_knots
        self._cognitive_updating_pump_on = snap.pump_on
        self._cognitive_updating_active_tank = snap.active_tank
        self._cognitive_updating_alpha_armed = snap.alpha_armed
        self._cognitive_updating_bravo_armed = snap.bravo_armed
        self._cognitive_updating_air_sensor_armed = snap.air_sensor_armed
        self._cognitive_updating_ground_sensor_armed = snap.ground_sensor_armed
        self._cognitive_updating_dispenser_lit = snap.dispenser_lit
        self._input = snap.comms_input
        return snap

    def _cognitive_updating_results_lines(self) -> tuple[str, ...]:
        events_fn = getattr(self._engine, "events", None)
        if not callable(events_fn):
            return ()
        try:
            events = list(events_fn())
        except Exception:
            return ()

        decoded = []
        for event in events:
            if getattr(event, "phase", None) is not Phase.SCORED:
                continue
            parsed = decode_cognitive_updating_submission_raw(getattr(event, "raw", ""))
            if parsed is not None:
                decoded.append(parsed)

        if not decoded:
            return ()

        count = len(decoded)
        controls = int(round(sum(item.controls_score for item in decoded) / count))
        navigation = int(round(sum(item.navigation_score for item in decoded) / count))
        engine = int(round(sum(item.engine_score for item in decoded) / count))
        sensors = int(round(sum(item.sensors_score for item in decoded) / count))
        objectives = int(round(sum(item.objectives_score for item in decoded) / count))
        warning_penalty = int(round(sum(item.warnings_penalty_points for item in decoded) / count))
        overall = int(round(sum(item.overall_score for item in decoded) / count))

        return (
            f"Controls: {controls}",
            f"Navigation: {navigation}",
            f"Engine: {engine}",
            f"Sensors: {sensors}",
            f"Objectives: {objectives}",
            f"Warning Penalty: -{warning_penalty}",
            f"Overall: {overall}",
        )

    def _render_system_logic_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: SystemLogicPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (3, 9, 78)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (186, 200, 224)
        panel_dark = (6, 13, 92)
        row_bg = (9, 20, 106)
        row_border = (62, 84, 152)
        active_bg = (244, 248, 255)
        active_text = (14, 26, 74)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        phase_text = self._tiny_font.render(phase_label, True, text_muted)
        surface.blit(
            phase_text, (header.x + 12, header.y + (header.h - phase_text.get_height()) // 2)
        )

        title = self._small_font.render(str(snap.title), True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 135, header.centery)))

        stats_text = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats_text, stats_text.get_rect(midright=(header.right - 12, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, panel_dark, content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        is_question_phase = snap.phase in (Phase.PRACTICE, Phase.SCORED)
        if payload is None or not is_question_phase:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-22, -22),
                color=text_main,
                font=self._small_font,
                max_lines=14,
            )
        else:
            self._system_logic_sync_payload(payload)
            folder_panel = pygame.Rect(
                content.x + 8,
                content.y + 8,
                max(160, min(240, content.w // 4)),
                content.h - 16,
            )
            pygame.draw.rect(surface, panel_bg, folder_panel)
            pygame.draw.rect(surface, row_border, folder_panel, 1)

            folder_label = self._tiny_font.render("Folders", True, text_muted)
            surface.blit(folder_label, (folder_panel.x + 10, folder_panel.y + 8))

            folder_row_top = folder_panel.y + 28
            folder_count = max(1, len(payload.folders))
            folder_gap = 6
            folder_row_h = max(
                30,
                min(38, (folder_panel.h - 44 - folder_gap * (folder_count - 1)) // folder_count),
            )

            y = folder_row_top
            for idx, folder in enumerate(payload.folders):
                row = pygame.Rect(folder_panel.x + 8, y, folder_panel.w - 16, folder_row_h)
                selected = idx == self._system_logic_folder_index
                pygame.draw.rect(surface, active_bg if selected else row_bg, row)
                pygame.draw.rect(surface, row_border, row, 1)
                color = active_text if selected else text_main
                label = self._fit_label(self._small_font, folder.name, row.w - 14)
                txt = self._small_font.render(label, True, color)
                surface.blit(txt, (row.x + 8, row.y + (row.h - txt.get_height()) // 2))
                y += folder_row_h + folder_gap

            right_x = folder_panel.right + 10
            right_w = max(120, content.right - right_x - 8)

            docs_h = max(88, min(144, content.h // 4))
            docs_rect = pygame.Rect(right_x, content.y + 8, right_w, docs_h)
            question_h = max(112, min(152, content.h // 3))
            question_rect = pygame.Rect(
                right_x, content.bottom - question_h - 8, right_w, question_h
            )
            doc_body_top = docs_rect.bottom + 8
            doc_body_h = max(84, question_rect.y - doc_body_top - 8)
            doc_body_rect = pygame.Rect(right_x, doc_body_top, right_w, doc_body_h)

            pygame.draw.rect(surface, panel_bg, docs_rect)
            pygame.draw.rect(surface, row_border, docs_rect, 1)
            pygame.draw.rect(surface, panel_bg, doc_body_rect)
            pygame.draw.rect(surface, row_border, doc_body_rect, 1)
            pygame.draw.rect(surface, panel_bg, question_rect)
            pygame.draw.rect(surface, row_border, question_rect, 1)

            active_folder = self._system_logic_active_folder(payload)
            if active_folder is not None and active_folder.documents:
                docs_label = self._tiny_font.render("Submenu", True, text_muted)
                surface.blit(docs_label, (docs_rect.x + 8, docs_rect.y + 6))

                doc_gap = 4
                doc_rows = max(1, len(active_folder.documents))
                doc_row_h = max(
                    24,
                    min(30, (docs_rect.h - 22 - doc_gap * (doc_rows - 1)) // doc_rows),
                )
                doc_y = docs_rect.y + 20
                for doc_idx, doc in enumerate(active_folder.documents):
                    row = pygame.Rect(docs_rect.x + 8, doc_y, docs_rect.w - 16, doc_row_h)
                    selected = doc_idx == self._system_logic_doc_index
                    pygame.draw.rect(surface, active_bg if selected else row_bg, row)
                    pygame.draw.rect(surface, row_border, row, 1)
                    color = active_text if selected else text_main
                    label = f"{doc_idx + 1}. {doc.title}"
                    clipped = self._fit_label(self._tiny_font, label, row.w - 10)
                    doc_text = self._tiny_font.render(clipped, True, color)
                    surface.blit(
                        doc_text, (row.x + 6, row.y + (row.h - doc_text.get_height()) // 2)
                    )
                    doc_y += doc_row_h + doc_gap

            active_doc = self._system_logic_active_document(payload)
            if active_doc is not None:
                title_label = self._small_font.render(active_doc.title, True, text_main)
                surface.blit(title_label, (doc_body_rect.x + 10, doc_body_rect.y + 8))
                kind_label = self._tiny_font.render(active_doc.kind.upper(), True, text_muted)
                surface.blit(
                    kind_label,
                    kind_label.get_rect(topright=(doc_body_rect.right - 10, doc_body_rect.y + 12)),
                )

                lines_rect = pygame.Rect(
                    doc_body_rect.x + 10,
                    doc_body_rect.y + 34,
                    doc_body_rect.w - 20,
                    doc_body_rect.h - 42,
                )
                line_y = lines_rect.y
                line_h = self._tiny_font.get_linesize() + 2
                for line in active_doc.lines:
                    if line_y + line_h > lines_rect.bottom:
                        break
                    to_draw = self._fit_label(self._tiny_font, line, lines_rect.w)
                    line_text = self._tiny_font.render(to_draw, True, text_main)
                    surface.blit(line_text, (lines_rect.x, line_y))
                    line_y += line_h
            else:
                self._draw_wrapped_text(
                    surface,
                    "No document available.",
                    doc_body_rect.inflate(-20, -20),
                    color=text_main,
                    font=self._small_font,
                    max_lines=3,
                )

            self._draw_wrapped_text(
                surface,
                payload.question,
                pygame.Rect(
                    question_rect.x + 10,
                    question_rect.y + 8,
                    question_rect.w - 20,
                    max(20, question_rect.h - 58),
                ),
                color=text_main,
                font=self._small_font,
                max_lines=3,
            )

            input_box = pygame.Rect(
                question_rect.x + 10, question_rect.bottom - 38, min(260, right_w - 20), 30
            )
            pygame.draw.rect(surface, (4, 11, 84), input_box)
            pygame.draw.rect(surface, row_border, input_box, 2)

            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            entry = self._small_font.render(self._input + caret, True, text_main)
            surface.blit(entry, (input_box.x + 8, input_box.y + 4))

            unit_text = self._tiny_font.render(f"Unit: {payload.answer_unit}", True, text_muted)
            surface.blit(
                unit_text,
                unit_text.get_rect(midleft=(input_box.right + 10, input_box.y + input_box.h // 2)),
            )

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "Left/Right or Tab: Folder  |  Up/Down: Submenu  |  Digits + Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(
            footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12))
        )

    def _render_cognitive_updating_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: CognitiveUpdatingPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (0, 0, 0)
        panel_bg = (10, 10, 14)
        card_bg = (90, 90, 92)
        header_blue = (49, 143, 220)
        accent_orange = (231, 160, 35)
        text_main = (238, 242, 248)
        text_muted = (188, 198, 214)
        border = (226, 233, 242)

        def _label(name: str, *, center: bool) -> str:
            if center and name == "Messages":
                return "Message"
            if center and name == "Objectives":
                return "Objective"
            return name

        def _draw_card(rect: pygame.Rect, title: str, *, title_h: int = 34) -> None:
            pygame.draw.rect(surface, panel_bg, rect)
            pygame.draw.rect(surface, border, rect, 2)
            top = pygame.Rect(rect.x, rect.y, rect.w, title_h)
            pygame.draw.rect(surface, header_blue, top)
            pygame.draw.rect(surface, border, top, 1)
            txt = self._small_font.render(title, True, text_main)
            surface.blit(txt, txt.get_rect(center=top.center))

        def _draw_tabbar(
            rect: pygame.Rect, names: tuple[str, str, str], active: int, prefix: str
        ) -> None:
            pygame.draw.rect(surface, header_blue, rect)
            pygame.draw.rect(surface, border, rect, 2)
            side_w = max(84, min(150, (rect.w - 12) // 3))
            left = pygame.Rect(rect.x + 2, rect.y + 2, side_w, rect.h - 4)
            right = pygame.Rect(rect.right - side_w - 2, rect.y + 2, side_w, rect.h - 4)
            center = pygame.Rect(left.right + 4, rect.y + 2, right.x - left.right - 8, rect.h - 4)
            pygame.draw.rect(surface, accent_orange, left, border_radius=4)
            pygame.draw.rect(surface, accent_orange, right, border_radius=4)
            left_name = names[(active - 1) % len(names)]
            center_name = names[active % len(names)]
            right_name = names[(active + 1) % len(names)]
            left_txt = self._small_font.render(_label(left_name, center=False), True, text_main)
            center_txt = self._small_font.render(_label(center_name, center=True), True, text_main)
            right_txt = self._small_font.render(_label(right_name, center=False), True, text_main)
            surface.blit(left_txt, left_txt.get_rect(center=left.center))
            surface.blit(center_txt, center_txt.get_rect(center=center.center))
            surface.blit(right_txt, right_txt.get_rect(center=right.center))
            self._cognitive_updating_hitboxes[f"{prefix}_prev"] = left
            self._cognitive_updating_hitboxes[f"{prefix}_next"] = right

        def _mmss(seconds: int) -> str:
            value = max(0, int(seconds))
            return f"{value // 60:02d}:{value % 60:02d}"

        surface.fill(bg)
        margin = max(10, min(22, min(w, h) // 30))
        frame = pygame.Rect(margin, margin, max(320, w - margin * 2), max(380, h - margin * 2))
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        self._cognitive_updating_hitboxes.clear()
        title_h = max(52, min(72, frame.h // 10))
        title_rect = pygame.Rect(frame.x + 8, frame.y + 8, frame.w - 16, title_h)
        arrow = self._big_font.render("←", True, text_main)
        surface.blit(
            arrow, (title_rect.x + 4, title_rect.y + (title_rect.h - arrow.get_height()) // 2)
        )
        title = self._app.font.render("Multitasking", True, text_main)
        surface.blit(title, title.get_rect(center=title_rect.center))

        content = pygame.Rect(
            frame.x + 6, title_rect.bottom + 6, frame.w - 12, frame.bottom - title_rect.bottom - 14
        )
        if payload is None or snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            pygame.draw.rect(surface, panel_bg, content)
            pygame.draw.rect(surface, border, content, 2)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-28, -28),
                color=text_main,
                font=self._app.font,
                max_lines=16,
            )
            if snap.phase is Phase.RESULTS:
                breakdown = self._cognitive_updating_results_lines()
                y = content.y + max(220, content.h // 2)
                for line in breakdown:
                    txt = self._small_font.render(line, True, text_main)
                    surface.blit(txt, (content.x + 20, y))
                    y += txt.get_height() + 6
            footer = (
                "Enter: Continue"
                if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE)
                else "Enter: Return to Tests"
            )
            footer_text = self._small_font.render(footer, True, text_muted)
            surface.blit(
                footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 8))
            )
            return

        self._cognitive_updating_sync_payload(payload)
        runtime_snap = self._cognitive_updating_refresh_runtime_state()
        if runtime_snap is None:
            runtime_snap = CognitiveUpdatingRuntimeSnapshot(
                elapsed_s=0,
                clock_hms=payload.clock_hms,
                current_knots=payload.current_knots,
                pump_on=payload.pump_on,
                active_tank=payload.active_tank,
                alpha_armed=False,
                bravo_armed=False,
                air_sensor_armed=False,
                ground_sensor_armed=False,
                parcel_values=("", "", ""),
                active_parcel_field=0,
                comms_input="",
                dispenser_lit=payload.dispenser_lit,
                air_time_left_s=payload.air_sensor_due_s,
                ground_time_left_s=payload.ground_sensor_due_s,
                comms_time_left_s=payload.comms_time_limit_s,
                objective_deadline_left_s=payload.objective_deadline_s,
                state_code=payload.comms_code,
                operation_score_hint=0.0,
                event_count=0,
                pressure_value=payload.pressure_value,
                tank_levels_l=payload.tank_levels_l,
                warning_lines=payload.warning_lines,
                message_lines=payload.message_lines,
                controls_score=100,
                navigation_score=100,
                engine_score=100,
                sensors_score=0,
                objectives_score=0,
                warnings_penalty_points=0,
                overall_score=0,
                objective_drop_ready=False,
                objective_drop_complete=False,
            )

        air_left = runtime_snap.air_time_left_s
        ground_left = runtime_snap.ground_time_left_s
        comms_left = runtime_snap.comms_time_left_s
        speed_knots = runtime_snap.current_knots
        pump_on = runtime_snap.pump_on
        active_tank = runtime_snap.active_tank
        dispenser_lit = runtime_snap.dispenser_lit

        warning_h = max(90, min(128, content.h // 7))
        warning = pygame.Rect(content.x + 2, content.y + 2, int(content.w * 0.72), warning_h)
        clock = pygame.Rect(
            warning.right + 8,
            warning.y,
            content.right - warning.right - 10,
            warning_h,
        )
        _draw_card(warning, "Warning")
        _draw_card(clock, "Clock")

        warning_rows = runtime_snap.warning_lines[:4]
        warn_col_w = max(90, (warning.w - 28) // 2)
        for idx, line in enumerate(warning_rows):
            row = idx // 2
            col = idx % 2
            warn_x = warning.x + 10 + (col * warn_col_w)
            warn_y = warning.y + 42 + (row * 30)
            line_txt = self._small_font.render(
                self._fit_label(self._small_font, line, warn_col_w - 10), True, text_main
            )
            surface.blit(line_txt, (warn_x, warn_y))

        clock_txt = self._big_font.render(runtime_snap.clock_hms, True, text_main)
        if clock_txt.get_width() > clock.w - 16:
            clock_txt = self._mid_font.render(runtime_snap.clock_hms, True, text_main)
        surface.blit(clock_txt, clock_txt.get_rect(center=(clock.centerx, clock.centery + 8)))
        if snap.time_remaining_s is not None:
            rem = int(max(0, round(snap.time_remaining_s)))
            rem_txt = self._tiny_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_muted)
            surface.blit(rem_txt, rem_txt.get_rect(midbottom=(clock.centerx, clock.bottom - 4)))

        tab_names = ("Messages", "Objectives", "Controls", "Navigation", "Sensors", "Engine")

        def _set_hitbox(scope: str, code: str, rect: pygame.Rect) -> None:
            self._cognitive_updating_hitboxes[f"{scope}|{code}"] = rect

        def _draw_page(panel_rect: pygame.Rect, page_idx: int, scope: str) -> None:
            if page_idx == 0:
                lines_rect = panel_rect.inflate(-22, -20)
                line_y = lines_rect.y
                for line in runtime_snap.message_lines:
                    rendered = self._app.font.render(
                        self._fit_label(self._app.font, line, lines_rect.w), True, text_main
                    )
                    surface.blit(rendered, (lines_rect.x, line_y))
                    line_y += max(34, self._app.font.get_linesize() + 8)
                    if line_y > lines_rect.bottom - 24:
                        break
                return

            if page_idx == 1:
                pad = 10
                parcel_h = max(138, int(panel_rect.h * 0.53))
                parcel_rect = pygame.Rect(
                    panel_rect.x + pad, panel_rect.y + pad, panel_rect.w - pad * 2, parcel_h
                )
                _draw_card(parcel_rect, "Parcel Drop")

                labels = ("Latitude", "Longitude", "Time")
                row_h = max(28, min(46, (parcel_rect.h - 54) // 3))
                for idx, label in enumerate(labels):
                    row_y = parcel_rect.y + 40 + idx * row_h
                    label_txt = self._app.font.render(label, True, text_main)
                    surface.blit(label_txt, (parcel_rect.x + 16, row_y + 2))
                    box_w = max(110, min(260, parcel_rect.w // 2))
                    box = pygame.Rect(parcel_rect.right - box_w - 24, row_y - 2, box_w, row_h - 4)
                    is_active = idx == self._cognitive_updating_active_parcel_field
                    pygame.draw.rect(
                        surface,
                        (239, 163, 33) if is_active else (226, 150, 30),
                        box,
                        border_radius=4,
                    )
                    value = self._cognitive_updating_parcel_values[idx]
                    if value != "":
                        value_txt = self._small_font.render(value, True, (22, 22, 24))
                        surface.blit(
                            value_txt, (box.x + 8, box.y + (box.h - value_txt.get_height()) // 2)
                        )
                    _set_hitbox(scope, f"parcel_field:{idx}", box)

                disp_y = parcel_rect.bottom + 8
                disp_h = panel_rect.bottom - disp_y - 50
                dispenser_rect = pygame.Rect(
                    panel_rect.x + pad, disp_y, panel_rect.w - pad * 2, disp_h
                )
                _draw_card(dispenser_rect, "Parcel Drop Dispenser")
                row_mid_y = dispenser_rect.y + dispenser_rect.h // 2 + 4
                act_txt = self._app.font.render("Drop", True, text_main)
                surface.blit(
                    act_txt, (dispenser_rect.x + 16, row_mid_y - act_txt.get_height() // 2)
                )

                circle_r = max(12, min(30, dispenser_rect.w // 28))
                activate_btn = pygame.Rect(dispenser_rect.right - 84, dispenser_rect.y + 44, 64, 64)
                cx = max(dispenser_rect.x + 130, dispenser_rect.x + (dispenser_rect.w // 3))
                max_cx = activate_btn.x - (circle_r * 2 + 12) * 5
                cx = min(cx, max_cx)
                for idx in range(5):
                    col = (0, 220, 0) if idx < dispenser_lit else (220, 220, 220)
                    pygame.draw.circle(
                        surface, col, (cx + idx * (circle_r * 2 + 14), row_mid_y), circle_r
                    )
                    pygame.draw.circle(
                        surface,
                        (180, 180, 180),
                        (cx + idx * (circle_r * 2 + 14), row_mid_y),
                        circle_r,
                        1,
                    )

                activate_color = (0, 176, 0) if runtime_snap.objective_drop_ready else header_blue
                if runtime_snap.objective_drop_complete:
                    activate_color = (90, 90, 90)
                pygame.draw.rect(surface, activate_color, activate_btn, border_radius=4)
                _set_hitbox(scope, "objective_activate", activate_btn)

                deadline_text = self._tiny_font.render(
                    f"Drop deadline: {runtime_snap.objective_deadline_left_s}s",
                    True,
                    text_muted,
                )
                surface.blit(deadline_text, (dispenser_rect.x + 12, dispenser_rect.y + 10))

                keypad_y = panel_rect.bottom - 42
                key_w = (panel_rect.w - 4) // 10
                for digit in range(10):
                    key = pygame.Rect(panel_rect.x + 2 + digit * key_w, keypad_y, key_w - 2, 40)
                    pygame.draw.rect(surface, (70, 70, 72), key)
                    pygame.draw.rect(surface, (40, 40, 42), key, 1)
                    txt = self._app.font.render(str(digit), True, text_main)
                    surface.blit(txt, txt.get_rect(center=key.center))
                    _set_hitbox(scope, f"objective_digit:{digit}", key)
                return

            if page_idx == 2:
                pad = 8
                min_right_w = 130
                max_left_w = max(140, panel_rect.w - (pad * 3) - min_right_w)
                left_w = max(140, min(int(panel_rect.w * 0.56), max_left_w))
                hyd_rect = pygame.Rect(
                    panel_rect.x + pad, panel_rect.y + pad, left_w, panel_rect.h - pad * 2
                )
                right_x = hyd_rect.right + 8
                right_w = panel_rect.right - right_x - pad
                pump_rect = pygame.Rect(
                    right_x, panel_rect.y + pad, right_w, max(112, int(panel_rect.h * 0.44))
                )
                comms_rect = pygame.Rect(
                    right_x,
                    pump_rect.bottom + 8,
                    right_w,
                    panel_rect.bottom - pump_rect.bottom - pad,
                )

                _draw_card(hyd_rect, "Hydraulics")
                _draw_card(pump_rect, "Hydraulic Pump")
                _draw_card(comms_rect, "Comms Code")

                scale_margin = max(14, min(52, hyd_rect.w // 6))
                scale = pygame.Rect(
                    hyd_rect.x + scale_margin,
                    hyd_rect.y + 42,
                    max(80, hyd_rect.w - (scale_margin * 2)),
                    max(80, hyd_rect.h - 78),
                )
                pygame.draw.rect(surface, card_bg, scale)
                section_h = scale.h // 3
                label_w = max(76, int(scale.w * 0.58))
                tick_w = max(18, min(30, int(scale.w * 0.16)))
                axis_w = max(22, scale.w - label_w - tick_w)
                label_col = pygame.Rect(scale.x, scale.y, label_w, scale.h)
                tick_col = pygame.Rect(label_col.right, scale.y, tick_w, scale.h)
                axis_col = pygame.Rect(tick_col.right, scale.y, axis_w, scale.h)

                high_zone = pygame.Rect(label_col.x, label_col.y, label_col.w, section_h)
                mid_zone = pygame.Rect(label_col.x, label_col.y + section_h, label_col.w, section_h)
                low_zone = pygame.Rect(
                    label_col.x,
                    label_col.y + section_h * 2,
                    label_col.w,
                    label_col.h - section_h * 2,
                )
                pygame.draw.rect(
                    surface,
                    (242, 68, 33)
                    if runtime_snap.pressure_value > payload.pressure_high
                    else (226, 226, 226),
                    high_zone,
                )
                pygame.draw.rect(
                    surface,
                    (0, 196, 0)
                    if payload.pressure_low <= runtime_snap.pressure_value <= payload.pressure_high
                    else (226, 226, 226),
                    mid_zone,
                )
                pygame.draw.rect(
                    surface,
                    (242, 68, 33)
                    if runtime_snap.pressure_value < payload.pressure_low
                    else (226, 226, 226),
                    low_zone,
                )
                pygame.draw.rect(surface, (232, 232, 232), tick_col)
                pygame.draw.rect(surface, (80, 80, 82), scale, 1)
                pygame.draw.line(
                    surface,
                    (126, 126, 126),
                    (label_col.right, label_col.y),
                    (label_col.right, label_col.bottom),
                    1,
                )
                pygame.draw.line(
                    surface,
                    (126, 126, 126),
                    (tick_col.right, tick_col.y),
                    (tick_col.right, tick_col.bottom),
                    1,
                )
                pygame.draw.line(
                    surface,
                    (126, 126, 126),
                    (label_col.x, label_col.y + section_h),
                    (tick_col.right, label_col.y + section_h),
                    1,
                )
                pygame.draw.line(
                    surface,
                    (126, 126, 126),
                    (label_col.x, label_col.y + section_h * 2),
                    (tick_col.right, label_col.y + section_h * 2),
                    1,
                )

                scale_min = float(payload.pressure_low - 20)
                scale_max = float(payload.pressure_high + 20)
                if scale_max <= scale_min:
                    scale_max = scale_min + 1.0
                pressure_ratio = (float(runtime_snap.pressure_value) - scale_min) / (
                    scale_max - scale_min
                )
                pressure_ratio = max(0.0, min(1.0, pressure_ratio))
                pressure_fill_y = tick_col.bottom - int(round(pressure_ratio * tick_col.h))
                pressure_fill_y = max(tick_col.y, min(tick_col.bottom, pressure_fill_y))

                if pressure_fill_y < tick_col.bottom:
                    fill_rect = pygame.Rect(
                        tick_col.x, pressure_fill_y, tick_col.w, tick_col.bottom - pressure_fill_y
                    )
                    pygame.draw.rect(surface, (49, 143, 220), fill_rect)

                total_ticks = 30
                for tick in range(total_ticks + 1):
                    y = tick_col.bottom - int(round((tick / total_ticks) * tick_col.h))
                    is_filled = y >= pressure_fill_y
                    tick_color = (29, 102, 170) if is_filled else (148, 148, 148)
                    pygame.draw.line(
                        surface, tick_color, (tick_col.x + 1, y), (tick_col.right - 1, y), 1
                    )

                high_txt_1 = self._app.font.render("High", True, (10, 10, 10))
                high_txt_2 = self._app.font.render("Pressure", True, (10, 10, 10))
                mid_txt_1 = self._app.font.render("Correct", True, (10, 10, 10))
                mid_txt_2 = self._app.font.render("Pressure", True, (10, 10, 10))
                low_txt_1 = self._app.font.render("Low", True, (10, 10, 10))
                low_txt_2 = self._app.font.render("Pressure", True, (10, 10, 10))
                surface.blit(high_txt_1, (high_zone.x + 8, high_zone.y + 8))
                surface.blit(
                    high_txt_2, (high_zone.x + 8, high_zone.y + 8 + high_txt_1.get_height() + 2)
                )
                surface.blit(mid_txt_1, (mid_zone.x + 8, mid_zone.y + 8))
                surface.blit(
                    mid_txt_2, (mid_zone.x + 8, mid_zone.y + 8 + mid_txt_1.get_height() + 2)
                )
                surface.blit(low_txt_1, (low_zone.x + 8, low_zone.y + 8))
                surface.blit(
                    low_txt_2, (low_zone.x + 8, low_zone.y + 8 + low_txt_1.get_height() + 2)
                )

                axis_vals = (
                    int(payload.pressure_high + 20),
                    int(payload.pressure_high),
                    int(payload.pressure_low),
                    int(payload.pressure_low - 20),
                )
                axis_positions = (
                    tick_col.y + 2,
                    tick_col.y + section_h - 8,
                    tick_col.y + section_h * 2 - 8,
                    tick_col.bottom - 16,
                )
                for value, y in zip(axis_vals, axis_positions, strict=True):
                    axis_txt = self._small_font.render(str(value), True, text_main)
                    surface.blit(axis_txt, (axis_col.x + 6, y))

                btn_w = max(42, pump_rect.w // 2 - 28)
                on_btn = pygame.Rect(pump_rect.x + 14, pump_rect.y + 58, btn_w, 56)
                off_btn = pygame.Rect(on_btn.right + 8, pump_rect.y + 58, btn_w, 56)
                pygame.draw.rect(
                    surface, (0, 196, 0) if pump_on else (80, 80, 80), on_btn, border_radius=4
                )
                pygame.draw.rect(
                    surface,
                    (136, 70, 56) if not pump_on else (80, 80, 80),
                    off_btn,
                    border_radius=4,
                )
                on_txt = self._app.font.render("On", True, text_main)
                off_txt = self._app.font.render("Off", True, text_main)
                surface.blit(on_txt, on_txt.get_rect(center=on_btn.center))
                surface.blit(off_txt, off_txt.get_rect(center=off_btn.center))
                _set_hitbox(scope, "pump_on", on_btn)
                _set_hitbox(scope, "pump_off", off_btn)

                submit_w = max(52, min(66, comms_rect.w // 3))
                box_w = max(70, comms_rect.w - submit_w - 36)
                box = pygame.Rect(comms_rect.x + 12, comms_rect.y + 52, box_w, 52)
                pygame.draw.rect(surface, (230, 157, 38), box, border_radius=4)
                comms_txt = self._app.font.render(
                    self._cognitive_updating_comms_input, True, (20, 20, 22)
                )
                surface.blit(comms_txt, (box.x + 8, box.y + (box.h - comms_txt.get_height()) // 2))
                submit = pygame.Rect(box.right + 8, box.y, submit_w, 52)
                submit_color = (
                    (0, 176, 0) if len(self._cognitive_updating_comms_input) == 4 else header_blue
                )
                pygame.draw.rect(surface, submit_color, submit, border_radius=4)
                _set_hitbox(scope, "comms_submit", submit)
                time_txt = self._small_font.render(f"Time remaining: {comms_left}", True, text_main)
                surface.blit(time_txt, (box.x, box.bottom + 8))

                key_y = comms_rect.bottom - 44
                key_w = max(26, min(72, (comms_rect.w - 28) // 4))
                for idx, digit in enumerate(("1", "2", "3", "4")):
                    key = pygame.Rect(comms_rect.x + 8 + idx * (key_w + 2), key_y, key_w, 38)
                    pygame.draw.rect(surface, (70, 70, 72), key)
                    pygame.draw.rect(surface, (40, 40, 42), key, 1)
                    key_txt = self._app.font.render(digit, True, text_main)
                    surface.blit(key_txt, key_txt.get_rect(center=key.center))
                    _set_hitbox(scope, f"comms_digit:{digit}", key)
                return

            if page_idx == 3:
                table_w = max(280, min(520, int(panel_rect.w * 0.58)))
                table_h = max(170, min(250, int(panel_rect.h * 0.56)))
                table = pygame.Rect(
                    panel_rect.centerx - table_w // 2, panel_rect.y + 52, table_w, table_h
                )
                pygame.draw.rect(surface, bg, table)
                pygame.draw.rect(surface, border, table, 2)
                pygame.draw.line(
                    surface, border, (table.centerx, table.y), (table.centerx, table.bottom), 2
                )
                header_split = table.y + 64
                pygame.draw.line(
                    surface, border, (table.x, header_split), (table.right, header_split), 2
                )
                left_header = self._small_font.render("Required Knots", True, text_main)
                right_header = self._small_font.render("Current Knots", True, text_main)
                surface.blit(
                    left_header, left_header.get_rect(center=(table.x + table.w // 4, table.y + 30))
                )
                surface.blit(
                    right_header,
                    right_header.get_rect(center=(table.x + table.w * 3 // 4, table.y + 30)),
                )
                left_value = self._app.font.render(str(payload.required_knots), True, text_main)
                right_value = self._app.font.render(str(speed_knots), True, text_main)
                surface.blit(
                    left_value,
                    left_value.get_rect(
                        center=(table.x + table.w // 4, table.y + table.h * 3 // 4)
                    ),
                )
                surface.blit(
                    right_value,
                    right_value.get_rect(
                        center=(table.x + table.w * 3 // 4, table.y + table.h * 3 // 4)
                    ),
                )

                dec = pygame.Rect(panel_rect.centerx - 90, panel_rect.bottom - 70, 58, 58)
                inc = pygame.Rect(panel_rect.centerx + 32, panel_rect.bottom - 70, 58, 58)
                pygame.draw.circle(surface, text_main, dec.center, 29, 2)
                pygame.draw.circle(surface, text_main, inc.center, 29, 2)
                dec_txt = self._app.font.render("−", True, text_main)
                inc_txt = self._app.font.render("+", True, text_main)
                surface.blit(dec_txt, dec_txt.get_rect(center=dec.center))
                surface.blit(inc_txt, inc_txt.get_rect(center=inc.center))
                _set_hitbox(scope, "knots_dec", dec)
                _set_hitbox(scope, "knots_inc", inc)
                return

            if page_idx == 4:
                pad = 10
                video_h = max(120, int(panel_rect.h * 0.46))
                video = pygame.Rect(
                    panel_rect.x + pad, panel_rect.y + pad, panel_rect.w - pad * 2, video_h
                )
                sensors = pygame.Rect(
                    video.x, video.bottom + 10, video.w, panel_rect.bottom - video.bottom - 12
                )
                _draw_card(video, "Video Recording")
                _draw_card(sensors, "Sensors")

                row1_y = video.y + 50
                row2_y = row1_y + 64
                alpha_txt = self._app.font.render("Alpha Camera Activation", True, text_main)
                bravo_txt = self._app.font.render("Bravo Camera Activation", True, text_main)
                surface.blit(alpha_txt, (video.x + 18, row1_y))
                surface.blit(bravo_txt, (video.x + 18, row2_y))
                cam_btn_w = max(96, min(140, video.w // 3))
                alpha_btn = pygame.Rect(video.right - cam_btn_w - 16, row1_y - 6, cam_btn_w, 48)
                bravo_btn = pygame.Rect(video.right - cam_btn_w - 16, row2_y - 6, cam_btn_w, 48)
                pygame.draw.rect(
                    surface,
                    (0, 176, 0) if self._cognitive_updating_alpha_armed else header_blue,
                    alpha_btn,
                    border_radius=4,
                )
                pygame.draw.rect(
                    surface,
                    (0, 176, 0) if self._cognitive_updating_bravo_armed else header_blue,
                    bravo_btn,
                    border_radius=4,
                )
                alpha_btn_txt = self._small_font.render("Activate", True, text_main)
                bravo_btn_txt = self._small_font.render("Activate", True, text_main)
                surface.blit(alpha_btn_txt, alpha_btn_txt.get_rect(center=alpha_btn.center))
                surface.blit(bravo_btn_txt, bravo_btn_txt.get_rect(center=bravo_btn.center))
                _set_hitbox(scope, "camera_alpha", alpha_btn)
                _set_hitbox(scope, "camera_bravo", bravo_btn)

                air_row = sensors.y + 52
                ground_row = air_row + 70
                air_txt = self._app.font.render("Air Sensor", True, text_main)
                ground_txt = self._app.font.render("Ground Sensor", True, text_main)
                surface.blit(air_txt, (sensors.x + 18, air_row))
                surface.blit(ground_txt, (sensors.x + 18, ground_row))

                air_time_label = f"Time Left: {_mmss(air_left)}"
                ground_time_label = f"Time Left: {_mmss(ground_left)}"
                air_time_txt = self._app.font.render(air_time_label, True, text_muted)
                ground_time_txt = self._app.font.render(ground_time_label, True, text_muted)
                if air_time_txt.get_width() > sensors.w // 3:
                    air_time_txt = self._small_font.render(air_time_label, True, text_muted)
                if ground_time_txt.get_width() > sensors.w // 3:
                    ground_time_txt = self._small_font.render(ground_time_label, True, text_muted)

                time_right = sensors.right - 14
                air_time_x = time_right - air_time_txt.get_width()
                ground_time_x = time_right - ground_time_txt.get_width()

                base_btn_w = max(88, min(126, sensors.w // 5))
                min_btn_x = sensors.x + 18 + max(air_txt.get_width(), ground_txt.get_width()) + 24
                air_btn_x = max(min_btn_x, air_time_x - 12 - base_btn_w)
                ground_btn_x = max(min_btn_x, ground_time_x - 12 - base_btn_w)
                sensor_btn_w = max(
                    72,
                    min(
                        base_btn_w,
                        min(air_time_x - air_btn_x - 12, ground_time_x - ground_btn_x - 12),
                    ),
                )
                air_btn = pygame.Rect(air_btn_x, air_row - 8, sensor_btn_w, 52)
                ground_btn = pygame.Rect(ground_btn_x, ground_row - 8, sensor_btn_w, 52)

                pygame.draw.rect(
                    surface,
                    (0, 176, 0) if self._cognitive_updating_air_sensor_armed else header_blue,
                    air_btn,
                    border_radius=4,
                )
                pygame.draw.rect(
                    surface,
                    (0, 176, 0) if self._cognitive_updating_ground_sensor_armed else header_blue,
                    ground_btn,
                    border_radius=4,
                )
                air_btn_txt = self._small_font.render("Activate", True, text_main)
                ground_btn_txt = self._small_font.render("Activate", True, text_main)
                surface.blit(air_btn_txt, air_btn_txt.get_rect(center=air_btn.center))
                surface.blit(ground_btn_txt, ground_btn_txt.get_rect(center=ground_btn.center))
                surface.blit(air_time_txt, (air_time_x, air_row + 2))
                surface.blit(ground_time_txt, (ground_time_x, ground_row + 2))
                _set_hitbox(scope, "sensor_air", air_btn)
                _set_hitbox(scope, "sensor_ground", ground_btn)
                return

            col_w = max(92, (panel_rect.w - 24) // 3)
            col_y = panel_rect.y + 32
            for idx in range(3):
                col_x = panel_rect.x + 8 + idx * col_w
                top = col_y + 24
                tank_top_x = col_x + 36
                pygame.draw.lines(
                    surface,
                    text_main,
                    False,
                    [
                        (tank_top_x, top),
                        (tank_top_x, top + 30),
                        (tank_top_x + col_w - 70, top + 30),
                        (tank_top_x + col_w - 70, top + 96),
                    ],
                    5,
                )
                tank_label = self._app.font.render(f"Tank {idx + 1} (L)", True, text_main)
                surface.blit(tank_label, (col_x + 44, top + 118))
                val = runtime_snap.tank_levels_l[idx]
                val_txt = self._app.font.render(str(val), True, text_main)
                surface.blit(val_txt, val_txt.get_rect(center=(col_x + col_w // 2, top + 210)))
                btn = pygame.Rect(col_x + 22, top + 260, col_w - 40, 58)
                is_active = active_tank == idx + 1
                pygame.draw.rect(
                    surface, (0, 96, 24) if is_active else (233, 70, 33), btn, border_radius=4
                )
                btn_txt = self._app.font.render("On" if is_active else "Off", True, text_main)
                surface.blit(btn_txt, btn_txt.get_rect(center=btn.center))
                _set_hitbox(scope, f"tank:{idx + 1}", btn)

        def _draw_mfd(mfd_rect: pygame.Rect, active_idx: int, scope: str) -> None:
            pygame.draw.rect(surface, card_bg, mfd_rect)
            pygame.draw.rect(surface, border, mfd_rect, 2)

            title_rect = pygame.Rect(mfd_rect.x + 2, mfd_rect.y + 2, mfd_rect.w - 4, 22)
            pygame.draw.rect(surface, header_blue, title_rect)
            pygame.draw.rect(surface, border, title_rect, 1)
            title_txt = self._small_font.render("Multifunction Display", True, text_main)
            surface.blit(title_txt, title_txt.get_rect(center=title_rect.center))

            tabs_rect = pygame.Rect(mfd_rect.x + 6, title_rect.bottom + 4, mfd_rect.w - 12, 66)
            pygame.draw.rect(surface, panel_bg, tabs_rect)
            pygame.draw.rect(surface, border, tabs_rect, 1)
            row_gap = 6
            col_gap = 6
            row_h = (tabs_rect.h - row_gap) // 2
            btn_w = (tabs_rect.w - (col_gap * 2)) // 3
            for idx, name in enumerate(tab_names):
                row = idx // 3
                col = idx % 3
                btn = pygame.Rect(
                    tabs_rect.x + col * (btn_w + col_gap),
                    tabs_rect.y + row * (row_h + row_gap),
                    btn_w,
                    row_h,
                )
                active = idx == active_idx
                pygame.draw.rect(
                    surface, accent_orange if active else header_blue, btn, border_radius=6
                )
                pygame.draw.rect(surface, border, btn, 1, border_radius=6)
                text = self._small_font.render(_label(name, center=False), True, text_main)
                surface.blit(text, text.get_rect(center=btn.center))
                _set_hitbox(scope, f"tab:{idx}", btn)

            page_rect = pygame.Rect(
                mfd_rect.x + 4,
                tabs_rect.bottom + 6,
                mfd_rect.w - 8,
                mfd_rect.bottom - tabs_rect.bottom - 10,
            )
            pygame.draw.rect(surface, panel_bg, page_rect)
            pygame.draw.rect(surface, border, page_rect, 2)
            _draw_page(page_rect, active_idx % 6, scope)

        panel_gap = 10
        panel_top = warning.bottom + 8
        panel_h = content.bottom - panel_top - 8
        panel_w = max(260, (content.w - panel_gap - 4) // 2)
        upper_rect = pygame.Rect(content.x + 2, panel_top, panel_w, panel_h)
        lower_rect = pygame.Rect(
            upper_rect.right + panel_gap,
            panel_top,
            content.right - upper_rect.right - panel_gap - 2,
            panel_h,
        )
        _draw_mfd(upper_rect, self._cognitive_updating_upper_tab_index % 6, "upper")
        _draw_mfd(lower_rect, self._cognitive_updating_lower_tab_index % 6, "lower")

        footer = (
            f"C:{runtime_snap.controls_score}  N:{runtime_snap.navigation_score}  "
            f"E:{runtime_snap.engine_score}  S:{runtime_snap.sensors_score}  "
            f"O:{runtime_snap.objectives_score}  W:-{runtime_snap.warnings_penalty_points}  "
            f"Total:{runtime_snap.overall_score}"
        )
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 8)))

    def _render_numerical_operations_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        w, h = surface.get_size()
        bg = (0, 0, 116)
        frame_color = (236, 243, 255)
        text_main = (244, 248, 255)
        text_muted = (214, 225, 244)
        accent = (164, 190, 235)

        surface.fill(bg)
        margin = max(8, min(18, w // 50))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_color, frame, 1)

        phase_label = "Practice" if snap.phase is Phase.PRACTICE else "Timed Test"
        left = self._num_header_font.render(phase_label, True, text_muted)
        surface.blit(left, (frame.x + 12, frame.y + 10))

        title = self._num_header_font.render("Numerical Operations Test", True, text_main)
        surface.blit(title, title.get_rect(midtop=(frame.centerx, frame.y + 10)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._num_header_font.render(f"{mm:02d}:{ss:02d}", True, text_muted)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, frame.y + 10)))

        stats = self._small_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            accent,
        )
        stats_rect = stats.get_rect(midtop=(frame.centerx, frame.y + 40))
        surface.blit(stats, stats_rect)

        prompt = str(snap.prompt).strip().split("\n", 1)[0]
        if prompt == "":
            prompt = "0 + 0 ="

        prompt_panel = pygame.Rect(
            frame.x + max(28, frame.w // 10),
            stats_rect.bottom + max(20, frame.h // 18),
            frame.w - max(56, frame.w // 5),
            max(120, min(190, frame.h // 4)),
        )
        pygame.draw.rect(surface, (10, 18, 92), prompt_panel, border_radius=14)
        pygame.draw.rect(surface, (118, 150, 214), prompt_panel, 2, border_radius=14)
        prompt_body = prompt_panel.inflate(-20, -18)

        prompt_surface = None
        for f in self._num_prompt_fonts:
            candidate = f.render(prompt, True, text_main)
            if candidate.get_width() <= prompt_body.w:
                prompt_surface = candidate
                break
        if prompt_surface is None:
            prompt_surface = self._num_prompt_fonts[-1].render(prompt, True, text_main)

        surface.blit(prompt_surface, prompt_surface.get_rect(center=prompt_body.center))

    def _render_numerical_operations_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
    ) -> None:
        w, h = surface.get_size()
        text_main = (244, 248, 255)
        box_fill = (246, 250, 255)
        box_border = (142, 168, 210)
        input_color = (12, 26, 88)

        box = pygame.Rect(
            (w - max(300, min(500, int(w * 0.50)))) // 2,
            int(h * 0.62),
            max(300, min(500, int(w * 0.50))),
            max(60, min(78, int(h * 0.13))),
        )
        self._render_centered_input_box(
            surface,
            box,
            label="Your Answer",
            hint=snap.input_hint,
            entry_text=self._input,
            fill=box_fill,
            border=box_border,
            label_color=text_main,
            input_color=input_color,
            hint_color=(194, 210, 236),
            label_font=self._small_font,
            input_font=self._num_input_font,
            hint_font=self._small_font,
        )

    def _render_angles_bearings_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: AnglesBearingsDegreesPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 12, 84)
        panel_bg = (8, 18, 104)
        header_bg = (18, 30, 118)
        border = (226, 236, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        active_bg = (244, 248, 255)
        active_text = (16, 32, 88)

        surface.fill(bg)

        margin = max(10, min(24, w // 34))
        frame = pygame.Rect(margin, margin, max(280, w - margin * 2), max(220, h - margin * 2))
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 2)

        header_h = max(40, min(56, h // 7))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        surface.blit(
            self._tiny_font.render(phase_label, True, text_muted),
            (header.x + 12, header.y + (header.h - self._tiny_font.get_height()) // 2),
        )

        title = self._small_font.render("Angles, Bearings and Degrees", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 145, header.centery)))

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 12, header.centery))
        surface.blit(stats, stats_rect)
        stats_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(
            stats_label, stats_label.get_rect(midright=(stats_rect.left - 6, header.centery))
        )

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._small_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        content = pygame.Rect(
            frame.x + max(14, w // 48),
            header.bottom + max(12, h // 36),
            frame.w - max(28, w // 24),
            frame.bottom - header.bottom - max(62, h // 9),
        )
        pygame.draw.rect(surface, (6, 13, 92), content)
        pygame.draw.rect(surface, (78, 102, 170), content, 1)

        if payload is not None and snap.phase in (Phase.PRACTICE, Phase.SCORED):
            stem_rect = pygame.Rect(
                content.x + 12, content.y + 10, content.w - 24, max(44, content.h // 6)
            )
            self._draw_wrapped_text(
                surface,
                payload.stem,
                stem_rect,
                color=text_main,
                font=self._small_font,
                max_lines=2,
            )

            work = pygame.Rect(
                content.x + 12,
                stem_rect.bottom + 8,
                content.w - 24,
                content.bottom - stem_rect.bottom - 16,
            )
            left_w = max(240, min(int(work.w * 0.60), work.w - 180))
            diagram_rect = pygame.Rect(work.x, work.y, left_w, work.h)
            options_rect = pygame.Rect(
                diagram_rect.right + 10, work.y, work.w - left_w - 10, work.h
            )

            self._render_angles_bearings_question(surface, diagram_rect, payload)

            selected = self._math_choice
            if self._input.isdigit():
                selected = int(self._input)

            pygame.draw.rect(surface, (9, 20, 106), options_rect)
            pygame.draw.rect(surface, (62, 84, 152), options_rect, 1)

            rows = max(1, len(payload.options))
            gap = 8
            row_h = max(36, min(58, (options_rect.h - gap * (rows + 1)) // rows))
            y = options_rect.y + gap
            for option in payload.options:
                row = pygame.Rect(options_rect.x + 8, y, options_rect.w - 16, row_h)
                is_selected = option.code == selected
                if is_selected:
                    pygame.draw.rect(surface, active_bg, row)
                    pygame.draw.rect(surface, (124, 148, 202), row, 2)
                else:
                    pygame.draw.rect(surface, (8, 18, 96), row)
                    pygame.draw.rect(surface, (62, 84, 152), row, 1)

                text_color = active_text if is_selected else text_main
                label = self._small_font.render(
                    f"{self._choice_key_label(option.code)}  {option.text}",
                    True,
                    text_color,
                )
                surface.blit(label, (row.x + 10, row.y + (row.h - label.get_height()) // 2))
                y += row_h + gap
        else:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                content.inflate(-24, -24),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            footer = "A/S/D/F/G: Select  |  Up/Down: Move  |  Enter: Submit"
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            footer = "Enter: Continue  |  Esc/Backspace: Back"
        else:
            footer = "Enter: Return to Tests"
        footer_text = self._tiny_font.render(footer, True, text_muted)
        surface.blit(
            footer_text, footer_text.get_rect(midbottom=(frame.centerx, frame.bottom - 12))
        )

    def _render_angles_bearings_question(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        pygame.draw.rect(surface, (16, 18, 64), panel)
        pygame.draw.rect(surface, (112, 134, 190), panel, 1)

        if payload.kind is AnglesBearingsQuestionKind.ANGLE_BETWEEN_LINES:
            self._draw_angle_trial(surface, panel, payload)
        else:
            self._draw_bearing_trial(surface, panel, payload)

    def _draw_angle_trial(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        cx = panel.centerx
        cy = panel.centery
        radius = max(44, (min(panel.w, panel.h) // 2) - 20)

        pygame.draw.circle(surface, (35, 35, 48), (cx, cy), radius)
        pygame.draw.circle(surface, (90, 90, 110), (cx, cy), radius, 2)

        p1 = self._bearing_point(cx, cy, radius, payload.reference_bearing_deg)
        p2 = self._bearing_point(cx, cy, radius, payload.target_bearing_deg)
        pygame.draw.line(surface, (235, 235, 245), (cx, cy), p1, 4)
        pygame.draw.line(surface, (140, 220, 140), (cx, cy), p2, 4)
        indicator_radius = max(20, min(radius - 12, int(radius * 0.34)))
        indicator_points = [
            self._bearing_point(cx, cy, indicator_radius, bearing)
            for bearing in self._angle_indicator_bearings(
                payload.reference_bearing_deg,
                payload.target_bearing_deg,
                payload.angle_measure,
            )
        ]
        if len(indicator_points) >= 2:
            pygame.draw.lines(surface, (255, 208, 104), False, indicator_points, 4)
        pygame.draw.circle(surface, (235, 235, 245), (cx, cy), 6)

    def _draw_bearing_trial(
        self,
        surface: pygame.Surface,
        panel: pygame.Rect,
        payload: AnglesBearingsDegreesPayload,
    ) -> None:
        cx = panel.centerx
        cy = panel.centery
        radius = max(44, (min(panel.w, panel.h) // 2) - 20)

        pygame.draw.circle(surface, (35, 35, 48), (cx, cy), radius)
        pygame.draw.circle(surface, (90, 90, 110), (cx, cy), radius, 2)

        for bearing in (0, 90, 180, 270):
            end = self._bearing_point(cx, cy, radius, bearing)
            pygame.draw.line(surface, (70, 70, 85), (cx, cy), end, 1)

        for label, bearing in (("000", 0), ("090", 90), ("180", 180), ("270", 270)):
            tx, ty = self._bearing_point(cx, cy, radius + 24, bearing)
            surf = self._tiny_font.render(label, True, (150, 150, 165))
            rect = surf.get_rect(center=(tx, ty))
            surface.blit(surf, rect)

        target = self._bearing_point(cx, cy, radius - 8, payload.target_bearing_deg)
        pygame.draw.line(surface, (140, 220, 140), (cx, cy), target, 4)
        pygame.draw.circle(surface, (235, 235, 245), target, 6)
        lbl = self._small_font.render(payload.object_label, True, (235, 235, 245))
        surface.blit(lbl, (target[0] + 8, target[1] - 12))
        pygame.draw.circle(surface, (235, 235, 245), (cx, cy), 6)

    def _render_target_recognition_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: TargetRecognitionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 10, 108)
        frame_bg = (4, 14, 96)
        panel_bg = (8, 16, 80)
        panel_header = (130, 10, 22)
        strip_header = (20, 148, 26)
        border = (228, 238, 255)
        text_main = (236, 244, 255)
        text_muted = (182, 198, 226)

        surface.fill(bg)

        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 15))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        left = self._tiny_font.render(f"Target Recognition - {phase_label}", True, text_main)
        surface.blit(left, left.get_rect(midleft=(header.x + 10, header.centery)))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 6)))

        content = pygame.Rect(
            frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16
        )
        if payload is None or snap.phase not in (Phase.PRACTICE, Phase.SCORED):
            self._tr_selector_hitboxes = {}
            self._tr_light_button_hitbox = None
            self._tr_scan_button_hitbox = None
            self._tr_system_string_hitboxes = []
            self._tr_scene_symbol_hitboxes = []
            card = content.inflate(-8, -8)
            pygame.draw.rect(surface, panel_bg, card)
            pygame.draw.rect(surface, border, card, 1)
            if snap.phase is Phase.PRACTICE_DONE:
                prompt_rect = pygame.Rect(card.x + 14, card.y + 12, card.w - 28, 40)
                self._draw_wrapped_text(
                    surface,
                    str(snap.prompt),
                    prompt_rect,
                    color=text_main,
                    font=self._small_font,
                    max_lines=2,
                )
                title = self._small_font.render("Practice Category Breakdown", True, text_main)
                surface.blit(title, (card.x + 14, prompt_rect.bottom + 8))

                y = prompt_rect.bottom + 34
                step = self._tiny_font.get_linesize() + 2
                for line in self._target_recognition_practice_breakdown_lines():
                    row = self._tiny_font.render(line, True, text_muted)
                    surface.blit(row, (card.x + 16, y))
                    y += step
            else:
                self._draw_wrapped_text(
                    surface,
                    str(snap.prompt),
                    card.inflate(-14, -14),
                    color=text_main,
                    font=self._small_font,
                    max_lines=12,
                )
            footer = (
                "Enter: Continue  |  Esc/Backspace: Back"
                if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE)
                else "Enter: Return to Tests"
            )
            footer_surf = self._tiny_font.render(footer, True, text_muted)
            surface.blit(
                footer_surf, footer_surf.get_rect(midbottom=(frame.centerx, frame.bottom - 10))
            )
            return

        self._target_recognition_sync_selection(payload)
        self._tr_selector_hitboxes = {}
        self._tr_light_button_hitbox = None
        self._tr_scan_button_hitbox = None
        self._tr_system_string_hitboxes = []
        self._tr_scene_symbol_hitboxes = []
        self._target_recognition_sync_scene_stream(payload)
        self._target_recognition_sync_light_stream(payload)
        self._target_recognition_sync_scan_stream(payload)
        self._target_recognition_sync_system_stream(payload)
        live_light_pattern = self._tr_light_current_pattern
        live_light_target = self._tr_light_target_pattern_live
        live_scan_pattern = self._tr_scan_current_pattern
        live_scan_target = self._tr_scan_target_pattern_live

        active_system_target, system_columns, system_row_frac = (
            self._target_recognition_system_view(payload)
        )

        target_strip_h = max(92, min(130, h // 4))
        panels_h = max(140, content.h - target_strip_h - 6)
        panels = pygame.Rect(content.x, content.y, content.w, panels_h)
        targets = pygame.Rect(content.x, panels.bottom + 6, content.w, target_strip_h)

        right_w = max(170, min(230, int(panels.w * 0.28)))
        left_rect = pygame.Rect(panels.x, panels.y, panels.w - right_w - 8, panels.h)
        right_rect = pygame.Rect(left_rect.right + 8, panels.y, right_w, panels.h)

        top_h = max(76, min(106, int(left_rect.h * 0.26)))
        top_row = pygame.Rect(left_rect.x, left_rect.y, left_rect.w, top_h)
        scene_rect = pygame.Rect(
            left_rect.x, top_row.bottom + 8, left_rect.w, left_rect.h - top_h - 8
        )

        gap = 8
        col_w = max(80, (top_row.w - gap * 2) // 3)
        info_rect = pygame.Rect(top_row.x, top_row.y, col_w, top_row.h)
        light_rect = pygame.Rect(info_rect.right + gap, top_row.y, col_w, top_row.h)
        scan_rect = pygame.Rect(
            light_rect.right + gap, top_row.y, top_row.right - (light_rect.right + gap), top_row.h
        )

        def draw_panel(rect: pygame.Rect, title: str) -> pygame.Rect:
            pygame.draw.rect(surface, panel_bg, rect)
            pygame.draw.rect(surface, border, rect, 1)
            bar = pygame.Rect(rect.x + 1, rect.y + 1, rect.w - 2, 18)
            pygame.draw.rect(surface, panel_header, bar)
            lbl = self._tiny_font.render(title, True, text_main)
            surface.blit(lbl, lbl.get_rect(center=bar.center))
            return pygame.Rect(rect.x + 6, rect.y + 24, rect.w - 12, rect.h - 30)

        info_inner = draw_panel(info_rect, "Information")
        light_inner = draw_panel(light_rect, "Light Panel")
        scan_inner = draw_panel(scan_rect, "Scan Panel")
        scene_inner = draw_panel(scene_rect, "Map Panel")
        system_inner = draw_panel(right_rect, "System Panel")

        self._draw_target_recognition_info_legend(surface, info_inner)

        light_bg = light_inner.inflate(-1, -2)
        pygame.draw.rect(surface, (44, 44, 52), light_bg)
        pygame.draw.rect(surface, (86, 104, 150), light_bg, 1)
        bulb_y = light_bg.centery
        bulb_r = max(8, min(12, (light_bg.h // 2) - 4))
        step = max(20, light_bg.w // 4)
        start_x = light_bg.x + max(16, (light_bg.w - (step * 2)) // 2)
        for idx, code in enumerate(live_light_pattern):
            cx = start_x + idx * step
            color = self._target_recognition_light_color(code)
            pygame.draw.circle(surface, color, (cx, bulb_y), bulb_r)
            pygame.draw.circle(surface, (16, 22, 44), (cx, bulb_y), bulb_r, 2)

        light_btn_w = max(56, min(72, light_bg.w // 3))
        light_btn_h = max(24, min(32, light_bg.h - 8))
        light_btn = pygame.Rect(
            light_bg.right - light_btn_w - 4,
            light_bg.centery - (light_btn_h // 2),
            light_btn_w,
            light_btn_h,
        )
        pygame.draw.rect(surface, (220, 174, 34), light_btn)
        pygame.draw.rect(surface, (248, 234, 184), light_btn, 1)
        light_btn_txt = self._tiny_font.render("PRESS", True, (28, 20, 6))
        surface.blit(light_btn_txt, light_btn_txt.get_rect(center=light_btn.center))
        self._tr_light_button_hitbox = light_btn

        scan_bg = scan_inner.inflate(-1, -2)
        pygame.draw.rect(surface, (28, 30, 36), scan_bg)
        pygame.draw.rect(surface, (86, 104, 150), scan_bg, 1)
        scan_token_w = max(22, min(32, (scan_bg.w - 84) // 4))
        scan_token_h = max(18, min(24, scan_bg.h - 8))
        scan_gap = 4
        scan_x0 = scan_bg.x + 4
        scan_y = scan_bg.centery - (scan_token_h // 2)
        for idx in range(4):
            tok_rect = pygame.Rect(
                scan_x0 + idx * (scan_token_w + scan_gap), scan_y, scan_token_w, scan_token_h
            )
            pygame.draw.rect(surface, (14, 20, 34), tok_rect)
            pygame.draw.rect(surface, (72, 92, 126), tok_rect, 1)
        reveal_idx = max(0, min(3, int(self._tr_scan_reveal_index)))
        show_rect = pygame.Rect(
            scan_x0 + reveal_idx * (scan_token_w + scan_gap),
            scan_y,
            scan_token_w,
            scan_token_h,
        )
        pygame.draw.rect(surface, (26, 42, 72), show_rect)
        pygame.draw.rect(surface, (118, 164, 226), show_rect, 1)
        tok_s = self._tiny_font.render(str(live_scan_pattern[reveal_idx]), True, text_main)
        surface.blit(tok_s, tok_s.get_rect(center=show_rect.center))

        scan_btn_w = max(56, min(72, scan_bg.w // 3))
        scan_btn_h = max(24, min(32, scan_bg.h - 8))
        scan_btn = pygame.Rect(
            scan_bg.right - scan_btn_w - 4,
            scan_bg.centery - (scan_btn_h // 2),
            scan_btn_w,
            scan_btn_h,
        )
        pygame.draw.rect(surface, (220, 174, 34), scan_btn)
        pygame.draw.rect(surface, (248, 234, 184), scan_btn, 1)
        scan_btn_txt = self._tiny_font.render("PRESS", True, (28, 20, 6))
        surface.blit(scan_btn_txt, scan_btn_txt.get_rect(center=scan_btn.center))
        self._tr_scan_button_hitbox = scan_btn

        self._draw_target_recognition_scene(surface, scene_inner, payload)

        pygame.draw.rect(surface, (30, 30, 38), system_inner)
        pygame.draw.rect(surface, (86, 104, 150), system_inner, 1)

        cols = max(1, len(system_columns))
        gap_x = 8
        inner_top = system_inner.y + 4
        inner_h = max(20, system_inner.bottom - inner_top - 2)
        col_w = max(46, (system_inner.w - gap_x * (cols + 1)) // cols)
        row_h = self._tiny_font.get_linesize() + 2
        max_rows = max(1, inner_h // row_h)

        for col_idx, col_values in enumerate(system_columns):
            x = system_inner.x + gap_x + col_idx * (col_w + gap_x)
            col_rect = pygame.Rect(x, inner_top, col_w, inner_h)
            pygame.draw.rect(surface, (18, 20, 26), col_rect)
            pygame.draw.rect(surface, (70, 86, 124), col_rect, 1)
            if not col_values:
                continue

            clip = col_rect.inflate(-2, -2)
            prev_clip = surface.get_clip()
            surface.set_clip(clip)
            n_rows = len(col_values)
            for slot in range(-1, max_rows + 2):
                row = col_values[slot % n_rows]
                y = clip.y + int((slot + system_row_frac) * row_h)
                row_surf = self._tiny_font.render(str(row), True, text_main)
                surface.blit(row_surf, (clip.x + 3, y))
                hit = pygame.Rect(
                    clip.x + 3, y, max(8, min(clip.w - 5, row_surf.get_width())), row_h
                )
                hit = hit.clip(clip)
                if hit.w > 0 and hit.h > 0:
                    self._tr_system_string_hitboxes.append((hit, str(row)))
            surface.set_clip(prev_clip)

        pygame.draw.rect(surface, panel_bg, targets)
        pygame.draw.rect(surface, border, targets, 1)

        controls_h = 26
        boxes_area = pygame.Rect(
            targets.x + 2, targets.y + 2, targets.w - 4, max(24, targets.h - controls_h - 4)
        )
        controls = pygame.Rect(targets.x + 2, boxes_area.bottom + 2, targets.w - 4, controls_h)

        target_gap = 6
        target_w = max(80, (boxes_area.w - target_gap * 3) // 4)
        target_labels = (
            ("scene", "Map Targets", ""),
            ("light", "Light Target", "-".join(live_light_target)),
            ("scan", "Scan Target", " ".join(live_scan_target)),
            ("system", "System Target", active_system_target),
        )
        for idx, (panel_key, label, value) in enumerate(target_labels):
            x = boxes_area.x + idx * (target_w + target_gap)
            if idx == 3:
                box_w = boxes_area.right - x
            else:
                box_w = target_w
            box = pygame.Rect(x, boxes_area.y, box_w, boxes_area.h)
            pygame.draw.rect(surface, (4, 9, 36), box)
            pygame.draw.rect(surface, border, box, 1)
            bar = pygame.Rect(box.x + 1, box.y + 1, box.w - 2, 16)
            pygame.draw.rect(surface, strip_header, bar)
            label_surf = self._tiny_font.render(label, True, text_main)
            surface.blit(label_surf, label_surf.get_rect(center=bar.center))
            value_rect = pygame.Rect(
                box.x + 6, bar.bottom + 4, box.w - 12, box.bottom - bar.bottom - 8
            )
            if panel_key == "scene":
                lines = list(self._tr_scene_active_targets)
                y = value_rect.y
                line_h = self._tiny_font.get_linesize() + 1
                self._tr_scene_target_cap = max(1, value_rect.h // max(1, line_h))
                lines = lines[: self._tr_scene_target_cap]
                for line in lines:
                    surf = self._tiny_font.render(line, True, text_main)
                    surface.blit(surf, (value_rect.x, y))
                    y += line_h
            elif panel_key == "light":
                dot_r = max(6, min(10, value_rect.h // 3))
                dot_step = max(dot_r * 2 + 8, value_rect.w // 3)
                dots_x0 = value_rect.x + max(
                    dot_r + 2, (value_rect.w - (dot_step * 2 + dot_r * 2)) // 2
                )
                cy = value_rect.centery
                for dot_idx, code in enumerate(live_light_target):
                    dcx = dots_x0 + dot_idx * dot_step
                    dcol = self._target_recognition_light_color(code)
                    pygame.draw.circle(surface, dcol, (dcx, cy), dot_r)
                    pygame.draw.circle(surface, (16, 22, 44), (dcx, cy), dot_r, 2)
            elif panel_key == "scan":
                tok_w = max(18, min(28, value_rect.w // 4))
                tok_h = max(18, min(24, value_rect.h))
                tok_gap = 4
                x0 = value_rect.x + max(2, (value_rect.w - (tok_w * 4 + tok_gap * 3)) // 2)
                y0 = value_rect.centery - (tok_h // 2)
                for tok_idx, tok in enumerate(live_scan_target):
                    tok_rect = pygame.Rect(x0 + tok_idx * (tok_w + tok_gap), y0, tok_w, tok_h)
                    pygame.draw.rect(surface, (14, 20, 34), tok_rect)
                    pygame.draw.rect(surface, (110, 132, 188), tok_rect, 1)
                    tok_s = self._tiny_font.render(str(tok), True, text_main)
                    surface.blit(tok_s, tok_s.get_rect(center=tok_rect.center))
            else:
                value_surf = self._small_font.render(value, True, text_main)
                value_pos = value_surf.get_rect(center=(value_rect.centerx, value_rect.centery))
                surface.blit(value_surf, value_pos)
            if panel_key not in ("scene", "light", "scan", "system"):
                self._tr_selector_hitboxes[panel_key] = box

        pygame.draw.rect(surface, (5, 12, 42), controls)
        pygame.draw.rect(surface, (72, 92, 138), controls, 1)

        hint = self._tiny_font.render(
            (
                "Mouse only: click active panel controls when matching. "
                f"Auto-advance on exact match.  "
                f"Scene Pts: {self._tr_scene_points}  "
                f"Light Pts: {self._tr_light_points}  "
                f"Scan Pts: {self._tr_scan_points}  "
                f"Sys Pts: {self._tr_system_points}"
            ),
            True,
            text_muted,
        )
        surface.blit(hint, (controls.x + 8, controls.y + 5))

    @staticmethod
    def _target_recognition_light_color(code: str) -> tuple[int, int, int]:
        key = str(code).strip().upper()
        if key == "G":
            return (42, 222, 68)
        if key == "B":
            return (64, 104, 242)
        if key == "Y":
            return (250, 214, 56)
        if key == "R":
            return (234, 72, 72)
        return (186, 190, 204)

    def _draw_target_recognition_info_legend(
        self, surface: pygame.Surface, rect: pygame.Rect
    ) -> None:
        text = (220, 230, 246)
        muted = (156, 176, 206)
        row_h = max(14, rect.h // 4)
        y0 = rect.y + 2
        x_l = rect.x + 8
        x_r = rect.x + (rect.w // 2) + 2

        # Shape legend (top row).
        shape_defs = (
            ("Trucks", TargetRecognitionSceneEntity("truck", "friendly", False, False)),
            ("Tanks", TargetRecognitionSceneEntity("tank", "friendly", False, False)),
            ("Buildings", TargetRecognitionSceneEntity("building", "friendly", False, False)),
        )
        for idx, (label, entity) in enumerate(shape_defs):
            cy = y0 + 7
            self._draw_target_recognition_symbol(
                surface,
                entity=entity,
                cx=x_l + 4 + idx * max(44, rect.w // 3),
                cy=cy,
                size=6,
                color=(230, 230, 230, 255),
            )
            surf = self._tiny_font.render(label, True, text)
            surface.blit(surf, (x_l + 14 + idx * max(44, rect.w // 3), cy - 7))

        # Affiliation row.
        aff_defs = (
            ("Hostile", (226, 90, 92)),
            ("Friendly", (96, 176, 232)),
            ("Neutral", (214, 206, 88)),
        )
        for idx, (label, color) in enumerate(aff_defs):
            cy = y0 + row_h + 7
            sw = pygame.Rect(x_l + idx * max(56, rect.w // 3), cy - 5, 8, 8)
            pygame.draw.rect(surface, color, sw)
            surf = self._tiny_font.render(label, True, text)
            surface.blit(surf, (sw.right + 4, cy - 7))

        # Modifiers row.
        flags_y = y0 + (2 * row_h) + 6
        dmg = self._tiny_font.render("X Damaged", True, muted)
        pri = self._tiny_font.render("+- High Priority", True, muted)
        surface.blit(dmg, (x_l, flags_y))
        surface.blit(pri, (x_r - 8, flags_y))

        # Standalone filler symbols.
        bot_y = y0 + (3 * row_h) + 6
        self._draw_target_recognition_beacon(
            surface, cx=x_l + 5, cy=bot_y + 5, size=5, color=(226, 90, 92, 255)
        )
        beacon = self._tiny_font.render("Beacon", True, muted)
        surface.blit(beacon, (x_l + 14, bot_y - 1))
        self._draw_target_recognition_unknown(
            surface, cx=x_r - 2, cy=bot_y + 5, size=5, color=(214, 206, 88, 255)
        )
        unknown = self._tiny_font.render("Unknown", True, muted)
        surface.blit(unknown, (x_r + 8, bot_y - 1))

    def _draw_target_recognition_scene(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        payload: TargetRecognitionPayload,
    ) -> None:
        if rect.w <= 0 or rect.h <= 0:
            return

        seed = self._target_recognition_scene_seed(payload)
        if (
            self._tr_scene_base_cache is None
            or self._tr_scene_base_cache_size != (rect.w, rect.h)
            or self._tr_scene_base_cache_seed != seed
        ):
            self._tr_scene_base_cache = self._target_recognition_build_scene_base(
                rect.w, rect.h, seed
            )
            self._tr_scene_base_cache_size = (rect.w, rect.h)
            self._tr_scene_base_cache_seed = seed

        assert self._tr_scene_base_cache is not None
        scene = self._tr_scene_base_cache.copy()
        self._draw_target_recognition_scene_compass(scene)
        self._tr_scene_symbol_hitboxes = []

        for glyph_id in self._tr_scene_glyph_order:
            glyph = self._tr_scene_glyphs.get(glyph_id)
            if glyph is None:
                continue
            cx = int(glyph.nx * float(rect.w))
            cy = int(glyph.ny * float(rect.h))
            size = max(5, int(min(rect.w, rect.h) * glyph.scale))
            alpha = max(20, min(255, int(round(glyph.alpha))))
            if glyph.kind == "entity" and glyph.entity is not None:
                rc, gc, bc = self._target_recognition_affiliation_color(glyph.entity.affiliation)
                self._draw_target_recognition_symbol(
                    scene,
                    entity=glyph.entity,
                    cx=cx,
                    cy=cy,
                    size=size,
                    color=(rc, gc, bc, alpha),
                    heading=glyph.heading,
                )
            elif glyph.kind == "beacon":
                self._draw_target_recognition_beacon(
                    scene,
                    cx=cx,
                    cy=cy,
                    size=max(4, int(size * 0.68)),
                    color=(226, 90, 92, alpha),
                )
            else:
                self._draw_target_recognition_unknown(
                    scene,
                    cx=cx,
                    cy=cy,
                    size=max(4, int(size * 0.72)),
                    color=(214, 206, 88, alpha),
                )

            hit_r = max(8, int(size * 1.7))
            hit = pygame.Rect(rect.x + cx - hit_r, rect.y + cy - hit_r, hit_r * 2, hit_r * 2)
            self._tr_scene_symbol_hitboxes.append((hit, glyph_id))

        self._draw_target_recognition_clouds(
            scene,
            payload,
            phase_s=float(self._tr_scene_anim_frame) / 60.0,
        )

        surface.blit(scene, rect.topleft)
        pygame.draw.rect(surface, (78, 98, 138), rect, 1)

    def _target_recognition_build_scene_base(
        self, width: int, height: int, seed: int
    ) -> pygame.Surface:
        base = pygame.Surface((width, height), pygame.SRCALPHA)
        rng = random.Random(seed ^ 0x7F4A7C15)
        base.fill((18, 34, 30, 255))

        for _ in range(24):
            cx = int(rng.uniform(0, width))
            cy = int(rng.uniform(0, height))
            radius = int(rng.uniform(max(36, width * 0.08), max(104, width * 0.25)))
            tone = int(rng.uniform(30, 78))
            alpha = int(rng.uniform(30, 74))
            pygame.draw.circle(base, (tone - 8, tone, tone - 11, alpha), (cx, cy), radius)

        line_palette = (
            (226, 86, 92, 78),
            (96, 176, 232, 76),
            (214, 206, 88, 74),
            (172, 184, 216, 62),
        )
        for _ in range(120):
            col = line_palette[int(rng.uniform(0, len(line_palette)))]
            x1 = int(rng.uniform(0, width))
            y1 = int(rng.uniform(0, height))
            x2 = int(x1 + rng.uniform(-42, 42))
            y2 = int(y1 + rng.uniform(-42, 42))
            w = int(rng.uniform(1, 3))
            pygame.draw.line(base, col, (x1, y1), (x2, y2), w)

        for _ in range(70):
            col = line_palette[int(rng.uniform(0, len(line_palette)))]
            kind = int(rng.uniform(0, 3))
            cx = int(rng.uniform(4, width - 4))
            cy = int(rng.uniform(4, height - 4))
            s = int(rng.uniform(7, 16))
            if kind == 0:
                pygame.draw.circle(base, col, (cx, cy), s, 1)
            elif kind == 1:
                pygame.draw.rect(base, col, pygame.Rect(cx - s, cy - s, s * 2, s * 2), 1)
            else:
                pts = ((cx, cy - s), (cx + s, cy + s), (cx - s, cy + s))
                pygame.draw.polygon(base, col, pts, 1)
        return base

    @staticmethod
    def _draw_target_recognition_scene_compass(scene: pygame.Surface) -> None:
        ring_c = (212, 214, 220, 230)
        mark_c = (240, 102, 106, 230)
        txt_c = (220, 226, 238, 210)
        cx = 16
        cy = 16
        r = 11
        pygame.draw.circle(scene, (200, 206, 214, 65), (cx, cy), r + 2)
        pygame.draw.circle(scene, ring_c, (cx, cy), r, 1)
        pygame.draw.line(scene, mark_c, (cx, cy), (cx, cy - r + 2), 2)
        pygame.draw.line(scene, txt_c, (cx - 2, cy), (cx + 2, cy), 1)

    @staticmethod
    def _target_recognition_scene_seed(payload: TargetRecognitionPayload) -> int:
        # Stable seed per trial payload (do not use Python hash()).
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= v & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(payload.scene_rows))
        mix(int(payload.scene_cols))
        for e in payload.scene_entities:
            for ch in f"{e.shape}:{e.affiliation}:{int(e.damaged)}:{int(e.high_priority)}":
                mix(ord(ch))
        for tok in payload.scene_cells:
            for ch in str(tok):
                mix(ord(ch))
        for opt in payload.scene_target_options:
            for ch in str(opt):
                mix(ord(ch))
        return seed

    def _draw_target_recognition_symbol(
        self,
        surface: pygame.Surface,
        entity: TargetRecognitionSceneEntity,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
        heading: float = 0.0,
    ) -> None:
        line_w = 2 if size >= 8 else 1
        s = max(5, int(size))

        if entity.shape == "truck":
            pygame.draw.circle(surface, color, (cx, cy), s, line_w)
            dx = math.cos(heading)
            dy = math.sin(heading)
            ex = int(cx + dx * (s + 7))
            ey = int(cy + dy * (s + 7))
            pygame.draw.line(surface, color, (cx, cy), (ex, ey), line_w)
            px = -dy
            py = dx
            p1 = (ex, ey)
            p2 = (int(ex - dx * 5 + px * 3), int(ey - dy * 5 + py * 3))
            p3 = (int(ex - dx * 5 - px * 3), int(ey - dy * 5 - py * 3))
            pygame.draw.polygon(surface, color, (p1, p2, p3), line_w)
        elif entity.shape == "tank":
            box = pygame.Rect(cx - s, cy - s, s * 2, s * 2)
            pygame.draw.rect(surface, color, box, line_w)
            dx = math.cos(heading)
            dy = math.sin(heading)
            pygame.draw.line(
                surface,
                color,
                (int(cx - dx * (s + 3)), int(cy - dy * (s + 3))),
                (int(cx + dx * (s + 3)), int(cy + dy * (s + 3))),
                line_w,
            )
        elif entity.shape == "building":
            a0 = heading - (math.pi / 2.0)
            pts = (
                (int(cx + math.cos(a0) * (s + 1)), int(cy + math.sin(a0) * (s + 1))),
                (int(cx + math.cos(a0 + 2.12) * (s + 2)), int(cy + math.sin(a0 + 2.12) * (s + 2))),
                (int(cx + math.cos(a0 - 2.12) * (s + 2)), int(cy + math.sin(a0 - 2.12) * (s + 2))),
            )
            pygame.draw.polygon(surface, color, pts, line_w)
        else:
            points = []
            for i in range(6):
                ang = (math.tau * i) / 6.0
                points.append((int(cx + math.cos(ang) * s), int(cy + math.sin(ang) * s)))
            pygame.draw.polygon(surface, color, points, line_w)

        if entity.damaged:
            xw = max(1, line_w)
            pygame.draw.line(surface, color, (cx - s + 1, cy - s + 1), (cx + s - 1, cy + s - 1), xw)
            pygame.draw.line(surface, color, (cx + s - 1, cy - s + 1), (cx - s + 1, cy + s - 1), xw)
        if entity.high_priority:
            pw = max(1, line_w)
            pygame.draw.line(surface, color, (cx - s - 3, cy), (cx + s + 3, cy), pw)
            pygame.draw.line(surface, color, (cx, cy - s - 3), (cx, cy + s + 3), pw)

    @staticmethod
    def _draw_target_recognition_beacon(
        surface: pygame.Surface,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
    ) -> None:
        s = max(3, int(size))
        box = pygame.Rect(cx - s, cy - s, s * 2, s * 2)
        pygame.draw.rect(surface, color, box)
        pygame.draw.rect(surface, (18, 22, 34, max(80, color[3])), box, 1)

    @staticmethod
    def _draw_target_recognition_unknown(
        surface: pygame.Surface,
        *,
        cx: int,
        cy: int,
        size: int,
        color: tuple[int, int, int, int],
    ) -> None:
        s = max(4, int(size))
        pts = ((cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy))
        pygame.draw.polygon(surface, color, pts, 2)

    @staticmethod
    def _target_recognition_affiliation_color(affiliation: str) -> tuple[int, int, int]:
        key = str(affiliation).strip().lower()
        if key == "hostile":
            return (224, 88, 90)
        if key == "friendly":
            return (96, 176, 232)
        return (214, 206, 88)

    @staticmethod
    def _target_recognition_entity_from_code(code: str) -> TargetRecognitionSceneEntity:
        text = str(code).upper()
        shape_code = text
        side = "N"
        flags = ""
        if ":" in text:
            shape_code, rest = text.split(":", 1)
            if rest:
                side = rest[0]
                flags = rest[1:]
        shape = {
            "TRK": "truck",
            "TNK": "tank",
            "BLD": "building",
        }.get(shape_code, "truck")
        affiliation = {
            "H": "hostile",
            "F": "friendly",
            "N": "neutral",
        }.get(side, "neutral")
        return TargetRecognitionSceneEntity(
            shape=shape,
            affiliation=affiliation,
            damaged=("D" in flags),
            high_priority=("P" in flags),
        )

    def _draw_target_recognition_clouds(
        self,
        scene: pygame.Surface,
        payload: TargetRecognitionPayload,
        *,
        phase_s: float,
    ) -> None:
        w, h = scene.get_size()
        seed = self._target_recognition_scene_seed(payload) ^ 0x9E3779B9
        rng = random.Random(seed)
        t = max(0.0, float(phase_s))

        # Bright haze base.
        haze = pygame.Surface((w, h), pygame.SRCALPHA)
        haze.fill((186, 190, 186, 28))

        for _ in range(22):
            bx = rng.uniform(0, w)
            by = rng.uniform(0, h)
            radius = rng.uniform(max(46, w * 0.08), max(120, w * 0.24))
            phase = rng.uniform(0.0, math.tau)
            speed = rng.uniform(0.05, 0.14)
            drift = rng.uniform(8.0, 28.0)
            cx = bx + math.sin((t * speed) + phase) * drift
            cy = by + math.cos((t * speed * 0.8) + phase * 1.3) * drift

            for i in range(3):
                rr = int(radius * (1.0 - i * 0.22))
                alpha = max(28, int(94 - i * 20))
                shade = 184 - (i * 6)
                pygame.draw.circle(haze, (shade, shade + 6, shade, alpha), (int(cx), int(cy)), rr)
                pygame.draw.circle(
                    haze,
                    (shade, shade + 4, shade, max(22, alpha - 24)),
                    (int(cx + rr * 0.34), int(cy - rr * 0.18)),
                    int(rr * 0.72),
                )

        # Dark cloud pockets to block symbol visibility.
        for _ in range(18):
            bx = rng.uniform(0, w)
            by = rng.uniform(0, h)
            radius = rng.uniform(max(34, w * 0.07), max(96, w * 0.16))
            phase = rng.uniform(0.0, math.tau)
            speed = rng.uniform(0.04, 0.11)
            drift = rng.uniform(6.0, 20.0)
            cx = bx + math.sin((t * speed) + phase) * drift
            cy = by + math.cos((t * speed * 0.9) + phase * 0.9) * drift
            pygame.draw.circle(haze, (16, 22, 20, 94), (int(cx), int(cy)), int(radius))
            pygame.draw.circle(
                haze,
                (10, 14, 12, 78),
                (int(cx + radius * 0.2), int(cy - radius * 0.1)),
                int(radius * 0.68),
            )

        haze.fill((170, 174, 170, 12), special_flags=pygame.BLEND_RGBA_ADD)

        scene.blit(haze, (0, 0))

    def _target_recognition_reset_scene_subtask(self) -> None:
        self._tr_scene_payload_id = None
        self._tr_scene_rng = None
        self._tr_scene_glyphs = {}
        self._tr_scene_glyph_order = []
        self._tr_scene_symbol_hitboxes = []
        self._tr_scene_next_glyph_id = 1
        self._tr_scene_target_queue = []
        self._tr_scene_active_targets = []
        self._tr_scene_target_cap = 5
        self._tr_scene_next_target_add_ms = 0
        self._tr_scene_points = 0
        self._tr_scene_hits = 0
        self._tr_scene_misses = 0
        self._tr_scene_beacon_hits = 0
        self._tr_scene_unknown_hits = 0
        self._tr_scene_anim_frame = 0
        self._tr_scene_last_update_ms = 0
        self._tr_scene_base_cache = None
        self._tr_scene_base_cache_size = (0, 0)
        self._tr_scene_base_cache_seed = 0

    def _target_recognition_sync_scene_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_scene_payload_id != pid:
            self._tr_scene_payload_id = pid
            self._tr_scene_rng = random.Random(
                self._target_recognition_scene_seed(payload) ^ 0xC0FFEE17
            )
            self._tr_scene_glyphs = {}
            self._tr_scene_glyph_order = []
            self._tr_scene_next_glyph_id = 1
            self._tr_scene_symbol_hitboxes = []
            self._tr_scene_target_queue = (
                list(payload.scene_target_options) if payload.scene_has_target else []
            )
            self._tr_scene_active_targets = []
            self._tr_scene_next_target_add_ms = now_ms + 1200
            self._tr_scene_anim_frame = 0
            self._tr_scene_last_update_ms = now_ms
            self._tr_scene_base_cache = None
            self._tr_scene_base_cache_size = (0, 0)
            self._tr_scene_base_cache_seed = 0

            scene_entities = payload.scene_entities
            if not scene_entities:
                scene_entities = tuple(
                    self._target_recognition_entity_from_code(code) for code in payload.scene_cells
                )
            rows = max(1, int(payload.scene_rows))
            cols = max(1, int(payload.scene_cols))
            max_items = min(rows * cols, len(scene_entities))
            assert self._tr_scene_rng is not None
            for idx in range(max_items):
                rr = idx // cols
                cc = idx % cols
                nx = (cc + 0.5) / float(cols)
                ny = (rr + 0.5) / float(rows)
                nx += (self._tr_scene_rng.random() - 0.5) * 0.08
                ny += (self._tr_scene_rng.random() - 0.5) * 0.08
                nx = max(0.04, min(0.96, nx))
                ny = max(0.04, min(0.96, ny))
                if self._target_recognition_scene_in_compass_zone(nx, ny):
                    ny = min(0.96, ny + 0.16)
                entity = scene_entities[idx]
                labels = self._target_recognition_scene_matching_labels(
                    entity=entity,
                    labels=payload.scene_target_options,
                )
                glyph_id = self._tr_scene_next_glyph_id
                self._tr_scene_next_glyph_id += 1
                self._tr_scene_glyphs[glyph_id] = _TargetRecognitionSceneGlyph(
                    glyph_id=glyph_id,
                    kind="entity",
                    entity=entity,
                    nx=nx,
                    ny=ny,
                    scale=float(self._tr_scene_rng.uniform(0.010, 0.024)),
                    heading=float(self._tr_scene_rng.uniform(0.0, math.tau)),
                    alpha=float(self._tr_scene_rng.uniform(88.0, 160.0)),
                    max_alpha=float(self._tr_scene_rng.uniform(128.0, 190.0)),
                    matching_labels=labels,
                )
                self._tr_scene_glyph_order.append(glyph_id)

            filler_count = max(2, min(5, (rows * cols) // 20))
            for _ in range(filler_count):
                self._target_recognition_scene_add_filler_glyph(kind="beacon")
                self._target_recognition_scene_add_filler_glyph(kind="unknown")

        if self._tr_scene_payload_id != pid:
            return
        dt_ms = max(0, min(120, now_ms - self._tr_scene_last_update_ms))
        self._tr_scene_last_update_ms = now_ms
        self._tr_scene_anim_frame += 1
        if dt_ms > 0:
            fade = float(dt_ms) * 0.040
            for glyph in self._tr_scene_glyphs.values():
                if glyph.alpha < glyph.max_alpha:
                    glyph.alpha = min(glyph.max_alpha, glyph.alpha + fade)

        if payload.scene_has_target:
            while now_ms >= self._tr_scene_next_target_add_ms:
                if (
                    self._tr_scene_target_queue
                    and len(self._tr_scene_active_targets) < self._tr_scene_target_cap
                ):
                    attempts = 0
                    queue_len = max(1, len(self._tr_scene_target_queue))
                    added = False
                    while attempts < queue_len:
                        nxt = self._tr_scene_target_queue.pop(0)
                        self._tr_scene_target_queue.append(nxt)
                        attempts += 1
                        if nxt in self._tr_scene_active_targets:
                            continue
                        self._tr_scene_active_targets.append(nxt)
                        self._target_recognition_scene_ensure_label_present(payload, nxt)
                        added = True
                        break
                    if not added and queue_len == 1 and not self._tr_scene_active_targets:
                        only = self._tr_scene_target_queue[0]
                        self._tr_scene_active_targets.append(only)
                        self._target_recognition_scene_ensure_label_present(payload, only)
                self._tr_scene_next_target_add_ms += (
                    self._target_recognition_scene_spawn_interval_ms()
                )

            self._target_recognition_scene_prune_completed_targets()

    def _target_recognition_scene_prune_completed_targets(self) -> None:
        if not self._tr_scene_active_targets:
            return
        active: list[str] = []
        for label in self._tr_scene_active_targets:
            has_any = any(
                glyph.kind == "entity" and label in glyph.matching_labels
                for glyph in self._tr_scene_glyphs.values()
            )
            if has_any:
                active.append(label)
        self._tr_scene_active_targets = active

    def _target_recognition_handle_scene_press(
        self,
        payload: TargetRecognitionPayload,
        *,
        glyph_id: int,
    ) -> bool:
        glyph = self._tr_scene_glyphs.get(int(glyph_id))
        if glyph is None:
            return False

        if glyph.kind == "beacon":
            self._tr_scene_points += 1
            self._tr_scene_beacon_hits += 1
            self._target_recognition_scene_reseed_glyph(
                payload, glyph, kind="beacon", force_non_target=True
            )
            return True
        if glyph.kind == "unknown":
            self._tr_scene_points += 1
            self._tr_scene_unknown_hits += 1
            self._target_recognition_scene_reseed_glyph(
                payload, glyph, kind="unknown", force_non_target=True
            )
            return True

        active = set(self._tr_scene_active_targets)
        hit_target = bool(active.intersection(glyph.matching_labels))
        if hit_target:
            self._tr_scene_points += 1
            self._tr_scene_hits += 1
            self._target_recognition_scene_reseed_glyph(
                payload, glyph, kind="entity", force_non_target=True
            )
            self._target_recognition_scene_prune_completed_targets()
            return True

        self._tr_scene_misses += 1
        return False

    def _target_recognition_scene_reseed_glyph(
        self,
        payload: TargetRecognitionPayload,
        glyph: _TargetRecognitionSceneGlyph,
        *,
        kind: str,
        force_non_target: bool,
    ) -> None:
        if self._tr_scene_rng is None:
            self._tr_scene_rng = random.Random(
                self._target_recognition_scene_seed(payload) ^ 0xC0FFEE17
            )

        glyph.kind = kind
        glyph.nx, glyph.ny = self._target_recognition_scene_random_position()
        glyph.scale = float(self._tr_scene_rng.uniform(0.010, 0.024))
        glyph.heading = float(self._tr_scene_rng.uniform(0.0, math.tau))
        glyph.alpha = 0.0
        glyph.max_alpha = float(self._tr_scene_rng.uniform(128.0, 186.0))
        glyph.matching_labels = ()
        glyph.entity = None

        if kind != "entity":
            return

        active = set(self._tr_scene_active_targets)
        for _ in range(72):
            damaged, high_priority = self._target_recognition_scene_roll_modifiers()
            candidate = TargetRecognitionSceneEntity(
                shape=str(self._tr_scene_rng.choice(("truck", "tank", "building"))),
                affiliation=str(self._tr_scene_rng.choice(("hostile", "friendly", "neutral"))),
                damaged=damaged,
                high_priority=high_priority,
            )
            labels = self._target_recognition_scene_matching_labels(
                entity=candidate,
                labels=payload.scene_target_options,
            )
            if force_non_target and active.intersection(labels):
                continue
            glyph.entity = candidate
            glyph.matching_labels = labels
            return

        glyph.entity = TargetRecognitionSceneEntity("truck", "neutral", False, False)
        glyph.matching_labels = ()

    def _target_recognition_scene_spawn_interval_ms(self) -> int:
        if self._tr_scene_rng is None:
            return 1800
        return int(round(self._tr_scene_rng.uniform(1400.0, 2400.0)))

    def _target_recognition_scene_ensure_label_present(
        self,
        payload: TargetRecognitionPayload,
        label: str,
    ) -> None:
        already_present = any(
            glyph.kind == "entity" and label in glyph.matching_labels
            for glyph in self._tr_scene_glyphs.values()
        )
        if already_present:
            return

        target_entity = self._target_recognition_scene_entity_from_label(label)
        if target_entity is None:
            return

        active_set = set(self._tr_scene_active_targets)
        preferred: _TargetRecognitionSceneGlyph | None = None
        for glyph_id in self._tr_scene_glyph_order:
            glyph = self._tr_scene_glyphs.get(glyph_id)
            if glyph is None or glyph.kind != "entity":
                continue
            if preferred is None and not active_set.intersection(glyph.matching_labels):
                preferred = glyph
                break
            if preferred is None:
                preferred = glyph
        if preferred is None:
            return

        preferred.kind = "entity"
        preferred.entity = target_entity
        preferred.matching_labels = self._target_recognition_scene_matching_labels(
            entity=target_entity,
            labels=payload.scene_target_options,
        )
        preferred.alpha = 0.0
        preferred.nx, preferred.ny = self._target_recognition_scene_random_position()

    def _target_recognition_scene_add_filler_glyph(self, *, kind: str) -> None:
        if self._tr_scene_rng is None:
            return
        glyph_id = self._tr_scene_next_glyph_id
        self._tr_scene_next_glyph_id += 1
        nx, ny = self._target_recognition_scene_random_position()
        self._tr_scene_glyphs[glyph_id] = _TargetRecognitionSceneGlyph(
            glyph_id=glyph_id,
            kind=kind,
            entity=None,
            nx=nx,
            ny=ny,
            scale=float(self._tr_scene_rng.uniform(0.008, 0.016)),
            heading=0.0,
            alpha=float(self._tr_scene_rng.uniform(120.0, 188.0)),
            max_alpha=float(self._tr_scene_rng.uniform(140.0, 196.0)),
            matching_labels=(),
        )
        self._tr_scene_glyph_order.append(glyph_id)

    def _target_recognition_scene_random_position(self) -> tuple[float, float]:
        assert self._tr_scene_rng is not None
        for _ in range(48):
            nx = float(self._tr_scene_rng.uniform(0.04, 0.96))
            ny = float(self._tr_scene_rng.uniform(0.05, 0.96))
            if not self._target_recognition_scene_in_compass_zone(nx, ny):
                return nx, ny
        return 0.5, 0.52

    @staticmethod
    def _target_recognition_scene_in_compass_zone(nx: float, ny: float) -> bool:
        dx = nx - 0.055
        dy = ny - 0.065
        return (dx * dx + dy * dy) <= (0.058 * 0.058)

    def _target_recognition_scene_matching_labels(
        self,
        *,
        entity: TargetRecognitionSceneEntity,
        labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            label for label in labels if self._target_recognition_scene_label_matches(entity, label)
        )

    @staticmethod
    def _target_recognition_scene_label_matches(
        entity: TargetRecognitionSceneEntity, label: str
    ) -> bool:
        txt = str(label).upper()
        if "UNKNOWN" in txt or "BEACON" in txt:
            return False
        if "TRUCK" in txt:
            shape = "truck"
        elif "TANK" in txt:
            shape = "tank"
        elif "BUILDING" in txt:
            shape = "building"
        else:
            return False
        if "HOSTILE" in txt:
            affiliation = "hostile"
        elif "FRIENDLY" in txt:
            affiliation = "friendly"
        elif "NEUTRAL" in txt:
            affiliation = "neutral"
        else:
            return False
        if entity.shape != shape or entity.affiliation != affiliation:
            return False
        label_damaged = "DAMAGED" in txt
        label_hp = "HP" in txt
        if entity.damaged != label_damaged:
            return False
        if entity.high_priority != label_hp:
            return False
        return True

    @staticmethod
    def _target_recognition_scene_entity_from_label(
        label: str,
    ) -> TargetRecognitionSceneEntity | None:
        txt = str(label).upper()
        if "UNKNOWN" in txt or "BEACON" in txt:
            return None
        if "TRUCK" in txt:
            shape = "truck"
        elif "TANK" in txt:
            shape = "tank"
        elif "BUILDING" in txt:
            shape = "building"
        else:
            return None
        if "HOSTILE" in txt:
            affiliation = "hostile"
        elif "FRIENDLY" in txt:
            affiliation = "friendly"
        elif "NEUTRAL" in txt:
            affiliation = "neutral"
        else:
            return None
        return TargetRecognitionSceneEntity(
            shape=shape,
            affiliation=affiliation,
            damaged=("DAMAGED" in txt),
            high_priority=("HP" in txt),
        )

    def _target_recognition_scene_roll_modifiers(self) -> tuple[bool, bool]:
        assert self._tr_scene_rng is not None
        damaged = bool(self._tr_scene_rng.random() < 0.24)
        high_priority = bool(self._tr_scene_rng.random() < 0.11)
        if damaged and high_priority and self._tr_scene_rng.random() < 0.82:
            if self._tr_scene_rng.random() < 0.66:
                high_priority = False
            else:
                damaged = False
        return damaged, high_priority

    def _target_recognition_sync_selection(self, payload: TargetRecognitionPayload) -> None:
        pid = id(payload)
        if self._tr_selection_payload_id == pid:
            return
        self._tr_selection_payload_id = pid
        self._tr_selected_panels.clear()
        self._input = ""

    def _target_recognition_reset_light_subtask(self) -> None:
        self._tr_light_payload_id = None
        self._tr_light_rng = None
        self._tr_light_next_change_ms = 0
        self._tr_light_current_pattern = ("G", "G", "G")
        self._tr_light_target_pattern_live = ("G", "G", "G")
        self._tr_light_points = 0
        self._tr_light_hits = 0
        self._tr_light_early_presses = 0
        self._tr_light_button_hitbox = None

    def _target_recognition_reset_scan_subtask(self) -> None:
        self._tr_scan_payload_id = None
        self._tr_scan_rng = None
        self._tr_scan_token_pool = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_next_change_ms = 0
        self._tr_scan_current_pattern = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_target_pattern_live = ("<>", "[]", "/\\", "\\/")
        self._tr_scan_points = 0
        self._tr_scan_hits = 0
        self._tr_scan_early_presses = 0
        self._tr_scan_button_hitbox = None
        self._tr_scan_reveal_index = 3
        self._tr_scan_next_step_ms = 0
        self._tr_scan_passes_left = 0

    def _target_recognition_sync_light_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_light_payload_id != pid:
            self._tr_light_payload_id = pid
            self._tr_light_rng = random.Random(self._target_recognition_light_seed(payload))
            self._tr_light_current_pattern = self._target_recognition_light_triplet(
                payload.light_pattern
            )
            self._tr_light_target_pattern_live = self._target_recognition_light_triplet(
                payload.light_target_pattern
            )
            self._tr_light_next_change_ms = now_ms + self._target_recognition_light_interval_ms()

        while now_ms >= self._tr_light_next_change_ms:
            assert self._tr_light_rng is not None
            if self._tr_light_rng.random() < 0.38:
                self._tr_light_current_pattern = self._tr_light_target_pattern_live
            else:
                self._tr_light_current_pattern = self._target_recognition_next_light_pattern(
                    exclude=self._tr_light_target_pattern_live
                )
            self._tr_light_next_change_ms += self._target_recognition_light_interval_ms()

    def _target_recognition_handle_light_press(self, payload: TargetRecognitionPayload) -> bool:
        self._target_recognition_sync_light_stream(payload)
        if self._tr_light_current_pattern == self._tr_light_target_pattern_live:
            self._tr_light_points += 1
            self._tr_light_hits += 1
            self._tr_light_target_pattern_live = self._target_recognition_next_light_pattern(
                exclude=self._tr_light_current_pattern
            )
            return True

        self._tr_light_points -= 1
        self._tr_light_early_presses += 1
        return False

    def _target_recognition_sync_scan_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_scan_payload_id != pid:
            self._tr_scan_payload_id = pid
            self._tr_scan_rng = random.Random(self._target_recognition_scan_seed(payload))
            pool = tuple(dict.fromkeys(str(tok) for tok in payload.scan_tokens))
            if payload.scan_target and str(payload.scan_target) not in pool:
                pool = (*pool, str(payload.scan_target))
            if len(pool) < 4:
                pool = ("<>", "<|", "|>", "[]", "{}", "()", "/\\", "\\/", "==", "=~")
            self._tr_scan_token_pool = tuple(pool)
            self._tr_scan_target_pattern_live = self._target_recognition_next_scan_pattern(
                exclude=None
            )
            self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                exclude=self._tr_scan_target_pattern_live
            )
            self._tr_scan_next_change_ms = now_ms + self._target_recognition_scan_interval_ms()
            self._tr_scan_reveal_index = 3
            self._tr_scan_next_step_ms = now_ms + 1000
            self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

        while now_ms >= self._tr_scan_next_change_ms:
            assert self._tr_scan_rng is not None
            if self._tr_scan_rng.random() < 0.34:
                self._tr_scan_current_pattern = self._tr_scan_target_pattern_live
            else:
                self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                    exclude=self._tr_scan_target_pattern_live
                )
            self._tr_scan_next_change_ms += self._target_recognition_scan_interval_ms()
            self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

        while now_ms >= self._tr_scan_next_step_ms:
            self._tr_scan_next_step_ms += 1000
            if self._tr_scan_reveal_index > 0:
                self._tr_scan_reveal_index -= 1
                continue

            self._tr_scan_reveal_index = 3
            self._tr_scan_passes_left -= 1
            if self._tr_scan_passes_left <= 0:
                assert self._tr_scan_rng is not None
                if self._tr_scan_rng.random() < 0.34:
                    self._tr_scan_current_pattern = self._tr_scan_target_pattern_live
                else:
                    self._tr_scan_current_pattern = self._target_recognition_next_scan_pattern(
                        exclude=self._tr_scan_target_pattern_live
                    )
                self._tr_scan_passes_left = self._target_recognition_scan_repeat_count()

    def _target_recognition_handle_scan_press(self, payload: TargetRecognitionPayload) -> bool:
        self._target_recognition_sync_scan_stream(payload)
        if self._tr_scan_current_pattern == self._tr_scan_target_pattern_live:
            self._tr_scan_points += 1
            self._tr_scan_hits += 1
            self._tr_scan_target_pattern_live = self._target_recognition_next_scan_pattern(
                exclude=self._tr_scan_current_pattern
            )
            return True

        self._tr_scan_points -= 1
        self._tr_scan_early_presses += 1
        return False

    def _target_recognition_next_scan_pattern(
        self,
        *,
        exclude: tuple[str, str, str, str] | None,
    ) -> tuple[str, str, str, str]:
        if self._tr_scan_rng is None:
            self._tr_scan_rng = random.Random(0)
        pool = self._tr_scan_token_pool or ("<>", "[]", "/\\", "\\/")
        for _ in range(64):
            cand = (
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
                self._target_recognition_scan_symbol(pool),
            )
            if exclude is None or cand != exclude:
                return cand
        if exclude is None:
            return (
                f"{pool[0]}{pool[1 % len(pool)]}",
                f"{pool[1 % len(pool)]}{pool[2 % len(pool)]}",
                f"{pool[2 % len(pool)]}{pool[0]}",
                f"{pool[0]}{pool[3 % len(pool)]}",
            )
        return (
            f"{pool[0]}{pool[1 % len(pool)]}",
            f"{pool[1 % len(pool)]}{pool[0]}",
            f"{pool[0]}{pool[0]}",
            f"{pool[2 % len(pool)]}{pool[3 % len(pool)]}",
        )

    def _target_recognition_scan_symbol(self, pool: tuple[str, ...]) -> str:
        assert self._tr_scan_rng is not None
        a = str(self._tr_scan_rng.choice(pool))
        b = str(self._tr_scan_rng.choice(pool))
        return f"{a}{b}"

    def _target_recognition_scan_repeat_count(self) -> int:
        if self._tr_scan_rng is None:
            return 3
        return int(self._tr_scan_rng.randint(2, 4))

    def _target_recognition_scan_interval_ms(self) -> int:
        if self._tr_scan_rng is None:
            return 6500
        return int(round(self._tr_scan_rng.uniform(5.0, 10.0) * 1000.0))

    @staticmethod
    def _target_recognition_scan_seed(payload: TargetRecognitionPayload) -> int:
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= v & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        for token in (
            payload.scan_target,
            payload.system_target,
            payload.scene_target,
            *payload.scan_tokens,
        ):
            for ch in str(token):
                mix(ord(ch))
        return seed ^ 0x13579BDF

    def _target_recognition_next_light_pattern(
        self,
        *,
        exclude: tuple[str, str, str] | None = None,
    ) -> tuple[str, str, str]:
        if self._tr_light_rng is None:
            self._tr_light_rng = random.Random(0)
        colors = ("G", "B", "Y", "R")
        for _ in range(48):
            cand = (
                str(self._tr_light_rng.choice(colors)),
                str(self._tr_light_rng.choice(colors)),
                str(self._tr_light_rng.choice(colors)),
            )
            if exclude is None or cand != exclude:
                return cand
        if exclude is None:
            return ("R", "G", "B")
        return ("R" if exclude[0] != "R" else "G", exclude[1], exclude[2])

    def _target_recognition_light_interval_ms(self) -> int:
        if self._tr_light_rng is None:
            return 7500
        return int(round(self._tr_light_rng.uniform(5.0, 10.0) * 1000.0))

    @staticmethod
    def _target_recognition_light_triplet(pattern: tuple[str, ...]) -> tuple[str, str, str]:
        vals = [str(v).strip().upper()[:1] for v in pattern]
        vals = [v if v in ("G", "B", "Y", "R") else "G" for v in vals]
        while len(vals) < 3:
            vals.append("G")
        return (vals[0], vals[1], vals[2])

    @staticmethod
    def _target_recognition_light_seed(payload: TargetRecognitionPayload) -> int:
        # Stable per-trial seed for light cadence and target switching.
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= v & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(payload.scene_rows))
        mix(int(payload.scene_cols))
        for token in (
            *payload.light_pattern,
            *payload.light_target_pattern,
            payload.scan_target,
            payload.system_target,
            payload.scene_target,
        ):
            for ch in str(token):
                mix(ord(ch))
        return seed ^ 0xA5A55A5A

    def _target_recognition_reset_system_subtask(self) -> None:
        self._tr_system_payload_id = None
        self._tr_system_rng = None
        self._tr_system_columns = [[], [], []]
        self._tr_system_row_offset = 0
        self._tr_system_row_frac = 0.0
        self._tr_system_target_code = "----"
        self._tr_system_step_interval_ms = 1700
        self._tr_system_last_step_ms = 0
        self._tr_system_points = 0
        self._tr_system_hits = 0
        self._tr_system_string_hitboxes = []

    def _target_recognition_sync_system_stream(self, payload: TargetRecognitionPayload) -> None:
        now_ms = pygame.time.get_ticks()
        pid = id(payload)
        if self._tr_system_payload_id != pid:
            self._tr_system_payload_id = pid
            self._tr_system_rng = random.Random(self._target_recognition_system_seed(payload))
            self._tr_system_columns = self._target_recognition_build_system_columns(payload)
            row_count = max(1, len(self._tr_system_columns[0])) if self._tr_system_columns else 1
            self._tr_system_row_offset = 0
            self._tr_system_row_frac = 0.0
            self._tr_system_last_step_ms = now_ms
            assert self._tr_system_rng is not None
            self._tr_system_step_interval_ms = int(
                round(self._tr_system_rng.uniform(1450.0, 2300.0))
            )
            self._tr_system_target_code = self._target_recognition_pick_initial_system_target(
                payload
            )
            if row_count <= 0:
                return

        step = max(1000, int(self._tr_system_step_interval_ms))
        row_count = max(1, len(self._tr_system_columns[0])) if self._tr_system_columns else 1
        while now_ms - self._tr_system_last_step_ms >= step:
            self._tr_system_last_step_ms += step
            self._tr_system_row_offset = (self._tr_system_row_offset + 1) % row_count
        self._tr_system_row_frac = max(
            0.0,
            min(1.0, float(now_ms - self._tr_system_last_step_ms) / float(step)),
        )

    def _target_recognition_build_system_columns(
        self,
        payload: TargetRecognitionPayload,
    ) -> list[list[str]]:
        cols: list[list[str]] = []
        if payload.system_cycles and payload.system_cycles[0].columns:
            source_cols = payload.system_cycles[0].columns
            for col in source_cols[:3]:
                cols.append([str(v) for v in col])
        else:
            base = [str(v) for v in payload.system_rows]
            if not base:
                base = ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"]
            while len(cols) < 3:
                cols.append(list(base))

        while len(cols) < 3:
            cols.append(
                list(cols[-1] if cols else ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"])
            )

        max_len = max(6, max((len(c) for c in cols), default=6))
        for idx, col in enumerate(cols):
            if not col:
                col = ["A1B2", "C3D4", "E5F6", "G7H8", "J9K1", "L2M3"]
            while len(col) < max_len:
                col.append(col[len(col) % max(1, len(col))])
            cols[idx] = col[:max_len]
        return cols[:3]

    def _target_recognition_pick_initial_system_target(
        self, payload: TargetRecognitionPayload
    ) -> str:
        pool = [code for col in self._tr_system_columns for code in col]
        if payload.system_target and str(payload.system_target) in pool:
            return str(payload.system_target)
        if not pool:
            return "A1B2"
        assert self._tr_system_rng is not None
        return str(self._tr_system_rng.choice(pool))

    def _target_recognition_handle_system_press(
        self,
        payload: TargetRecognitionPayload,
        *,
        clicked_code: str,
    ) -> bool:
        self._target_recognition_sync_system_stream(payload)
        if str(clicked_code) != str(self._tr_system_target_code):
            return False
        self._tr_system_points += 1
        self._tr_system_hits += 1
        next_target = self._target_recognition_pick_next_system_target()
        self._tr_system_target_code = next_target
        return True

    def _target_recognition_pick_next_system_target(self) -> str:
        pool = [
            code
            for col in self._tr_system_columns
            for code in col
            if code != self._tr_system_target_code
        ]
        if not pool:
            return self._tr_system_target_code
        assert self._tr_system_rng is not None
        return str(self._tr_system_rng.choice(pool))

    @staticmethod
    def _target_recognition_system_seed(payload: TargetRecognitionPayload) -> int:
        seed = 2166136261

        def mix(v: int) -> None:
            nonlocal seed
            seed ^= v & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        for token in (
            payload.system_target,
            payload.scene_target,
            payload.scan_target,
            *payload.light_pattern,
            *payload.light_target_pattern,
        ):
            for ch in str(token):
                mix(ord(ch))
        for cycle in payload.system_cycles:
            for col in cycle.columns:
                for row in col:
                    for ch in str(row):
                        mix(ord(ch))
        return seed ^ 0x5A5AA55A

    def _target_recognition_reset_practice_breakdown(self) -> None:
        self._tr_practice_trials = 0
        self._tr_practice_panel_correct = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }
        self._tr_practice_panel_present = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }
        self._tr_practice_panel_hits = {
            "scene": 0,
            "light": 0,
            "scan": 0,
            "system": 0,
        }

    def _target_recognition_record_practice_trial(
        self,
        *,
        selected: set[str],
        expected: set[str],
    ) -> None:
        self._tr_practice_trials += 1
        for panel in ("scene", "light", "scan", "system"):
            exp = panel in expected
            sel = panel in selected
            if exp:
                self._tr_practice_panel_present[panel] += 1
            if exp and sel:
                self._tr_practice_panel_hits[panel] += 1
            if exp == sel:
                self._tr_practice_panel_correct[panel] += 1

    def _target_recognition_practice_breakdown_lines(self) -> tuple[str, ...]:
        trials = max(0, int(self._tr_practice_trials))
        if trials == 0:
            return ("No practice data recorded.",)

        labels = (
            ("scene", "Map"),
            ("light", "Light"),
            ("scan", "Scan"),
            ("system", "System"),
        )
        lines: list[str] = []
        for key, label in labels:
            correct = int(self._tr_practice_panel_correct.get(key, 0))
            present = int(self._tr_practice_panel_present.get(key, 0))
            hits = int(self._tr_practice_panel_hits.get(key, 0))
            acc = (correct / float(trials)) * 100.0
            if present > 0:
                lines.append(f"{label}: {correct}/{trials} ({acc:.0f}%)  Hits {hits}/{present}")
            else:
                lines.append(f"{label}: {correct}/{trials} ({acc:.0f}%)  Hits n/a")
        return tuple(lines)

    @staticmethod
    def _target_recognition_expected_panels(payload: TargetRecognitionPayload) -> set[str]:
        expected: set[str] = set()
        if payload.scene_has_target:
            expected.add("scene")
        if payload.light_has_target:
            expected.add("light")
        if payload.scan_has_target:
            expected.add("scan")
        if payload.system_has_target:
            expected.add("system")
        return expected

    def _target_recognition_system_view(
        self,
        payload: TargetRecognitionPayload,
    ) -> tuple[str, tuple[tuple[str, ...], ...], float]:
        self._target_recognition_sync_system_stream(payload)
        columns = self._tr_system_columns
        if not columns:
            return payload.system_target, (payload.system_rows,), 0.0

        row_offset = self._tr_system_row_offset
        step_frac = self._tr_system_row_frac
        rotated_cols: list[tuple[str, ...]] = []
        for col in columns:
            if not col:
                rotated_cols.append(())
                continue
            n = len(col)
            rotated = tuple(col[(row - row_offset) % n] for row in range(n))
            rotated_cols.append(rotated)
        return self._tr_system_target_code, tuple(rotated_cols), step_frac

    def _render_situational_awareness_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: SituationalAwarenessPayload | None,
    ) -> None:
        self._sa_option_hitboxes = {}
        self._sa_grid_hitboxes = {}

        bg = (6, 16, 86)
        panel_bg = (12, 28, 116)
        panel_dark = (10, 22, 96)
        border = (224, 236, 255)
        text_main = (238, 246, 255)
        text_muted = (182, 202, 230)
        text_dark = (12, 20, 36)

        w, h = surface.get_size()
        surface.fill(bg)

        margin = max(8, min(18, w // 48))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, panel_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(30, min(40, h // 16))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, panel_dark, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        title = self._tiny_font.render("Situational Awareness Test", True, text_main)
        surface.blit(title, title.get_rect(midleft=(header.x + 10, header.centery)))
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._tiny_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(midright=(header.right - 10, header.centery)))

        scored = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(scored, (frame.x + 12, header.bottom + 6))

        if payload is None:
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                pygame.Rect(
                    frame.x + 12, header.bottom + 30, frame.w - 24, frame.h - header_h - 42
                ),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
            return

        left_w = int(frame.w * 0.62)
        left_panel = pygame.Rect(
            frame.x + 10, header.bottom + 28, left_w - 16, frame.bottom - header.bottom - 38
        )
        right_panel = pygame.Rect(
            left_panel.right + 10,
            left_panel.y,
            frame.right - (left_panel.right + 20),
            left_panel.h,
        )
        pygame.draw.rect(surface, panel_dark, left_panel)
        pygame.draw.rect(surface, border, left_panel, 1)
        pygame.draw.rect(surface, panel_dark, right_panel)
        pygame.draw.rect(surface, border, right_panel, 1)

        # Grid view with live contact markers.
        grid_box = left_panel.inflate(-14, -14)
        grid_box_top = grid_box.y + 4
        label_pad = 20
        grid_size_px = min(grid_box.w - label_pad - 4, grid_box.h - label_pad - 4)
        cell_size = max(14, min(44, grid_size_px // 10))
        grid_w = cell_size * 10
        grid_h = cell_size * 10
        start_x = grid_box.x + label_pad + max(0, (grid_box.w - label_pad - grid_w) // 2)
        start_y = grid_box_top + label_pad + max(0, (grid_box.h - label_pad - grid_h) // 2)
        grid_rect = pygame.Rect(start_x, start_y, grid_w, grid_h)
        pygame.draw.rect(surface, (242, 248, 255), grid_rect)
        pygame.draw.rect(surface, (164, 182, 216), grid_rect, 1)

        selected_code: int | None = None
        if self._input.isdigit():
            try:
                selected_code = int(self._input)
            except ValueError:
                selected_code = None
        if selected_code is None:
            selected_code = int(self._math_choice)

        option_cell_by_label = {
            str(option.cell_label): int(option.code)
            for option in payload.options
            if option.cell_label is not None
        }
        contact_color = (
            (30, 81, 209),
            (22, 131, 77),
            (189, 83, 23),
            (142, 36, 170),
            (196, 56, 114),
        )
        heading_vectors = {
            "N": (0, -1),
            "NE": (1, -1),
            "E": (1, 0),
            "SE": (1, 1),
            "S": (0, 1),
            "SW": (-1, 1),
            "W": (-1, 0),
            "NW": (-1, -1),
        }

        for x in range(10):
            col_text = self._tiny_font.render(str(x), True, text_muted)
            cx = grid_rect.x + (x * cell_size) + (cell_size // 2)
            surface.blit(col_text, col_text.get_rect(center=(cx, start_y - 10)))
        for y in range(10):
            row_text = self._tiny_font.render(chr(ord("A") + y), True, text_muted)
            cy = grid_rect.y + (y * cell_size) + (cell_size // 2)
            surface.blit(row_text, row_text.get_rect(center=(start_x - 10, cy)))

        for y in range(10):
            for x in range(10):
                cell = pygame.Rect(
                    grid_rect.x + x * cell_size, grid_rect.y + y * cell_size, cell_size, cell_size
                )
                cell_label = cell_label_from_xy(x, y)
                shade = (247, 251, 255) if ((x + y) % 2 == 0) else (237, 244, 255)
                pygame.draw.rect(surface, shade, cell)
                pygame.draw.rect(surface, (176, 194, 224), cell, 1)

                if payload.kind is SituationalAwarenessQuestionKind.POSITION_PROJECTION:
                    option_code = option_cell_by_label.get(cell_label)
                    if option_code is not None:
                        glow = (255, 207, 92) if option_code == selected_code else (255, 228, 138)
                        pygame.draw.rect(surface, glow, cell, 3)
                        code_txt = self._tiny_font.render(
                            self._choice_key_label(option_code),
                            True,
                            (114, 68, 8),
                        )
                        surface.blit(code_txt, code_txt.get_rect(midtop=(cell.centerx, cell.y + 1)))
                        self._sa_grid_hitboxes[cell_label] = cell.copy()

        for idx, contact in enumerate(payload.contacts):
            cx = grid_rect.x + int(contact.x) * cell_size + (cell_size // 2)
            cy = grid_rect.y + int(contact.y) * cell_size + (cell_size // 2)
            color = contact_color[idx % len(contact_color)]
            radius = max(4, min(10, cell_size // 4))
            pygame.draw.circle(surface, color, (cx, cy), radius)
            pygame.draw.circle(surface, (0, 0, 0), (cx, cy), radius, 1)

            hd_dx, hd_dy = heading_vectors.get(str(contact.heading).upper(), (0, 0))
            if hd_dx != 0 or hd_dy != 0:
                line_len = max(5, cell_size // 3)
                end = (cx + int(hd_dx * line_len), cy + int(hd_dy * line_len))
                pygame.draw.line(surface, color, (cx, cy), end, 2)

            label = self._tiny_font.render(contact.callsign, True, text_dark)
            surface.blit(label, label.get_rect(midbottom=(cx, cy - radius - 1)))

        # Right panel: feed + options.
        info_box = pygame.Rect(right_panel.x + 8, right_panel.y + 8, right_panel.w - 16, 126)
        pygame.draw.rect(surface, (18, 36, 126), info_box)
        pygame.draw.rect(surface, border, info_box, 1)

        contacts_short = "  ".join(
            (
                f"{contact.callsign}@{chr(ord('A') + int(contact.y))}{int(contact.x)} "
                f"{contact.heading}{contact.speed_cells_per_min}"
            )
            for contact in payload.contacts[:4]
        )
        self._draw_wrapped_text(
            surface,
            f"Query: {payload.stem}",
            pygame.Rect(info_box.x + 8, info_box.y + 8, info_box.w - 16, 30),
            color=text_main,
            font=self._tiny_font,
            max_lines=2,
        )
        self._draw_wrapped_text(
            surface,
            f"Contacts: {contacts_short}",
            pygame.Rect(info_box.x + 8, info_box.y + 42, info_box.w - 16, 40),
            color=text_muted,
            font=self._tiny_font,
            max_lines=2,
        )
        self._draw_wrapped_text(
            surface,
            f"Horizon: +{payload.horizon_min} min",
            pygame.Rect(info_box.x + 8, info_box.bottom - 26, info_box.w - 16, 20),
            color=text_muted,
            font=self._tiny_font,
            max_lines=1,
        )

        option_box = pygame.Rect(
            right_panel.x + 8,
            info_box.bottom + 8,
            right_panel.w - 16,
            right_panel.h - 60 - info_box.h,
        )
        pygame.draw.rect(surface, (18, 36, 126), option_box)
        pygame.draw.rect(surface, border, option_box, 1)

        row_h = 34
        y = option_box.y + 8
        for option in payload.options:
            row = pygame.Rect(option_box.x + 8, y, option_box.w - 16, row_h)
            is_selected = int(option.code) == selected_code
            row_fill = (255, 236, 166) if is_selected else (238, 246, 255)
            row_border = (160, 128, 34) if is_selected else (176, 194, 224)
            pygame.draw.rect(surface, row_fill, row)
            pygame.draw.rect(surface, row_border, row, 1)
            self._sa_option_hitboxes[int(option.code)] = row.copy()

            label = self._fit_label(
                self._tiny_font,
                f"{self._choice_key_label(option.code)}  {option.text}",
                row.w - 10,
            )
            color = (70, 50, 0) if is_selected else text_dark
            row_text = self._tiny_font.render(label, True, color)
            surface.blit(row_text, (row.x + 6, row.y + 9))
            y += row_h + 6
            if y > option_box.bottom - row_h:
                break

        footer = pygame.Rect(frame.x + 10, frame.bottom - 28, frame.w - 20, 18)
        hint = "Click highlighted grid cells or options. A/S/D/F/G + Enter also works."
        if payload.kind is not SituationalAwarenessQuestionKind.POSITION_PROJECTION:
            hint = "Click an option or press A/S/D/F/G, then Enter."
        footer_text = self._tiny_font.render(hint, True, text_muted)
        surface.blit(footer_text, footer_text.get_rect(midleft=(footer.x + 2, footer.centery)))

    def _render_visual_search_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: VisualSearchPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (3, 8, 148)
        frame_border = (232, 240, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        answer_label = (242, 216, 74)
        tile_red = (230, 78, 72)
        tile_num = (70, 96, 188)

        surface.fill(bg)
        margin = max(8, min(18, w // 48))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header_h = max(28, min(36, h // 16))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, frame_border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        if snap.phase is Phase.INSTRUCTIONS:
            title_text = "Visual Search - Instructions"
        elif snap.phase in (Phase.PRACTICE, Phase.SCORED):
            title_text = "VISS"
        else:
            title_text = "Visual Search"
        title = self._tiny_font.render(title_text, True, text_main)
        surface.blit(title, title.get_rect(center=header.center))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            mm = rem // 60
            ss = rem % 60
            timer = self._tiny_font.render(f"{mm:02d}:{ss:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(midright=(header.right - 10, header.centery)))

        footer = pygame.Rect(frame.x + 1, frame.bottom - 28, frame.w - 2, 27)
        pygame.draw.rect(surface, (0, 0, 0), footer)
        pygame.draw.line(surface, frame_border, (footer.x, footer.y), (footer.right, footer.y), 1)

        board_rect = pygame.Rect(
            frame.x + 12, header.bottom + 10, frame.w - 24, footer.y - header.bottom - 22
        )
        board_payload = payload
        if board_payload is None and snap.phase is Phase.INSTRUCTIONS:
            board_payload = self._visual_search_demo_payload(VisualSearchTaskKind.ALPHANUMERIC)

        if board_payload is not None and snap.phase in (
            Phase.INSTRUCTIONS,
            Phase.PRACTICE,
            Phase.SCORED,
        ):
            self._draw_visual_search_board(
                surface,
                board_rect,
                board_payload,
                tile_red=tile_red,
                tile_num=tile_num,
            )
        else:
            panel = board_rect.inflate(-max(24, board_rect.w // 8), -max(18, board_rect.h // 10))
            pygame.draw.rect(surface, (12, 18, 106), panel)
            pygame.draw.rect(surface, frame_border, panel, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                panel.inflate(-16, -14),
                color=text_main,
                font=self._small_font,
                max_lines=14,
            )

        label = self._tiny_font.render("Answer", True, answer_label)
        answer_box = pygame.Rect(footer.centerx - 18, footer.y + 6, 36, 14)
        pygame.draw.rect(surface, (18, 18, 18), answer_box)
        pygame.draw.rect(surface, frame_border, answer_box, 1)
        surface.blit(label, label.get_rect(midright=(answer_box.x - 6, answer_box.centery)))

        entry_value = ""
        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            entry_value = self._input + caret
        entry = self._tiny_font.render(entry_value, True, text_main)
        surface.blit(entry, entry.get_rect(center=answer_box.center))

        if snap.phase is Phase.INSTRUCTIONS:
            hint = self._tiny_font.render("Press Enter to start practice", True, text_muted)
            surface.blit(hint, hint.get_rect(midleft=(answer_box.right + 14, answer_box.centery)))
        elif snap.phase in (Phase.PRACTICE, Phase.SCORED):
            hint = self._tiny_font.render("Enter the matching block number", True, text_muted)
            surface.blit(hint, hint.get_rect(midleft=(answer_box.right + 14, answer_box.centery)))

    @staticmethod
    def _visual_search_demo_payload(kind: VisualSearchTaskKind) -> VisualSearchPayload:
        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            cells = ("A", "F", "K", "R", "B", "S", "L", "H", "G", "P", "E", "G")
            target = "R"
        else:
            cells = (
                "X_MARK",
                "DOUBLE_CROSS",
                "S_BEND",
                "RING_SPOKE",
                "L_HOOK",
                "BOX",
                "STAR",
                "BOLT",
                "TRIANGLE",
                "LOLLIPOP",
                "PIN",
                "FORK",
            )
            target = "L_HOOK"
        return VisualSearchPayload(
            kind=kind,
            rows=3,
            cols=4,
            target=target,
            cells=cells,
            cell_codes=tuple(range(10, 22)),
            full_credit_error=0,
            zero_credit_error=1,
        )

    def _draw_visual_search_board(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        payload: VisualSearchPayload,
        *,
        tile_red: tuple[int, int, int],
        tile_num: tuple[int, int, int],
    ) -> None:
        rows = max(1, int(payload.rows))
        cols = max(1, int(payload.cols))
        gap_x = max(8, min(16, rect.w // 42))
        gap_y = max(8, min(16, rect.h // 28))
        tile_size = min(
            max(36, (rect.w - gap_x * (cols + 1)) // cols),
            max(36, (rect.h - gap_y * (rows + 4)) // (rows + 1)),
        )
        grid_w = cols * tile_size + (cols - 1) * gap_x
        grid_h = rows * tile_size + (rows - 1) * gap_y
        target_gap = max(20, min(44, tile_size // 2))
        total_h = grid_h + target_gap + tile_size
        start_x = rect.x + max(0, (rect.w - grid_w) // 2)
        start_y = rect.y + max(0, (rect.h - total_h) // 2)

        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cell = pygame.Rect(
                    start_x + c * (tile_size + gap_x),
                    start_y + r * (tile_size + gap_y),
                    tile_size,
                    tile_size,
                )
                token = payload.cells[idx] if idx < len(payload.cells) else ""
                code = payload.cell_codes[idx] if idx < len(payload.cell_codes) else 0
                self._draw_visual_search_tile(
                    surface,
                    cell,
                    token=str(token),
                    code_text=str(code),
                    kind=payload.kind,
                    tile_red=tile_red,
                    tile_num=tile_num,
                )

        target_rect = pygame.Rect(
            rect.centerx - (tile_size // 2),
            start_y + grid_h + target_gap,
            tile_size,
            tile_size,
        )
        self._draw_visual_search_tile(
            surface,
            target_rect,
            token=str(payload.target),
            code_text="??",
            kind=payload.kind,
            tile_red=tile_red,
            tile_num=tile_num,
        )

    def _draw_visual_search_tile(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        token: str,
        code_text: str,
        kind: VisualSearchTaskKind,
        tile_red: tuple[int, int, int],
        tile_num: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, (246, 247, 252), rect)
        pygame.draw.rect(surface, (164, 180, 222), rect, 1)

        content_rect = pygame.Rect(
            rect.x + 4,
            rect.y + 4,
            rect.w - 8,
            max(12, int(rect.h * 0.62)),
        )
        code_rect = pygame.Rect(
            rect.x + 2,
            content_rect.bottom,
            rect.w - 4,
            rect.bottom - content_rect.bottom - 2,
        )

        if kind is VisualSearchTaskKind.ALPHANUMERIC:
            letter_font = pygame.font.Font(None, max(22, min(48, int(rect.h * 0.74))))
            glyph = letter_font.render(token[:1], True, tile_red)
            surface.blit(
                glyph,
                glyph.get_rect(center=(content_rect.centerx, content_rect.centery + 1)),
            )
        else:
            self._draw_visual_search_symbol(surface, content_rect, token, tile_red)

        code_font = pygame.font.Font(None, max(14, min(22, int(rect.h * 0.26))))
        code = code_font.render(code_text, True, tile_num)
        surface.blit(code, code.get_rect(center=code_rect.center))

    def _draw_visual_search_symbol(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        token: str,
        color: tuple[int, int, int],
    ) -> None:
        left = rect.x + 4
        right = rect.right - 4
        top = rect.y + 4
        bottom = rect.bottom - 4
        mid_x = (left + right) // 2
        mid_y = (top + bottom) // 2
        lw = max(2, min(4, rect.w // 10))

        if token == "X_MARK":
            pygame.draw.line(surface, color, (left, top), (right, bottom), lw)
            pygame.draw.line(surface, color, (right, top), (left, bottom), lw)
        elif token == "DOUBLE_CROSS":
            pygame.draw.line(surface, color, (mid_x, top), (mid_x, bottom), lw)
            pygame.draw.line(surface, color, (left, mid_y - 4), (right, mid_y - 4), lw)
            pygame.draw.line(surface, color, (left + 6, mid_y + 6), (right - 6, mid_y + 6), lw)
        elif token == "S_BEND":
            points = [
                (right - 4, top + 4),
                (mid_x + 2, top),
                (mid_x - 6, mid_y - 2),
                (right - 6, mid_y + 2),
                (mid_x + 2, bottom),
                (left + 4, bottom - 4),
            ]
            pygame.draw.lines(surface, color, False, points, lw)
        elif token == "RING_SPOKE":
            radius = max(6, min(rect.w, rect.h) // 4)
            pygame.draw.circle(surface, color, (mid_x, mid_y), radius, lw)
            pygame.draw.line(
                surface,
                color,
                (mid_x + radius - 2, mid_y - radius + 2),
                (right, top),
                lw,
            )
        elif token == "L_HOOK":
            pygame.draw.line(surface, color, (mid_x, top), (mid_x, bottom - 4), lw)
            pygame.draw.line(surface, color, (mid_x, bottom - 4), (left + 4, bottom - 4), lw)
        elif token == "BOX":
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(left + 4, top + 4, rect.w - 16, rect.h - 16),
                lw,
            )
        elif token == "STAR":
            pygame.draw.line(surface, color, (left, mid_y), (right, mid_y), lw)
            pygame.draw.line(surface, color, (mid_x, top), (mid_x, bottom), lw)
            pygame.draw.line(surface, color, (left + 2, top + 2), (right - 2, bottom - 2), lw)
            pygame.draw.line(surface, color, (right - 2, top + 2), (left + 2, bottom - 2), lw)
        elif token == "TRIANGLE":
            pygame.draw.polygon(
                surface,
                color,
                [(mid_x, top), (right - 2, bottom - 2), (left + 2, bottom - 2)],
                lw,
            )
        elif token == "LOLLIPOP":
            radius = max(5, min(rect.w, rect.h) // 6)
            pygame.draw.circle(surface, color, (mid_x, top + radius + 2), radius, lw)
            pygame.draw.line(surface, color, (mid_x, top + radius * 2 + 2), (mid_x, bottom), lw)
        elif token == "PIN":
            pygame.draw.circle(surface, color, (mid_x, top + 8), max(4, rect.w // 10), lw)
            pygame.draw.line(surface, color, (mid_x, top + 12), (mid_x, bottom), lw)
        elif token == "BOLT":
            points = [
                (left + 8, top + 2),
                (mid_x, top + 2),
                (mid_x - 4, mid_y),
                (right - 8, mid_y),
                (mid_x + 2, bottom - 2),
                (mid_x + 4, mid_y + 2),
            ]
            pygame.draw.lines(surface, color, False, points, lw)
        elif token == "FORK":
            pygame.draw.line(surface, color, (mid_x, top + 8), (mid_x, bottom), lw)
            pygame.draw.line(surface, color, (mid_x, top + 8), (left + 4, top), lw)
            pygame.draw.line(surface, color, (mid_x, top + 8), (right - 4, top), lw)
        else:
            fallback_font = pygame.font.Font(None, max(16, min(32, int(rect.h * 0.52))))
            glyph = fallback_font.render(token[:2], True, color)
            surface.blit(glyph, glyph.get_rect(center=rect.center))

    def _render_vigilance_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: VigilancePayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (3, 8, 148)
        frame_border = (232, 240, 255)
        text_main = (238, 245, 255)
        text_muted = (188, 204, 228)
        answer_label = (242, 216, 74)

        self._vigilance_row_hitbox = None
        self._vigilance_col_hitbox = None

        surface.fill(bg)
        margin = max(8, min(18, w // 48))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, frame_border, frame, 1)

        header_h = max(28, min(36, h // 16))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, frame_border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Scored",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        title = self._tiny_font.render(f"Vigilance - {phase_label}", True, text_main)
        surface.blit(title, title.get_rect(center=header.center))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._tiny_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(midright=(header.right - 10, header.centery)))

        footer = pygame.Rect(frame.x + 1, frame.bottom - 28, frame.w - 2, 27)
        pygame.draw.rect(surface, (0, 0, 0), footer)
        pygame.draw.line(surface, frame_border, (footer.x, footer.y), (footer.right, footer.y), 1)

        work_rect = pygame.Rect(
            frame.x + 10,
            header.bottom + 8,
            frame.w - 20,
            footer.y - header.bottom - 16,
        )
        pygame.draw.rect(surface, (7, 14, 118), work_rect)
        pygame.draw.rect(surface, frame_border, work_rect, 1)

        if payload is not None:
            self._draw_vigilance_board(surface, work_rect, payload)

        stats = self._tiny_font.render(
            f"Captures {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_main,
        )
        surface.blit(stats, (footer.x + 8, footer.y + 7))
        if payload is not None:
            points = self._tiny_font.render(f"Points {payload.points_total}", True, answer_label)
            surface.blit(points, points.get_rect(midright=(footer.right - 8, footer.centery)))

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            row_box = pygame.Rect(footer.centerx - 112, footer.y + 5, 52, 16)
            col_box = pygame.Rect(footer.centerx + 18, footer.y + 5, 52, 16)
            self._vigilance_row_hitbox = row_box.copy()
            self._vigilance_col_hitbox = col_box.copy()

            row_active = self._vigilance_focus == "row"
            col_active = self._vigilance_focus == "col"
            row_fill = (246, 232, 154) if row_active else (18, 18, 18)
            col_fill = (246, 232, 154) if col_active else (18, 18, 18)
            row_color = (36, 24, 0) if row_active else text_main
            col_color = (36, 24, 0) if col_active else text_main

            pygame.draw.rect(surface, row_fill, row_box)
            pygame.draw.rect(surface, frame_border, row_box, 1)
            pygame.draw.rect(surface, col_fill, col_box)
            pygame.draw.rect(surface, frame_border, col_box, 1)

            row_label = self._tiny_font.render("Row", True, answer_label)
            col_label = self._tiny_font.render("Col", True, answer_label)
            surface.blit(row_label, row_label.get_rect(midright=(row_box.x - 6, row_box.centery)))
            surface.blit(col_label, col_label.get_rect(midright=(col_box.x - 6, col_box.centery)))

            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            row_value = self._vigilance_row_input + (caret if row_active else "")
            col_value = self._vigilance_col_input + (caret if col_active else "")
            row_text = self._tiny_font.render(row_value, True, row_color)
            col_text = self._tiny_font.render(col_value, True, col_color)
            surface.blit(row_text, row_text.get_rect(center=row_box.center))
            surface.blit(col_text, col_text.get_rect(center=col_box.center))

            hint = self._tiny_font.render(
                "Click Row/Col or use arrows. Type 1-9 (auto-submit on both).",
                True,
                text_muted,
            )
            surface.blit(hint, hint.get_rect(midleft=(col_box.right + 16, col_box.centery)))
        elif snap.phase is Phase.INSTRUCTIONS:
            hint = self._tiny_font.render("Press Enter to start practice", True, text_muted)
            surface.blit(hint, hint.get_rect(midleft=(footer.centerx - 80, footer.centery)))
        elif snap.phase is Phase.PRACTICE_DONE:
            hint = self._tiny_font.render("Press Enter to start scored block", True, text_muted)
            surface.blit(hint, hint.get_rect(midleft=(footer.centerx - 90, footer.centery)))
        elif snap.phase is Phase.RESULTS:
            hint = self._tiny_font.render("Enter: Return to Tests", True, text_muted)
            surface.blit(hint, hint.get_rect(midleft=(footer.centerx - 50, footer.centery)))

    def _draw_vigilance_board(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        payload: VigilancePayload,
    ) -> None:
        rows = max(1, int(payload.rows))
        cols = max(1, int(payload.cols))

        panel_w = max(170, min(280, rect.w // 3))
        gap = max(10, min(28, rect.w // 36))
        matrix_w = max(120, rect.w - panel_w - gap)
        grid_side = min(max(120, matrix_w), max(120, rect.h - 18))

        total_w = grid_side + gap + panel_w
        start_x = rect.x + max(0, (rect.w - total_w) // 2)
        grid_rect = pygame.Rect(start_x, rect.y + max(0, (rect.h - grid_side) // 2), grid_side, grid_side)
        legend_rect = pygame.Rect(
            grid_rect.right + gap,
            grid_rect.y + 4,
            panel_w,
            grid_rect.h - 8,
        )

        pygame.draw.rect(surface, (8, 14, 112), grid_rect)
        pygame.draw.rect(surface, (220, 232, 255), grid_rect, 1)

        cell_size = max(10, min(grid_rect.w // cols, grid_rect.h // rows))
        grid_w = cell_size * cols
        grid_h = cell_size * rows
        board = pygame.Rect(0, 0, grid_w, grid_h)
        board.center = grid_rect.center
        board.x += max(8, min(18, grid_rect.w // 15))
        if board.right > grid_rect.right - 4:
            board.right = grid_rect.right - 4

        label_color = (204, 218, 248)
        for c in range(cols):
            txt = self._tiny_font.render(str(c + 1), True, label_color)
            x = board.x + (c * cell_size) + (cell_size // 2)
            surface.blit(txt, txt.get_rect(center=(x, board.y - 10)))
        for r in range(rows):
            txt = self._tiny_font.render(str(r + 1), True, label_color)
            y = board.y + (r * cell_size) + (cell_size // 2)
            surface.blit(txt, txt.get_rect(center=(board.x - 12, y)))

        for r in range(rows + 1):
            y = board.y + (r * cell_size)
            pygame.draw.line(surface, (198, 210, 244), (board.x, y), (board.right, y), 1)
        for c in range(cols + 1):
            x = board.x + (c * cell_size)
            pygame.draw.line(surface, (198, 210, 244), (x, board.y), (x, board.bottom), 1)

        for symbol in payload.symbols:
            cx = board.x + ((int(symbol.col) - 1) * cell_size) + (cell_size // 2)
            cy = board.y + ((int(symbol.row) - 1) * cell_size) + (cell_size // 2)
            self._draw_vigilance_symbol(
                surface=surface,
                rect=pygame.Rect(
                    cx - max(3, int(cell_size * 0.24)),
                    cy - max(3, int(cell_size * 0.24)),
                    max(6, int(cell_size * 0.48)),
                    max(6, int(cell_size * 0.48)),
                ),
                symbol_kind=symbol.kind,
            )

        pygame.draw.rect(surface, (5, 12, 102), legend_rect)
        pygame.draw.rect(surface, (220, 232, 255), legend_rect, 1)

        title = self._tiny_font.render("Symbol Points", True, (238, 245, 255))
        surface.blit(title, title.get_rect(midtop=(legend_rect.centerx, legend_rect.y + 8)))

        row_h = max(28, min(34, legend_rect.h // 8))
        y = legend_rect.y + 30
        for kind, points in payload.legend:
            row = pygame.Rect(legend_rect.x + 10, y, legend_rect.w - 20, row_h)
            pygame.draw.rect(surface, (10, 24, 120), row)
            pygame.draw.rect(surface, (146, 174, 236), row, 1)

            icon_rect = pygame.Rect(row.x + 8, row.y + 4, row_h - 8, row_h - 8)
            self._draw_vigilance_symbol(surface=surface, rect=icon_rect, symbol_kind=kind)

            rarity = {
                VigilanceSymbolKind.STAR: "common",
                VigilanceSymbolKind.DIAMOND: "less common",
                VigilanceSymbolKind.TRIANGLE: "rare",
                VigilanceSymbolKind.HEXAGON: "very rare",
            }.get(kind, "")
            label = self._tiny_font.render(f"{points} pt ({rarity})", True, (228, 240, 255))
            surface.blit(label, (icon_rect.right + 8, row.y + 7))
            y += row_h + 6

        stats = [
            f"Points: {payload.points_total}",
            f"Captured: {payload.captured_total}",
            f"Missed: {payload.missed_total}",
            f"Visible: {len(payload.symbols)}",
        ]
        stats_y = min(legend_rect.bottom - 72, y + 4)
        for line in stats:
            txt = self._tiny_font.render(line, True, (198, 216, 246))
            surface.blit(txt, (legend_rect.x + 12, stats_y))
            stats_y += 16

    def _draw_vigilance_symbol(
        self,
        *,
        surface: pygame.Surface,
        rect: pygame.Rect,
        symbol_kind: VigilanceSymbolKind,
    ) -> None:
        cx, cy = rect.center
        radius = max(3, min(rect.w, rect.h) // 2 - 1)
        if symbol_kind is VigilanceSymbolKind.STAR:
            self._draw_vigilance_star(surface, center=(cx, cy), radius=radius, color=(246, 248, 255))
            return
        if symbol_kind is VigilanceSymbolKind.DIAMOND:
            pygame.draw.polygon(
                surface,
                (255, 230, 166),
                ((cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)),
            )
            return
        if symbol_kind is VigilanceSymbolKind.TRIANGLE:
            pygame.draw.polygon(
                surface,
                (255, 214, 130),
                ((cx, cy - radius), (cx + radius, cy + radius), (cx - radius, cy + radius)),
            )
            return

        points: list[tuple[float, float]] = []
        for i in range(6):
            angle = (-math.pi / 2.0) + (i * (math.pi / 3.0))
            points.append((cx + (math.cos(angle) * radius), cy + (math.sin(angle) * radius)))
        pygame.draw.polygon(surface, (255, 198, 108), points)

    @staticmethod
    def _draw_vigilance_star(
        surface: pygame.Surface,
        *,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        outer = max(3, int(radius))
        inner = max(2, int(round(outer * 0.46)))
        points: list[tuple[float, float]] = []
        for i in range(10):
            angle = (-math.pi / 2.0) + (i * (math.pi / 5.0))
            rr = outer if i % 2 == 0 else inner
            points.append((cx + (math.cos(angle) * rr), cy + (math.sin(angle) * rr)))
        pygame.draw.polygon(surface, color, points)

    def _render_trace_test_1_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: TraceTest1Payload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 10, 140)
        frame_bg = (8, 18, 124)
        panel_bg = (12, 20, 92)
        border = (226, 236, 255)
        text_main = (236, 244, 255)
        text_muted = (176, 196, 226)
        input_bg = (4, 10, 36)

        surface.fill(bg)

        margin = max(10, min(18, w // 46))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 17))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Scored",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        title = self._tiny_font.render(f"Trace Test 1 - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 10, header.y + 7))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        work = pygame.Rect(
            frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16
        )
        pygame.draw.rect(surface, panel_bg, work)
        pygame.draw.rect(surface, border, work, 1)

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED) or payload is None:
            info = work.inflate(-20, -20)
            pygame.draw.rect(surface, (18, 28, 108), info)
            pygame.draw.rect(surface, border, info, 1)

            scene = pygame.Rect(info.x + 12, info.y + 12, info.w - 24, max(160, int(info.h * 0.56)))
            self._draw_trace_test_1_scene(
                surface,
                scene,
                reference=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0),
                candidate=TraceTest1Attitude(roll_deg=0.0, pitch_deg=0.0, yaw_deg=270.0),
                correct_code=1,
                viewpoint_bearing_deg=180,
                scene_turn_index=0,
                animate=False,
                motion_progress=0.0,
            )

            text_rect = pygame.Rect(
                info.x + 12, scene.bottom + 10, info.w - 24, info.bottom - scene.bottom - 22
            )
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                text_rect,
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
            return

        scene_h = int(work.h * 0.68)
        scene = pygame.Rect(work.x + 8, work.y + 8, work.w - 16, scene_h - 12)
        footer = pygame.Rect(work.x + 8, scene.bottom + 8, work.w - 16, work.bottom - scene.bottom - 16)

        self._draw_trace_test_1_scene(
            surface,
            scene,
            reference=payload.reference,
            candidate=payload.candidate,
            correct_code=int(payload.correct_code),
            viewpoint_bearing_deg=int(payload.viewpoint_bearing_deg),
            scene_turn_index=int(payload.scene_turn_index),
            animate=True,
            motion_progress=float(payload.observe_progress),
        )

        tag = pygame.Rect(scene.x + 12, scene.y + 12, 148, 28)
        pygame.draw.rect(surface, (78, 10, 18), tag)
        pygame.draw.rect(surface, (255, 216, 220), tag, 1)
        label = self._tiny_font.render("RED AIRCRAFT = TARGET", True, (255, 234, 236))
        surface.blit(label, label.get_rect(center=tag.center))

        pygame.draw.rect(surface, (18, 28, 108), footer)
        pygame.draw.rect(surface, border, footer, 1)

        prompt_box = pygame.Rect(footer.x + 8, footer.y + 8, footer.w - 16, 48)
        pygame.draw.rect(surface, (14, 20, 88), prompt_box)
        pygame.draw.rect(surface, border, prompt_box, 1)
        self._draw_wrapped_text(
            surface,
            "Which way did the stick move for the red aircraft?",
            prompt_box.inflate(-8, -8),
            color=text_main,
            font=self._tiny_font,
            max_lines=2,
        )

        pad = pygame.Rect(footer.x + 8, prompt_box.bottom + 8, footer.w - 16, footer.h - 58)
        pygame.draw.rect(surface, (14, 20, 88), pad)
        pygame.draw.rect(surface, border, pad, 1)

        card_w = max(86, min(132, pad.w // 4))
        card_h = max(38, min(52, pad.h // 2))
        up_rect = pygame.Rect(0, 0, card_w, card_h)
        up_rect.midtop = (pad.centerx, pad.y + 10)
        left_rect = pygame.Rect(0, 0, card_w, card_h)
        left_rect.midleft = (pad.x + 14, pad.centery + 8)
        right_rect = pygame.Rect(0, 0, card_w, card_h)
        right_rect.midright = (pad.right - 14, pad.centery + 8)
        down_rect = pygame.Rect(0, 0, card_w, card_h)
        down_rect.midbottom = (pad.centerx, pad.bottom - 10)

        cards = (
            (up_rect, "UP", "Push"),
            (left_rect, "LEFT", "Left"),
            (right_rect, "RIGHT", "Right"),
            (down_rect, "DOWN", "Pull"),
        )
        for card, key_name, label_text in cards:
            pygame.draw.rect(surface, (28, 42, 122), card)
            pygame.draw.rect(surface, border, card, 1)
            key_txt = self._tiny_font.render(key_name, True, text_main)
            label_txt = self._tiny_font.render(label_text, True, text_muted)
            surface.blit(key_txt, key_txt.get_rect(midtop=(card.centerx, card.y + 6)))
            surface.blit(label_txt, label_txt.get_rect(midbottom=(card.centerx, card.bottom - 6)))

        entry_card = pygame.Rect(footer.x + 8, footer.bottom - 34, footer.w - 16, 26)
        pygame.draw.rect(surface, (14, 20, 88), entry_card)
        pygame.draw.rect(surface, border, entry_card, 1)
        hint = self._tiny_font.render(
            "Answer as soon as the turn starts; maneuver keeps running.",
            True,
            text_muted,
        )
        surface.blit(hint, (entry_card.x + 8, entry_card.y + 10))

    def _draw_trace_test_1_scene(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        reference: TraceTest1Attitude,
        candidate: TraceTest1Attitude,
        correct_code: int,
        viewpoint_bearing_deg: int,
        scene_turn_index: int = 0,
        animate: bool = True,
        motion_progress: float | None = None,
    ) -> None:
        sky_top = (82, 124, 196)
        sky_bottom = (238, 170, 142)
        ground_top = (136, 152, 176)
        ground_bottom = (112, 126, 146)
        border = (208, 222, 248)

        pygame.draw.rect(surface, (8, 14, 62), rect)
        pygame.draw.rect(surface, border, rect, 1)
        inner = rect.inflate(-8, -8)
        if inner.w <= 0 or inner.h <= 0:
            return
        if self._render_trace_test_1_panda_view(
            surface=surface,
            world=inner,
            reference=reference,
            candidate=candidate,
            correct_code=correct_code,
            viewpoint_bearing_deg=viewpoint_bearing_deg,
            scene_turn_index=scene_turn_index,
            animate=animate,
            motion_progress=motion_progress,
        ):
            return

        anim_s = float(pygame.time.get_ticks()) / 1000.0 if animate else 0.0
        scene_progress = (
            max(0.0, min(1.0, float(motion_progress)))
            if motion_progress is not None
            else 0.5 + (0.5 * math.sin(anim_s * 0.55))
        )
        horizon = inner.y + int(inner.h * 0.58)
        sky_h = max(1, horizon - inner.y)
        for i in range(sky_h):
            t = i / max(1, sky_h - 1)
            color = (
                int(round(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t)),
                int(round(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t)),
                int(round(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t)),
            )
            pygame.draw.line(surface, color, (inner.x, inner.y + i), (inner.right, inner.y + i))

        ground_h = max(1, inner.bottom - horizon)
        for i in range(ground_h):
            t = i / max(1, ground_h - 1)
            color = (
                int(round(ground_top[0] + (ground_bottom[0] - ground_top[0]) * t)),
                int(round(ground_top[1] + (ground_bottom[1] - ground_top[1]) * t)),
                int(round(ground_top[2] + (ground_bottom[2] - ground_top[2]) * t)),
            )
            pygame.draw.line(surface, color, (inner.x, horizon + i), (inner.right, horizon + i))

        # Keep the camera view stable; only the aircraft should move.
        for idx in range(4):
            cloud_w = max(26, int(inner.w * (0.12 + (idx * 0.02))))
            cloud_h = max(10, int(cloud_w * 0.34))
            cx = inner.x + int(inner.w * (0.08 + (idx * 0.19)))
            cy = inner.y + int(inner.h * (0.09 + (idx * 0.08)))
            cloud_rect = pygame.Rect(cx, cy, cloud_w, cloud_h)
            pygame.draw.ellipse(surface, (252, 246, 238), cloud_rect)
            pygame.draw.ellipse(surface, (218, 226, 242), cloud_rect, 1)

        for idx in range(6):
            t = idx / 5.0
            yy = horizon + int(round((t**1.18) * max(1, inner.bottom - horizon)))
            left_x = inner.x + int(round(inner.w * (0.02 + (t * 0.04))))
            right_x = inner.right - int(round(inner.w * (0.02 + (t * 0.01))))
            pygame.draw.line(surface, (182, 194, 214), (left_x, yy), (right_x, yy), 1)
        for lane in (-2, -1, 0, 1, 2):
            xx = inner.centerx + int(round(lane * inner.w * 0.15))
            pygame.draw.line(surface, (168, 182, 204), (xx, horizon + 2), (xx, inner.bottom), 1)

        target_frame, distractor_frames = trace_test_1_scene_frames(
            reference=reference,
            candidate=candidate,
            correct_code=correct_code,
            progress=scene_progress,
            scene_turn_index=scene_turn_index,
        )
        future_target_frame, future_distractor_frames = trace_test_1_scene_frames(
            reference=reference,
            candidate=candidate,
            correct_code=correct_code,
            progress=min(1.0, scene_progress + 0.03),
            scene_turn_index=scene_turn_index,
        )
        anchor_x, anchor_y, anchor_z = target_frame.position

        def project_scene_frame(position: tuple[float, float, float]) -> tuple[tuple[int, int], float]:
            x, forward_y, altitude_z = position
            rel_x = x - anchor_x
            rel_forward = forward_y - anchor_y
            rel_alt = altitude_z - anchor_z
            forward_scale_x = max(2.0, inner.w / 176.0)
            depth_parallax_x = max(0.12, inner.w / 760.0)
            altitude_scale = max(1.45, inner.h / 84.0)
            center = (
                inner.centerx
                + int(round((rel_forward * forward_scale_x) + (rel_x * depth_parallax_x))),
                horizon
                - int(round(rel_alt * altitude_scale)),
            )
            scale = _clamp(0.82 - (rel_x / 210.0), 0.54, 1.20)
            return center, scale

        def frame_screen_heading(
            frame: TraceTest1SceneFrame,
            future_frame: TraceTest1SceneFrame,
            center: tuple[int, int],
        ) -> float:
            _ = future_frame
            heading_rad = math.radians(float(frame.travel_heading_deg))
            ahead_world = (
                frame.position[0] + (math.sin(heading_rad) * 5.4),
                frame.position[1] + (math.cos(heading_rad) * 5.4),
                frame.position[2],
            )
            ahead_center, _ = project_scene_frame(ahead_world)
            dx = float(ahead_center[0] - center[0])
            dy = float(ahead_center[1] - center[1])
            if abs(dx) + abs(dy) < 0.5:
                return 0.0
            return float(math.degrees(math.atan2(dy, dx)))

        distractor_specs = (
            ((122, 148, 184), (220, 232, 246)),
            ((108, 132, 150), (210, 220, 232)),
            ((96, 126, 168), (206, 220, 244)),
        )
        for frame, future_frame, spec in zip(
            distractor_frames,
            future_distractor_frames,
            distractor_specs,
            strict=False,
        ):
            color, outline = spec
            center, scale = project_scene_frame(frame.position)
            self._draw_trace_test_1_aircraft(
                surface,
                center=center,
                attitude=frame.attitude,
                screen_heading_deg=frame_screen_heading(frame, future_frame, center),
                color=color,
                outline=outline,
                scale=scale,
                anim_s=anim_s,
            )

        target_center, target_scale = project_scene_frame(target_frame.position)
        self._draw_trace_test_1_aircraft(
            surface,
            center=target_center,
            attitude=target_frame.attitude,
            screen_heading_deg=frame_screen_heading(
                target_frame,
                future_target_frame,
                target_center,
            ),
            color=(228, 54, 56),
            outline=(255, 210, 206),
            scale=max(0.68, min(1.28, target_scale)),
            anim_s=anim_s,
        )

    def _draw_trace_test_1_aircraft(
        self,
        surface: pygame.Surface,
        *,
        center: tuple[int, int],
        attitude: TraceTest1Attitude,
        screen_heading_deg: float,
        color: tuple[int, int, int],
        outline: tuple[int, int, int],
        scale: float,
        anim_s: float,
    ) -> None:
        cx, cy = center
        heading_rad = math.radians(float(screen_heading_deg))
        forward = (math.cos(heading_rad), math.sin(heading_rad))
        right_axis = (-forward[1], forward[0])
        body_len = max(16.0, 27.0 * scale * (1.0 - (abs(attitude.pitch_deg) / 180.0)))
        wing_span = max(12.0, 22.0 * scale * (1.0 - (abs(attitude.pitch_deg) / 105.0)))
        bank_shift = math.sin(math.radians(float(attitude.roll_deg))) * scale * 5.5
        tail_span = max(6.0, wing_span * 0.32)
        nose = (
            int(round(cx + (forward[0] * body_len * 0.86))),
            int(round(cy + (forward[1] * body_len * 0.86))),
        )
        tail = (
            int(round(cx - (forward[0] * body_len * 0.76))),
            int(round(cy - (forward[1] * body_len * 0.76))),
        )
        wing_root = (
            int(round(cx - (forward[0] * body_len * 0.06))),
            int(round(cy - (forward[1] * body_len * 0.06))),
        )
        left = (
            int(round(wing_root[0] - (right_axis[0] * wing_span) + (forward[0] * bank_shift))),
            int(round(wing_root[1] - (right_axis[1] * wing_span) + (forward[1] * bank_shift))),
        )
        right = (
            int(round(wing_root[0] + (right_axis[0] * wing_span) - (forward[0] * bank_shift))),
            int(round(wing_root[1] + (right_axis[1] * wing_span) - (forward[1] * bank_shift))),
        )

        body_mid = ((nose[0] + tail[0]) // 2, (nose[1] + tail[1]) // 2)
        tail_root = (
            int(round((nose[0] * 0.14) + (tail[0] * 0.86))),
            int(round((nose[1] * 0.14) + (tail[1] * 0.86))),
        )
        tail_left = (
            int(round(tail_root[0] - (right_axis[0] * tail_span))),
            int(round(tail_root[1] - (right_axis[1] * tail_span))),
        )
        tail_right = (
            int(round(tail_root[0] + (right_axis[0] * tail_span))),
            int(round(tail_root[1] + (right_axis[1] * tail_span))),
        )

        shadow_offset = (2, 3)
        fuselage = (
            (nose[0], nose[1]),
            (wing_root[0], wing_root[1]),
            (tail[0], tail[1]),
            (wing_root[0], wing_root[1]),
        )
        wing = (left, wing_root, right)
        tail_plane = (tail_left, tail_root, tail_right)

        shadow_color = (16, 18, 28)
        fuselage_shadow = tuple((x + shadow_offset[0], y + shadow_offset[1]) for x, y in fuselage)
        wing_shadow = tuple((x + shadow_offset[0], y + shadow_offset[1]) for x, y in wing)
        tail_shadow = tuple((x + shadow_offset[0], y + shadow_offset[1]) for x, y in tail_plane)
        pygame.draw.polygon(surface, shadow_color, wing_shadow)
        pygame.draw.polygon(surface, shadow_color, tail_shadow)
        pygame.draw.lines(surface, shadow_color, False, fuselage_shadow, 4)

        pygame.draw.polygon(surface, color, wing)
        pygame.draw.polygon(surface, color, tail_plane)
        pygame.draw.lines(surface, color, False, fuselage, 4)
        pygame.draw.polygon(surface, outline, wing, 1)
        pygame.draw.polygon(surface, outline, tail_plane, 1)
        pygame.draw.lines(surface, outline, False, fuselage, 1)
        pygame.draw.circle(surface, outline, body_mid, 2)

        prop_r = max(4, int(round(scale * 4.6)))
        prop_phase = anim_s * 18.0
        prop_dx = int(round(math.cos(prop_phase) * prop_r))
        prop_dy = int(round(math.sin(prop_phase) * prop_r))
        pygame.draw.line(
            surface,
            (255, 245, 186),
            (nose[0] - prop_dx, nose[1] - prop_dy),
            (nose[0] + prop_dx, nose[1] + prop_dy),
            1,
        )
        pygame.draw.circle(surface, (255, 245, 186), nose, 2)

    @staticmethod
    def _trace_test_2_track_position(
        *,
        track: TraceTest2AircraftTrack,
        progress: float,
    ) -> TraceTest2Point3:
        return trace_test_2_track_position(track=track, progress=progress)

    def _draw_trace_test_2_aircraft_fallback(
        self,
        surface: pygame.Surface,
        *,
        center: tuple[int, int],
        color: tuple[int, int, int],
        scale: float,
        heading_deg: float,
        bank_deg: float,
    ) -> None:
        tx, ty = center
        heading_rad = math.radians(float(heading_deg))
        forward = (math.cos(heading_rad), math.sin(heading_rad))
        right_axis = (-forward[1], forward[0])
        span = max(10.0, 18.0 * scale)
        body = max(8.0, 20.0 * scale)
        tail = max(5.0, 7.0 * scale)
        bank_shift = math.sin(math.radians(bank_deg)) * 3.5 * scale

        def point(px: float, py: float) -> tuple[int, int]:
            return (int(round(tx + px)), int(round(ty + py)))

        nose = point(forward[0] * body, forward[1] * body)
        left = point(
            (-forward[0] * (body * 0.02)) - (right_axis[0] * span) + (forward[0] * bank_shift),
            (-forward[1] * (body * 0.02)) - (right_axis[1] * span) + (forward[1] * bank_shift),
        )
        right = point(
            (-forward[0] * (body * 0.02)) + (right_axis[0] * span) - (forward[0] * bank_shift),
            (-forward[1] * (body * 0.02)) + (right_axis[1] * span) - (forward[1] * bank_shift),
        )
        tail_left = point(
            (-forward[0] * (body * 0.72)) - (right_axis[0] * tail),
            (-forward[1] * (body * 0.72)) - (right_axis[1] * tail),
        )
        tail_root = point(-forward[0] * (body * 0.82), -forward[1] * (body * 0.82))
        tail_right = point(
            (-forward[0] * (body * 0.72)) + (right_axis[0] * tail),
            (-forward[1] * (body * 0.72)) + (right_axis[1] * tail),
        )
        points = (nose, right, tail_right, tail_root, tail_left, left)
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (236, 242, 252), points, 1)
        canopy = (
            point(forward[0] * (body * 0.28), forward[1] * (body * 0.28)),
            point(
                (forward[0] * (body * 0.05)) + (right_axis[0] * max(2.0, scale * 3.0)),
                (forward[1] * (body * 0.05)) + (right_axis[1] * max(2.0, scale * 3.0)),
            ),
            point(-forward[0] * max(1.0, scale * 2.0), -forward[1] * max(1.0, scale * 2.0)),
            point(
                (forward[0] * (body * 0.05)) - (right_axis[0] * max(2.0, scale * 3.0)),
                (forward[1] * (body * 0.05)) - (right_axis[1] * max(2.0, scale * 3.0)),
            ),
        )
        pygame.draw.polygon(surface, (240, 244, 250), canopy)

    def _render_trace_test_2_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: TraceTest2Payload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 10, 140)
        frame_bg = (8, 18, 124)
        panel_bg = (12, 20, 92)
        border = (226, 236, 255)
        text_main = (236, 244, 255)
        text_muted = (176, 196, 226)
        input_bg = (4, 10, 36)

        surface.fill(bg)

        margin = max(10, min(18, w // 46))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 17))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Scored",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        title = self._tiny_font.render(f"Trace Test 2 - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 10, header.y + 7))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        work = pygame.Rect(
            frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16
        )
        pygame.draw.rect(surface, panel_bg, work)
        pygame.draw.rect(surface, border, work, 1)

        def render_scene(scene_rect: pygame.Rect, scene_payload: TraceTest2Payload | None) -> None:
            pygame.draw.rect(surface, (8, 14, 62), scene_rect)
            pygame.draw.rect(surface, border, scene_rect, 1)
            inner = scene_rect.inflate(-8, -8)
            if inner.w <= 0 or inner.h <= 0:
                return
            if self._render_trace_test_2_panda_view(surface=surface, world=inner, payload=scene_payload):
                return

            sky_top = (84, 126, 198)
            sky_bottom = (238, 172, 144)
            ground_top = (138, 154, 176)
            ground_bottom = (112, 126, 146)
            horizon = inner.y + int(inner.h * 0.58)
            sky_h = max(1, horizon - inner.y)
            for i in range(sky_h):
                t = i / max(1, sky_h - 1)
                color = (
                    int(round(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t)),
                    int(round(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t)),
                    int(round(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t)),
                )
                pygame.draw.line(surface, color, (inner.x, inner.y + i), (inner.right, inner.y + i))
            ground_h = max(1, inner.bottom - horizon)
            for i in range(ground_h):
                t = i / max(1, ground_h - 1)
                color = (
                    int(round(ground_top[0] + (ground_bottom[0] - ground_top[0]) * t)),
                    int(round(ground_top[1] + (ground_bottom[1] - ground_top[1]) * t)),
                    int(round(ground_top[2] + (ground_bottom[2] - ground_top[2]) * t)),
                )
                pygame.draw.line(surface, color, (inner.x, horizon + i), (inner.right, horizon + i))

            for idx in range(6):
                t = idx / 5.0
                yy = horizon + int(round((t**1.18) * max(1, inner.bottom - horizon)))
                left_x = inner.x + int(round(inner.w * (0.02 + (t * 0.04))))
                right_x = inner.right - int(round(inner.w * (0.02 + (t * 0.01))))
                pygame.draw.line(surface, (182, 194, 214), (left_x, yy), (right_x, yy), 1)
            for lane in (-2, -1, 0, 1, 2):
                xx = inner.centerx + int(round(lane * inner.w * 0.15))
                pygame.draw.line(surface, (168, 182, 204), (xx, horizon + 2), (xx, inner.bottom), 1)

            tracks = scene_payload.aircraft if scene_payload is not None else ()
            if not tracks:
                return
            progress = float(scene_payload.observe_progress) if scene_payload is not None else 0.5
            forward_scale_x = max(2.8, inner.w / 128.0)
            depth_parallax_x = max(0.20, inner.w / 520.0)
            altitude_scale = max(2.0, inner.h / 50.0)

            def project_point(point: TraceTest2Point3) -> tuple[float, float]:
                return (
                    inner.centerx
                    + (((point.y - 88.0) * forward_scale_x) + (point.x * depth_parallax_x)),
                    horizon - ((point.z - 8.0) * altitude_scale),
                )

            for track in tracks:
                pos = self._trace_test_2_track_position(track=track, progress=progress)
                future = self._trace_test_2_track_position(
                    track=track,
                    progress=min(1.0, progress + 0.03),
                )
                dx = future.x - pos.x
                dy = future.y - pos.y
                dz = future.z - pos.z
                if (dx * dx) + (dy * dy) + (dz * dz) <= 1e-8:
                    past = self._trace_test_2_track_position(
                        track=track,
                        progress=max(0.0, progress - 0.03),
                    )
                    dx = pos.x - past.x
                    dy = pos.y - past.y
                    dz = pos.z - past.z

                px, py = project_point(pos)
                fx, fy = project_point(
                    TraceTest2Point3(
                        x=pos.x + dx,
                        y=pos.y + dy,
                        z=pos.z + dz,
                    )
                )
                heading_deg = 0.0
                if abs(fx - px) + abs(fy - py) >= 0.01:
                    heading_deg = math.degrees(math.atan2(fy - py, fx - px))

                screen_x = int(round(px))
                screen_y = int(round(py))
                bank = _clamp(dx * 2.8, -32.0, 32.0)
                scale = max(0.50, min(1.18, 0.92 - (pos.x / 120.0)))
                self._draw_trace_test_2_aircraft_fallback(
                    surface,
                    center=(screen_x, screen_y),
                    color=track.color_rgb,
                    scale=scale,
                    heading_deg=heading_deg,
                    bank_deg=bank,
                )

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED) or payload is None:
            info = work.inflate(-20, -20)
            pygame.draw.rect(surface, (18, 28, 108), info)
            pygame.draw.rect(surface, border, info, 1)

            scene = pygame.Rect(info.x + 12, info.y + 12, info.w - 24, max(170, int(info.h * 0.56)))
            render_scene(scene, None)
            text_rect = pygame.Rect(
                info.x + 12, scene.bottom + 10, info.w - 24, info.bottom - scene.bottom - 22
            )
            intro = (
                "Watch a short 3-D aircraft scene, then answer a recall question about it.\n\n"
                "Questions may ask which color stayed on screen, started lowest, ended left-most, "
                "or how many left turns the red aircraft made."
            )
            if snap.phase is Phase.RESULTS:
                intro = str(snap.prompt)
            elif snap.phase is Phase.PRACTICE_DONE:
                intro = "Practice complete. Press Enter to start the timed Trace Test 2 block."
            self._draw_wrapped_text(
                surface,
                intro,
                text_rect,
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
            return

        panel = work.inflate(-20, -20)
        pygame.draw.rect(surface, (18, 28, 108), panel)
        pygame.draw.rect(surface, border, panel, 1)

        scene = pygame.Rect(
            panel.x + 12,
            panel.y + 12,
            panel.w - 24,
            max(150, int(panel.h * 0.44)),
        )
        render_scene(scene, payload)

        rem = 0.0 if payload.stage_time_remaining_s is None else float(payload.stage_time_remaining_s)
        trial_text = (
            f"{payload.block_kind.title()} {payload.trial_index_in_block}/{payload.trials_in_block}"
        )
        status = pygame.Rect(panel.x + 12, scene.bottom + 8, panel.w - 24, 24)
        pygame.draw.rect(surface, (14, 20, 88), status)
        pygame.draw.rect(surface, border, status, 1)
        status_label = self._tiny_font.render(trial_text, True, text_main)
        status_time = self._tiny_font.render(f"{rem:0.1f}s", True, text_main)
        surface.blit(status_label, (status.x + 8, status.y + 6))
        surface.blit(status_time, status_time.get_rect(midright=(status.right - 8, status.centery)))

        prog_card = pygame.Rect(panel.x + 12, status.bottom + 6, panel.w - 24, 16)
        pygame.draw.rect(surface, (10, 14, 54), prog_card, border_radius=4)
        pygame.draw.rect(surface, border, prog_card, 1, border_radius=4)
        fill_w = int(round((prog_card.w - 4) * max(0.0, min(1.0, float(payload.observe_progress)))))
        if fill_w > 0:
            fill = pygame.Rect(prog_card.x + 2, prog_card.y + 2, fill_w, prog_card.h - 4)
            pygame.draw.rect(surface, (214, 222, 98), fill, border_radius=3)

        question_top = prog_card.bottom + 8
        question = pygame.Rect(panel.x + 12, question_top, panel.w - 24, panel.bottom - question_top - 8)
        pygame.draw.rect(surface, (18, 28, 108), question)
        pygame.draw.rect(surface, border, question, 1)

        header_h = max(44, min(62, int(question.h * 0.28)))
        header_box = pygame.Rect(question.x + 12, question.y + 8, question.w - 24, header_h)
        pygame.draw.rect(surface, (14, 20, 88), header_box)
        pygame.draw.rect(surface, border, header_box, 1)
        self._draw_wrapped_text(
            surface,
            str(payload.stem),
            header_box.inflate(-10, -8),
            color=text_main,
            font=self._small_font,
            max_lines=3,
        )

        selected_code = int(self._math_choice)
        raw = self._input.strip()
        if raw.isdigit():
            selected_code = int(raw)

        entry_card = pygame.Rect(question.x + 12, question.bottom - 34, question.w - 24, 26)
        options_rect = pygame.Rect(
            question.x + 12,
            header_box.bottom + 8,
            question.w - 24,
            max(36, entry_card.y - header_box.bottom - 16),
        )
        pygame.draw.rect(surface, (14, 20, 88), options_rect)
        pygame.draw.rect(surface, border, options_rect, 1)

        row_h = max(28, (options_rect.h - 8) // max(1, len(payload.options)))
        y = options_rect.y + 4
        for option in payload.options:
            row = pygame.Rect(options_rect.x + 8, y, options_rect.w - 16, row_h - 2)
            selected = int(option.code) == selected_code
            row_fill = (58, 84, 154) if selected else (20, 30, 106)
            pygame.draw.rect(surface, row_fill, row)
            pygame.draw.rect(surface, border, row, 1)
            swatch = pygame.Rect(row.x + 10, row.y + max(4, (row.h - 18) // 2), 18, 18)
            swatch_color = option.color_rgb if option.color_rgb is not None else (34, 44, 124)
            pygame.draw.rect(surface, swatch_color, swatch)
            pygame.draw.rect(surface, (236, 244, 255), swatch, 1)
            label = self._small_font.render(option.label, True, text_main)
            surface.blit(label, label.get_rect(midleft=(swatch.right + 12, row.centery)))
            code_tag = pygame.Rect(row.right - 34, row.y + 4, 24, max(14, row.h - 8))
            pygame.draw.rect(surface, input_bg, code_tag)
            pygame.draw.rect(surface, border, code_tag, 1)
            code_txt = self._tiny_font.render(self._choice_key_label(option.code), True, text_main)
            surface.blit(code_txt, code_txt.get_rect(center=code_tag.center))
            y += row_h

        pygame.draw.rect(surface, (14, 20, 88), entry_card)
        pygame.draw.rect(surface, border, entry_card, 1)
        hint = self._tiny_font.render("A/S/D/F, Up/Down, Enter", True, text_muted)
        surface.blit(hint, (entry_card.x + 8, entry_card.y + 7))

    def _render_spatial_integration_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: SpatialIntegrationPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 10, 90)
        frame_bg = (8, 20, 112)
        header_bg = (18, 34, 132)
        work_bg = (68, 72, 84)
        card_bg = (42, 47, 58)
        border = (228, 238, 255)
        text_main = (236, 244, 255)
        text_muted = (180, 198, 226)
        input_bg = (6, 12, 30)

        surface.fill(bg)

        margin = max(10, min(18, w // 42))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(30, min(38, h // 15))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Scored",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        title = self._tiny_font.render(f"Spatial Integration - {phase_label}", True, text_main)
        surface.blit(title, (header.x + 10, header.y + 7))

        stats = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}",
            True,
            text_muted,
        )
        surface.blit(stats, stats.get_rect(midright=(header.right - 10, header.centery)))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 12, header.bottom + 8)))

        work = pygame.Rect(
            frame.x + 8, header.bottom + 8, frame.w - 16, frame.bottom - header.bottom - 16
        )
        pygame.draw.rect(surface, work_bg, work)
        pygame.draw.rect(surface, border, work, 1)

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE):
            panel = work.inflate(-18, -18)
            scene = pygame.Rect(panel.x + 14, panel.y + 64, panel.w - 28, panel.h - 74)
            self._draw_spatial_terrain_scene(surface, scene, payload=None)

            msg = pygame.Rect(scene.x + 10, panel.y + 10, scene.w - 20, 46)
            pygame.draw.rect(surface, (205, 203, 132), msg, border_radius=6)
            pygame.draw.rect(surface, (18, 18, 22), msg, 1, border_radius=6)
            if snap.phase is Phase.INSTRUCTIONS:
                line1 = "Three sections: A (top-down), B (oblique), C (aircraft motion)."
                line2 = "Each section has short practice before scored."
            else:
                line1 = "Section transition."
                line2 = "Press Enter to continue."
            surface.blit(self._tiny_font.render(line1, True, (20, 20, 22)), (msg.x + 8, msg.y + 8))
            surface.blit(self._tiny_font.render(line2, True, (20, 20, 22)), (msg.x + 8, msg.y + 24))

            help_rect = pygame.Rect(panel.x + 12, panel.y + 10, panel.w - 24, 44)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                help_rect,
                color=(236, 244, 255),
                font=self._tiny_font,
                max_lines=2,
            )
            return

        if snap.phase not in (Phase.PRACTICE, Phase.SCORED) or payload is None:
            info = work.inflate(-20, -20)
            pygame.draw.rect(surface, card_bg, info)
            pygame.draw.rect(surface, border, info, 1)
            text = str(snap.prompt)
            self._draw_wrapped_text(
                surface,
                text,
                info.inflate(-16, -16),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
            return

        left_w = int(work.w * 0.58)
        left = pygame.Rect(work.x + 8, work.y + 8, left_w - 12, work.h - 16)
        right = pygame.Rect(left.right + 8, work.y + 8, work.right - left.right - 16, work.h - 16)
        if payload.section is SpatialIntegrationSection.PART_A:
            section_text = "Section A"
            section_desc = "Top-down scene memory"
        elif payload.section is SpatialIntegrationSection.PART_B:
            section_text = "Section B"
            section_desc = "Oblique scene memory"
        else:
            section_text = "Section C"
            section_desc = "Oblique scene with moving aircraft"

        if payload.trial_stage is SpatialIntegrationTrialStage.MEMORIZE:
            scene_panel = left
            self._draw_spatial_terrain_scene(surface, scene_panel, payload=payload)

            rem = payload.stage_time_remaining_s
            rem_text = "--"
            if rem is not None:
                rem_text = f"{max(0.0, rem):.1f}s"
            note_rect = pygame.Rect(scene_panel.x + 10, scene_panel.y + 10, scene_panel.w - 20, 56)
            pygame.draw.rect(surface, (205, 203, 132), note_rect, border_radius=6)
            pygame.draw.rect(surface, (18, 18, 22), note_rect, 1, border_radius=6)
            l1 = (
                f"{section_text} {payload.block_kind.title()} "
                f"({payload.trial_index_in_block}/{payload.trials_in_block})"
            )
            l2 = f"Memorize now. Question appears in {rem_text}"
            surface.blit(
                self._tiny_font.render(l1, True, (20, 20, 22)), (note_rect.x + 8, note_rect.y + 9)
            )
            surface.blit(
                self._tiny_font.render(l2, True, (20, 20, 22)), (note_rect.x + 8, note_rect.y + 27)
            )

            pygame.draw.rect(surface, card_bg, right)
            pygame.draw.rect(surface, border, right, 1)
            self._draw_wrapped_text(
                surface,
                f"{section_text}: {section_desc}\n\n"
                f"Target focus: {payload.query_label}\n"
                "Remember positions and relative spacing.\n\n"
                "Testing shortcuts:\nF10 skip practice  |  F11 skip section  |  F8 skip all",
                right.inflate(-12, -12),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
            return

        scene_panel = left
        pygame.draw.rect(surface, (10, 16, 58), scene_panel)
        pygame.draw.rect(surface, border, scene_panel, 1)
        mem_title = self._small_font.render("Scene Hidden", True, (228, 238, 252))
        surface.blit(
            mem_title, mem_title.get_rect(center=(scene_panel.centerx, scene_panel.y + 32))
        )
        self._draw_wrapped_text(
            surface,
            f"{section_text} ({payload.block_kind.title()} "
            f"{payload.trial_index_in_block}/{payload.trials_in_block})\n"
            "Answer from memory only.",
            pygame.Rect(scene_panel.x + 14, scene_panel.y + 58, scene_panel.w - 28, 64),
            color=(214, 224, 248),
            font=self._small_font,
            max_lines=3,
        )

        pygame.draw.rect(surface, card_bg, right)
        pygame.draw.rect(surface, border, right, 1)

        prompt_rect = pygame.Rect(right.x + 10, right.y + 10, right.w - 20, 84)
        pygame.draw.rect(surface, (34, 40, 52), prompt_rect)
        pygame.draw.rect(surface, border, prompt_rect, 1)
        self._draw_wrapped_text(
            surface,
            f"{section_text}: {payload.stem}",
            prompt_rect.inflate(-8, -8),
            color=text_main,
            font=self._tiny_font,
            max_lines=4,
        )

        selected_code = int(self._math_choice)
        raw = self._input.strip()
        if raw.isdigit():
            selected_code = int(raw)

        options_rect = pygame.Rect(
            right.x + 10, prompt_rect.bottom + 8, right.w - 20, right.h - 172
        )
        pygame.draw.rect(surface, (34, 40, 52), options_rect)
        pygame.draw.rect(surface, border, options_rect, 1)

        row_h = max(34, (options_rect.h - 24) // max(1, len(payload.options)))
        y = options_rect.y + 6
        for option in payload.options:
            row = pygame.Rect(options_rect.x + 6, y, options_rect.w - 12, row_h - 4)
            selected = int(option.code) == selected_code
            if selected:
                pygame.draw.rect(surface, (66, 86, 136), row)
            else:
                pygame.draw.rect(surface, (25, 31, 42), row)
            pygame.draw.rect(surface, border, row, 1)

            badge = pygame.Rect(row.x + 6, row.y + 5, 24, row.h - 10)
            pygame.draw.rect(surface, (16, 28, 112), badge)
            pygame.draw.rect(surface, border, badge, 1)
            num = self._tiny_font.render(self._choice_key_label(option.code), True, text_main)
            surface.blit(num, num.get_rect(center=badge.center))

            label = self._small_font.render(option.label, True, text_main)
            surface.blit(label, (badge.right + 10, row.y + 6))
            y += row_h

        entry_card = pygame.Rect(right.x + 10, right.bottom - 60, right.w - 20, 50)
        pygame.draw.rect(surface, (34, 40, 52), entry_card)
        pygame.draw.rect(surface, border, entry_card, 1)
        surface.blit(
            self._tiny_font.render("Answer (A/S/D/F/G):", True, text_muted),
            (entry_card.x + 10, entry_card.y + 7),
        )
        entry_box = pygame.Rect(entry_card.x + 104, entry_card.y + 6, 92, 28)
        pygame.draw.rect(surface, input_bg, entry_box)
        pygame.draw.rect(surface, border, entry_box, 1)
        caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        entry_text = self._choice_input_label(self._input) + caret
        entry = self._small_font.render(entry_text, True, text_main)
        surface.blit(entry, (entry_box.x + 6, entry_box.y + 3))
        hint = self._tiny_font.render(snap.input_hint, True, text_muted)
        surface.blit(hint, (entry_card.x + 10, entry_card.y + 32))

    def _draw_spatial_terrain_scene(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        payload: SpatialIntegrationPayload | None,
    ) -> None:
        panel_bg = (8, 12, 36)
        panel_border = (190, 204, 236)
        surface.fill(panel_bg, rect)
        pygame.draw.rect(surface, panel_border, rect, 1)

        view = rect.inflate(-8, -8)
        scene_view = (
            payload.scene_view if payload is not None else SpatialIntegrationSceneView.OBLIQUE
        )
        horizon_ratio = 0.16 if scene_view is SpatialIntegrationSceneView.TOPDOWN else 0.42
        horizon = view.y + int(view.h * horizon_ratio)

        if self._render_spatial_integration_panda_view(
            surface=surface,
            world=view,
            payload=payload,
        ):
            tag = self._tiny_font.render("3D air/ground picture", True, (224, 236, 252))
            surface.blit(tag, (view.x + 6, view.y + 6))
            return

        sky_top = (86, 132, 198)
        sky_bottom = (238, 168, 142)
        sky_h = max(1, horizon - view.y)
        for i in range(sky_h):
            t = i / max(1, sky_h - 1)
            color = (
                int(round(sky_top[0] + (sky_bottom[0] - sky_top[0]) * t)),
                int(round(sky_top[1] + (sky_bottom[1] - sky_top[1]) * t)),
                int(round(sky_top[2] + (sky_bottom[2] - sky_top[2]) * t)),
            )
            pygame.draw.line(surface, color, (view.x, view.y + i), (view.right, view.y + i))

        seed = self._spatial_scene_seed(payload=payload)
        rng = random.Random(seed)
        anim_s = float(pygame.time.get_ticks()) / 1000.0

        # Gentle cloud drift to make the scene continuously alive.
        cloud_count = 6
        cloud_lane = max(12, int((horizon - view.y) * 0.18))
        for idx in range(cloud_count):
            lane_y = view.y + 10 + ((idx % 3) * cloud_lane)
            wobble = int(round(3.0 * math.sin((anim_s * 0.42) + (idx * 1.3))))
            drift = int(round((anim_s * (12.0 + (idx * 1.7))) + (idx * 93)))
            cx = view.x + ((drift + (idx * 71)) % (view.w + 120)) - 60
            cy = lane_y + wobble
            cw = 32 + (idx * 5)
            ch = 10 + (idx % 3) * 3
            alpha = max(58, min(110, 90 - (idx * 4)))
            cloud_surf = pygame.Surface((cw + 18, ch + 10), pygame.SRCALPHA)
            pygame.draw.ellipse(cloud_surf, (252, 244, 236, alpha), pygame.Rect(0, 2, cw, ch))
            pygame.draw.ellipse(cloud_surf, (248, 240, 228, alpha), pygame.Rect(12, 0, cw // 2, ch))
            pygame.draw.ellipse(
                cloud_surf, (248, 240, 228, alpha), pygame.Rect(cw // 2, 3, cw // 2 + 6, ch)
            )
            surface.blit(cloud_surf, (cx, cy))

        if scene_view is SpatialIntegrationSceneView.TOPDOWN:
            ridge: list[tuple[int, int]] = [(view.x, horizon + 8)]
            ridge_points = 10
            for idx in range(ridge_points):
                x = view.x + int((idx / max(1, ridge_points - 1)) * view.w)
                y = horizon + rng.randint(-8, 6)
                ridge.append((x, y))
            ridge.append((view.right, horizon + 8))
            mountains = [*ridge, (view.right, horizon + 36), (view.x, horizon + 36)]
            pygame.draw.polygon(surface, (120, 136, 104), mountains)
            pygame.draw.lines(surface, (142, 156, 128), False, ridge, 2)

            ground_rect = pygame.Rect(view.x, horizon, view.w, view.bottom - horizon)
            pygame.draw.rect(surface, (126, 150, 74), ground_rect)

            for idx in range(8):
                t = idx / 7.0
                y = horizon + int((view.bottom - horizon) * t)
                shade = int(round(154 - (idx * 6)))
                pygame.draw.line(surface, (108, shade, 68), (view.x, y), (view.right, y), 1)

            for idx in range(-6, 7):
                x0 = view.centerx + int(idx * view.w * 0.075)
                x1 = x0 + int(view.w * 0.12 * math.sin(idx * 0.85))
                pygame.draw.line(surface, (118, 142, 70), (x0, horizon), (x1, view.bottom), 1)

            contour_centers = (
                (0.24, 0.42),
                (0.64, 0.55),
                (0.48, 0.72),
            )
            for idx, (cxn, cyn) in enumerate(contour_centers):
                cx = view.x + int(view.w * cxn)
                cy = horizon + int((view.bottom - horizon) * cyn)
                rx = max(12, int(view.w * (0.12 + (idx * 0.03))))
                ry = max(8, int(rx * 0.58))
                pygame.draw.ellipse(
                    surface, (136, 162, 82), pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2), 1
                )
        else:
            ridge: list[tuple[int, int]] = [(view.x, horizon + 12)]
            ridge_points = 10
            for idx in range(ridge_points):
                x = view.x + int((idx / max(1, ridge_points - 1)) * view.w)
                y = horizon + rng.randint(-18, 10)
                ridge.append((x, y))
            ridge.append((view.right, horizon + 14))
            mountains = [*ridge, (view.right, horizon + 56), (view.x, horizon + 56)]
            pygame.draw.polygon(surface, (116, 130, 106), mountains)
            pygame.draw.lines(surface, (142, 156, 134), False, ridge, 2)

            ground_rect = pygame.Rect(view.x, horizon, view.w, view.bottom - horizon)
            pygame.draw.rect(surface, (120, 144, 66), ground_rect)

            path = [
                (view.centerx - int(view.w * 0.16), view.bottom),
                (view.centerx + int(view.w * 0.20), view.bottom),
                (view.centerx + int(view.w * 0.05), horizon + 2),
                (view.centerx - int(view.w * 0.03), horizon + 2),
            ]
            pygame.draw.polygon(surface, (146, 166, 82), path)

            for idx in range(1, 8):
                t = idx / 8.0
                y = horizon + int((view.bottom - horizon) * (t**1.35))
                x_off = int((1.0 - t) * view.w * 0.45)
                pygame.draw.line(
                    surface, (136, 158, 78), (view.centerx - x_off, y), (view.centerx + x_off, y), 1
                )

            for idx in range(-4, 5):
                t = abs(idx) / 4.0
                top_x = view.centerx + int(idx * view.w * 0.04)
                bottom_x = view.centerx + int(idx * view.w * (0.13 + (0.18 * t)))
                pygame.draw.line(
                    surface, (126, 150, 70), (top_x, horizon), (bottom_x, view.bottom), 1
                )

        grid_cols = int(payload.grid_cols) if payload is not None else 5
        grid_rows = int(payload.grid_rows) if payload is not None else 5
        alt_levels = int(payload.alt_levels) if payload is not None else 4

        if payload is None:
            landmarks = (
                ("HGR", 1, 0),
                ("TWR", 3, 1),
                ("WDM", 4, 2),
            )
            now_point = (2, 1, 1)
            prev_point = (1, 0, 1)
            velocity = (1, 1, 0)
            show_motion = True
        else:
            landmarks = tuple(
                (str(landmark.label), int(landmark.x), int(landmark.y))
                for landmark in payload.landmarks
            )
            now_point = (
                int(payload.aircraft_now.x),
                int(payload.aircraft_now.y),
                int(payload.aircraft_now.z),
            )
            prev_point = (
                int(payload.aircraft_prev.x),
                int(payload.aircraft_prev.y),
                int(payload.aircraft_prev.z),
            )
            velocity = (
                int(payload.velocity.dx),
                int(payload.velocity.dy),
                int(payload.velocity.dz),
            )
            show_motion = bool(payload.show_aircraft_motion) and prev_point != now_point

        asset_specs: list[tuple[float, str, float, float, float, float, float, float]] = []

        def add_asset(
            kind: str, *, gx: int, gy: int, air: bool = False, scale_bias: float = 1.0
        ) -> None:
            wx, wy, _wz, terrain = self._spatial_grid_to_world(
                x=gx,
                y=gy,
                z=0,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
            )
            wx += rng.uniform(-0.26, 0.26)
            wy = max(0.72, min(7.8, wy + rng.uniform(-0.28, 0.34)))
            wz = terrain + 0.02
            if air:
                wz = terrain + rng.uniform(0.30, 1.00)
                wy = max(0.72, min(7.8, wy + rng.uniform(-0.22, 0.24)))
            scale = rng.uniform(0.76, 1.22) * float(scale_bias)
            heading = rng.uniform(0.0, 359.0)
            anim_phase = rng.uniform(0.0, math.tau)
            anim_rate = rng.uniform(0.65, 1.85)
            asset_specs.append((wy, kind, wx, wz, scale, heading, anim_phase, anim_rate))

        required_kinds = (
            "building",
            "forest",
            "helicopter",
            "fast_jet",
            "foot_soldiers",
            "truck",
            "tent",
        )
        for kind in required_kinds:
            add_asset(
                kind,
                gx=int(rng.randint(0, max(0, grid_cols - 1))),
                gy=int(rng.randint(0, max(0, grid_rows - 1))),
                air=kind in {"helicopter", "fast_jet"},
                scale_bias=1.08 if kind in {"helicopter", "fast_jet"} else 1.0,
            )

        landmark_to_kind = {
            "TWR": "tower",
            "HGR": "building",
            "WDM": "forest",
            "VLG": "foot_soldiers",
            "LKE": "tent",
            "RDG": "truck",
        }
        for label, gx, gy in landmarks:
            add_asset(
                landmark_to_kind.get(label, "building"),
                gx=int(gx),
                gy=int(gy),
                air=False,
                scale_bias=1.15 if label in {"TWR", "HGR"} else 1.0,
            )

        extra_count = 6 + max(grid_cols, grid_rows)
        ambient_kinds = ("building", "forest", "truck", "tent", "foot_soldiers", "radar")
        for _ in range(extra_count):
            kind = str(rng.choice(ambient_kinds))
            add_asset(
                kind,
                gx=int(rng.randint(0, max(0, grid_cols - 1))),
                gy=int(rng.randint(0, max(0, grid_rows - 1))),
                air=False,
                scale_bias=0.92,
            )

        for depth_y, kind, wx, wz, scale, heading, anim_phase, anim_rate in sorted(
            asset_specs, key=lambda it: it[0]
        ):
            self._draw_spatial_scene_asset(
                surface,
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                kind=kind,
                wx=wx,
                wy=depth_y,
                wz=wz,
                scale=scale,
                heading_deg=heading,
                anim_s=anim_s,
                anim_phase=anim_phase,
                anim_rate=anim_rate,
            )

        landmark_data: list[tuple[float, str, tuple[int, int], tuple[int, int]]] = []
        for label, gx, gy in landmarks:
            wx, wy, wz, terrain = self._spatial_grid_to_world(
                x=gx,
                y=gy,
                z=0,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
            )
            base = self._spatial_project_point(
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                wx=wx,
                wy=wy,
                wz=terrain + 0.01,
            )
            top = self._spatial_project_point(
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                wx=wx,
                wy=wy,
                wz=terrain + 0.22,
            )
            landmark_data.append((wy, label, base, top))

        for _, label, base, top in sorted(landmark_data, key=lambda it: it[0]):
            bx, by = base
            tx, ty = top
            if (
                bx < view.x - 20
                or bx > view.right + 20
                or by < view.y - 20
                or by > view.bottom + 20
            ):
                continue
            pygame.draw.line(surface, (232, 226, 118), (bx, by), (tx, ty), 2)
            pygame.draw.circle(surface, (244, 236, 130), (tx, ty), 3)
            surface.blit(self._tiny_font.render(label, True, (236, 236, 176)), (tx + 4, ty - 6))

        now_wx, now_wy, now_wz, now_terrain = self._spatial_grid_to_world(
            x=now_point[0],
            y=now_point[1],
            z=now_point[2],
            grid_cols=grid_cols,
            grid_rows=grid_rows,
            alt_levels=alt_levels,
        )
        live_wx = now_wx
        live_wy = now_wy
        live_wz = now_wz
        prev_wx = now_wx
        prev_wy = now_wy
        prev_wz = now_wz
        if show_motion:
            prev_wx, prev_wy, prev_wz, _ = self._spatial_grid_to_world(
                x=prev_point[0],
                y=prev_point[1],
                z=prev_point[2],
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                alt_levels=alt_levels,
            )
            future_wx = now_wx + (now_wx - prev_wx)
            future_wy = now_wy + (now_wy - prev_wy)
            future_wz = now_wz + (now_wz - prev_wz)
            motion_phase = (anim_s * 0.32 + ((seed % 997) / 997.0)) % 1.0
            if motion_phase < 0.333:
                t = motion_phase / 0.333
                live_wx = prev_wx + ((now_wx - prev_wx) * t)
                live_wy = prev_wy + ((now_wy - prev_wy) * t)
                live_wz = prev_wz + ((now_wz - prev_wz) * t)
            elif motion_phase < 0.666:
                t = (motion_phase - 0.333) / 0.333
                live_wx = now_wx + ((future_wx - now_wx) * t)
                live_wy = now_wy + ((future_wy - now_wy) * t)
                live_wz = now_wz + ((future_wz - now_wz) * t)
            else:
                t = (motion_phase - 0.666) / 0.334
                live_wx = future_wx + ((now_wx - future_wx) * t)
                live_wy = future_wy + ((now_wy - future_wy) * t)
                live_wz = future_wz + ((now_wz - future_wz) * t)

        live_terrain = self._spatial_terrain_height(wx=live_wx, wy=live_wy)
        now_screen = self._spatial_project_point(
            rect=view,
            horizon_y=horizon,
            scene_view=scene_view,
            wx=live_wx,
            wy=live_wy,
            wz=live_wz,
        )
        now_ground = self._spatial_project_point(
            rect=view,
            horizon_y=horizon,
            scene_view=scene_view,
            wx=live_wx,
            wy=live_wy,
            wz=live_terrain + 0.02,
        )
        craft_bob = int(round(2.0 * math.sin((anim_s * 3.1) + (seed * 0.0007))))
        now_screen_vis = (now_screen[0], now_screen[1] + craft_bob)

        if show_motion:
            prev_screen = self._spatial_project_point(
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                wx=prev_wx,
                wy=prev_wy,
                wz=prev_wz,
            )
            ref_now_screen = self._spatial_project_point(
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                wx=now_wx,
                wy=now_wy,
                wz=now_wz,
            )
            pygame.draw.circle(surface, (210, 216, 236), prev_screen, 5, 1)
            pygame.draw.circle(surface, (194, 208, 232), ref_now_screen, 4, 1)
            pygame.draw.line(surface, (210, 216, 236), prev_screen, ref_now_screen, 1)

        pygame.draw.line(surface, (204, 222, 242), now_ground, now_screen_vis, 1)
        size = 7
        craft = [
            (now_screen_vis[0], now_screen_vis[1] - size),
            (now_screen_vis[0] + int(size * 0.65), now_screen_vis[1] + size),
            (now_screen_vis[0] - int(size * 0.65), now_screen_vis[1] + size),
        ]
        pygame.draw.polygon(surface, (76, 212, 236), craft)
        pygame.draw.polygon(surface, (224, 246, 254), craft, 1)
        pygame.draw.circle(surface, (232, 248, 255), now_screen_vis, 2)

        if show_motion and (velocity[0] != 0 or velocity[1] != 0 or velocity[2] != 0):
            pred = self._spatial_project_point(
                rect=view,
                horizon_y=horizon,
                scene_view=scene_view,
                wx=now_wx + (0.35 * velocity[0]),
                wy=max(0.6, now_wy + (0.35 * velocity[1])),
                wz=now_wz + (0.20 * velocity[2]),
            )
            pygame.draw.line(surface, (140, 226, 246), now_screen_vis, pred, 2)

        tag = self._tiny_font.render("3D air/ground picture", True, (224, 236, 252))
        surface.blit(tag, (view.x + 6, view.y + 6))

    def _draw_spatial_scene_asset(
        self,
        surface: pygame.Surface,
        *,
        rect: pygame.Rect,
        horizon_y: int,
        scene_view: SpatialIntegrationSceneView,
        kind: str,
        wx: float,
        wy: float,
        wz: float,
        scale: float,
        heading_deg: float,
        anim_s: float,
        anim_phase: float = 0.0,
        anim_rate: float = 1.0,
    ) -> None:
        if wy <= 0.01:
            return

        phase = (float(anim_s) * max(0.10, float(anim_rate))) + float(anim_phase)
        wx_anim = float(wx)
        wy_anim = max(0.35, float(wy))
        wz_anim = float(wz)
        heading_anim = float(heading_deg)

        if kind == "truck":
            drift = 0.11 * math.sin(phase * 0.86)
            hd = math.radians(float(heading_deg))
            wx_anim += drift * math.cos(hd)
            wy_anim = max(0.55, wy_anim + (drift * 0.30 * math.sin(hd)))
        elif kind == "foot_soldiers":
            wx_anim += 0.06 * math.sin(phase * 0.78)
        elif kind == "helicopter":
            wx_anim += 0.18 * math.cos(phase * 0.68)
            wy_anim = max(0.55, wy_anim + 0.10 * math.sin(phase * 0.64))
            wz_anim += 0.08 * math.sin(phase * 2.4)
        elif kind == "fast_jet":
            wx_anim += 0.42 * math.sin(phase * 1.20)
            wy_anim = max(0.55, wy_anim + 0.22 * math.cos(phase * 0.94))
            wz_anim += 0.12 * math.sin(phase * 1.86)
            heading_anim += 24.0 * math.cos(phase * 0.52)

        terrain = self._spatial_terrain_height(wx=wx_anim, wy=wy_anim)
        ground = self._spatial_project_point(
            rect=rect,
            horizon_y=horizon_y,
            scene_view=scene_view,
            wx=wx_anim,
            wy=wy_anim,
            wz=terrain + 0.01,
        )
        pos = self._spatial_project_point(
            rect=rect,
            horizon_y=horizon_y,
            scene_view=scene_view,
            wx=wx_anim,
            wy=wy_anim,
            wz=wz_anim,
        )

        if (
            pos[0] < rect.x - 30
            or pos[0] > rect.right + 30
            or pos[1] < rect.y - 30
            or pos[1] > rect.bottom + 30
        ):
            return

        if scene_view is SpatialIntegrationSceneView.TOPDOWN:
            depth_for_size = max(0.90, 1.35 + (wy_anim * 0.22))
            size_scale = 0.092
        else:
            depth_for_size = max(0.85, wy_anim)
            size_scale = 0.080
        base = max(
            2,
            int(
                round((min(rect.w, rect.h) * size_scale * max(0.55, float(scale))) / depth_for_size)
            ),
        )
        facing = 1 if math.cos(math.radians(heading_anim)) >= 0.0 else -1

        shadow_w = max(3, int(base * 1.20))
        shadow_h = max(2, int(base * 0.48))
        shadow_rect = pygame.Rect(
            ground[0] - (shadow_w // 2), ground[1] - (shadow_h // 2), shadow_w, shadow_h
        )
        pygame.draw.ellipse(surface, (64, 82, 46), shadow_rect)

        if kind == "building":
            bw = max(5, int(base * 1.15))
            bh = max(6, int(base * 1.48))
            body = pygame.Rect(ground[0] - (bw // 2), ground[1] - bh, bw, bh)
            pygame.draw.rect(surface, (172, 180, 172), body)
            pygame.draw.rect(surface, (88, 92, 86), body, 1)
            roof = [
                (body.x - 1, body.y),
                (body.centerx, body.y - max(2, base // 2)),
                (body.right + 1, body.y),
            ]
            pygame.draw.polygon(surface, (134, 110, 96), roof)
            if bw >= 8 and bh >= 8:
                window = pygame.Rect(body.x + 2, body.y + 2, 2, 2)
                while window.y < body.bottom - 2:
                    wx0 = body.x + 2
                    while wx0 < body.right - 2:
                        pygame.draw.rect(surface, (230, 232, 202), pygame.Rect(wx0, window.y, 2, 2))
                        wx0 += 4
                    window.y += 4
            return

        if kind == "tower":
            tw = max(2, int(base * 0.40))
            th = max(10, int(base * 2.35))
            shaft = pygame.Rect(ground[0] - (tw // 2), ground[1] - th, tw, th)
            pygame.draw.rect(surface, (204, 202, 172), shaft)
            pygame.draw.rect(surface, (90, 88, 76), shaft, 1)
            pygame.draw.line(
                surface,
                (238, 228, 132),
                (shaft.centerx, shaft.y),
                (shaft.centerx, shaft.y - max(3, base // 2)),
                1,
            )
            if math.sin(phase * 6.0) >= 0.0:
                pygame.draw.circle(
                    surface, (250, 232, 132), (shaft.centerx, shaft.y - max(3, base // 2)), 2
                )
            return

        if kind == "forest":
            offsets = (-base, 0, base)
            for idx, dx in enumerate(offsets):
                tx = ground[0] + dx
                ty = ground[1] + (idx % 2)
                th = max(5, int(base * (1.30 + (idx * 0.20))))
                pygame.draw.line(surface, (90, 70, 40), (tx, ty), (tx, ty - max(2, th // 4)), 1)
                sway = int(round(math.sin((phase * 1.5) + idx) * max(1, th // 8)))
                tree = [
                    (tx + sway, ty - th),
                    (tx - max(2, th // 3), ty - max(2, th // 4)),
                    (tx + max(2, th // 3), ty - max(2, th // 4)),
                ]
                pygame.draw.polygon(surface, (68, 128, 66), tree)
            return

        if kind == "truck":
            bw = max(7, int(base * 1.65))
            bh = max(4, int(base * 0.62))
            bob = int(round(0.6 * math.sin(phase * 6.2)))
            body = pygame.Rect(ground[0] - (bw // 2), ground[1] - bh - 1 + bob, bw, bh)
            pygame.draw.rect(surface, (112, 136, 92), body)
            pygame.draw.rect(surface, (62, 78, 52), body, 1)
            cab_w = max(3, bw // 3)
            if facing > 0:
                cab = pygame.Rect(body.right - cab_w, body.y - 1, cab_w, bh - 1)
            else:
                cab = pygame.Rect(body.x, body.y - 1, cab_w, bh - 1)
            pygame.draw.rect(surface, (140, 158, 112), cab)
            wheel_r = 1 if base < 4 else 2
            pygame.draw.circle(surface, (26, 26, 28), (body.x + 2, body.bottom), wheel_r)
            pygame.draw.circle(surface, (26, 26, 28), (body.right - 2, body.bottom), wheel_r)
            return

        if kind == "tent":
            tw = max(5, int(base * 1.36))
            th = max(4, int(base * 0.92))
            p1 = (ground[0] - (tw // 2), ground[1])
            p2 = (ground[0] + (tw // 2), ground[1])
            flap_wobble = int(round(math.sin(phase * 1.9) * max(1, th // 6)))
            p3 = (ground[0] + flap_wobble, ground[1] - th)
            pygame.draw.polygon(surface, (198, 186, 112), [p1, p2, p3])
            pygame.draw.polygon(surface, (110, 98, 62), [p1, p2, p3], 1)
            flap = [
                (ground[0], ground[1] - th),
                (ground[0], ground[1]),
                (ground[0] + (tw // 3), ground[1]),
            ]
            pygame.draw.polygon(surface, (168, 156, 88), flap)
            return

        if kind == "foot_soldiers":
            offsets = (-base, 0, base)
            for idx, dx in enumerate(offsets):
                sx = ground[0] + dx
                sy = ground[1]
                body_h = max(3, int(base * 0.92))
                head_r = 1 if base < 4 else 2
                pygame.draw.circle(surface, (26, 30, 34), (sx, sy - body_h), head_r)
                pygame.draw.line(surface, (26, 30, 34), (sx, sy - body_h + head_r), (sx, sy - 1), 1)
                leg_phase = phase * 5.2 + idx
                step = 1 if math.sin(leg_phase) >= 0.0 else -1
                pygame.draw.line(surface, (26, 30, 34), (sx, sy - 1), (sx - step, sy + 1), 1)
                pygame.draw.line(surface, (26, 30, 34), (sx, sy - 1), (sx + step, sy + 1), 1)
            return

        if kind == "radar":
            mast_h = max(6, int(base * 1.45))
            pygame.draw.line(
                surface, (176, 190, 198), (ground[0], ground[1]), (ground[0], ground[1] - mast_h), 1
            )
            dish_rect = pygame.Rect(
                ground[0] - max(2, base // 2),
                ground[1] - mast_h - max(2, base // 3),
                max(4, base),
                max(3, base // 2),
            )
            pygame.draw.arc(surface, (198, 210, 220), dish_rect, math.pi * 1.05, math.pi * 1.95, 1)
            sweep = phase * 2.2
            sx = int(round(dish_rect.centerx + math.cos(sweep) * max(2, dish_rect.w * 0.45)))
            sy = int(round(dish_rect.centery + math.sin(sweep) * max(1, dish_rect.h * 0.35)))
            pygame.draw.line(surface, (228, 240, 248), dish_rect.center, (sx, sy), 1)
            return

        if kind == "helicopter":
            rw = max(6, int(base * 1.45))
            rh = max(3, int(base * 0.62))
            body = pygame.Rect(pos[0] - (rw // 2), pos[1] - (rh // 2), rw, rh)
            pygame.draw.ellipse(surface, (96, 150, 96), body)
            pygame.draw.ellipse(surface, (42, 84, 42), body, 1)
            tail = (body.left - max(3, base), body.centery)
            pygame.draw.line(surface, (70, 112, 70), (body.left, body.centery), tail, 1)
            rotor_len = rw + max(3, base // 2)
            rotor_a = phase * 11.0
            rx = int(round(math.cos(rotor_a) * rotor_len))
            ry = int(round(math.sin(rotor_a) * max(1, rotor_len // 5)))
            pygame.draw.line(
                surface,
                (224, 236, 248),
                (pos[0] - rx, pos[1] - rh - ry),
                (pos[0] + rx, pos[1] - rh + ry),
                1,
            )
            pygame.draw.line(
                surface,
                (214, 226, 238),
                (pos[0] - ry, pos[1] - rh + rx // 5),
                (pos[0] + ry, pos[1] - rh - rx // 5),
                1,
            )
            skid_y = body.bottom
            pygame.draw.line(
                surface, (54, 66, 60), (body.left + 1, skid_y), (body.right - 1, skid_y), 1
            )
            return

        if kind == "fast_jet":
            span = max(6, int(base * 1.55))
            length = max(8, int(base * 1.88))
            nose = (pos[0] + (facing * length), pos[1])
            wing_l = (pos[0] - (facing * max(2, length // 4)), pos[1] - span // 2)
            wing_r = (pos[0] - (facing * max(2, length // 4)), pos[1] + span // 2)
            tail = (pos[0] - (facing * length // 2), pos[1])
            pygame.draw.polygon(surface, (174, 182, 194), [nose, wing_l, tail, wing_r])
            pygame.draw.polygon(surface, (86, 92, 104), [nose, wing_l, tail, wing_r], 1)
            contrail = (
                tail[0] - (facing * max(6, int(base * 2.4))),
                tail[1] + int(round(math.sin(phase * 7.0) * 1.6)),
            )
            pygame.draw.line(surface, (224, 230, 236), tail, contrail, 1)
            return

        pygame.draw.circle(surface, (186, 198, 210), pos, max(1, base // 3))

    def _spatial_scene_seed(self, *, payload: SpatialIntegrationPayload | None) -> int:
        if payload is None:
            return 7919

        seed = 2166136261

        def mix(value: int) -> None:
            nonlocal seed
            seed ^= int(value) & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(payload.grid_cols)
        mix(payload.grid_rows)
        mix(payload.alt_levels)
        mix(payload.aircraft_now.x)
        mix(payload.aircraft_now.y)
        mix(payload.aircraft_now.z)
        mix(payload.aircraft_prev.x)
        mix(payload.aircraft_prev.y)
        mix(payload.aircraft_prev.z)
        for landmark in payload.landmarks:
            for ch in landmark.label:
                mix(ord(ch))
            mix(landmark.x)
            mix(landmark.y)
        return int(seed)

    def _spatial_grid_to_world(
        self,
        *,
        x: int,
        y: int,
        z: int,
        grid_cols: int,
        grid_rows: int,
        alt_levels: int,
    ) -> tuple[float, float, float, float]:
        cols = max(1, int(grid_cols))
        rows = max(1, int(grid_rows))
        levels = max(1, int(alt_levels))

        x_norm = 0.5 if cols <= 1 else float(x) / float(cols - 1)
        y_norm = 0.5 if rows <= 1 else float(y) / float(rows - 1)
        z_norm = 0.0 if levels <= 1 else float(z) / float(levels - 1)

        wx = (x_norm - 0.5) * 3.2
        wy = 1.1 + (y_norm * 6.2)
        terrain = self._spatial_terrain_height(wx=wx, wy=wy)
        wz = terrain + 0.05 + (z_norm * 0.95)
        return wx, wy, wz, terrain

    def _spatial_project_point(
        self,
        *,
        rect: pygame.Rect,
        horizon_y: int,
        scene_view: SpatialIntegrationSceneView = SpatialIntegrationSceneView.OBLIQUE,
        wx: float,
        wy: float,
        wz: float,
    ) -> tuple[int, int]:
        if scene_view is SpatialIntegrationSceneView.TOPDOWN:
            wy_norm = (float(wy) - 0.75) / 7.5
            ny = max(0.0, min(1.0, wy_norm))
            lateral = min(rect.w, rect.h) * 0.22 * (0.98 - (ny * 0.42))
            sx = int(round(rect.centerx + (float(wx) * lateral)))
            ground_top = max(rect.y + 2, int(horizon_y))
            ground_span = max(1, rect.bottom - ground_top)
            y_base = float(ground_top) + (ny * float(ground_span))
            alt_lift = (float(wz) * min(rect.w, rect.h) * 0.11) / (1.0 + (ny * 0.8))
            sy = int(round(y_base - alt_lift))
            return sx, sy

        depth = max(0.45, float(wy))
        scale = min(rect.w, rect.h) * 1.12
        cam_z = 0.58
        sx = int(round(rect.centerx + (float(wx) / depth) * scale))
        sy = int(round(float(horizon_y) + ((cam_z - float(wz)) / depth) * scale))
        return sx, sy

    def _spatial_terrain_height(self, *, wx: float, wy: float) -> float:
        base = 0.02
        ridge = 0.08 * math.sin((wx * 1.55) + 0.7) * math.exp(-((wy - 3.8) ** 2) * 0.16)
        hill_a = 0.12 * math.exp(-(((wx + 0.8) ** 2) * 1.9) - (((wy - 2.9) ** 2) * 0.24))
        hill_b = 0.10 * math.exp(-(((wx - 1.1) ** 2) * 1.5) - (((wy - 5.1) ** 2) * 0.30))
        return float(base + ridge + hill_a + hill_b)

    def _draw_spatial_projection_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        payload: SpatialIntegrationPayload,
        view: str,
    ) -> None:
        panel_bg = (38, 44, 56)
        panel_border = (170, 184, 212)
        grid_bg = (18, 22, 32)
        grid_line = (76, 90, 118)
        text_main = (232, 240, 252)
        text_muted = (160, 176, 206)

        pygame.draw.rect(surface, panel_bg, rect)
        pygame.draw.rect(surface, panel_border, rect, 1)

        titles = {
            "top": "Top View (X/Y)",
            "side": "Side View (X/ALT)",
            "front": "Front View (Y/ALT)",
        }
        title = self._tiny_font.render(titles.get(view, "View"), True, text_main)
        surface.blit(title, (rect.x + 8, rect.y + 6))

        inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
        pygame.draw.rect(surface, grid_bg, inner)
        pygame.draw.rect(surface, panel_border, inner, 1)

        if view == "top":
            cols = max(1, int(payload.grid_cols))
            rows = max(1, int(payload.grid_rows))
        elif view == "side":
            cols = max(1, int(payload.grid_cols))
            rows = max(1, int(payload.alt_levels))
        else:
            cols = max(1, int(payload.grid_rows))
            rows = max(1, int(payload.alt_levels))

        cell = min(max(10, inner.w // cols), max(10, inner.h // rows))
        grid_w = cols * cell
        grid_h = rows * cell
        gx = inner.x + max(0, (inner.w - grid_w) // 2)
        gy = inner.y + max(0, (inner.h - grid_h) // 2)

        for c in range(cols + 1):
            x = gx + c * cell
            pygame.draw.line(surface, grid_line, (x, gy), (x, gy + grid_h), 1)
        for r in range(rows + 1):
            y = gy + r * cell
            pygame.draw.line(surface, grid_line, (gx, y), (gx + grid_w, y), 1)

        def center_for(col: int, row: int) -> tuple[int, int]:
            return (
                gx + col * cell + (cell // 2),
                gy + row * cell + (cell // 2),
            )

        def point_to_grid(point: object) -> tuple[int, int]:
            px = int(getattr(point, "x", 0))
            py = int(getattr(point, "y", 0))
            pz = int(getattr(point, "z", 0))
            if view == "top":
                return px, (rows - 1) - py
            if view == "side":
                return px, (rows - 1) - pz
            return py, (rows - 1) - pz

        if view == "top":
            for landmark in payload.landmarks:
                col = int(landmark.x)
                row = (rows - 1) - int(landmark.y)
                cx, cy = center_for(col, row)
                box = pygame.Rect(cx - 4, cy - 4, 8, 8)
                pygame.draw.rect(surface, (252, 224, 82), box)
                pygame.draw.rect(surface, (18, 18, 20), box, 1)
                tag = self._tiny_font.render(landmark.label, True, text_muted)
                surface.blit(tag, (box.right + 2, box.y - 1))

        now_col, now_row = point_to_grid(payload.aircraft_now)
        now_x, now_y = center_for(now_col, now_row)

        if payload.show_aircraft_motion and payload.aircraft_prev != payload.aircraft_now:
            prev_col, prev_row = point_to_grid(payload.aircraft_prev)
            prev_x, prev_y = center_for(prev_col, prev_row)
            pygame.draw.circle(surface, (210, 220, 236), (prev_x, prev_y), max(3, cell // 5), 1)
            pygame.draw.line(surface, (210, 220, 236), (prev_x, prev_y), (now_x, now_y), 1)

        pygame.draw.circle(surface, (248, 64, 80), (now_x, now_y), max(4, cell // 4))
        pygame.draw.circle(surface, (255, 246, 246), (now_x, now_y), max(2, cell // 8))

        dims = self._tiny_font.render(f"{cols} x {rows}", True, text_muted)
        surface.blit(dims, dims.get_rect(bottomright=(inner.right - 4, inner.bottom - 2)))

    def _render_instrument_comprehension_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: InstrumentComprehensionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (4, 9, 90)
        frame_bg = (6, 14, 112)
        header_bg = (12, 22, 124)
        border = (228, 238, 255)
        text_main = (236, 244, 255)
        text_muted = (186, 202, 228)

        surface.fill(bg)

        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_bg, frame)
        pygame.draw.rect(surface, border, frame, 1)

        header_h = max(28, min(36, h // 15))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, header_bg, header)
        pygame.draw.line(
            surface, border, (header.x, header.bottom), (header.right, header.bottom), 1
        )

        if (
            payload is not None
            and payload.kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT
        ):
            part_title = "Instrument Interpretation Test Part 1"
        elif payload is not None:
            part_title = "Instrument Interpretation Test Part 2"
        else:
            part_title = "Instrument Interpretation Test"
        mode_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Transition",
            Phase.SCORED: "Testing",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")
        title_text = f"{part_title} - {mode_label}"
        title = self._tiny_font.render(title_text, True, text_main)
        surface.blit(title, (header.x + 10, header.y + 6))

        stats_text = f"{snap.correct_scored}/{snap.attempted_scored}"
        stats = self._tiny_font.render(stats_text, True, text_muted)
        stats_rect = stats.get_rect(midright=(header.right - 10, header.centery))
        surface.blit(stats, stats_rect)
        scored_label = self._tiny_font.render("Scored", True, text_muted)
        surface.blit(
            scored_label, scored_label.get_rect(midright=(stats_rect.left - 6, header.centery))
        )

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            timer = self._small_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_main)
            surface.blit(timer, timer.get_rect(topright=(frame.right - 14, header.bottom + 8)))

        work = pygame.Rect(
            frame.x + 4,
            header.bottom + 8,
            frame.w - 8,
            frame.bottom - header.bottom - 12,
        )
        pygame.draw.rect(surface, frame_bg, work)
        pygame.draw.rect(surface, border, work, 1)

        footer_h = 18
        footer = pygame.Rect(work.x, work.bottom - footer_h, work.w, footer_h)
        question_rect = pygame.Rect(work.x + 6, work.y + 6, work.w - 12, work.h - footer_h - 8)

        if snap.phase in (Phase.PRACTICE, Phase.SCORED) and payload is not None:
            self._render_instrument_comprehension_question(surface, snap, payload, question_rect)
            self._render_instrument_answer_entry(surface, snap, footer)
            return

        info_card = question_rect.inflate(-20, -16)
        pygame.draw.rect(surface, (8, 18, 104), info_card)
        pygame.draw.rect(surface, border, info_card, 1)
        prompt = str(snap.prompt)
        if snap.phase is Phase.INSTRUCTIONS:
            prompt = (
                "Part 1: Read the attitude and heading instruments, "
                "then choose the matching aircraft image.\n\n"
                "Part 2: Read the full instrument panel, "
                "then choose the matching flight description.\n\n"
                "Controls: press A, S, D, F, or G, then Enter."
            )
        self._draw_wrapped_text(
            surface,
            prompt,
            info_card.inflate(-18, -18),
            color=text_main,
            font=self._small_font,
            max_lines=12,
        )
        self._render_instrument_answer_entry(surface, snap, footer)

    def _render_instrument_answer_entry(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        rect: pygame.Rect,
    ) -> None:
        text_main = (236, 244, 255)
        text_muted = (184, 200, 224)
        pygame.draw.rect(surface, (0, 0, 0), rect)
        answer_label = self._tiny_font.render("Answer", True, (244, 232, 122))
        surface.blit(answer_label, answer_label.get_rect(midleft=(rect.x + 44, rect.centery)))

        entry_rect = pygame.Rect(rect.centerx - 14, rect.y + 2, 28, rect.h - 4)
        pygame.draw.rect(surface, (10, 14, 28), entry_rect)
        pygame.draw.rect(surface, (196, 206, 228), entry_rect, 1)
        entry = self._tiny_font.render(self._choice_input_label(self._input), True, text_main)
        surface.blit(entry, entry.get_rect(center=entry_rect.center))

        left_text = self._tiny_font.render(
            "Practice" if snap.phase is Phase.PRACTICE else "Testing",
            True,
            text_muted,
        )
        surface.blit(left_text, (rect.x + 8, rect.y + 3))

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            time_text = self._tiny_font.render(f"{rem // 60:02d}:{rem % 60:02d}", True, text_muted)
            surface.blit(time_text, time_text.get_rect(midright=(rect.right - 8, rect.centery)))
        else:
            help_text = self._tiny_font.render("A/S/D/F/G", True, text_muted)
            surface.blit(help_text, help_text.get_rect(midright=(rect.right - 8, rect.centery)))

    def _render_instrument_comprehension_question(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: InstrumentComprehensionPayload,
        panel: pygame.Rect,
    ) -> None:
        text_main = (236, 244, 255)
        border = (214, 226, 248)
        pygame.draw.rect(surface, (4, 12, 96), panel)
        pygame.draw.rect(surface, border, panel, 1)

        selected_code = int(self._math_choice)
        raw = self._input.strip()
        if raw.isdigit():
            selected_code = int(raw)

        if payload.kind is InstrumentComprehensionTrialKind.INSTRUMENTS_TO_AIRCRAFT:
            dials_rect = pygame.Rect(
                panel.x + panel.w // 8,
                panel.y + 10,
                panel.w // 3,
                max(92, panel.h // 4),
            )
            self._draw_orientation_prompt_dials(surface, dials_rect, payload.prompt_state)

            cards_top = dials_rect.bottom + 18
            gap = 10
            top_w = (panel.w - gap * 3) // 2
            top_h = max(92, int(panel.h * 0.24))
            bottom_w = (panel.w - gap * 4) // 3
            bottom_h = max(84, int(panel.h * 0.22))
            card_rects = [
                pygame.Rect(panel.x + gap, cards_top, top_w, top_h),
                pygame.Rect(panel.x + gap * 2 + top_w, cards_top, top_w, top_h),
                pygame.Rect(panel.x + gap, cards_top + top_h + gap, bottom_w, bottom_h),
                pygame.Rect(
                    panel.x + gap * 2 + bottom_w,
                    cards_top + top_h + gap,
                    bottom_w,
                    bottom_h,
                ),
                pygame.Rect(
                    panel.x + gap * 3 + bottom_w * 2,
                    cards_top + top_h + gap,
                    bottom_w,
                    bottom_h,
                ),
            ]
            for option, card in zip(payload.options, card_rects, strict=False):
                selected = int(option.code) == selected_code
                frame_color = (248, 228, 122) if selected else border
                pygame.draw.rect(surface, frame_color, card, 2)
                inner = card.inflate(-4, -4)
                self._draw_aircraft_orientation_card(surface, inner, option.state)
                badge = pygame.Rect(card.x + 4, card.bottom - 18, 18, 14)
                pygame.draw.rect(surface, (0, 0, 0), badge)
                label = self._tiny_font.render(
                    self._choice_key_label(option.code),
                    True,
                    (255, 255, 255),
                )
                surface.blit(label, label.get_rect(center=badge.center))
            return

        cluster_rect = pygame.Rect(
            panel.x + 14,
            panel.y + 12,
            panel.w - 28,
            max(132, int(panel.h * 0.42)),
        )
        self._draw_instrument_cluster(surface, cluster_rect, payload.prompt_state, compact=False)

        options_rect = pygame.Rect(
            panel.x + 8,
            cluster_rect.bottom + 10,
            panel.w - 16,
            panel.bottom - cluster_rect.bottom - 18,
        )
        row_gap = 6
        row_h = max(30, min(38, (options_rect.h - row_gap * 6) // 5))
        y = options_rect.y + row_gap
        for option in payload.options:
            row = pygame.Rect(options_rect.x + 6, y, options_rect.w - 12, row_h)
            selected = int(option.code) == selected_code
            pygame.draw.rect(surface, (18, 28, 108), row)
            pygame.draw.rect(surface, (248, 228, 122) if selected else border, row, 1)
            badge = pygame.Rect(row.x + 4, row.y + 4, 18, row.h - 8)
            pygame.draw.rect(surface, (0, 0, 0), badge)
            key_txt = self._tiny_font.render(
                self._choice_key_label(option.code),
                True,
                (255, 255, 255),
            )
            surface.blit(key_txt, key_txt.get_rect(center=badge.center))
            self._draw_wrapped_text(
                surface,
                option.description,
                pygame.Rect(row.x + 28, row.y + 5, row.w - 34, row.h - 8),
                color=text_main,
                font=self._tiny_font,
                max_lines=2,
            )
            y += row_h + row_gap

    def _draw_orientation_prompt_dials(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
    ) -> None:
        panel_bg = (36, 40, 50)
        panel_border = (170, 184, 212)
        text_muted = (184, 200, 224)

        pygame.draw.rect(surface, panel_bg, rect)
        pygame.draw.rect(surface, panel_border, rect, 1)

        gap = 12
        dial_size = max(42, min(rect.h - 20, (rect.w - gap * 3) // 2))
        total_w = dial_size * 2 + gap
        start_x = rect.x + (rect.w - total_w) // 2
        y = rect.y + (rect.h - dial_size) // 2
        att_rect = pygame.Rect(start_x, y, dial_size, dial_size)
        hdg_rect = pygame.Rect(att_rect.right + gap, y, dial_size, dial_size)

        self._draw_attitude_dial(
            surface, att_rect, bank_deg=state.bank_deg, pitch_deg=state.pitch_deg
        )
        self._draw_heading_dial(surface, hdg_rect, state.heading_deg)

        att_label = self._tiny_font.render("ATTITUDE", True, text_muted)
        hdg_label = self._tiny_font.render("HEADING", True, text_muted)
        surface.blit(att_label, att_label.get_rect(midbottom=(att_rect.centerx, rect.bottom - 3)))
        surface.blit(hdg_label, hdg_label.get_rect(midbottom=(hdg_rect.centerx, rect.bottom - 3)))

    def _draw_wrapped_text(
        self,
        surface: pygame.Surface,
        text: str,
        rect: pygame.Rect,
        *,
        color: tuple[int, int, int],
        font: pygame.font.Font,
        max_lines: int,
    ) -> None:
        words = str(text).split()
        lines: list[str] = []
        cur = ""
        for word in words:
            trial = word if cur == "" else f"{cur} {word}"
            if font.size(trial)[0] <= rect.w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            cur = word
        if cur:
            lines.append(cur)

        y = rect.y
        line_h = font.get_linesize() + 2
        for line in lines[: max(0, max_lines)]:
            to_draw = line
            if font.size(to_draw)[0] > rect.w:
                while to_draw and font.size(f"{to_draw}...")[0] > rect.w:
                    to_draw = to_draw[:-1]
                to_draw = f"{to_draw}..." if to_draw else "..."
            surface.blit(font.render(to_draw, True, color), (rect.x, y))
            y += line_h

    def _draw_instrument_cluster(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
        *,
        compact: bool = False,
    ) -> None:
        panel_border = (210, 214, 224)

        for y in range(rect.y, rect.bottom):
            shade = 126 + ((y - rect.y) % 3) * 2
            pygame.draw.line(surface, (shade, shade, shade), (rect.x, y), (rect.right, y))
        pygame.draw.rect(surface, panel_border, rect, 1)
        inner = rect.inflate(-8, -8)

        gap = 5 if compact else 8
        cols = 3
        rows = 2
        cell_w = max(46, (inner.w - gap * (cols - 1)) // cols)
        cell_h = max(46, (inner.h - gap * (rows - 1)) // rows)

        cells: list[pygame.Rect] = []
        for row in range(rows):
            for col in range(cols):
                x = inner.x + col * (cell_w + gap)
                y = inner.y + row * (cell_h + gap)
                cells.append(pygame.Rect(x, y, cell_w, cell_h))

        self._draw_speed_dial(surface, cells[0], state.speed_kts)
        self._draw_attitude_dial(
            surface, cells[1], bank_deg=state.bank_deg, pitch_deg=state.pitch_deg
        )
        self._draw_heading_dial(surface, cells[2], state.heading_deg)
        self._draw_altimeter_dial(surface, cells[3], state.altitude_ft)
        self._draw_vertical_dial(surface, cells[4], state.vertical_rate_fpm, state.slip)
        self._draw_slip_indicator(surface, cells[5], bank_deg=state.bank_deg, slip=state.slip)

    def _draw_speed_dial(self, surface: pygame.Surface, rect: pygame.Rect, speed_kts: int) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for knots in range(0, 360, 10):
                turn = knots / 360.0
                ang = math.radians(-90.0 + 360.0 * turn)
                outer = inner_ring - 1
                inner = inner_ring - (9 if knots % 30 == 0 else 5)
                ox = int(round(c + math.cos(ang) * outer))
                oy = int(round(c + math.sin(ang) * outer))
                ix = int(round(c + math.cos(ang) * inner))
                iy = int(round(c + math.sin(ang) * inner))
                pygame.draw.line(base, (188, 196, 212), (ix, iy), (ox, oy), 1)

            labels = (0, 60, 120, 180, 240, 300) if size >= 64 else (0, 120, 240)
            label_font = self._tiny_font
            for knots in labels:
                turn = knots / 360.0
                ang = math.radians(-90.0 + 360.0 * turn)
                radius = inner_ring - (18 if size >= 64 else 14)
                tx = int(round(c + math.cos(ang) * radius))
                ty = int(round(c + math.sin(ang) * radius))
                label = label_font.render(str(knots), True, (226, 232, 244))
                base.blit(label, label.get_rect(center=(tx, ty)))

            title_surf = self._tiny_font.render("KNOTS", True, (166, 180, 206))
            base.blit(title_surf, title_surf.get_rect(center=(c, c - inner_ring + 12)))
            return base

        key = ("airspeed_base", size)
        surface.blit(self._get_instrument_sprite(key, build_base), dial_rect.topleft)

        ang = math.radians(-90.0 + 360.0 * airspeed_turn(int(speed_kts)))
        needle_len = max(6, face_r - 10)
        tail_len = max(4, int(round(face_r * 0.16)))
        tip = (
            int(round(cx + math.cos(ang) * needle_len)),
            int(round(cy + math.sin(ang) * needle_len)),
        )
        tail = (
            int(round(cx - math.cos(ang) * tail_len)),
            int(round(cy - math.sin(ang) * tail_len)),
        )
        pygame.draw.line(surface, (246, 248, 252), tail, tip, 4 if size >= 84 else 3)
        pygame.draw.circle(surface, (10, 10, 12), (cx, cy), max(2, size // 18))
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(1, size // 24))

    def _draw_altimeter_dial(
        self, surface: pygame.Surface, rect: pygame.Rect, altitude_ft: int
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for idx in range(50):
                turn = idx / 50.0
                ang = math.radians(-90.0 + 360.0 * turn)
                outer = inner_ring - 1
                inner = inner_ring - (9 if idx % 5 == 0 else 5)
                ox = int(round(c + math.cos(ang) * outer))
                oy = int(round(c + math.sin(ang) * outer))
                ix = int(round(c + math.cos(ang) * inner))
                iy = int(round(c + math.sin(ang) * inner))
                pygame.draw.line(base, (188, 196, 212), (ix, iy), (ox, oy), 1)

            if size >= 56:
                label_font = self._tiny_font
                for numeral in range(10):
                    turn = numeral / 10.0
                    ang = math.radians(-90.0 + 360.0 * turn)
                    radius = inner_ring - 16
                    tx = int(round(c + math.cos(ang) * radius))
                    ty = int(round(c + math.sin(ang) * radius))
                    label = label_font.render(str(numeral), True, (226, 232, 244))
                    base.blit(label, label.get_rect(center=(tx, ty)))

            title_surf = self._tiny_font.render("ALT", True, (166, 180, 206))
            base.blit(title_surf, title_surf.get_rect(center=(c, c - inner_ring + 12)))
            return base

        key = ("altimeter_base", size)
        surface.blit(self._get_instrument_sprite(key, build_base), dial_rect.topleft)

        thousands_turn, hundreds_turn = altimeter_hand_turns(int(altitude_ft))
        long_ang = math.radians(-90.0 + 360.0 * hundreds_turn)
        short_ang = math.radians(-90.0 + 360.0 * thousands_turn)
        long_len = max(7, int(round(face_r * 0.82)))
        short_len = max(5, int(round(face_r * 0.56)))
        tail_len = max(4, int(round(face_r * 0.14)))

        long_tip = (
            int(round(cx + math.cos(long_ang) * long_len)),
            int(round(cy + math.sin(long_ang) * long_len)),
        )
        long_tail = (
            int(round(cx - math.cos(long_ang) * tail_len)),
            int(round(cy - math.sin(long_ang) * tail_len)),
        )
        short_tip = (
            int(round(cx + math.cos(short_ang) * short_len)),
            int(round(cy + math.sin(short_ang) * short_len)),
        )
        short_tail = (
            int(round(cx - math.cos(short_ang) * tail_len)),
            int(round(cy - math.sin(short_ang) * tail_len)),
        )
        pygame.draw.line(surface, (214, 224, 242), short_tail, short_tip, 5 if size >= 84 else 4)
        pygame.draw.line(surface, (246, 248, 252), long_tail, long_tip, 3 if size >= 84 else 2)
        pygame.draw.circle(surface, (10, 10, 12), (cx, cy), max(2, size // 18))
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(1, size // 24))

    def _draw_vertical_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        vertical_rate_fpm: int,
        slip: int,
    ) -> None:
        _ = slip
        self._draw_scalar_dial(surface, rect, "V/S", int(vertical_rate_fpm), vmin=-2000, vmax=2000)

    def _draw_scalar_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        title: str,
        value: int,
        *,
        vmin: int,
        vmax: int,
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for idx in range(36):
                ang = math.radians(-130.0 + (260.0 / 35.0) * idx)
                outer = inner_ring - 1
                inner = inner_ring - (9 if idx % 6 == 0 else 5)
                ox = int(round(c + math.cos(ang) * outer))
                oy = int(round(c + math.sin(ang) * outer))
                ix = int(round(c + math.cos(ang) * inner))
                iy = int(round(c + math.sin(ang) * inner))
                pygame.draw.line(base, (188, 196, 212), (ix, iy), (ox, oy), 1)

            if size >= 64:
                for idx in range(6):
                    t = idx / 5.0
                    raw_v = int(round(float(vmin) + float(vmax - vmin) * t))
                    label = self._format_scalar_tick(title, raw_v)
                    ang = math.radians(-130.0 + 260.0 * t)
                    tx = int(round(c + math.cos(ang) * (inner_ring - 16)))
                    ty = int(round(c + math.sin(ang) * (inner_ring - 16)))
                    label_surf = self._tiny_font.render(label, True, (226, 232, 244))
                    base.blit(label_surf, label_surf.get_rect(center=(tx, ty)))

            title_surf = self._tiny_font.render(title, True, (166, 180, 206))
            base.blit(title_surf, title_surf.get_rect(center=(c, c - inner_ring + 12)))
            return base

        key = ("scalar_base", size, title, int(vmin), int(vmax))
        surface.blit(self._get_instrument_sprite(key, build_base), dial_rect.topleft)

        t = (float(value) - float(vmin)) / max(1.0, float(vmax - vmin))
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        ang = math.radians(-130.0 + 260.0 * t)
        needle_len = max(6, face_r - 10)
        tip = (
            int(round(cx + math.cos(ang) * needle_len)),
            int(round(cy + math.sin(ang) * needle_len)),
        )
        tail = (
            int(round(cx - math.cos(ang) * max(5, face_r * 0.16))),
            int(round(cy - math.sin(ang) * max(5, face_r * 0.16))),
        )
        pygame.draw.line(surface, (246, 248, 252), tail, tip, 4 if size >= 84 else 3)
        pygame.draw.circle(surface, (10, 10, 12), (cx, cy), max(2, size // 18))
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(1, size // 24))

    def _draw_attitude_dial(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        bank_deg: int,
        pitch_deg: int,
    ) -> None:
        dial_rect, _, _, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        # Dynamic horizon/pitch ladder layer.
        horizon_side = max(size * 3, 96)
        horizon = pygame.Surface((horizon_side, horizon_side), pygame.SRCALPHA)
        hc = horizon_side // 2
        pitch = max(-20.0, min(20.0, float(pitch_deg)))
        horizon_y = hc + int(round((pitch / 20.0) * (face_r * 0.90)))
        sky = (30, 176, 238)
        ground = (174, 108, 36)
        pygame.draw.rect(horizon, sky, pygame.Rect(0, 0, horizon_side, horizon_y))
        pygame.draw.rect(
            horizon,
            ground,
            pygame.Rect(0, horizon_y, horizon_side, max(0, horizon_side - horizon_y)),
        )
        pygame.draw.line(horizon, (246, 246, 248), (0, horizon_y), (horizon_side, horizon_y), 3)

        for mark in (-15, -10, -5, 5, 10, 15):
            y = horizon_y - int(round((mark / 20.0) * (face_r * 0.90)))
            half = int(round(face_r * (0.45 if mark % 10 == 0 else 0.30)))
            pygame.draw.line(horizon, (242, 244, 248), (hc - half, y), (hc + half, y), 2)

        rotated = pygame.transform.rotozoom(horizon, -float(bank_deg), 1.0)
        self._draw_circular_layer(surface, dial_rect, rotated, radius=face_r - 1)

        def build_overlay() -> pygame.Surface:
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)
            rim_w = max(3, outer_r - inner_ring + 1)

            # Draw only the bezel/rim so the dynamic horizon remains visible.
            pygame.draw.circle(overlay, (198, 204, 214), (c, c), outer_r, rim_w)
            pygame.draw.circle(overlay, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(overlay, (24, 30, 44), (c, c), inner_ring, 2)

            for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (9 if deg % 30 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(overlay, (204, 214, 230), (ix, iy), (ox, oy), 1)

            # Fixed airplane cue.
            wing_y = c + int(round(inner_ring * 0.10))
            wing_half = max(8, int(round(inner_ring * 0.35)))
            pygame.draw.line(
                overlay, (248, 250, 255), (c - wing_half, wing_y), (c + wing_half, wing_y), 3
            )
            pygame.draw.line(overlay, (248, 250, 255), (c, wing_y - 6), (c, wing_y + 6), 2)

            cue = [
                (c - int(inner_ring * 0.42), c + int(inner_ring * 0.10)),
                (c - int(inner_ring * 0.26), c + int(inner_ring * 0.23)),
                (c - int(inner_ring * 0.14), c + int(inner_ring * 0.08)),
                (c, c + int(inner_ring * 0.22)),
                (c + int(inner_ring * 0.14), c + int(inner_ring * 0.08)),
                (c + int(inner_ring * 0.26), c + int(inner_ring * 0.23)),
                (c + int(inner_ring * 0.42), c + int(inner_ring * 0.10)),
            ]
            pygame.draw.lines(overlay, (242, 244, 248), False, cue, 2)
            return overlay

        overlay_key = ("attitude_overlay", size)
        surface.blit(self._get_instrument_sprite(overlay_key, build_overlay), dial_rect.topleft)

    def _draw_heading_dial(
        self, surface: pygame.Surface, rect: pygame.Rect, heading_deg: int
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for deg in range(0, 360, 15):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (10 if deg % 90 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(base, (192, 202, 220), (ix, iy), (ox, oy), 1)

            for label, deg in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
                rad = math.radians(float(deg - 90))
                tx = int(round(c + math.cos(rad) * (inner_ring - 18)))
                ty = int(round(c + math.sin(rad) * (inner_ring - 18)))
                label_font = self._small_font if size >= 88 else self._tiny_font
                surf = label_font.render(label, True, (236, 244, 255))
                base.blit(surf, surf.get_rect(center=(tx, ty)))
            return base

        base_key = ("heading_base", size)
        surface.blit(self._get_instrument_sprite(base_key, build_base), dial_rect.topleft)

        def build_aircraft_icon() -> pygame.Surface:
            icon = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            wing = max(7, int(round(face_r * 0.40)))
            body = max(12, int(round(face_r * 0.82)))
            aircraft = [
                (c, c - body),
                (c + 4, c - body + 8),
                (c + 4, c - 4),
                (c + wing, c - 1),
                (c + wing, c + 3),
                (c + 4, c + 2),
                (c + 4, c + body - 10),
                (c + 9, c + body - 2),
                (c + 9, c + body + 2),
                (c + 3, c + body),
                (c - 3, c + body),
                (c - 9, c + body + 2),
                (c - 9, c + body - 2),
                (c - 4, c + body - 10),
                (c - 4, c + 2),
                (c - wing, c + 3),
                (c - wing, c - 1),
                (c - 4, c - 4),
                (c - 4, c - body + 8),
            ]
            pygame.draw.polygon(icon, (245, 248, 255), aircraft)
            arrow = [(c, c - body - 12), (c - 5, c - body - 2), (c + 5, c - body - 2)]
            pygame.draw.polygon(icon, (245, 248, 255), arrow)
            return icon

        icon_key = ("heading_icon", size)
        icon = self._get_instrument_sprite(icon_key, build_aircraft_icon)
        rotation = -float(int(heading_deg) % 360)
        rot_icon = pygame.transform.rotozoom(icon, rotation, 1.0)
        surface.blit(rot_icon, rot_icon.get_rect(center=(cx, cy)))

    def _draw_slip_indicator(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        bank_deg: int,
        slip: int,
    ) -> None:
        dial_rect, cx, cy, _, face_r = self._dial_geometry(rect)
        size = dial_rect.w

        def build_base() -> pygame.Surface:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            c = size // 2
            outer_r = size // 2 - 1
            inner_ring = max(10, outer_r - 6)

            pygame.draw.circle(base, (198, 204, 214), (c, c), outer_r)
            pygame.draw.circle(base, (86, 96, 116), (c, c), outer_r, 2)
            pygame.draw.circle(base, (0, 0, 0), (c, c), inner_ring)
            pygame.draw.circle(base, (32, 38, 52), (c, c), inner_ring, 1)

            for deg in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
                rad = math.radians(float(deg - 90))
                outer = inner_ring - 1
                inner = inner_ring - (9 if deg % 30 == 0 else 6)
                ox = int(round(c + math.cos(rad) * outer))
                oy = int(round(c + math.sin(rad) * outer))
                ix = int(round(c + math.cos(rad) * inner))
                iy = int(round(c + math.sin(rad) * inner))
                pygame.draw.line(base, (192, 202, 220), (ix, iy), (ox, oy), 1)

            left = self._tiny_font.render("L", True, (236, 244, 255))
            right = self._tiny_font.render("R", True, (236, 244, 255))
            base.blit(
                left, left.get_rect(center=(c - int(inner_ring * 0.55), c - int(inner_ring * 0.22)))
            )
            base.blit(
                right,
                right.get_rect(center=(c + int(inner_ring * 0.55), c - int(inner_ring * 0.22))),
            )
            return base

        base_key = ("slip_base", size)
        surface.blit(self._get_instrument_sprite(base_key, build_base), dial_rect.topleft)

        bank_norm = max(-1.0, min(1.0, float(bank_deg) / 35.0))
        angle = math.radians(-90.0 + bank_norm * 58.0)
        pointer_len = max(6, face_r - 8)
        tip = (
            int(round(cx + math.cos(angle) * pointer_len)),
            int(round(cy + math.sin(angle) * pointer_len)),
        )
        pygame.draw.line(surface, (246, 248, 252), (cx, cy), tip, 4 if size >= 84 else 3)
        pygame.draw.circle(surface, (246, 248, 252), (cx, cy), max(2, size // 24))

        tube_w = min(dial_rect.w - 8, max(14, int(round(face_r * 1.40))))
        tube_h = max(7, int(round(face_r * 0.46)))
        track = pygame.Rect(0, 0, tube_w, tube_h)
        track.centerx = cx
        track.centery = cy + int(round(face_r * 0.62))
        pygame.draw.rect(surface, (8, 10, 16), track)
        pygame.draw.rect(surface, (130, 146, 176), track, 1)
        pygame.draw.line(
            surface, (172, 184, 208), (track.centerx, track.y), (track.centerx, track.bottom), 1
        )

        ball_r = max(2, min(4, tube_h // 2 - 1))
        max_offset = max(1, track.w // 2 - ball_r - 2)
        offset = int(max(-1, min(1, int(slip))) * max_offset)
        ball_center = (track.centerx + offset, track.centery)
        pygame.draw.circle(surface, (244, 248, 255), ball_center, ball_r)
        pygame.draw.circle(surface, (120, 132, 156), ball_center, 1)

    def _dial_geometry(self, rect: pygame.Rect) -> tuple[pygame.Rect, int, int, int, int]:
        size = max(24, min(rect.w, rect.h))
        dial_rect = pygame.Rect(0, 0, size, size)
        dial_rect.center = rect.center
        cx = dial_rect.centerx
        cy = dial_rect.centery
        outer_r = size // 2 - 1
        face_r = max(8, outer_r - 7)
        return dial_rect, cx, cy, outer_r, face_r

    def _format_scalar_tick(self, title: str, value: int) -> str:
        if title == "ALT":
            return str((abs(int(value)) // 1000) % 10)
        if title == "V/S":
            v = int(round(float(value) / 1000.0))
            return "0" if v == 0 else f"{v:+d}"
        return str(int(value))

    def _get_instrument_sprite(
        self,
        key: tuple[object, ...],
        builder: Callable[[], pygame.Surface],
    ) -> pygame.Surface:
        cached = self._instrument_sprite_cache.get(key)
        if cached is not None:
            return cached
        built = builder()
        self._instrument_sprite_cache[key] = built
        return built

    def _draw_circular_layer(
        self,
        surface: pygame.Surface,
        dial_rect: pygame.Rect,
        layer: pygame.Surface,
        *,
        radius: int,
    ) -> None:
        face = pygame.Surface(dial_rect.size, pygame.SRCALPHA)
        face.blit(layer, layer.get_rect(center=(dial_rect.w // 2, dial_rect.h // 2)))
        mask = pygame.Surface(dial_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(
            mask, (255, 255, 255, 255), (dial_rect.w // 2, dial_rect.h // 2), max(1, radius)
        )
        face.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(face, dial_rect.topleft)

    def _draw_aircraft_orientation_card(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        state: InstrumentState,
    ) -> None:
        border = (170, 184, 212)
        sprite = self._instrument_card_bank.get_scaled_surface(
            state=state,
            size=(max(8, rect.w), max(8, rect.h)),
        )
        if sprite is not None:
            surface.blit(sprite, rect.topleft)
            pygame.draw.rect(surface, border, rect, 1)
            return

        pitch = max(-20.0, min(20.0, float(state.pitch_deg)))
        bank = max(-45.0, min(45.0, float(state.bank_deg)))
        heading = float(int(state.heading_deg) % 360)

        for y in range(rect.y, rect.bottom):
            t = (y - rect.y) / max(1, rect.h - 1)
            shade = int(round(230 - (t * 72)))
            pygame.draw.line(surface, (shade, shade, shade), (rect.x, y), (rect.right, y))

        inset_x = max(10, rect.w // 7)
        inset_y = max(10, rect.h // 7)
        rear = rect.inflate(-inset_x * 2, -inset_y * 2)
        rear.centery = rect.centery - max(2, rect.h // 18)

        left_wall = [rect.topleft, rear.topleft, rear.bottomleft, rect.bottomleft]
        right_wall = [rear.topright, rect.topright, rect.bottomright, rear.bottomright]
        ceiling = [rect.topleft, rect.topright, rear.topright, rear.topleft]
        floor = [rear.bottomleft, rear.bottomright, rect.bottomright, rect.bottomleft]

        pygame.draw.polygon(surface, (202, 202, 202), left_wall)
        pygame.draw.polygon(surface, (198, 198, 198), right_wall)
        pygame.draw.polygon(surface, (220, 220, 220), ceiling)
        pygame.draw.polygon(surface, (170, 170, 170), floor)
        pygame.draw.rect(surface, (188, 188, 188), rear)
        pygame.draw.rect(surface, (152, 152, 152), rear, 1)

        for step in (0.25, 0.5, 0.75):
            ly = int(round(rear.bottom + ((rect.bottom - rear.bottom) * step)))
            left_x = int(round(rear.left - ((rear.left - rect.left) * step)))
            right_x = int(round(rear.right + ((rect.right - rear.right) * step)))
            pygame.draw.line(surface, (146, 146, 146), (left_x, ly), (right_x, ly), 1)

        for front, back in (
            (rect.topleft, rear.topleft),
            (rect.topright, rear.topright),
            (rect.bottomleft, rear.bottomleft),
            (rect.bottomright, rear.bottomright),
        ):
            pygame.draw.line(surface, (154, 154, 154), front, back, 1)
        pygame.draw.rect(surface, border, rect, 1)

        cx = rect.centerx
        cy = rect.centery + int(round(rect.h * 0.03))
        scale = max(12.0, float(min(rect.w, rect.h)) * 0.20)

        fuselage = [
            (0.00, 3.35, 0.02),
            (0.56, 2.62, 0.36),
            (0.82, 1.10, 0.42),
            (0.70, -1.90, 0.30),
            (0.00, -3.12, 0.18),
            (-0.70, -1.90, 0.30),
            (-0.82, 1.10, 0.42),
            (-0.56, 2.62, 0.36),
        ]
        wings = [
            (-3.65, 0.44, 0.12),
            (-1.18, 0.95, 0.22),
            (-0.18, 0.36, 0.26),
            (0.18, 0.36, 0.26),
            (1.18, 0.95, 0.22),
            (3.65, 0.44, 0.12),
            (2.72, -0.56, -0.04),
            (-2.72, -0.56, -0.04),
        ]
        wing_shadow = [
            (-3.40, 0.34, -0.14),
            (-1.00, 0.68, -0.06),
            (1.00, 0.68, -0.06),
            (3.40, 0.34, -0.14),
            (2.56, -0.52, -0.18),
            (-2.56, -0.52, -0.18),
        ]
        tailplane = [
            (-1.55, -2.12, 0.10),
            (-0.32, -1.82, 0.16),
            (0.32, -1.82, 0.16),
            (1.55, -2.12, 0.10),
            (0.86, -2.66, 0.02),
            (-0.86, -2.66, 0.02),
        ]
        fin = [
            (0.00, -2.34, 0.12),
            (0.00, -1.28, 1.42),
            (0.34, -2.02, 0.34),
            (-0.34, -2.02, 0.34),
        ]
        canopy = [
            (-0.30, 1.94, 0.48),
            (0.30, 1.94, 0.48),
            (0.24, 0.98, 0.66),
            (-0.24, 0.98, 0.66),
        ]

        parts: list[
            tuple[str, list[tuple[float, float, float]], tuple[int, int, int], tuple[int, int, int]]
        ] = [
            ("wing_shadow", wing_shadow, (124, 28, 32), (154, 44, 48)),
            ("wings", wings, (214, 52, 56), (246, 110, 112)),
            ("tailplane", tailplane, (206, 50, 54), (238, 98, 102)),
            ("fuselage", fuselage, (224, 64, 66), (252, 118, 120)),
            ("fin", fin, (198, 44, 50), (232, 84, 90)),
            ("canopy", canopy, (156, 226, 232), (104, 188, 208)),
        ]

        projected_parts: list[
            tuple[float, list[tuple[int, int]], tuple[int, int, int], tuple[int, int, int]]
        ] = []
        for _, local_pts, fill_color, edge_color in parts:
            depth_total = 0.0
            pts_2d: list[tuple[int, int]] = []
            for pt in local_pts:
                rot = self._rotate_aircraft_point(
                    pt,
                    heading_deg=heading,
                    pitch_deg=pitch,
                    bank_deg=bank,
                )
                sx, sy, depth = self._project_aircraft_point(rot, cx=cx, cy=cy, scale=scale)
                pts_2d.append((sx, sy))
                depth_total += depth
            avg_depth = depth_total / float(max(1, len(local_pts)))
            projected_parts.append((avg_depth, pts_2d, fill_color, edge_color))

        # Draw far geometry first.
        for _, pts_2d, fill_color, edge_color in sorted(
            projected_parts, key=lambda t: t[0], reverse=True
        ):
            pygame.draw.polygon(surface, fill_color, pts_2d)
            pygame.draw.polygon(surface, edge_color, pts_2d, 2)

        def project_point(pt: tuple[float, float, float]) -> tuple[int, int]:
            rot = self._rotate_aircraft_point(
                pt,
                heading_deg=heading,
                pitch_deg=pitch,
                bank_deg=bank,
            )
            sx, sy, _ = self._project_aircraft_point(rot, cx=cx, cy=cy, scale=scale)
            return sx, sy

        nose = project_point((0.0, 3.38, 0.02))
        tail = project_point((0.0, -3.00, 0.18))
        pygame.draw.line(surface, (255, 238, 230), tail, nose, 2)
        pygame.draw.circle(surface, (244, 248, 255), nose, max(2, int(scale * 0.12)))

        engine_left = project_point((-1.04, 0.74, 0.02))
        engine_right = project_point((1.04, 0.74, 0.02))
        nacelle_r = max(2, int(scale * 0.14))
        pygame.draw.circle(surface, (152, 18, 24), engine_left, nacelle_r)
        pygame.draw.circle(surface, (152, 18, 24), engine_right, nacelle_r)
        pygame.draw.circle(surface, (244, 104, 110), engine_left, nacelle_r, 1)
        pygame.draw.circle(surface, (244, 104, 110), engine_right, nacelle_r, 1)

    def _rotate_aircraft_point(
        self,
        point: tuple[float, float, float],
        *,
        heading_deg: float,
        pitch_deg: float,
        bank_deg: float,
    ) -> tuple[float, float, float]:
        x, y, z = point

        # Roll around the aircraft longitudinal axis (forward/y).
        roll = math.radians(bank_deg)
        cos_r = math.cos(roll)
        sin_r = math.sin(roll)
        x1 = x * cos_r + z * sin_r
        y1 = y
        z1 = -x * sin_r + z * cos_r

        # Pitch around right axis (x).
        pitch = math.radians(pitch_deg)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        x2 = x1
        y2 = y1 * cos_p - z1 * sin_p
        z2 = y1 * sin_p + z1 * cos_p

        # Yaw around up axis (z). Heading is clockwise from North.
        yaw = math.radians(-heading_deg)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x3 = x2 * cos_y - y2 * sin_y
        y3 = x2 * sin_y + y2 * cos_y
        z3 = z2
        return x3, y3, z3

    def _project_aircraft_point(
        self,
        point: tuple[float, float, float],
        *,
        cx: int,
        cy: int,
        scale: float,
    ) -> tuple[int, int, float]:
        x, y, z = point
        sx = int(round(cx + (x + y * 0.10) * scale))
        sy = int(round(cy - (z + y * 0.34) * scale))
        return sx, sy, y

    def _visual_search_cell_color(
        self, kind: VisualSearchTaskKind, token: str
    ) -> tuple[int, int, int]:
        _ = (kind, token)
        # Keep scan field visually consistent across token types.
        return (36, 78, 70)

    def _color_pattern_cell_color(self, token: str) -> tuple[int, int, int]:
        palette = {
            "R": (200, 70, 70),
            "G": (70, 180, 100),
            "B": (80, 110, 200),
            "Y": (210, 190, 80),
            "W": (220, 220, 220),
        }
        t = str(token)
        c1 = palette.get(t[0], (90, 90, 110)) if len(t) >= 1 else (90, 90, 110)
        c2 = palette.get(t[1], c1) if len(t) >= 2 else c1
        return ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2, (c1[2] + c2[2]) // 2)

    @staticmethod
    def _angle_indicator_bearings(
        reference_bearing_deg: int,
        target_bearing_deg: int,
        angle_measure: str | None,
    ) -> list[float]:
        start = float(reference_bearing_deg) % 360.0
        end = float(target_bearing_deg) % 360.0
        clockwise_delta = (end - start) % 360.0
        counterclockwise_delta = (start - end) % 360.0

        use_clockwise = clockwise_delta <= counterclockwise_delta
        sweep_delta = clockwise_delta if use_clockwise else counterclockwise_delta

        if angle_measure == "larger":
            use_clockwise = not use_clockwise
            sweep_delta = 360.0 - sweep_delta

        steps = max(8, min(40, int(round(sweep_delta / 8.0)) + 2))
        bearings: list[float] = []
        for idx in range(steps + 1):
            t = idx / float(steps)
            offset = sweep_delta * t
            bearing = start + offset if use_clockwise else start - offset
            bearings.append(bearing % 360.0)
        return bearings

    def _bearing_point(
        self, cx: int, cy: int, radius: int, bearing_deg: int | float
    ) -> tuple[int, int]:
        rad = math.radians(float(bearing_deg))
        x = int(round(cx + math.sin(rad) * radius))
        y = int(round(cy - math.cos(rad) * radius))
        return x, y

    def _render_airborne_question(
        self, surface: pygame.Surface, snap: TestSnapshot, scenario: AirborneScenario
    ) -> None:
        w, h = surface.get_size()

        bg = (2, 7, 122)
        frame_border = (232, 240, 255)
        text_main = (238, 245, 255)
        text_muted = (210, 220, 240)
        dark_panel = (3, 8, 18)
        green_panel = (0, 120, 16)
        green_panel_dark = (0, 88, 14)
        white_panel = (244, 244, 242)

        surface.fill(bg)

        margin = max(10, min(20, w // 40))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, frame_border, frame, 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Test")

        header_h = max(24, min(30, h // 18))
        header = pygame.Rect(frame.x + 2, frame.y + 2, frame.w - 4, header_h)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface,
            frame_border,
            (header.x, header.bottom),
            (header.right, header.bottom),
            1,
        )

        title = self._tiny_font.render(
            f"Airborne Numerical Test - {phase_label}", True, text_main
        )
        surface.blit(title, title.get_rect(center=header.center))

        footer_h = max(28, min(34, h // 18))
        footer = pygame.Rect(frame.x + 1, frame.bottom - footer_h - 1, frame.w - 2, footer_h)
        pygame.draw.rect(surface, dark_panel, footer)
        pygame.draw.line(surface, frame_border, (footer.x, footer.y), (footer.right, footer.y), 1)

        work = pygame.Rect(
            frame.x + 1,
            header.bottom + 1,
            frame.w - 2,
            footer.y - header.bottom - 2,
        )
        pygame.draw.line(surface, frame_border, (work.x, work.bottom), (work.right, work.bottom), 1)

        left_w = max(280, min(int(work.w * 0.38), 420))
        left = pygame.Rect(work.x, work.y, left_w, work.h)
        right = pygame.Rect(left.right, work.y, work.w - left_w, work.h)
        pygame.draw.line(surface, frame_border, (left.right, work.y), (left.right, work.bottom), 1)

        left_pad = max(16, min(20, left.w // 20))
        right_pad = max(16, min(22, right.w // 24))
        menu_h = max(124, min(160, int(left.h * 0.24)))
        formula_h = max(54, min(72, int(left.h * 0.10)))
        menu_rect = pygame.Rect(left.x + left_pad, left.y + 10, left.w - left_pad * 2, menu_h)
        formula_rect = pygame.Rect(
            left.x + 18,
            left.bottom - formula_h - 12,
            left.w - 36,
            formula_h,
        )
        info_rect = pygame.Rect(
            left.x + 18,
            menu_rect.bottom + 12,
            left.w - 36,
            formula_rect.y - menu_rect.bottom - 24,
        )

        map_h = max(240, min(int(right.h * 0.56), right.h - 180))
        mission_h = max(92, min(126, int(right.h * 0.22)))
        table_h = max(96, right.h - map_h - mission_h - 40)

        map_rect = pygame.Rect(
            right.x + right_pad,
            right.y + 14,
            right.w - right_pad * 2,
            map_h,
        )
        mission_rect = pygame.Rect(
            right.x + max(120, int(right.w * 0.26)),
            map_rect.bottom + 18,
            right.w - max(240, int(right.w * 0.52)),
            mission_h,
        )
        table_rect = pygame.Rect(
            right.x + max(72, int(right.w * 0.12)),
            mission_rect.bottom + 18,
            right.w - max(144, int(right.w * 0.24)),
            table_h,
        )

        active_page = self._air_overlay or "intro"

        self._draw_airborne_menu_panel(
            surface,
            menu_rect,
            active_page=active_page,
            show_distances=self._air_show_distances,
            text_main=text_main,
            text_muted=text_muted,
        )

        if active_page == "intro":
            self._draw_airborne_reference_panel(
                surface,
                info_rect,
                scenario=scenario,
                active_page=active_page,
                green_panel=green_panel,
                green_panel_dark=green_panel_dark,
                text_main=text_main,
            )
            self._draw_airborne_formula_panel(
                surface,
                formula_rect,
                green_panel=green_panel,
                green_panel_dark=green_panel_dark,
                text_main=text_main,
            )
            self._draw_airborne_map_guide_panel(
                surface,
                map_rect,
                scenario=scenario,
                panel_bg=white_panel,
                text_main=text_main,
            )
            self._draw_airborne_mission_panel(
                surface,
                mission_rect,
                snap=snap,
                scenario=scenario,
                dark_panel=dark_panel,
                text_main=text_main,
            )
            self._draw_airborne_summary_panel(
                surface,
                table_rect,
                scenario=scenario,
                green_panel=green_panel,
                green_panel_dark=green_panel_dark,
                text_main=text_main,
            )
        else:
            overlay_rect = pygame.Rect(
                work.x + 12,
                menu_rect.bottom + 12,
                work.w - 24,
                work.bottom - menu_rect.bottom - 24,
            )
            self._draw_airborne_overlay_panel(
                surface,
                overlay_rect,
                scenario=scenario,
                active_page=active_page,
                green_panel=green_panel,
                green_panel_dark=green_panel_dark,
                dark_panel=dark_panel,
                text_main=text_main,
            )
        self._draw_airborne_footer(
            surface,
            footer,
            snap=snap,
            scenario=scenario,
            dark_panel=dark_panel,
            text_main=text_main,
        )

    def _draw_airborne_menu_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        active_page: str,
        show_distances: bool,
        text_main: tuple[int, int, int],
        text_muted: tuple[int, int, int],
    ) -> None:
        title = self._small_font.render("Menu", True, text_main)
        surface.blit(title, (rect.x, rect.y))
        pygame.draw.line(surface, text_main, (rect.x, rect.y + 20), (rect.right, rect.y + 20), 1)

        entries = (
            ("S", "Introduction", "intro"),
            ("D", "Speed and Fuel Consumption", "fuel"),
            ("F", "Speed and Parcel Weight", "parcel"),
            ("A", "Hold for Distances", "distances"),
        )
        row_y = rect.y + 34
        key_box = 18
        for key_label, label, token in entries:
            selected = show_distances if token == "distances" else token == active_page
            chip = pygame.Rect(rect.x + 2, row_y - 1, key_box, key_box)
            chip_fill = (
                (255, 218, 220)
                if token == "intro"
                else (190, 122, 24)
                if token == "fuel"
                else (216, 220, 248)
                if token == "parcel"
                else (180, 214, 255)
            )
            if selected:
                pygame.draw.rect(surface, (255, 255, 255), chip.inflate(4, 4), 1)
            pygame.draw.rect(surface, chip_fill, chip)
            pygame.draw.rect(surface, (26, 30, 38), chip, 1)
            key_text = self._tiny_font.render(key_label, True, (18, 18, 24))
            surface.blit(key_text, key_text.get_rect(center=chip.center))
            label_color = text_main if selected else text_muted
            text = self._small_font.render(label, True, label_color)
            surface.blit(text, (chip.right + 12, row_y - 3))
            row_y += 34

    def _draw_airborne_reference_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        scenario: AirborneScenario,
        active_page: str,
        green_panel: tuple[int, int, int],
        green_panel_dark: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel, rect)
        pygame.draw.rect(surface, (208, 236, 208), rect, 1)

        inner = rect.inflate(-12, -12)
        title_map = {
            "intro": "Introduction",
            "fuel": "Speed and Fuel Consumption",
            "parcel": "Speed and Parcel Weight",
        }
        title = self._small_font.render(title_map.get(active_page, "Reference"), True, text_main)
        surface.blit(title, (inner.x, inner.y))

        body = pygame.Rect(inner.x, inner.y + 30, inner.w, inner.h - 30)
        if active_page == "fuel":
            self._draw_airborne_reference_table(
                surface,
                body,
                rows=[
                    (str(speed), str(burn))
                    for speed, burn in scenario.speed_fuel_table
                ],
                headers=(f"Speed ({scenario.speed_unit})", "Fuel"),
                panel_fill=green_panel_dark,
                text_main=text_main,
            )
            return
        if active_page == "parcel":
            self._draw_airborne_reference_table(
                surface,
                body,
                rows=[
                    (str(weight), str(speed))
                    for weight, speed in scenario.weight_speed_table
                ],
                headers=("Weight (kg)", f"Speed ({scenario.speed_unit})"),
                panel_fill=green_panel_dark,
                text_main=text_main,
            )
            return

        intro_text = (
            "You are the operator of a large Remote Controlled Aerial Vehicle.\n\n"
            "The Aerial Vehicle operates as part of a high-tech mail delivery service.\n\n"
            "Always assume refuelling and deliveries and collections take no time.\n\n"
            "Answer in whole numbers only. Always round up if the number is 0.5 or greater "
            "and round down if the answer is 0.49 or less.\n\n"
            "Work as quickly and as accurately as possible. However you will be given marks "
            "for estimated answers."
        )
        self._draw_wrapped_text(
            surface,
            intro_text,
            body,
            color=text_main,
            font=self._small_font,
            max_lines=11,
        )

    def _draw_airborne_reference_table(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        rows: list[tuple[str, str]],
        headers: tuple[str, str],
        panel_fill: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, panel_fill, rect)
        pygame.draw.rect(surface, (208, 236, 208), rect, 1)
        left_x = rect.x + 8
        right_x = rect.x + rect.w // 2 + 4
        y = rect.y + 8
        left_head = self._tiny_font.render(headers[0], True, text_main)
        right_head = self._tiny_font.render(headers[1], True, text_main)
        surface.blit(left_head, (left_x, y))
        surface.blit(right_head, (right_x, y))
        y += 20
        pygame.draw.line(surface, (208, 236, 208), (rect.x + 6, y), (rect.right - 6, y), 1)
        y += 6
        for left_text, right_text in rows[:8]:
            left = self._tiny_font.render(left_text, True, text_main)
            right = self._tiny_font.render(right_text, True, text_main)
            surface.blit(left, (left_x, y))
            surface.blit(right, (right_x, y))
            y += 18

    def _draw_airborne_formula_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        green_panel: tuple[int, int, int],
        green_panel_dark: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel, rect)
        pygame.draw.rect(surface, (208, 236, 208), rect, 1)
        formula = self._small_font.render("Speed =", True, text_main)
        surface.blit(formula, (rect.x + 14, rect.y + rect.h // 2 - 10))
        frac_x = rect.x + 100
        distance = self._small_font.render("Distance", True, text_main)
        time_txt = self._small_font.render("Time", True, text_main)
        surface.blit(distance, (frac_x, rect.y + 8))
        pygame.draw.line(
            surface,
            text_main,
            (frac_x - 2, rect.centery),
            (frac_x + 92, rect.centery),
            2,
        )
        surface.blit(time_txt, (frac_x + 24, rect.centery + 6))
        pygame.draw.rect(surface, green_panel_dark, rect, 1)

    def _draw_airborne_map_guide_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        scenario: AirborneScenario,
        panel_bg: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, (232, 240, 255), rect, 1)
        canvas = rect.inflate(-28, -24)
        pygame.draw.rect(surface, panel_bg, canvas)
        pygame.draw.rect(surface, (14, 26, 84), canvas, 1)

        step = max(12, min(18, canvas.w // 28))
        for x in range(canvas.x, canvas.right + 1, step):
            pygame.draw.line(surface, (222, 224, 226), (x, canvas.y), (x, canvas.bottom), 1)
        for y in range(canvas.y, canvas.bottom + 1, step):
            pygame.draw.line(surface, (222, 224, 226), (canvas.x, y), (canvas.right, y), 1)

        template = TEMPLATES_BY_NAME.get(scenario.template_name)
        if template is None:
            return

        plot = canvas.inflate(-22, -26)
        node_px: list[tuple[int, int]] = []
        for nx, ny in template.nodes:
            x = int(plot.x + nx * plot.w)
            y = int(plot.y + ny * plot.h)
            node_px.append((x, y))

        for idx, (ea, eb) in enumerate(template.edges):
            a = node_px[ea]
            b = node_px[eb]
            pygame.draw.line(surface, (158, 160, 164), a, b, 2)
            if self._air_show_distances:
                self._draw_airborne_edge_distance(
                    surface,
                    a,
                    b,
                    value=str(scenario.edge_distances[idx]),
                )

        start_idx = scenario.route[0] if scenario.route else 0
        for idx, (x, y) in enumerate(node_px):
            if idx == start_idx:
                fill = (255, 18, 24)
                outline = (128, 0, 0)
            else:
                fill = (244, 244, 244)
                outline = (156, 156, 156)
            pygame.draw.circle(surface, fill, (x, y), 10)
            pygame.draw.circle(surface, outline, (x, y), 10, 2)

            label = self._small_font.render(scenario.node_names[idx], True, (42, 42, 42))
            lx = x + 14 if x < canvas.centerx else x - label.get_width() - 14
            ly = y - label.get_height() // 2
            surface.blit(label, (lx, ly))

        if scenario.route:
            start_x, start_y = node_px[start_idx]
            parcel = pygame.Rect(start_x + 18, start_y - 18, 40, 32)
            pygame.draw.rect(surface, (72, 170, 214), parcel)
            pygame.draw.rect(surface, (28, 98, 136), parcel, 2)
            box = pygame.Rect(parcel.x + 8, parcel.y + 6, 22, 16)
            pygame.draw.rect(surface, (214, 182, 132), box)
            pygame.draw.rect(surface, (118, 88, 48), box, 1)
            pygame.draw.line(surface, (118, 88, 48), (box.x, box.y + 5), (box.right, box.y + 5), 1)
            pygame.draw.line(
                surface,
                (118, 88, 48),
                (box.centerx, box.y),
                (box.centerx, box.bottom),
                1,
            )

        unit_label = {
            "km": "kilometres",
            "NM": "nautical miles",
        }.get(str(scenario.distance_unit), str(scenario.distance_unit))
        note = self._small_font.render(f"All measurements in {unit_label}", True, (42, 42, 42))
        surface.blit(note, (canvas.x + 12, canvas.bottom - note.get_height() - 10))

    def _draw_airborne_edge_distance(
        self,
        surface: pygame.Surface,
        a: tuple[int, int],
        b: tuple[int, int],
        *,
        value: str,
    ) -> None:
        midx = (a[0] + b[0]) / 2.0
        midy = (a[1] + b[1]) / 2.0
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = max(1.0, math.hypot(dx, dy))
        ox = int(round((-dy / length) * 14))
        oy = int(round((dx / length) * 14))
        text = self._small_font.render(value, True, (42, 42, 42))
        surface.blit(text, text.get_rect(center=(int(midx) + ox, int(midy) + oy)))

    def _draw_airborne_overlay_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        scenario: AirborneScenario,
        active_page: str,
        green_panel: tuple[int, int, int],
        green_panel_dark: tuple[int, int, int],
        dark_panel: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel, rect)
        pygame.draw.rect(surface, (232, 240, 255), rect, 1)

        title_map = {
            "fuel": "Speed and Fuel Consumption",
            "parcel": "Speed and Parcel Weight",
        }
        title = self._app.font.render(title_map.get(active_page, "Reference"), True, text_main)
        surface.blit(title, title.get_rect(midtop=(rect.centerx, rect.y + 18)))

        subtitle = self._small_font.render(
            "Reference page open. Release the key to return to the introduction view.",
            True,
            text_main,
        )
        surface.blit(subtitle, subtitle.get_rect(midtop=(rect.centerx, rect.y + 58)))

        content = rect.inflate(-24, -92)
        if active_page == "fuel":
            rows = [
                (str(speed), f"{burn} {scenario.fuel_unit}")
                for speed, burn in scenario.speed_fuel_table
            ]
            headers = (f"Speed ({scenario.speed_unit})", f"Fuel Burn ({scenario.fuel_unit})")
            reference_format = scenario.fuel_reference_format
            chart_x_labels = [str(speed) for speed, _ in scenario.speed_fuel_table]
            chart_values = [burn for _, burn in scenario.speed_fuel_table]
            chart_x_axis = f"Speed ({scenario.speed_unit})"
            chart_y_axis = f"Fuel Burn ({scenario.fuel_unit})"
            chart_step = scenario.fuel_chart_step
        else:
            rows = [
                (f"{weight} kg", f"{speed} {scenario.speed_unit}")
                for weight, speed in scenario.weight_speed_table
            ]
            headers = ("Parcel Weight", "Speed")
            reference_format = scenario.parcel_reference_format
            chart_x_labels = [str(weight) for weight, _ in scenario.weight_speed_table]
            chart_values = [speed for _, speed in scenario.weight_speed_table]
            chart_x_axis = "Parcel Weight (kg)"
            chart_y_axis = f"Speed ({scenario.speed_unit})"
            chart_step = scenario.parcel_chart_step

        table_rect = pygame.Rect(
            content.x,
            content.y,
            content.w,
            max(180, content.h - 64),
        )
        if reference_format == "chart":
            self._draw_airborne_overlay_chart(
                surface,
                table_rect,
                x_labels=chart_x_labels,
                values=chart_values,
                x_axis_label=chart_x_axis,
                y_axis_label=chart_y_axis,
                tick_step=chart_step,
                green_panel_dark=green_panel_dark,
                text_main=text_main,
            )
        else:
            self._draw_airborne_overlay_table(
                surface,
                table_rect,
                headers=headers,
                rows=rows,
                green_panel_dark=green_panel_dark,
                text_main=text_main,
            )

        note_rect = pygame.Rect(
            content.x,
            table_rect.bottom + 12,
            content.w,
            content.bottom - table_rect.bottom - 12,
        )
        pygame.draw.rect(surface, dark_panel, note_rect)
        pygame.draw.rect(surface, (232, 240, 255), note_rect, 1)
        note_lines = (
            [
                f"Current route starts at {scenario.node_names[scenario.route[0]]}.",
                "Map, mission box, and journey table are hidden while this page is open.",
                (
                    "Estimate between grid lines where exact bar values are not printed."
                    if reference_format == "chart"
                    else "This page lists exact reference values."
                ),
                "Release the held reference key and answer from memory.",
            ]
            if active_page == "fuel"
            else [
                f"Parcel on board: {scenario.parcel_weight_kg} kg.",
                "Map, mission box, and journey table are hidden while this page is open.",
                (
                    "Estimate between grid lines where exact bar values are not printed."
                    if reference_format == "chart"
                    else "This page lists exact reference values."
                ),
                "Release the held reference key and answer from memory.",
            ]
        )
        self._draw_wrapped_text(
            surface,
            "\n".join(note_lines),
            note_rect.inflate(-12, -10),
            color=text_main,
            font=self._small_font,
            max_lines=5,
        )

    def _draw_airborne_overlay_table(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        headers: tuple[str, str],
        rows: list[tuple[str, str]],
        green_panel_dark: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel_dark, rect)
        pygame.draw.rect(surface, (232, 240, 255), rect, 1)

        header_h = 34
        left_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.w // 2 - 1, header_h)
        right_rect = pygame.Rect(left_rect.right, rect.y + 1, rect.w - (left_rect.w + 1), header_h)
        for cell_rect, label in ((left_rect, headers[0]), (right_rect, headers[1])):
            pygame.draw.rect(surface, (0, 110, 18), cell_rect)
            pygame.draw.rect(surface, (232, 240, 255), cell_rect, 1)
            label_surf = self._small_font.render(label, True, text_main)
            surface.blit(label_surf, label_surf.get_rect(center=cell_rect.center))

        row_h = max(28, min(42, (rect.h - header_h - 8) // max(1, len(rows))))
        y = rect.y + header_h + 4
        for left_text, right_text in rows:
            row_left = pygame.Rect(rect.x + 1, y, rect.w // 2 - 1, row_h)
            row_right = pygame.Rect(row_left.right, y, rect.w - (row_left.w + 1), row_h)
            for cell_rect, label in ((row_left, left_text), (row_right, right_text)):
                pygame.draw.rect(surface, green_panel_dark, cell_rect)
                pygame.draw.rect(surface, (208, 236, 208), cell_rect, 1)
                txt = self._small_font.render(label, True, text_main)
                surface.blit(txt, txt.get_rect(center=cell_rect.center))
            y += row_h
            if y + row_h > rect.bottom:
                break

    def _draw_airborne_overlay_chart(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        x_labels: list[str],
        values: list[int],
        x_axis_label: str,
        y_axis_label: str,
        tick_step: int,
        green_panel_dark: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel_dark, rect)
        pygame.draw.rect(surface, (232, 240, 255), rect, 1)

        if not values:
            return

        plot_bg = pygame.Rect(rect.x + 54, rect.y + 18, rect.w - 74, rect.h - 68)
        pygame.draw.rect(surface, (242, 246, 245), plot_bg)
        pygame.draw.rect(surface, (208, 236, 208), plot_bg, 1)

        tick_step = max(1, tick_step)
        top_value = max(tick_step, int(math.ceil(max(values) / float(tick_step))) * tick_step)
        tick_count = max(1, top_value // tick_step)
        for tick in range(tick_count + 1):
            value = tick * tick_step
            y = plot_bg.bottom - int(round((value / float(top_value)) * plot_bg.h))
            pygame.draw.line(surface, (196, 206, 200), (plot_bg.x, y), (plot_bg.right, y), 1)
            tick_label = self._tiny_font.render(str(value), True, (26, 40, 34))
            surface.blit(tick_label, tick_label.get_rect(midright=(plot_bg.x - 8, y)))

        pygame.draw.line(
            surface,
            (26, 40, 34),
            (plot_bg.x, plot_bg.bottom),
            (plot_bg.right, plot_bg.bottom),
            2,
        )
        pygame.draw.line(
            surface,
            (26, 40, 34),
            (plot_bg.x, plot_bg.y),
            (plot_bg.x, plot_bg.bottom),
            2,
        )

        bar_gap = max(10, min(18, plot_bg.w // max(8, len(values) * 3)))
        bar_w = max(18, (plot_bg.w - bar_gap * (len(values) + 1)) // max(1, len(values)))
        x = plot_bg.x + bar_gap
        for label_text, value in zip(x_labels, values, strict=False):
            bar_h = int(round((value / float(top_value)) * (plot_bg.h - 6)))
            bar = pygame.Rect(x, plot_bg.bottom - bar_h, bar_w, bar_h)
            pygame.draw.rect(surface, (30, 112, 74), bar)
            pygame.draw.rect(surface, (20, 72, 48), bar, 1)
            label = self._tiny_font.render(label_text, True, text_main)
            surface.blit(label, label.get_rect(midtop=(bar.centerx, plot_bg.bottom + 6)))
            x += bar_w + bar_gap

        x_axis = self._tiny_font.render(x_axis_label, True, text_main)
        y_axis = self._tiny_font.render(y_axis_label, True, text_main)
        surface.blit(x_axis, x_axis.get_rect(midbottom=(rect.centerx, rect.bottom - 6)))
        surface.blit(y_axis, (rect.x + 10, rect.y + 6))

    def _draw_airborne_mission_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        snap: TestSnapshot,
        scenario: AirborneScenario,
        dark_panel: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, dark_panel, rect)
        pygame.draw.rect(surface, (232, 240, 255), rect, 1)
        if scenario.question_kind == "arrival_time":
            mission = f"Mission: Deliver parcel to {scenario.target_label}."
            task = "Task: Calculate arrival time."
        elif scenario.question_kind == "takeoff_time":
            mission = (
                f"Mission: Deliver parcel to {scenario.target_label} "
                "by the stated arrival time."
            )
            task = "Task: Calculate take off time."
        elif scenario.question_kind == "empty_time":
            mission = "Mission: Remain airborne until fuel is exhausted."
            task = "Task: Calculate empty time."
        elif scenario.question_kind == "fuel_endurance":
            mission = "Mission: Determine endurance using the current fuel state."
            task = f"Task: Calculate fuel endurance ({scenario.answer_unit_label})."
        elif scenario.question_kind == "fuel_burned":
            mission = f"Mission: Deliver parcel to {scenario.target_label}."
            task = f"Task: Calculate fuel used ({scenario.answer_unit_label})."
        elif scenario.question_kind == "parcel_weight":
            mission = (
                f"Mission: Depart at {scenario.start_time_hhmm} and deliver parcel "
                f"to {scenario.target_label}."
            )
            task = f"Task: Calculate parcel weight ({scenario.answer_unit_label})."
        elif scenario.question_kind == "parcel_effect":
            mission = f"Mission: Deliver parcel to {scenario.target_label}."
            task = f"Task: Calculate parcel effect on speed ({scenario.answer_unit_label})."
        else:
            mission = f"Mission: Deliver parcel to {scenario.target_label}."
            task = f"Task: Calculate distance travelled ({scenario.answer_unit_label})."
        mission_text = self._small_font.render(mission, True, text_main)
        surface.blit(mission_text, (rect.x + 20, rect.y + 12))
        prompt_text = str(snap.prompt).strip() or task
        prompt_rect = pygame.Rect(rect.x + 20, rect.y + 38, rect.w - 40, rect.h - 46)
        self._draw_wrapped_text(
            surface,
            prompt_text,
            prompt_rect,
            color=text_main,
            font=self._small_font,
            max_lines=5,
        )

    def _draw_airborne_summary_panel(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        scenario: AirborneScenario,
        green_panel: tuple[int, int, int],
        green_panel_dark: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, green_panel, rect)
        pygame.draw.rect(surface, (208, 236, 208), rect, 1)

        via_label = "-"
        if len(scenario.route) > 2:
            via_label = "/".join(scenario.node_names[idx] for idx in scenario.route[1:-1])
        given_heading = getattr(scenario, "given_time_label", "Time Now")
        if scenario.answer_format == "hhmm":
            answer_heading = scenario.answer_label
        else:
            answer_heading = f"{scenario.answer_label}\n({scenario.answer_unit_label})"

        cols = (
            ("Start point", 0.18),
            ("Destination", 0.21),
            ("Via", 0.16),
            (given_heading, 0.17),
            (answer_heading, 0.15),
            ("Yes/No", 0.07),
            ("Weight\n(kg)", 0.06),
        )
        x = rect.x + 2
        col_rects: list[pygame.Rect] = []
        remaining = rect.w - 4
        for idx, (_, frac) in enumerate(cols):
            if idx == len(cols) - 1:
                width = remaining
            else:
                width = int(round((rect.w - 4) * frac))
                remaining -= width
            col_rects.append(pygame.Rect(x, rect.y + 28, width, rect.h - 30))
            x += width

        groups = (
            ("Journey", (0, 3)),
            ("Timings", (3, 5)),
            ("Parcel", (5, 7)),
        )
        for label, (start, end) in groups:
            group_rect = pygame.Rect(
                col_rects[start].x,
                rect.y + 2,
                col_rects[end - 1].right - col_rects[start].x,
                24,
            )
            pygame.draw.rect(surface, green_panel_dark, group_rect)
            pygame.draw.rect(surface, (208, 236, 208), group_rect, 1)
            group_text = self._small_font.render(label, True, text_main)
            surface.blit(group_text, group_text.get_rect(center=group_rect.center))

        headers = [label for label, _ in cols]
        values = [
            scenario.node_names[scenario.route[0]] if scenario.route else "-",
            scenario.node_names[scenario.route[-1]] if scenario.route else "-",
            via_label,
            scenario.given_time_hhmm,
            "",
            "Y",
            "" if scenario.question_kind == "parcel_weight" else str(scenario.parcel_weight_kg),
        ]
        for rect_cell, head, value in zip(col_rects, headers, values, strict=True):
            pygame.draw.rect(surface, green_panel, rect_cell)
            pygame.draw.rect(surface, (208, 236, 208), rect_cell, 1)
            header_rect = pygame.Rect(rect_cell.x, rect_cell.y, rect_cell.w, 28)
            pygame.draw.rect(surface, green_panel_dark, header_rect)
            pygame.draw.rect(surface, (208, 236, 208), header_rect, 1)
            head_font = self._tiny_font
            head_lines = str(head).split("\n")
            if len(head_lines) == 1:
                surf = head_font.render(head_lines[0], True, text_main)
                surface.blit(surf, surf.get_rect(center=header_rect.center))
            else:
                y = header_rect.y + 2
                for line in head_lines:
                    surf = head_font.render(line, True, text_main)
                    surface.blit(surf, surf.get_rect(centerx=header_rect.centerx, y=y))
                    y += surf.get_height() - 1
            value_text = self._small_font.render(value, True, text_main)
            value_y = rect_cell.bottom - value_text.get_height() - 6
            surface.blit(value_text, value_text.get_rect(centerx=rect_cell.centerx, y=value_y))

    def _draw_airborne_footer(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        snap: TestSnapshot,
        scenario: AirborneScenario,
        dark_panel: tuple[int, int, int],
        text_main: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surface, dark_panel, rect)

        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            rem_txt = f"{rem // 60:02d}:{rem % 60:02d}"
        else:
            rem_txt = "--:--"

        practice_total = int(getattr(self._engine, "_practice_questions", 0))
        practice_answered = int(getattr(self._engine, "_practice_answered", 0))
        if snap.phase is Phase.PRACTICE and practice_total > 0:
            practice_idx = min(practice_total, practice_answered + 1)
            left_text = f"Practice: {practice_idx} of {practice_total}"
        elif snap.phase is Phase.SCORED:
            left_text = f"Scored: {snap.correct_scored}/{snap.attempted_scored}"
        else:
            left_text = {
                Phase.INSTRUCTIONS: "Instructions",
                Phase.PRACTICE_DONE: "Practice Complete",
                Phase.RESULTS: "Results",
            }.get(snap.phase, "Airborne Numerical")

        left = self._tiny_font.render(left_text, True, text_main)
        surface.blit(left, (rect.x + 8, rect.y + 7))

        if getattr(scenario, "answer_format", "hhmm") == "hhmm":
            answer_label = "Answer (HHMM)"
        else:
            unit = str(getattr(scenario, "answer_unit_label", "")).strip()
            answer_label = f"Answer ({unit}, 4 digits)" if unit else "Answer (4 digits)"
        label = self._tiny_font.render(answer_label, True, text_main)
        surface.blit(label, label.get_rect(midtop=(rect.centerx, rect.y + 2)))

        slot_count = max(4, int(getattr(scenario, "answer_digits", 4)))
        slot_w = max(18, min(24, rect.w // 30))
        gap = max(3, min(6, rect.w // 180))
        total_w = slot_w * slot_count + gap * (slot_count - 1)
        start_x = rect.centerx - total_w // 2
        box_y = rect.y + 16
        show_input = snap.phase in (Phase.PRACTICE, Phase.SCORED)
        caret_on = (pygame.time.get_ticks() // 500) % 2 == 0
        for idx in range(slot_count):
            box = pygame.Rect(start_x + idx * (slot_w + gap), box_y, slot_w, 14)
            pygame.draw.rect(surface, (0, 0, 0), box)
            pygame.draw.rect(surface, (216, 224, 236), box, 1)
            ch = self._input[idx] if idx < len(self._input) else ""
            if show_input and ch:
                txt = self._tiny_font.render(ch, True, text_main)
                surface.blit(txt, txt.get_rect(center=box.center))
            elif (
                show_input
                and idx == len(self._input)
                and caret_on
                and len(self._input) < slot_count
            ):
                pygame.draw.line(
                    surface,
                    text_main,
                    (box.centerx, box.y + 2),
                    (box.centerx, box.bottom - 2),
                    1,
                )

        right = self._tiny_font.render(f"Time Left: {rem_txt}", True, text_main)
        surface.blit(right, right.get_rect(topright=(rect.right - 8, rect.y + 7)))

    def _render_colours_letters_numbers_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: ColoursLettersNumbersPayload | None,
    ) -> None:
        self._cln_option_hitboxes = {}

        w, h = surface.get_size()
        bg = (2, 8, 118)
        frame_edge = (228, 236, 255)
        gray_panel = (176, 176, 176)
        dark_panel = (8, 8, 10)
        text_light = (238, 245, 255)
        text_dark = (14, 14, 18)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        rem_txt = "--:--"
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            rem_txt = f"{rem // 60:02d}:{rem % 60:02d}"

        surface.fill(bg)
        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, frame_edge, frame, 1)

        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, max(26, min(34, h // 18)))
        footer = pygame.Rect(frame.x + 1, frame.bottom - 34, frame.w - 2, 33)
        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(
            surface, frame_edge, (header.x, header.bottom), (header.right, header.bottom), 1
        )
        pygame.draw.rect(surface, dark_panel, footer)
        pygame.draw.line(surface, frame_edge, (footer.x, footer.y), (footer.right, footer.y), 1)

        title = self._tiny_font.render(
            f"Colours, Letters and Numbers - {phase_label}", True, text_light
        )
        surface.blit(title, title.get_rect(center=header.center))

        body = pygame.Rect(
            frame.x + 1, header.bottom + 1, frame.w - 2, footer.y - header.bottom - 2
        )

        bar_colors = {
            "RED": (255, 44, 48),
            "YELLOW": (228, 232, 84),
            "GREEN": (92, 236, 96),
            "BLUE": (60, 114, 242),
        }
        key_for_color = {
            "RED": "Q",
            "YELLOW": "W",
            "GREEN": "E",
            "BLUE": "R",
        }

        if snap.phase is Phase.INSTRUCTIONS:
            pad_x = max(24, min(120, body.w // 8))
            pad_y = max(20, min(90, body.h // 8))
            panel = pygame.Rect(
                body.x + pad_x,
                body.y + pad_y,
                body.w - (pad_x * 2),
                body.h - (pad_y * 2),
            )
            if panel.w < 280 or panel.h < 180:
                panel = body.inflate(-20, -20)

            pygame.draw.rect(surface, dark_panel, panel)
            pygame.draw.rect(surface, frame_edge, panel, 1)

            head = self._small_font.render("How this test works", True, text_light)
            surface.blit(head, (panel.x + 14, panel.y + 12))

            help_text = "\n".join(
                [
                    "1) Memorize the letter sequence at the top.",
                    "2) Hold it in memory during a random blank gap.",
                    "3) Pick the matching corner with A/S/D/F or mouse click.",
                    "4) Type the math answer and press Enter (no math timer).",
                    "5) Clear diamonds inside the color lanes with Q/W/E/R.",
                    "6) Blank gaps vary between 5 and 60 seconds.",
                    "7) Memory, math, and colours run independently.",
                    "8) Missed diamonds reduce your score.",
                ]
            )
            self._draw_wrapped_text(
                surface,
                help_text,
                pygame.Rect(panel.x + 14, panel.y + 44, panel.w - 28, max(64, panel.h - 122)),
                color=text_light,
                font=self._small_font,
                max_lines=8,
            )

            legend = pygame.Rect(panel.x + 14, panel.bottom - 70, panel.w - 28, 54)
            pygame.draw.rect(surface, (16, 18, 30), legend)
            pygame.draw.rect(surface, frame_edge, legend, 1)
            legend_title = self._tiny_font.render(
                "Color keys (left -> right): Q / W / E / R", True, text_light
            )
            surface.blit(legend_title, (legend.x + 8, legend.y + 6))

            chips = [("RED", "Q"), ("YELLOW", "W"), ("GREEN", "E"), ("BLUE", "R")]
            chip_w = max(56, min(90, (legend.w - 18) // 4))
            chip_h = 22
            gap = max(4, (legend.w - (chip_w * 4)) // 5)
            cy = legend.bottom - chip_h - 8
            x = legend.x + gap
            for color_name, key_lbl in chips:
                chip = pygame.Rect(x, cy, chip_w, chip_h)
                pygame.draw.rect(surface, bar_colors.get(color_name, (120, 120, 120)), chip)
                pygame.draw.rect(surface, frame_edge, chip, 1)
                k = self._tiny_font.render(key_lbl, True, text_dark)
                surface.blit(k, k.get_rect(center=chip.center))
                x += chip_w + gap

        elif snap.phase in (Phase.PRACTICE_DONE, Phase.RESULTS):
            panel = body.inflate(-max(32, body.w // 10), -max(22, body.h // 10))
            pygame.draw.rect(surface, dark_panel, panel)
            pygame.draw.rect(surface, frame_edge, panel, 1)
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                panel.inflate(-18, -16),
                color=text_light,
                font=self._small_font,
                max_lines=14,
            )

        else:
            corner_w = max(164, min(228, int(body.w * 0.25)))
            corner_h = max(96, min(142, int(body.h * 0.24)))
            top_left = pygame.Rect(body.x, body.y, corner_w, corner_h)
            top_right = pygame.Rect(body.right - corner_w, body.y, corner_w, corner_h)
            mid_left = pygame.Rect(body.x, body.centery - (corner_h // 2), corner_w, corner_h)
            mid_right = pygame.Rect(
                body.right - corner_w, body.centery - (corner_h // 2), corner_w, corner_h
            )
            bottom_center = pygame.Rect(
                body.centerx - (corner_w // 2),
                body.bottom - (corner_h * 2) - max(18, body.h // 20),
                corner_w,
                corner_h,
            )
            option_rects = [top_left, top_right, mid_left, mid_right, bottom_center]
            labels = ["A", "S", "D", "F", "G"]
            options = payload.options if payload is not None else tuple()
            options_visible = bool(payload is not None and payload.options_active)

            for i, rect in enumerate(option_rects):
                pygame.draw.rect(surface, gray_panel, rect)
                pygame.draw.rect(surface, frame_edge, rect, 1)

                shown_value = ""
                if options_visible and i < len(options):
                    shown_value = options[i].label
                    self._cln_option_hitboxes[i + 1] = rect.copy()
                if shown_value:
                    text = self._mid_font.render(shown_value, True, text_dark)
                    if text.get_width() > rect.w - 16:
                        text = self._small_font.render(shown_value, True, text_dark)
                    surface.blit(text, text.get_rect(center=(rect.centerx, rect.centery + 2)))

                badge = pygame.Rect(rect.right - 20, rect.bottom - 20, 18, 18)
                pygame.draw.rect(surface, dark_panel, badge)
                pygame.draw.rect(surface, frame_edge, badge, 1)
                badge_text = self._tiny_font.render(labels[i], True, text_light)
                surface.blit(badge_text, badge_text.get_rect(center=badge.center))

            max_center_w = max(280, body.w - (corner_w * 2) - max(16, body.w // 24))
            center_w = max(280, min(max_center_w, int(body.w * 0.50)))
            top_mid_w = max(220, min(340, center_w - 24))
            top_mid_h = max(78, min(96, int(body.h * 0.18)))
            top_mid = pygame.Rect(
                body.centerx - (top_mid_w // 2),
                body.y + max(12, body.h // 30),
                top_mid_w,
                top_mid_h,
            )
            pygame.draw.rect(surface, bg, top_mid)
            pygame.draw.rect(surface, frame_edge, top_mid, 1)

            eq_w = max(224, min(300, int(body.w * 0.28)))
            eq_h = max(52, min(74, int(corner_h * 0.58)))
            eq_rect = pygame.Rect(
                body.centerx - (eq_w // 2),
                body.bottom - corner_h + max(12, (corner_h - eq_h) // 2),
                eq_w,
                eq_h,
            )

            center_gap = max(10, body.h // 38)
            center_top = top_mid.bottom + center_gap
            center_bottom = eq_rect.y - center_gap
            center_h = max(192, min(int(body.h * 0.63), center_bottom - center_top))
            center_rect = pygame.Rect(
                body.centerx - (center_w // 2), center_top, center_w, center_h
            )

            pygame.draw.rect(surface, dark_panel, center_rect)
            pygame.draw.rect(surface, frame_edge, center_rect, 1)

            lane_colors = (
                payload.lane_colors if payload is not None else ("RED", "YELLOW", "GREEN", "BLUE")
            )
            lane_start_norm = payload.lane_start_norm if payload is not None else 0.54
            lane_end_norm = payload.lane_end_norm if payload is not None else 0.98
            lane_start_norm = max(0.0, min(1.0, lane_start_norm))
            lane_end_norm = max(lane_start_norm + 0.01, min(1.0, lane_end_norm))
            lane_start_x = center_rect.x + int(lane_start_norm * center_rect.w)
            lane_end_x = center_rect.x + int(lane_end_norm * center_rect.w)
            lane_zone = pygame.Rect(
                lane_start_x,
                center_rect.y + 1,
                max(1, lane_end_x - lane_start_x),
                center_rect.h - 2,
            )
            pygame.draw.rect(surface, frame_edge, lane_zone, 1)
            lane_count = max(1, len(lane_colors))
            for i, color_name in enumerate(lane_colors):
                color = bar_colors.get(color_name, (128, 128, 128))
                lx0 = lane_zone.x + int((i * lane_zone.w) / lane_count)
                lx1 = lane_zone.x + int(((i + 1) * lane_zone.w) / lane_count)
                lane = pygame.Rect(lx0, lane_zone.y, max(1, lx1 - lx0), lane_zone.h)
                pygame.draw.rect(surface, color, lane)
                key_lbl = key_for_color.get(color_name, "?")
                key_s = self._tiny_font.render(key_lbl, True, (14, 14, 18))
                surface.blit(key_s, key_s.get_rect(midtop=(lane.centerx, lane.y + 4)))

            diamonds = payload.diamonds if payload is not None else tuple()
            row_y = [
                center_rect.y + int(center_rect.h * 0.40),
                center_rect.y + int(center_rect.h * 0.61),
                center_rect.y + int(center_rect.h * 0.82),
            ]
            diamond_size = max(7, min(10, center_rect.h // 24))
            for d in diamonds:
                x = center_rect.x + int(d.x_norm * max(1, center_rect.w - 1))
                y = row_y[max(0, min(len(row_y) - 1, int(d.row)))]
                poly = [
                    (x, y - diamond_size),
                    (x + diamond_size, y),
                    (x, y + diamond_size),
                    (x - diamond_size, y),
                ]
                color = bar_colors.get(d.color, (180, 180, 180))
                pygame.draw.polygon(surface, color, poly)
                pygame.draw.polygon(surface, frame_edge, poly, 1)

            pygame.draw.rect(surface, (100, 100, 100), eq_rect)
            pygame.draw.rect(surface, frame_edge, eq_rect, 1)
            eq_text = payload.math_prompt if payload is not None else "0 + 0 ="
            eq_text = eq_text.replace("SOLVE:", "").strip()
            eq_s = self._mid_font.render(eq_text, True, text_dark)
            if eq_s.get_width() > eq_rect.w - 12:
                eq_s = self._small_font.render(eq_text, True, text_dark)
            surface.blit(eq_s, eq_s.get_rect(center=eq_rect.center))

            if payload is not None and payload.target_sequence is not None:
                seq = self._mid_font.render(payload.target_sequence, True, text_light)
                surface.blit(seq, seq.get_rect(center=top_mid.center))
                hint_text = "Memorize sequence"
            elif payload is not None and not payload.options_active and not payload.memory_answered:
                hint_text = "Hold sequence in memory"
            elif payload is not None and not payload.memory_answered:
                hint_text = "Pick with A/S/D/F/G or mouse"
            elif payload is not None and payload.memory_answered:
                hint_text = "Sequence selected"
            else:
                hint_text = ""
            if hint_text:
                hint = self._tiny_font.render(hint_text, True, text_light)
                surface.blit(hint, hint.get_rect(midbottom=(top_mid.centerx, top_mid.bottom - 6)))

        attempted = max(0, int(snap.attempted_scored))
        correct = max(0, int(snap.correct_scored))
        accuracy = 0.0 if attempted == 0 else (correct / attempted) * 100.0
        misses = payload.missed_diamonds if payload is not None else 0
        cleared = payload.cleared_diamonds if payload is not None else 0
        points = payload.points if payload is not None else 0.0

        if snap.phase in (Phase.PRACTICE, Phase.SCORED):
            caret = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
            answer_value = self._input + caret
            control_text = (
                "Memory: A/S/D/F/G or mouse  |  Colour lanes: Q/W/E/R  |  Enter: math"
            )
        elif snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            answer_value = "--"
            control_text = "Press Enter to continue"
        else:
            answer_value = "--"
            control_text = ""

        left = self._tiny_font.render(control_text, True, text_light)
        center = self._small_font.render(f"Math Answer: {answer_value}", True, text_light)
        right = self._tiny_font.render(
            f"Scored {correct}/{attempted} ({accuracy:.1f}%)  "
            f"Clear {cleared}  Miss {misses}  Pts {points:.1f}  T {rem_txt}",
            True,
            text_light,
        )
        surface.blit(left, (footer.x + 8, footer.y + 3))
        surface.blit(center, center.get_rect(midleft=(footer.x + 10, footer.bottom - 10)))
        surface.blit(right, right.get_rect(midright=(footer.right - 8, footer.y + 9)))

    def _render_digit_recognition_screen(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        w, h = surface.get_size()
        bg = (2, 8, 114)
        edge = (232, 240, 255)
        text_main = (236, 244, 255)
        text_muted = (184, 198, 224)

        surface.fill(bg)
        margin = max(8, min(16, w // 56))
        frame = pygame.Rect(margin, margin, w - margin * 2, h - margin * 2)
        pygame.draw.rect(surface, bg, frame)
        pygame.draw.rect(surface, edge, frame, 1)

        header_h = max(24, min(32, h // 18))
        footer_h = max(30, min(38, h // 14))
        header = pygame.Rect(frame.x + 1, frame.y + 1, frame.w - 2, header_h)
        footer = pygame.Rect(frame.x + 1, frame.bottom - footer_h - 1, frame.w - 2, footer_h)
        body = pygame.Rect(
            frame.x + 1, header.bottom + 1, frame.w - 2, footer.y - header.bottom - 2
        )

        pygame.draw.rect(surface, bg, header)
        pygame.draw.line(surface, edge, (header.x, header.bottom), (header.right, header.bottom), 1)
        pygame.draw.rect(surface, (0, 0, 0), footer)
        pygame.draw.line(surface, edge, (footer.x, footer.y), (footer.right, footer.y), 1)

        phase_label = {
            Phase.INSTRUCTIONS: "Instructions",
            Phase.PRACTICE: "Practice",
            Phase.PRACTICE_DONE: "Practice Complete",
            Phase.SCORED: "Timed Test",
            Phase.RESULTS: "Results",
        }.get(snap.phase, "Task")

        title = self._tiny_font.render(f"Digit Recognition - {phase_label}", True, text_main)
        surface.blit(title, title.get_rect(center=header.center))

        rem_txt = "--:--"
        if snap.time_remaining_s is not None:
            rem = int(round(snap.time_remaining_s))
            rem_txt = f"{rem // 60:02d}:{rem % 60:02d}"

        if snap.phase in (Phase.INSTRUCTIONS, Phase.PRACTICE_DONE, Phase.RESULTS):
            self._draw_wrapped_text(
                surface,
                str(snap.prompt),
                body.inflate(-40, -34),
                color=text_main,
                font=self._small_font,
                max_lines=12,
            )
        elif payload is not None and (
            payload.display_digits is not None
            or getattr(payload, "display_lines", None) is not None
        ):
            display_lines = tuple(getattr(payload, "display_lines", ()) or ())
            if not display_lines and payload.display_digits is not None:
                display_lines = (payload.display_digits,)

            if len(display_lines) == 1:
                digits = self._big_font.render(display_lines[0], True, text_main)
                if digits.get_width() > int(body.w * 0.9):
                    digits = self._mid_font.render(display_lines[0], True, text_main)
                surface.blit(digits, digits.get_rect(center=body.center))
            else:
                line_surfaces: list[pygame.Surface] = []
                max_width = int(body.w * 0.9)
                for line in display_lines:
                    surf = self._mid_font.render(line, True, text_main)
                    if surf.get_width() > max_width:
                        surf = self._small_font.render(line, True, text_main)
                    line_surfaces.append(surf)

                total_h = sum(surf.get_height() for surf in line_surfaces) + (
                    (len(line_surfaces) - 1) * 16
                )
                y = body.centery - total_h // 2
                for surf in line_surfaces:
                    surface.blit(surf, surf.get_rect(centerx=body.centerx, y=y))
                    y += surf.get_height() + 16
        elif payload is not None and not payload.accepting_input:
            mask = self._mid_font.render("X X X X X X X X", True, text_muted)
            surface.blit(mask, mask.get_rect(center=body.center))
        else:
            prompt_box = pygame.Rect(
                body.x + max(24, body.w // 10),
                body.y + max(34, body.h // 6),
                body.w - max(48, body.w // 5),
                max(116, min(180, body.h // 3)),
            )
            self._render_centered_prompt_panel(
                surface,
                prompt_box,
                str(snap.prompt),
                fill=(8, 18, 120),
                border=edge,
                text_color=text_main,
                preferred_size=max(30, min(50, h // 14)),
                min_size=max(20, min(32, h // 22)),
            )

        info = self._tiny_font.render(
            f"Scored {snap.correct_scored}/{snap.attempted_scored}  |  Time Left: {rem_txt}",
            True,
            text_main,
        )
        surface.blit(info, (footer.x + 12, footer.y + (footer.h - info.get_height()) // 2))

    def _render_digit_recognition_answer_box(
        self,
        surface: pygame.Surface,
        snap: TestSnapshot,
        payload: DigitRecognitionPayload | None,
    ) -> None:
        if payload is None or not payload.accepting_input:
            return

        w, h = surface.get_size()
        box = pygame.Rect(
            (w - max(320, min(560, int(w * 0.56)))) // 2,
            int(h * 0.70),
            max(320, min(560, int(w * 0.56))),
            max(58, min(74, int(h * 0.11))),
        )
        self._render_centered_input_box(
            surface,
            box,
            label="Your Answer",
            hint=snap.input_hint,
            entry_text=self._input,
            fill=(6, 15, 92),
            border=(180, 196, 230),
            label_color=(236, 244, 255),
            input_color=(236, 244, 255),
            hint_color=(168, 184, 214),
            label_font=self._small_font,
            input_font=self._mid_font,
            hint_font=self._tiny_font,
        )

    def _airborne_graph_seed(self, scenario: AirborneScenario) -> int:
        # Stable per-scenario seed (no Python hash()).
        seed = 2166136261

        def mix(x: int) -> None:
            nonlocal seed
            seed ^= x & 0xFFFFFFFF
            seed = (seed * 16777619) & 0xFFFFFFFF

        mix(int(getattr(scenario, "speed_value", 0)))
        mix(int(getattr(scenario, "fuel_burn_per_hr", 0)))
        mix(int(getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))))
        for name in getattr(scenario, "node_names", ()):
            for ch in name:
                mix(ord(ch))
        for idx in getattr(scenario, "route", ()):
            mix(int(idx))
        return seed

    def _draw_airborne_bar_chart(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        title: str,
        x_labels: list[str],
        values: list[int],
        value_unit: str,
    ) -> None:
        pygame.draw.rect(surface, (18, 18, 26), rect)
        pygame.draw.rect(surface, (120, 120, 140), rect, 2)

        surface.blit(
            self._small_font.render(title, True, (235, 235, 245)), (rect.x + 12, rect.y + 10)
        )

        chart = pygame.Rect(rect.x + 40, rect.y + 42, rect.w - 56, rect.h - 70)
        pygame.draw.line(
            surface, (140, 140, 150), (chart.x, chart.bottom), (chart.right, chart.bottom), 1
        )
        pygame.draw.line(surface, (140, 140, 150), (chart.x, chart.y), (chart.x, chart.bottom), 1)

        if not values:
            return

        vmax = max(values) or 1
        n = len(values)
        gap = 8
        bar_w = max(10, (chart.w - gap * (n + 1)) // n)

        for i, (lbl, v) in enumerate(zip(x_labels, values, strict=False)):
            x = chart.x + gap + i * (bar_w + gap)
            hh = int(round((v / vmax) * (chart.h - 20)))
            bar = pygame.Rect(x, chart.bottom - hh, bar_w, hh)
            pygame.draw.rect(surface, (90, 90, 110), bar)

            t = self._tiny_font.render(f"{v}{value_unit}", True, (200, 200, 210))
            surface.blit(t, t.get_rect(midbottom=(bar.centerx, bar.y - 2)))

            xl = self._tiny_font.render(lbl, True, (150, 150, 165))
            surface.blit(xl, xl.get_rect(midtop=(bar.centerx, chart.bottom + 4)))

    def _draw_airborne_table_small(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        *,
        title: str,
        headers: tuple[str, str],
        rows: list[tuple[str, str]],
    ) -> None:
        pygame.draw.rect(surface, (18, 18, 26), rect)
        pygame.draw.rect(surface, (120, 120, 140), rect, 2)
        surface.blit(
            self._small_font.render(title, True, (235, 235, 245)), (rect.x + 12, rect.y + 10)
        )

        x1 = rect.x + 14
        x2 = rect.x + rect.w // 2 + 10
        y = rect.y + 46

        surface.blit(self._tiny_font.render(headers[0], True, (150, 150, 165)), (x1, y))
        surface.blit(self._tiny_font.render(headers[1], True, (150, 150, 165)), (x2, y))
        y += 20

        for a, b in rows[:10]:
            surface.blit(self._tiny_font.render(a, True, (235, 235, 245)), (x1, y))
            surface.blit(self._tiny_font.render(b, True, (235, 235, 245)), (x2, y))
            y += 20

    def _draw_airborne_fuel_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario
    ) -> None:
        seed = self._airborne_graph_seed(scenario)
        base_speed = max(1, int(getattr(scenario, "speed_value", 1)))
        base_burn = int(getattr(scenario, "fuel_burn_per_hr", 0))
        speeds = [int(round(base_speed * f)) for f in (0.8, 0.9, 1.0, 1.1, 1.2)]
        exp = 2.0 if (seed & 2) == 0 else 2.2
        burns = [int(round(base_burn * ((s / base_speed) ** exp))) for s in speeds]
        labels = [str(s) for s in speeds]

        if (seed & 1) == 0:
            self._draw_airborne_bar_chart(
                surface,
                rect,
                title="Fuel burn vs speed",
                x_labels=labels,
                values=burns,
                value_unit="",
            )
        else:
            rows = [
                (f"{sp} {getattr(scenario, 'speed_unit', '')}", f"{bn} L/hr")
                for sp, bn in zip(speeds, burns, strict=False)
            ]
            self._draw_airborne_table_small(
                surface, rect, title="Fuel burn table", headers=("SPEED", "BURN"), rows=rows
            )

    def _draw_airborne_parcel_panel(
        self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario
    ) -> None:
        seed = self._airborne_graph_seed(scenario) ^ 0x9E3779B9
        base_speed = max(1, int(getattr(scenario, "speed_value", 1)))
        weights = [0, 200, 400, 600, 800, 1000]
        slope = 6 + (seed % 7)  # speed drop per 100kg
        speeds = [max(1, base_speed - int((w / 100) * slope)) for w in weights]
        wlabels = [str(w) for w in weights]

        if (seed & 1) == 0:
            self._draw_airborne_bar_chart(
                surface,
                rect,
                title="Speed vs parcel weight",
                x_labels=wlabels,
                values=speeds,
                value_unit="",
            )
        else:
            rows = [
                (f"{w} kg", f"{sp} {getattr(scenario, 'speed_unit', '')}")
                for w, sp in zip(weights, speeds, strict=False)
            ]
            self._draw_airborne_table_small(
                surface, rect, title="Parcel weight table", headers=("WEIGHT", "SPEED"), rows=rows
            )

    def _draw_airborne_map(
        self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario
    ) -> None:
        template = TEMPLATES_BY_NAME.get(scenario.template_name)
        if template is None:
            return

        node_px: list[tuple[int, int]] = []
        for nx, ny in template.nodes:
            x = int(rect.x + nx * rect.w)
            y = int(rect.y + ny * rect.h)
            node_px.append((x, y))

        for idx, (ea, eb) in enumerate(template.edges):
            a = node_px[ea]
            b = node_px[eb]
            pygame.draw.line(surface, (70, 70, 85), a, b, 2)

            if self._air_show_distances:
                midx = (a[0] + b[0]) / 2
                midy = (a[1] + b[1]) / 2
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                length = max(1.0, (dx * dx + dy * dy) ** 0.5)
                ox = int(-dy / length * 10)
                oy = int(dx / length * 10)

                text = self._tiny_font.render(str(scenario.edge_distances[idx]), True, (12, 12, 18))
                bg = text.get_rect(center=(int(midx) + ox, int(midy) + oy))
                bg.inflate_ip(10, 6)
                pygame.draw.rect(surface, (235, 235, 245), bg)
                surface.blit(text, text.get_rect(center=bg.center))

        for i, (x, y) in enumerate(node_px):
            pygame.draw.circle(surface, (12, 12, 18), (x, y), 10)
            pygame.draw.circle(surface, (235, 235, 245), (x, y), 10, 2)

            lx = x + 18 if x < rect.right - 80 else x - 18
            ly = y
            label = self._tiny_font.render(scenario.node_names[i], True, (12, 12, 18))
            bg = label.get_rect(midleft=(lx, ly))
            bg.inflate_ip(10, 6)
            pygame.draw.rect(surface, (235, 235, 245), bg)
            surface.blit(label, label.get_rect(midleft=bg.midleft))

    def _draw_airborne_table(
        self, surface: pygame.Surface, rect: pygame.Rect, scenario: AirborneScenario
    ) -> None:
        text_main = (238, 245, 255)
        text_muted = (176, 192, 218)

        header = self._tiny_font.render("Journey Table", True, text_main)
        surface.blit(header, (rect.x + 10, rect.y + 8))

        inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
        pygame.draw.rect(surface, (54, 58, 65), inner)
        pygame.draw.rect(surface, (156, 170, 198), inner, 1)

        # Responsive column anchors (fractions of inner width).
        col_defs = [
            ("LEG", 0.03),
            ("FROM", 0.13),
            ("TO", 0.30),
            ("DIST", 0.46),
            ("SPEED", 0.60),
            ("TIME", 0.75),
            ("PARCEL", 0.88),
        ]
        cols = [(name, inner.x + int(inner.w * frac)) for name, frac in col_defs]

        y = inner.y + 6
        for label, x in cols:
            surface.blit(self._tiny_font.render(label, True, text_muted), (x, y))
        pygame.draw.line(
            surface, (120, 132, 157), (inner.x + 4, y + 16), (inner.right - 4, y + 16), 1
        )
        y += 20

        pw = getattr(scenario, "parcel_weight_kg", getattr(scenario, "parcel_weight", 0))
        row_h = 20
        max_rows = max(3, min(5, (inner.h - 26) // row_h))
        for i in range(max_rows):
            if i < len(scenario.legs):
                leg = scenario.legs[i]
                dist = str(getattr(leg, "distance", "----")) if self._air_show_distances else "----"
                row = [
                    str(i + 1),
                    scenario.node_names[leg.frm],
                    scenario.node_names[leg.to],
                    dist,
                    "----",
                    "----",
                    str(pw),
                ]
            else:
                row = ["", "", "", "", "", "", ""]

            for text, (_, x) in zip(row, cols, strict=True):
                surface.blit(self._tiny_font.render(text, True, text_main), (x, y))
            y += row_h


def _init_joysticks() -> None:
    # Safe on platforms with no joystick support.
    try:
        count = pygame.joystick.get_count()
    except Exception:
        return

    for i in range(count):
        try:
            js = pygame.joystick.Joystick(i)
            js.init()
        except Exception:
            continue


def _new_seed() -> int:
    return random.SystemRandom().randint(1, 2**31 - 1)


def _resolve_window_mode(*, video_driver: str, platform_name: str | None = None) -> str:
    platform_name = sys.platform if platform_name is None else str(platform_name)
    window_mode_env = os.environ.get("CFAST_WINDOW_MODE", "").strip().lower()

    if video_driver == "dummy":
        return "windowed"
    if window_mode_env in {"windowed", "resizable"}:
        return "windowed"
    if window_mode_env in {"fullscreen", "exclusive"}:
        return "fullscreen"
    if window_mode_env in {"borderless", "windowed_fullscreen", "desktop"}:
        return "borderless"

    want_fullscreen = os.environ.get("CFAST_FULLSCREEN", "0").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    if want_fullscreen:
        return "borderless" if platform_name == "darwin" else "fullscreen"
    return "windowed"


def run(
    *, max_frames: int | None = None, event_injector: Callable[[int], None] | None = None
) -> int:
    pygame.init()
    _init_joysticks()

    pygame.display.set_caption("RCAF CFAST Trainer")

    video_driver = os.environ.get("SDL_VIDEODRIVER", "").strip().lower()
    window_mode = _resolve_window_mode(video_driver=video_driver)

    window_size = WINDOW_SIZE
    if window_mode in {"fullscreen", "borderless"}:
        desktop_sizes: list[tuple[int, int]] = []
        try:
            desktop_sizes = list(pygame.display.get_desktop_sizes())
        except Exception:
            desktop_sizes = []
        if desktop_sizes:
            window_size = desktop_sizes[0]
        else:
            try:
                info = pygame.display.Info()
                if info.current_w > 0 and info.current_h > 0:
                    window_size = (int(info.current_w), int(info.current_h))
            except Exception:
                window_size = WINDOW_SIZE

    if window_mode == "fullscreen":
        window_flags = pygame.FULLSCREEN
    elif window_mode == "borderless":
        window_flags = pygame.NOFRAME
    else:
        window_flags = pygame.RESIZABLE

    opengl_window_flags = window_flags | pygame.OPENGL | pygame.DOUBLEBUF

    default_gl_pref = "0" if sys.platform == "darwin" else "1"
    want_gl = os.environ.get("CFAST_USE_OPENGL", default_gl_pref).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    can_try_gl = want_gl and video_driver != "dummy"

    display_surface: pygame.Surface
    app_surface: pygame.Surface
    gl_renderer: _OpenGLAuditoryRenderer | None = None
    active_window_flags = window_flags

    if can_try_gl:
        try:
            display_surface = pygame.display.set_mode(window_size, opengl_window_flags)
            gl_renderer = _OpenGLAuditoryRenderer(window_size=display_surface.get_size())
            app_surface = pygame.Surface(display_surface.get_size(), pygame.SRCALPHA)
            active_window_flags = opengl_window_flags
        except Exception:
            display_surface = pygame.display.set_mode(window_size, window_flags)
            app_surface = display_surface
            gl_renderer = None
            active_window_flags = window_flags
    else:
        display_surface = pygame.display.set_mode(window_size, window_flags)
        app_surface = display_surface

    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    app_version = os.environ.get("CFAST_APP_VERSION", "dev").strip() or "dev"
    input_profiles_store = InputProfilesStore(InputProfilesStore.default_path())
    difficulty_settings_store = DifficultySettingsStore(DifficultySettingsStore.default_path())
    results_store = ResultsStore(ResultsStore.default_path())

    app = App(
        surface=app_surface,
        font=font,
        opengl_enabled=(gl_renderer is not None),
        results_store=results_store,
        input_profiles_store=input_profiles_store,
        difficulty_settings_store=difficulty_settings_store,
        app_version=app_version,
    )

    axis_calibration = AxisCalibrationScreen(app, profiles=input_profiles_store)
    axis_visualizer = AxisVisualizerScreen(app, profiles=input_profiles_store)
    input_profiles = InputProfilesScreen(app, profiles=input_profiles_store)
    difficulty_settings = DifficultySettingsScreen(app)

    hotas_menu = MenuScreen(
        app,
        "HOTAS & Input",
        [
            MenuItem("Axis Calibration", lambda: app.push(axis_calibration)),
            MenuItem("Axis Visualizer", lambda: app.push(axis_visualizer)),
            MenuItem("Input Profiles", lambda: app.push(input_profiles)),
            MenuItem("Back", app.pop),
        ],
    )

    settings_hub = MenuScreen(
        app,
        "Settings",
        [
            MenuItem("Difficulty Settings", lambda: app.push(difficulty_settings)),
            MenuItem("HOTAS & Input", lambda: app.push(hotas_menu)),
            MenuItem("Back", app.pop),
        ],
    )

    real_clock = RealClock()

    def open_loading_screen(
        *,
        title: str,
        detail: str,
        target_factory: Callable[[], Screen],
    ) -> None:
        app.push(
            LoadingScreen(
                app,
                title=title,
                detail=detail,
                target_factory=target_factory,
            )
        )

    def open_test(
        *,
        test_code: str,
        title: str,
        engine_factory: Callable[[float], CognitiveEngine],
    ) -> None:
        _ = title
        app.push(
            CognitiveTestScreen(
                app,
                engine_factory=lambda: engine_factory(app.effective_difficulty_ratio(test_code)),
                test_code=test_code,
            )
        )

    def open_numerical_ops() -> None:
        seed = _new_seed()
        open_test(
            test_code="numerical_operations",
            title="Numerical Operations",
            engine_factory=lambda difficulty: build_numerical_operations_test(
                clock=real_clock, seed=seed, difficulty=difficulty
            ),
        )

    def open_airborne_numerical() -> None:
        seed = _new_seed()
        open_test(
            test_code="airborne_numerical",
            title="Airborne Numerical Test",
            engine_factory=lambda difficulty: build_airborne_numerical_test(
                clock=real_clock, seed=seed, difficulty=difficulty
            ),
        )

    def open_ant_snap_facts_sprint(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_snap_facts_sprint",
            title="Airborne Numerical: Snap Facts Sprint",
            engine_factory=lambda difficulty: build_ant_snap_facts_sprint_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_time_flip(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_time_flip",
            title="Airborne Numerical: Time Flip",
            engine_factory=lambda difficulty: build_ant_time_flip_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_mixed_tempo_set(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_mixed_tempo_set",
            title="Airborne Numerical: Mixed Tempo Set",
            engine_factory=lambda difficulty: build_ant_mixed_tempo_set_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_route_time_solve(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_route_time_solve",
            title="Airborne Numerical: Route Time Solve",
            engine_factory=lambda difficulty: build_ant_route_time_solve_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_endurance_solve(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_endurance_solve",
            title="Airborne Numerical: Endurance Solve",
            engine_factory=lambda difficulty: build_ant_endurance_solve_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_fuel_burn_solve(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_fuel_burn_solve",
            title="Airborne Numerical: Fuel Burn Solve",
            engine_factory=lambda difficulty: build_ant_fuel_burn_solve_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_distance_scan(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_distance_scan",
            title="Airborne Numerical: Distance Scan",
            engine_factory=lambda difficulty: build_ant_distance_scan_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_payload_reference(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_payload_reference",
            title="Airborne Numerical: Payload Reference",
            engine_factory=lambda difficulty: build_ant_payload_reference_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_info_grabber(mode: AntDrillMode) -> None:
        seed = _new_seed()
        open_test(
            test_code="ant_info_grabber",
            title="Airborne Numerical: Info Grabber",
            engine_factory=lambda difficulty: build_ant_info_grabber_drill(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
                mode=mode,
            ),
        )

    def open_ant_workout(workout_code: str) -> None:
        seed = _new_seed()

        def _build_screen() -> Screen:
            plan = build_ant_workout_plan(workout_code)
            return AntWorkoutScreen(
                app,
                session=AntWorkoutSession(
                    clock=real_clock,
                    seed=seed,
                    plan=plan,
                    starting_level=app.effective_difficulty_level(workout_code),
                ),
                test_code=workout_code,
            )

        open_loading_screen(
            title="90-minute workouts",
            detail="Building Airborne Numerical workout",
            target_factory=_build_screen,
        )

    def open_math_reasoning() -> None:
        seed = _new_seed()
        open_test(
            test_code="math_reasoning",
            title="Mathematics Reasoning",
            engine_factory=lambda difficulty: build_math_reasoning_test(
                clock=real_clock, seed=seed, difficulty=difficulty
            ),
        )

    def open_digit_recognition() -> None:
        seed = _new_seed()
        open_test(
            test_code="digit_recognition",
            title="Digit Recognition",
            engine_factory=lambda difficulty: build_digit_recognition_test(
                clock=real_clock, seed=seed, difficulty=difficulty
            ),
        )

    def open_colours_letters_numbers() -> None:
        seed = _new_seed()
        open_test(
            test_code="colours_letters_numbers",
            title="Colours, Letters and Numbers",
            engine_factory=lambda difficulty: build_colours_letters_numbers_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_angles_bearings_degrees() -> None:
        seed = _new_seed()
        open_test(
            test_code="angles_bearings_degrees",
            title="Angles, Bearings and Degrees",
            engine_factory=lambda difficulty: build_angles_bearings_degrees_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_visual_search() -> None:
        seed = _new_seed()
        open_test(
            test_code="visual_search",
            title="Visual Search",
            engine_factory=lambda difficulty: build_visual_search_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_vigilance() -> None:
        seed = _new_seed()
        open_test(
            test_code="vigilance",
            title="Vigilance",
            engine_factory=lambda difficulty: build_vigilance_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_instrument_comprehension() -> None:
        seed = _new_seed()
        open_test(
            test_code="instrument_comprehension",
            title="Instrument Comprehension",
            engine_factory=lambda difficulty: build_instrument_comprehension_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_target_recognition() -> None:
        seed = _new_seed()
        open_test(
            test_code="target_recognition",
            title="Target Recognition",
            engine_factory=lambda difficulty: build_target_recognition_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_system_logic() -> None:
        seed = _new_seed()
        open_test(
            test_code="system_logic",
            title="System Logic",
            engine_factory=lambda difficulty: build_system_logic_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_table_reading() -> None:
        seed = _new_seed()
        open_test(
            test_code="table_reading",
            title="Table Reading",
            engine_factory=lambda difficulty: build_table_reading_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_sensory_motor_apparatus() -> None:
        seed = _new_seed()
        open_test(
            test_code="sensory_motor_apparatus",
            title="Sensory Motor Apparatus",
            engine_factory=lambda difficulty: build_sensory_motor_apparatus_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_rapid_tracking() -> None:
        seed = _new_seed()
        open_test(
            test_code="rapid_tracking",
            title="Rapid Tracking",
            engine_factory=lambda difficulty: build_rapid_tracking_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_spatial_integration() -> None:
        seed = _new_seed()
        open_test(
            test_code="spatial_integration",
            title="Spatial Integration",
            engine_factory=lambda difficulty: build_spatial_integration_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_trace_test_1() -> None:
        seed = _new_seed()
        open_test(
            test_code="trace_test_1",
            title="Trace Test 1",
            engine_factory=lambda difficulty: build_trace_test_1_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_trace_test_2() -> None:
        seed = _new_seed()
        open_test(
            test_code="trace_test_2",
            title="Trace Test 2",
            engine_factory=lambda difficulty: build_trace_test_2_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_auditory_capacity() -> None:
        seed = _new_seed()
        open_test(
            test_code="auditory_capacity",
            title="Auditory Capacity",
            engine_factory=lambda difficulty: build_auditory_capacity_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_cognitive_updating() -> None:
        seed = _new_seed()
        open_test(
            test_code="cognitive_updating",
            title="Cognitive Updating",
            engine_factory=lambda difficulty: build_cognitive_updating_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    def open_situational_awareness() -> None:
        seed = _new_seed()
        open_test(
            test_code="situational_awareness",
            title="Situational Awareness",
            engine_factory=lambda difficulty: build_situational_awareness_test(
                clock=real_clock,
                seed=seed,
                difficulty=difficulty,
            ),
        )

    tests_menu = MenuScreen(
        app,
        "Tests",
        [
            MenuItem("Numerical Operations", open_numerical_ops),
            MenuItem("Mathematics Reasoning", open_math_reasoning),
            MenuItem("Airborne Numerical Test", open_airborne_numerical),
            MenuItem("Digit Recognition", open_digit_recognition),
            MenuItem("Colours, Letters and Numbers", open_colours_letters_numbers),
            MenuItem("Angles, Bearings and Degrees", open_angles_bearings_degrees),
            MenuItem("Visual Search", open_visual_search),
            MenuItem("Instrument Comprehension", open_instrument_comprehension),
            MenuItem("Target Recognition", open_target_recognition),
            MenuItem("System Logic", open_system_logic),
            MenuItem("Table Reading", open_table_reading),
            MenuItem("Sensory Motor Apparatus", open_sensory_motor_apparatus),
            MenuItem("Auditory Capacity", open_auditory_capacity),
            MenuItem("Cognitive Updating", open_cognitive_updating),
            MenuItem("Situational Awareness", open_situational_awareness),
            MenuItem("Rapid Tracking", open_rapid_tracking),
            MenuItem("Spatial Integration", open_spatial_integration),
            MenuItem("Trace Test 1", open_trace_test_1),
            MenuItem("Trace Test 2", open_trace_test_2),
            MenuItem("Vigilance", open_vigilance),
            MenuItem("Back", app.pop),
        ],
    )

    ant_drills_menu = MenuScreen(
        app,
        "Airborne Numerical Drills",
        [
            MenuItem(
                "Snap Facts Sprint - Build",
                lambda: open_ant_snap_facts_sprint(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Snap Facts Sprint - Tempo",
                lambda: open_ant_snap_facts_sprint(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Snap Facts Sprint - Stress",
                lambda: open_ant_snap_facts_sprint(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Time Flip - Build",
                lambda: open_ant_time_flip(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Time Flip - Tempo",
                lambda: open_ant_time_flip(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Time Flip - Stress",
                lambda: open_ant_time_flip(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Mixed Tempo Set - Build",
                lambda: open_ant_mixed_tempo_set(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Mixed Tempo Set - Tempo",
                lambda: open_ant_mixed_tempo_set(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Mixed Tempo Set - Stress",
                lambda: open_ant_mixed_tempo_set(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Route Time Solve - Build",
                lambda: open_ant_route_time_solve(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Route Time Solve - Tempo",
                lambda: open_ant_route_time_solve(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Route Time Solve - Stress",
                lambda: open_ant_route_time_solve(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Fuel Burn Solve - Build",
                lambda: open_ant_fuel_burn_solve(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Fuel Burn Solve - Tempo",
                lambda: open_ant_fuel_burn_solve(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Fuel Burn Solve - Stress",
                lambda: open_ant_fuel_burn_solve(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Endurance Solve - Build",
                lambda: open_ant_endurance_solve(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Endurance Solve - Tempo",
                lambda: open_ant_endurance_solve(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Endurance Solve - Stress",
                lambda: open_ant_endurance_solve(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Distance Scan - Build",
                lambda: open_ant_distance_scan(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Distance Scan - Tempo",
                lambda: open_ant_distance_scan(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Distance Scan - Stress",
                lambda: open_ant_distance_scan(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Payload Reference - Build",
                lambda: open_ant_payload_reference(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Payload Reference - Tempo",
                lambda: open_ant_payload_reference(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Payload Reference - Stress",
                lambda: open_ant_payload_reference(AntDrillMode.STRESS),
            ),
            MenuItem(
                "Info Grabber - Build",
                lambda: open_ant_info_grabber(AntDrillMode.BUILD),
            ),
            MenuItem(
                "Info Grabber - Tempo",
                lambda: open_ant_info_grabber(AntDrillMode.TEMPO),
            ),
            MenuItem(
                "Info Grabber - Stress",
                lambda: open_ant_info_grabber(AntDrillMode.STRESS),
            ),
            MenuItem("Back", app.pop),
        ],
    )

    workout_items = [
        MenuItem(label, lambda code=code: open_ant_workout(code))
        for code, label in ant_workout_menu_entries()
    ]
    workout_items.append(MenuItem("Back", app.pop))
    ant_workouts_menu = MenuScreen(
        app,
        "90-minute workouts",
        workout_items,
    )

    drill_menu = MenuScreen(
        app,
        "Drill",
        [
            MenuItem("Airborne Numerical Drills", lambda: app.push(ant_drills_menu)),
            MenuItem("Back", app.pop),
        ],
    )

    main_items = [
        MenuItem("90-minute workouts", lambda: app.push(ant_workouts_menu)),
        MenuItem(
            "Drill",
            lambda: app.push(drill_menu),
        ),
        MenuItem("Tests", lambda: app.push(tests_menu)),
        MenuItem("Settings", lambda: app.push(settings_hub)),
        MenuItem("Quit", app.quit),
    ]

    app.push(MenuScreen(app, "Main Menu", main_items, is_root=True))

    frame = 0
    resize_events: set[int] = {pygame.VIDEORESIZE}
    for token in ("WINDOWRESIZED", "WINDOWSIZECHANGED"):
        ev = getattr(pygame, token, None)
        if isinstance(ev, int):
            resize_events.add(ev)
    try:
        while app.running:
            if event_injector is not None:
                event_injector(frame)

            for event in pygame.event.get():
                if event.type in resize_events:
                    next_w = int(
                        getattr(
                            event,
                            "w",
                            getattr(event, "x", display_surface.get_width()),
                        )
                    )
                    next_h = int(
                        getattr(
                            event,
                            "h",
                            getattr(event, "y", display_surface.get_height()),
                        )
                    )
                    if next_w > 0 and next_h > 0:
                        if gl_renderer is not None:
                            try:
                                display_surface = pygame.display.set_mode(
                                    (next_w, next_h), active_window_flags
                                )
                            except Exception:
                                display_surface = pygame.display.set_mode(
                                    (next_w, next_h), window_flags
                                )
                                gl_renderer = None
                                active_window_flags = window_flags
                                app.set_opengl_enabled(False)
                                app.set_surface(display_surface)
                        else:
                            try:
                                display_surface = pygame.display.set_mode(
                                    (next_w, next_h), active_window_flags
                                )
                            except Exception:
                                display_surface = pygame.display.set_mode(
                                    (next_w, next_h), window_flags
                                )
                                active_window_flags = window_flags
                            app.set_surface(display_surface)
                app.handle_event(event)

            if gl_renderer is not None:
                current_display = pygame.display.get_surface()
                if current_display is not None:
                    display_surface = current_display
                window_size = display_surface.get_size()
                if app.surface.get_size() != window_size:
                    app.set_surface(pygame.Surface(window_size, pygame.SRCALPHA))
                gl_renderer.resize(window_size=window_size)
                app.surface.fill((0, 0, 0, 0))
            else:
                current_display = pygame.display.get_surface()
                if current_display is not None and app.surface is not current_display:
                    display_surface = current_display
                    app.set_surface(display_surface)

            app.render()

            if gl_renderer is not None:
                try:
                    gl_renderer.render_frame(
                        ui_surface=app.surface,
                        scene=app.consume_auditory_gl_scene(),
                    )
                except Exception:
                    window_size = display_surface.get_size()
                    display_surface = pygame.display.set_mode(window_size, window_flags)
                    active_window_flags = window_flags
                    gl_renderer = None
                    app.set_opengl_enabled(False)
                    display_surface.blit(app.surface, (0, 0))
                    app.set_surface(display_surface)
            pygame.display.flip()

            frame += 1
            if max_frames is not None and frame >= max_frames:
                break

            clock.tick(TARGET_FPS)
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
