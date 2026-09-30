"""One landmark, one agent walking the volume until it converges on it.

The search is deliberately simple: start at the centre of the coarse volume,
step in whichever of six directions the network scores highest, and declare
the point found when the agent revisits a position it has been to recently
(it is circling). Then move up to the finer scale and repeat; when there is no
finer scale left, average a short exploration around the final position to
smooth out the last voxel of jitter.

Ported from ALI_CBCT_utils/agent.py, with the training half removed and three
real defects fixed -- see `_move`, `search` and `_respawn`.
"""

import logging
import time
import zlib
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# The six moves the networks were trained to score, in the order their output
# neurons are indexed. Order is part of the weights' contract.
MOVEMENT_MATRIX = np.array(
    [
        [1, 0, 0],   # up
        [-1, 0, 0],  # down
        [0, 1, 0],   # back
        [0, -1, 0],  # front
        [0, 0, 1],   # left
        [0, 0, -1],  # right
    ]
)
MOVEMENT_COUNT = len(MOVEMENT_MATRIX)

# Fixed in the original CLI and never exposed to the user; kept fixed here for
# the same reason -- they describe the trained agents, not a user preference.
AGENT_FOV = (64, 64, 64)
SPEED_PER_SCALE = (1, 1)
SPAWN_RADIUS = 10

# How many times an agent may walk out of the volume and be respawned before
# the landmark is declared not found.
_MAX_ATTEMPTS = 3

# Directions and distance of the final averaging pass.
_FOCUS_OFFSETS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int16
)
_FOCUS_RADIUS = 4


def rng_for(target: str, seed: int) -> np.random.Generator:
    """The random stream one landmark's agent respawns from.

    Derived from the run's seed AND the landmark's name, rather than taken
    from a single stream shared by the whole scan, so that a landmark's result
    depends only on (scan, weights, seed) -- never on which OTHER landmarks
    were asked for. The same weights are called both ways: ASO asks for seven
    points through the supervisor, a standalone run asks for 119, and one
    shared stream would put C4 somewhere different in each.

    `zlib.crc32` and not the built-in `hash()`: hashing of `str` is salted per
    process by PYTHONHASHSEED, which would hand the seed straight back to the
    entropy it is here to remove.
    """
    return np.random.default_rng([seed, zlib.crc32(target.encode("utf-8"))])


class NotFound(Exception):
    """The agent did not converge on its landmark in this scan.

    An exception rather than the original's `return -1`, whose caller then had
    to remember to compare against a sentinel -- and, on one of the two paths,
    did not.
    """


class Agent:
    def __init__(
        self,
        target: str,
        scale_keys,
        brain,
        environment,
        field_of_view=AGENT_FOV,
        speed_per_scale=SPEED_PER_SCALE,
        spawn_radius: int = SPAWN_RADIUS,
        short_memory: int = 10,
        *,
        rng: np.random.Generator,
    ):
        # Keyword-only and without a default on purpose. Defaulting to a fresh
        # unseeded generator would let a caller reintroduce the very defect
        # `_respawn` documents just by forgetting an argument, and it would do
        # so silently -- the run still succeeds, it just answers differently
        # next time. There is no way to construct an Agent that does not know
        # where its randomness comes from.
        self.rng = rng
        self.target = target
        self.scale_keys = tuple(scale_keys)
        self.brain = brain
        self.environment = environment
        self.field_of_view = np.array(field_of_view, dtype=np.int16)
        self.speed_per_scale = tuple(speed_per_scale)
        self.spawn_radius = spawn_radius

        self.scale_state = 0
        self.speed = self.speed_per_scale[0]
        self.position = np.array([0, 0, 0], dtype=np.int16)
        self.start_position = np.array([0, 0, 0], dtype=np.int16)
        self.attempts = 0
        self._recent = [deque(maxlen=short_memory) for _ in self.scale_keys]

    # -- movement ----------------------------------------------------------

    def _current_scale(self) -> str:
        return self.scale_keys[self.scale_state]

    def _state(self):
        return self.environment.field_of_view(
            self._current_scale(), self.position, self.field_of_view
        )

    def _move(self, movement_index: int) -> None:
        """Step one move, or respawn if that would leave the volume.

        The bounds check reads `(new_position >= 0).all()`. The original wrote
        `new_pos.all() > 0`, which calls `.all()` FIRST -- reducing the whole
        array to one boolean, then comparing True > 0. It therefore tested
        "no component is exactly zero" and let negative coordinates through,
        where the crop silently wrapped around the far side of the volume.
        """
        new_position = self.position + MOVEMENT_MATRIX[movement_index] * self.speed
        inside = (new_position >= 0).all() and (
            new_position < self.environment.size(self._current_scale())
        ).all()

        if inside:
            self.position = new_position
            return

        self._recent[self.scale_state].clear()
        self._respawn()
        self.attempts += 1

    def _respawn(self) -> None:
        """Put the agent back somewhere plausible after it walked out.

        At the coarse scale that is anywhere in the volume; at a finer one it
        is near where the previous scale left off, since that position is
        already approximately right.

        Drawn from `self.rng` and not from the global `np.random`, which is the
        third defect carried over from the original. Nothing in this tool ever
        seeded that global, so the spawn position came from OS entropy: on a
        landmark that sits at the edge of the field of view -- where respawning
        is not the rare accident it is elsewhere -- the same scan, the same
        container and the same weights answered differently from one run to the
        next. On the reference scan, C4 came back at one of two positions 53 mm
        apart, or not at all, across six identical runs. Every other landmark
        was bit-identical, because every other landmark never leaves the volume.

        This is the same defect ASO had in `geometry.best_triplet`, and it has
        the same second symptom: a global stream is shared, so two concurrent
        requests in one process consume each other's randomness.
        """
        if self.scale_state == 0:
            self.position = self.rng.integers(
                1, self.environment.size(self._current_scale()), dtype=np.int16
            )
            self.start_position = self.position
            return

        offset = self.rng.integers([1, 1, 1], self.spawn_radius * 2) - self.spawn_radius
        self.position = np.clip(self.start_position + offset, 0, None).astype(np.int16)

    def _has_circled(self) -> bool:
        return any(
            np.array_equal(self.position, previous) for previous in self._recent[self.scale_state]
        )

    def _remember(self) -> None:
        self._recent[self.scale_state].append(self.position)

    def _upscale(self) -> bool:
        """Move to the next finer scale, converting the position into its grid."""
        if self.scale_state >= self.environment.scale_count - 1:
            return False

        current = self.environment.spacing(self._current_scale())
        finer = self.environment.spacing(self.scale_keys[self.scale_state + 1])
        self.position = (self.position * (current / finer)).astype(np.int16)
        self.scale_state += 1
        self.speed = self.speed_per_scale[self.scale_state]
        self.attempts = 0
        self.start_position = self.position
        return True

    def _step(self) -> bool:
        """One move. True when the agent has started circling, i.e. converged.

        Whether the new position has been seen before is decided BEFORE it is
        remembered -- checking afterwards would compare the position against
        itself and report convergence on the very first step.

        Counted here and nowhere else: a step is one forward pass, which is
        97.8 % of what a search costs (measured on this engine: 15.16 ms of a
        15.50 ms pass). So the budget below counts exactly the thing that takes
        the time.
        """
        self._steps_left -= 1
        self._move(self.brain.predict(self.scale_state, self._state()))
        circled = self._has_circled()
        self._remember()
        return circled

    def _focus(self, start_position):
        """Average where the agent settles from six nearby starting points.

        Bounded by the same step budget as the search itself. The original
        looped `while not found` with nothing to stop it: an agent that
        respawns (which clears the memory convergence is detected from) can
        circle forever, and in a server worker thread that is a request that
        never returns.
        """
        limits = self.environment.size(self._current_scale()) - 1
        final = np.array([0.0, 0.0, 0.0])

        for offset in _FOCUS_OFFSETS:
            self._recent[self.scale_state].clear()
            self.position = np.clip(
                start_position + _FOCUS_RADIUS * offset, 0, limits
            ).astype(np.int16)
            self._remember()

            while not self._step():
                if self._steps_left <= 0:
                    raise NotFound(
                        f"converged but ran out of steps while refining the "
                        f"position ({self._step_budget} steps)"
                    )
            final += self.position

        return final / len(_FOCUS_OFFSETS)

    # -- search ------------------------------------------------------------

    def search(self, max_steps: int):
        """Walk until the landmark is found; return its voxel position.

        Raises `NotFound` when the agent runs out of STEPS or respawns too many
        times. Both are normal outcomes on a difficult scan, and the run report
        distinguishes them from a landmark whose weights were simply missing.

        **Steps, not seconds, and that is the whole reason this engine can be
        widened at all.** The bound was a per-agent wall clock, which made the
        budget depend on how busy the machine was: two agents sharing one card
        each advance at half the rate, so a budget that was generous alone
        expires on a landmark that would have been found. It did not report
        "I was interrupted" -- it reported "this landmark did not converge",
        which is indistinguishable from a point legitimately outside a
        truncated field of view. Measured: four agents wide dropped C4, and a
        thread pool of six lost nine landmarks of twelve.

        A step count is a property of the search and of nothing else. It does
        not move with the width, the device or the load, and it is the same
        number on a laptop and on this card -- which is why the cuda/cpu split
        the seconds needed is gone.
        """
        self._step_budget = max(1, int(max_steps))
        self._steps_left = self._step_budget

        self.scale_state = 0
        self.speed = self.speed_per_scale[0]
        self.attempts = 0
        self.position = (self.environment.size(self._current_scale()) / 2).astype(np.int16)
        self._remember()

        found = False
        while not found:
            if self._steps_left <= 0:
                raise NotFound(
                    f"did not converge within {self._step_budget} steps "
                    f"(see search_steps)"
                )

            if self._step():
                # Converged at this scale. Going finer restarts the walk there;
                # when there is no finer scale left, this is the answer.
                found = not self._upscale()

            if self.attempts >= _MAX_ATTEMPTS:
                raise NotFound(f"left the volume {self.attempts} times without converging")

        return self._focus(self.position)
