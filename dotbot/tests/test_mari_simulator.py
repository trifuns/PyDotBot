"""Tests for the mari network and association simulators."""

import threading
import time
from heapq import heappop, heappush

import pytest

from dotbot.dotbot_simulator import (
    MariAssociationSimulator,
    MariNetworkSimulator,
    MariSimulator,
    SimulatedNetworkSettings,
    _interpolate_pdr,
    _next_downlink_cell,
)
from dotbot.mari_schedules import (
    MariSourcesNotFound,
    _load_from_firmware,
    _load_generated,
    load_schedules,
    resolve_mari_dir,
    select_schedule,
)

SCHEDULES = load_schedules()


class _FakeDotBot:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y


class _FakeRandom:
    """Stands in for the `random` module, counting calls to `randint`."""

    def __init__(self, return_value):
        self.calls = 0
        self._return_value = return_value

    def randint(self, a, b):
        self.calls += 1
        return self._return_value


def _network(schedule_name, dotbots=None, enqueue=None, **settings_kwargs):
    return MariNetworkSimulator(
        schedule=SCHEDULES[schedule_name],
        settings=SimulatedNetworkSettings(**settings_kwargs),
        dotbots=dotbots or [],
        enqueue=enqueue or (lambda delay, fn: None),
        on_frame_received=lambda frame: None,
    )


def _run_association_to_completion(schedule_name, n_bots, timeout_s=5.0):
    """Drives MariAssociationSimulator on its own thread/heap, mirroring the
    facade's event loop, and waits for every bot to join or for the timeout."""
    schedule = SCHEDULES[schedule_name]
    joined_at = {}
    done = threading.Event()
    start = time.monotonic()

    def on_joined(index):
        joined_at[index] = time.monotonic() - start
        if len(joined_at) == n_bots:
            done.set()

    heap = []
    seq = [0]
    cond = threading.Condition()
    stop_event = threading.Event()

    def enqueue(delay_s, fn):
        with cond:
            heappush(heap, (time.monotonic() + delay_s, seq[0], fn))
            seq[0] += 1
            cond.notify()

    def run():
        with cond:
            while not stop_event.is_set():
                now = time.monotonic()
                if heap:
                    deadline, _, fn = heap[0]
                    if deadline <= now:
                        heappop(heap)
                        cond.release()
                        try:
                            fn()
                        finally:
                            cond.acquire()
                        continue
                    wait = deadline - now
                else:
                    wait = None
                cond.wait(timeout=wait)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assoc = MariAssociationSimulator(
        schedule=schedule,
        mari_indices=list(range(n_bots)),
        enqueue=enqueue,
        on_joined=on_joined,
    )
    assoc.start()
    completed = done.wait(timeout=timeout_s)
    stop_event.set()
    with cond:
        cond.notify_all()
    thread.join()
    return joined_at, completed


# ------------------------------------------------------------- schedules --


@pytest.mark.parametrize(
    "n_bots,expected",
    [
        (10, "tiny"),
        (11, "medium"),
        (44, "medium"),
        (45, "big"),
        (66, "big"),
        (67, "huge"),
        (102, "huge"),
    ],
)
def test_schedule_selection_boundaries(n_bots, expected):
    assert select_schedule(n_bots).name == expected


def test_schedule_selection_above_max_capacity_raises():
    with pytest.raises(ValueError):
        select_schedule(103)


def test_generated_schedule_snapshot_matches_live_firmware_parse():
    try:
        resolve_mari_dir()
    except MariSourcesNotFound:
        pytest.skip("no mari firmware checkout available")
    assert _load_from_firmware() == _load_generated()


# ------------------------------------------------------------------- pdr --


def test_pdr_is_independent_of_node_count_at_fixed_distance():
    """The regression phase 1 guarded against: PDR must track distance only,
    never the fleet size sharing the schedule."""
    anchors = [(2.0, 98.6), (10.0, 94.3)]
    for n_bots in (1, 5, 100):
        dotbots = [_FakeDotBot(pos_x=1000, pos_y=1000) for _ in range(n_bots)]
        dotbots[0] = _FakeDotBot(pos_x=1000 + 2000, pos_y=1000)  # 2m from gateway
        net = _network("huge", dotbots=dotbots, pdr_by_distance_m=anchors)
        assert net._pdr_percent(0, flat_default=100) == round(98.6)


def test_pdr_interpolates_linearly_between_anchors():
    anchors = [(2.0, 98.6), (10.0, 94.3)]
    assert _interpolate_pdr(2.0, anchors) == round(98.6)
    assert _interpolate_pdr(10.0, anchors) == round(94.3)
    assert round(94.3) <= _interpolate_pdr(6.0, anchors) <= round(98.6)


def test_pdr_clamps_outside_the_anchor_range():
    anchors = [(2.0, 98.6), (10.0, 94.3)]
    assert _interpolate_pdr(0.5, anchors) == round(98.6)
    assert _interpolate_pdr(50.0, anchors) == round(94.3)


def test_pdr_falls_back_to_flat_default_when_unset():
    net = _network("tiny", dotbots=[_FakeDotBot(0, 0)])
    assert net._pdr_percent(0, flat_default=42) == 42


def test_uplink_makes_exactly_one_pdr_draw_no_retry(monkeypatch):
    """mari has no link-layer ACK/retransmission — a delivery attempt is
    single-shot, never retried."""
    fake_random = _FakeRandom(return_value=50)
    monkeypatch.setattr("dotbot.dotbot_simulator.random", fake_random)
    net = _network("tiny", pdr=0)  # flat PDR 0 -> always "lost"
    net.assign_cell(0)
    enqueued = []
    net._enqueue = lambda delay, fn: enqueued.append(fn)
    net.schedule_uplink(frame="x", dotbot_index=0)
    assert fake_random.calls == 1
    assert enqueued == []


# ------------------------------------------------------------- latency ----


def test_cell_delay_upper_bound_tracks_schedule_size_not_node_count():
    for name in SCHEDULES:
        net = _network(name)
        # node count isn't even a constructor input any more — the wait for
        # any cell is bounded purely by this schedule's own slotframe.
        for cell in range(net._schedule.n_cells):
            assert 0 <= net._cell_delay_s(cell) < net._schedule.slotframe_ms / 1000


def test_calibrated_overhead_is_schedule_derived_and_deterministic():
    assert _network("huge")._uplink_overhead_s == _network("huge")._uplink_overhead_s
    assert _network("tiny")._uplink_overhead_s > 0
    # huge's theoretical per-cell wait already exceeds the measured RTT, so
    # the calibration must clamp to 0 rather than go negative.
    assert _network("huge")._uplink_overhead_s == 0.0


# --------------------------------------------------------- cell assignment --


def test_assign_cell_gives_unique_real_uplink_cells_in_order():
    net = _network("tiny")  # 10 uplink cells, matches max_nodes exactly
    uplink_indices = set(net._schedule.uplink_cell_indices())
    assigned = []
    for i in range(10):
        net.assign_cell(i)
        assigned.append(net._uplink_cell[i])
    assert set(assigned) == uplink_indices
    assert len(assigned) == len(set(assigned))


def test_assign_cell_raises_once_schedule_capacity_is_exhausted():
    net = _network("tiny")
    for i in range(10):
        net.assign_cell(i)
    with pytest.raises(ValueError):
        net.assign_cell(10)


def test_downlink_cell_is_the_nearest_real_d_cell_after_uplink():
    net = _network("big")
    downlink_cells = net._schedule.downlink_cell_indices()
    for i in range(20):
        net.assign_cell(i)
        expected = _next_downlink_cell(net._uplink_cell[i], downlink_cells)
        assert net._downlink_cell[i] == expected
        assert net._downlink_cell[i] in downlink_cells


# ------------------------------------------------------------ association --


def test_unjoined_bot_never_gets_scheduled_traffic():
    """End to end through the facade: before start()/join, a bot has no
    assigned cell and its frames are silently dropped, not delivered."""
    received = []
    sim = MariSimulator(
        settings=SimulatedNetworkSettings(),
        dotbots=[],
        mari_indices=[0],
        on_frame_received=lambda frame: received.append(frame),
    )
    sim.schedule_uplink(frame="too-early", dotbot_index=0)
    assert received == []


def test_all_bots_eventually_join_at_schedule_capacity():
    joined_at, completed = _run_association_to_completion(
        "tiny", n_bots=10, timeout_s=5.0
    )
    assert completed, "not every bot joined within the timeout"
    assert set(joined_at) == set(range(10))
    assert all(t >= 0 for t in joined_at.values())
