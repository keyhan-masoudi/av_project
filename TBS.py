#!/usr/bin/env python3
"""
TBS + EDF simulator (updated as per specification).

Updates:
- Periodic tasks are released at the start of their period with random dataSize and cycles_per_bit.
- Each has deadline equal to end of its period.
- Execution time is computed as dataSize * 8 * cycles_per_bit / CPU_FREQ.
- Aperiodic tasks are read from XML and given new deadlines using TBS formula:
      d_k = max(r_k, d_{k-1}) + (C_k / U_s)
- EDF scheduler runs both periodic and aperiodic tasks until SIMULATION_END.
"""

import xml.etree.ElementTree as ET
import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------
# Configuration / parameters
# ---------------------------
CPU_FREQ = 2e9          # CPU frequency (cycles/sec)
SIMULATION_END = 60.0    # simulation end time (sec)
USE_USER_US = None       # manually specify U_s if desired
RNG_SEED = 42

# Periodic tasks definitions:
# Each entry: (period_sec, (data_kb_min, data_kb_max), (cycles_per_bit_min, cycles_per_bit_max))
PERIODIC_TASKS = [
    (7.0,  (800.0, 1200.0), (1000.0, 1200.0)),
    (5.0,  (1000.0, 5000.0), (1000.0, 1200.0)),
    (6.0,  (500.0, 1000.0), (500.0, 1000.0)),
]

random.seed(RNG_SEED)

# ---------------------------
# Data structures
# ---------------------------
@dataclass(order=True)
class Job:
    deadline: float
    arrival: float = field(compare=False)
    remaining: float = field(compare=False)
    job_id: str = field(compare=False, default="")
    kind: str = field(compare=False, default="aperiodic")  # 'periodic' or 'aperiodic'
    absolute_deadline: float = field(compare=False, default=0.0)

    def __post_init__(self):
        self.absolute_deadline = self.deadline

# ---------------------------
# Parse Aperiodic XML
# ---------------------------
def parse_aperiodic_xml(xml_path: str) -> List[Tuple[float, float, Optional[float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    jobs = []

    for timestep in root.findall('timestep'):
        release_time = float(timestep.get('time', 0))
        for task in timestep.findall('task'):
            exec_time = float(task.get('exec_time', 0))
            deadline = float(task.get('deadline', 0))
            jobs.append((release_time, exec_time, deadline))

    jobs.sort(key=lambda x: x[0])
    return jobs

# ---------------------------
# Generate periodic jobs
# ---------------------------
def generate_periodic_jobs(sim_end: float) -> List[Tuple[float, float, float, str]]:
    periodic_jobs = []
    for idx, (period, data_range, cycles_range) in enumerate(PERIODIC_TASKS, start=1):
        release = 0.0
        job_count = 0
        while release < sim_end + 1e-9:
            data_kb = random.uniform(*data_range)
            cycles_per_bit = random.uniform(*cycles_range)
            bits = data_kb * 1024  # convert KB → bits
            exec_time = (bits * cycles_per_bit) / CPU_FREQ
            deadline = release + period
            jid = f"P{idx}_{job_count}"
            periodic_jobs.append((release, exec_time, deadline, jid))
            job_count += 1
            release += period
    periodic_jobs.sort(key=lambda x: x[0])
    return periodic_jobs

# ---------------------------
# EDF + TBS simulation
# ---------------------------
def simulate_tbs(periodic_jobs, aperiodic_jobs, sim_end, user_us=None):
    # Compute Up ≈ Σ(Ci/Ti) for each periodic task type
    total_util = 0.0
    for (period, data_range, cycles_range) in PERIODIC_TASKS:
        max_data = max(data_range)
        max_cycles = max(cycles_range)
        max_bits = max_data * 1024
        Ci = (max_bits * max_cycles) / CPU_FREQ
        total_util += Ci / period
    Up = total_util

    Us = user_us if user_us is not None else max(0.0, 1.0 - Up)
    if Us <= 0:
        raise RuntimeError(f"Invalid U_s={Us}, reduce periodic load or specify manually.")

    events = []
    for (r, c, d, jid) in periodic_jobs:
        if r <= sim_end:
            events.append(("periodic", r, {"exec": c, "deadline": d, "job_id": jid}))
    for (r, c, d) in aperiodic_jobs:
        if r <= sim_end:
            events.append(("aperiodic", r, {"exec": c, "xml_deadline": d}))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "periodic" else 1))

    ready_heap = []
    running_job: Optional[Job] = None
    last_tbs_deadline = 0.0
    t = 0.0
    event_idx = 0
    job_records = []

    def push_job(job):
        heapq.heappush(ready_heap, job)

    while True:
        next_event_time = events[event_idx][1] if event_idx < len(events) else None
        if next_event_time is None and not ready_heap and running_job is None:
            break
        if running_job is None and not ready_heap and next_event_time is not None:
            t = max(t, next_event_time)
        while event_idx < len(events) and events[event_idx][1] <= t + 1e-12:
            etype, etime, info = events[event_idx]
            if etype == "periodic":
                job = Job(deadline=info["deadline"], arrival=etime,
                          remaining=info["exec"], job_id=info["job_id"], kind="periodic")
                push_job(job)
            else:
                Ck = info["exec"]
                rk = etime
                dk = max(rk, last_tbs_deadline) + (Ck / Us)
                last_tbs_deadline = dk
                job_id = f"A_{event_idx}"
                job = Job(deadline=dk, arrival=rk, remaining=Ck, job_id=job_id, kind="aperiodic")
                push_job(job)
            event_idx += 1

        # choose EDF job
        if running_job is None:
            if ready_heap:
                running_job = heapq.heappop(ready_heap)
        else:
            if ready_heap and ready_heap[0].deadline < running_job.deadline - 1e-12:
                heapq.heappush(ready_heap, running_job)
                running_job = heapq.heappop(ready_heap)

        next_arrival = events[event_idx][1] if event_idx < len(events) else None
        if running_job is None:
            if next_arrival is None:
                break
            t = next_arrival
            continue

        time_to_complete = running_job.remaining
        if next_arrival is not None and next_arrival < t + time_to_complete:
            run_for = next_arrival - t
            running_job.remaining -= run_for
            t = next_arrival
            continue
        else:
            t += time_to_complete
            missed = t > running_job.deadline + 1e-12
            job_records.append((running_job.job_id, running_job.kind, running_job.arrival,
                                t, running_job.deadline, missed))
            running_job = None
            if t > sim_end:
                break

    missed = sum(1 for r in job_records if r[5])
    return {
        "Up": Up,
        "Us": Us,
        "sim_time": sim_end,
        "total_jobs": len(job_records),
        "missed": missed,
        "details": job_records,
    }

# ---------------------------
# Runner
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TBS + EDF Simulator")
    parser.add_argument("--xml", required=True, help="Path to aperiodic XML file")
    parser.add_argument("--sim", type=float, default=SIMULATION_END, help="Simulation end time")
    parser.add_argument("--cpu", type=float, default=CPU_FREQ, help="CPU frequency (cycles/sec)")
    parser.add_argument("--us", type=float, default=None, help="Server utilization Us")
    args = parser.parse_args()

    CPU_FREQ = args.cpu
    SIMULATION_END = args.sim

    print("Reading aperiodic XML...")
    aperiodic_jobs = parse_aperiodic_xml(args.xml)
    print(f"Loaded {len(aperiodic_jobs)} aperiodic jobs")

    print("Generating periodic tasks...")
    periodic_jobs = generate_periodic_jobs(SIMULATION_END)
    print(f"Generated {len(periodic_jobs)} periodic jobs")

    print("Running simulation...")
    result = simulate_tbs(periodic_jobs, aperiodic_jobs, SIMULATION_END, args.us)

    print("\n=== Summary ===")
    print(f"Up = {result['Up']:.4f}, Us = {result['Us']:.4f}")
    print(f"Total jobs = {result['total_jobs']}, missed = {result['missed']}")
    print(f"Simulated {result['sim_time']} seconds")

    for rec in result['details'][-10:]:
        jid, kind, arr, fin, dl, miss = rec
        print(f"{jid:8s} | {kind:9s} | arr={arr:6.2f} fin={fin:7.3f} dl={dl:7.3f} missed={miss}")
