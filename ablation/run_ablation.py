"""Paired ablation for multi-station routing and nearest-neighbour heuristic.

Conditions
----------
RL-MultiStation-NNH
    The original shared-fleet policy: dynamic return workstation and NNH route
    construction/re-optimisation.
RL-NoNNH
    Keeps the multi-station fleet, but visits picks in FIFO order.  A mid-trip
    order is appended after the existing remaining route; no route reordering
    is performed.
RL-SingleStation-NNH
    Keeps NNH but forces every trip to return to the centre workstation.  This
    isolates the contribution of choosing among multiple workstations.

Every condition receives the same training episode seeds, load curriculum and
evaluation seeds.  Each condition is trained from a freshly seeded agent.
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from agents import DQNAgent, Robot
from environment import (
    LAMBDA_DEFAULT,
    ROBOT_CAPACITY,
    SIM_TIME,
    WarehouseLayout,
    nearest_neighbor_route,
    route_total_distance,
)
from metrics import MetricsStore, print_comparison
from run import RLSimulator


CURRICULUM_LAM_MARGIN = 0.7


@dataclass(frozen=True)
class AblationSpec:
    label: str
    use_nnh: bool
    fixed_workstation_id: Optional[int] = None


class RoutingAblationRobot(Robot):
    """Robot whose route order and/or return workstation can be constrained."""

    def __init__(self, *args, use_nnh: bool,
                 fixed_workstation_id: Optional[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nnh = use_nnh
        self.fixed_workstation_id = fixed_workstation_id

    def _route(self, start: Tuple[float, float], destinations: Sequence[Tuple[float, float]],
               layout: WarehouseLayout) -> List[Tuple[float, float]]:
        if self.use_nnh:
            return nearest_neighbor_route(start, list(destinations), layout.aisle_distance)
        # Deliberately preserve assignment order for the NoNNH condition.
        return list(destinations)

    def _workstation_for(self, start: Tuple[float, float], route: Sequence[Tuple[float, float]],
                         layout: WarehouseLayout) -> int:
        if self.fixed_workstation_id is not None:
            return self.fixed_workstation_id
        return min(
            range(len(layout.workstations)),
            key=lambda wid: route_total_distance(
                start, list(route), layout.workstations[wid], layout.aisle_distance),
        )

    def assign_trip(self, destinations: List[Tuple[float, float]], layout: WarehouseLayout):
        route = self._route(self.pos, destinations, layout)
        wid = self._workstation_for(self.pos, route, layout)
        self.workstation_id = wid
        self.workstation = layout.workstations[wid]
        self.route = route
        self.idle_slots = self.capacity - len(destinations)
        self.is_idle = False
        self._start_move()

    def insert_and_reorder(self, loc: Tuple[float, float], layout: WarehouseLayout):
        # current_dest is already scheduled/in progress and cannot be replaced.
        start = self.current_dest if self.current_dest else self.pos
        route = self._route(start, list(self.route) + [loc], layout)
        wid = self._workstation_for(start, route, layout)
        self.workstation_id = wid
        self.workstation = layout.workstations[wid]
        self.route = route
        self.idle_slots = max(0, self.idle_slots - 1)

    def delta_distance_insert(self, loc: Tuple[float, float], layout: WarehouseLayout,
                              target_ws_coords=None, dynamic_ws: bool = False) -> float:
        """Estimate the same insertion policy that this robot will execute.

        The ignored arguments make this method compatible with DQNAgent's
        candidate builder while preventing it from accidentally using the
        original NNH/dynamic-workstation estimator in the NoNNH ablation.
        """
        start = self.current_dest if self.current_dest else self.pos
        before = route_total_distance(start, self.route, self.workstation,
                                      layout.aisle_distance)
        route = self._route(start, list(self.route) + [loc], layout)
        wid = self._workstation_for(start, route, layout)
        after = route_total_distance(start, route, layout.workstations[wid],
                                     layout.aisle_distance)
        return after - before


class AblationRLSimulator(RLSimulator):
    """RLSimulator with route behaviour swapped only for one ablation arm."""

    def __init__(self, *args, spec: AblationSpec, **kwargs):
        self._ablation_spec = spec
        super().__init__(*args, **kwargs)

    def _build_robots(self) -> List[Robot]:
        fixed = self._ablation_spec.fixed_workstation_id
        robots: List[Robot] = []
        for rid in range(self.num_robots):
            # A single-station fleet must also start at that same station.
            wid = fixed if fixed is not None else rid % self.num_workstations
            robots.append(RoutingAblationRobot(
                rid, self.layout.workstations[wid], self.layout, self.robot_capacity,
                workstation_id=wid,
                use_nnh=self._ablation_spec.use_nnh,
                fixed_workstation_id=fixed,
            ))
        return robots


def make_simulator(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                   lam: float, seed: int, sim_time: float, num_robots: int,
                   robot_capacity: int, eval_mode: bool):
    if spec.use_nnh and spec.fixed_workstation_id is None:
        return RLSimulator(layout, agent, lam=lam, seed=seed, sim_time=sim_time,
                           num_robots=num_robots, robot_capacity=robot_capacity,
                           eval_mode=eval_mode)
    return AblationRLSimulator(layout, agent, lam=lam, seed=seed, sim_time=sim_time,
                                num_robots=num_robots, robot_capacity=robot_capacity,
                                eval_mode=eval_mode, spec=spec)


def train_condition(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                    lam: float, seed: int, episodes: int, sim_time: float,
                    num_robots: int, robot_capacity: int) -> MetricsStore:
    """Train one arm with a local RNG so all arms share identical episode seeds."""
    store = MetricsStore()
    sim = make_simulator(spec, layout, agent, lam, seed, sim_time, num_robots,
                         robot_capacity, eval_mode=False)
    rng = random.Random(seed)
    lam_lo = lam * CURRICULUM_LAM_MARGIN

    for episode in range(1, episodes + 1):
        episode_seed = rng.randint(0, 2**31 - 1)
        episode_lam = rng.uniform(lam_lo, lam) if episode <= episodes // 2 else lam
        sim.lam = episode_lam
        sim.order_mgr.lam = episode_lam
        sim.reset(episode_seed)
        for transition in sim.run():
            agent.push(*transition)
        agent.train_step()
        agent.decay_epsilon(episodes)
        completed = max(1, sim.order_mgr.completed_count())
        store.add(sim.build_result(spec.label, episode, episode_lam,
                                   reward=sim.total_reward / completed,
                                   epsilon=agent.epsilon))
    return store


def evaluate_condition(spec: AblationSpec, layout: WarehouseLayout, agent: DQNAgent,
                       lam: float, seeds: Sequence[int], sim_time: float,
                       num_robots: int, robot_capacity: int) -> MetricsStore:
    store = MetricsStore()
    for episode, seed in enumerate(seeds):
        sim = make_simulator(spec, layout, agent, lam, seed, sim_time, num_robots,
                             robot_capacity, eval_mode=True)
        sim.reset(seed)
        sim.run()
        completed = max(1, sim.order_mgr.completed_count())
        store.add(sim.build_result(spec.label, episode, lam,
                                   reward=sim.total_reward / completed, epsilon=0.0))
    return store


def write_paired_summary(store: MetricsStore, specs: Sequence[AblationSpec], lam: float,
                         num_robots: int, path: str) -> None:
    """Write per-seed differences against the full model, not only averages."""
    full = specs[0].label
    metrics = ["n_completed", "avg_wait_time", "avg_flow_time", "dist_per_order",
               "total_distance", "completion_rate", "reward"]
    by_label = {
        spec.label: store.filter(label=spec.label, lam=lam, num_robots=num_robots)
        for spec in specs
    }
    rows = []
    for i, ref in enumerate(by_label[full]):
        for spec in specs[1:]:
            other = by_label[spec.label][i]
            row = {"seed_index": i, "reference": full, "ablation": spec.label}
            for metric in metrics:
                ref_value = getattr(ref, metric)
                other_value = getattr(other, metric)
                row[f"{metric}_full_minus_ablation"] = ref_value - other_value
            rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired routing ablation runner")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--lam", type=float, default=LAMBDA_DEFAULT,
                        help="Arrival rate in orders/second (e.g. 850/3600).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-robots", type=int, default=14)
    parser.add_argument("--robot-capacity", type=int, default=ROBOT_CAPACITY)
    parser.add_argument("--sim-time", type=float, default=SIM_TIME)
    parser.add_argument("--out-dir", default="logs/ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layout = WarehouseLayout(seed=args.seed, num_robots=args.num_robots,
                             robot_capacity=args.robot_capacity)
    centre_station = (layout.num_workstations - 1) // 2
    specs = [
        AblationSpec("RL-MultiStation-NNH", use_nnh=True),
        AblationSpec("RL-NoNNH", use_nnh=False),
        AblationSpec("RL-SingleStation-NNH", use_nnh=True,
                      fixed_workstation_id=centre_station),
    ]
    eval_seeds = list(range(10_000, 10_000 + args.n_eval))
    all_eval = MetricsStore()

    print("Paired ablation: every arm uses the same training/evaluation seeds.")
    print(f"load={args.lam * 3600:.0f}/hr, episodes={args.episodes}, eval={args.n_eval}")
    for spec in specs:
        print(f"\nTraining {spec.label} ...")
        # DQNAgent seeds its epsilon RNG, but PyTorch initialisation and NumPy
        # prioritized-replay sampling are global RNGs.  Reset all three so the
        # three arms begin from the same learned-policy initial condition.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        agent = DQNAgent(num_robots=args.num_robots,
                         robot_capacity=args.robot_capacity, seed=args.seed,
                         sim_time=args.sim_time,
                         num_workstations=layout.num_workstations)
        train_store = train_condition(spec, layout, agent, args.lam, args.seed,
                                      args.episodes, args.sim_time, args.num_robots,
                                      args.robot_capacity)
        train_store.save_csv(os.path.join(args.out_dir, f"train_{spec.label}.csv"))
        eval_store = evaluate_condition(spec, layout, agent, args.lam, eval_seeds,
                                        args.sim_time, args.num_robots,
                                        args.robot_capacity)
        for record in eval_store.records:
            all_eval.add(record)

    all_eval.save_csv(os.path.join(args.out_dir, "evaluation_all.csv"))
    write_paired_summary(all_eval, specs, args.lam, args.num_robots,
                         os.path.join(args.out_dir, "paired_full_minus_ablation.csv"))
    print_comparison(all_eval, [s.label for s in specs], lam=args.lam,
                     num_robots=args.num_robots, last_n=args.n_eval,
                     title="Routing ablation")
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
