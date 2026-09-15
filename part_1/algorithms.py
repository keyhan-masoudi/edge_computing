import random
import numpy as np
 
from core import EdgeNode
from task import Task
from scheduler import Scheduler
from config import (RL_N_EPISODES, RL_ALPHA, RL_GAMMA, RL_EPS_START, RL_EPS_MIN, RL_EPS_DECAY, 
                     SEDM_W_ENERGY, SEDM_W_FINISH)



class EDFScheduler(Scheduler):
    
    def __init__(self, nodes):
        super().__init__("EDF", nodes)

    def schedule(self, tasks):
        self.reset()
        pending = tasks.copy()

        while pending:
            current_time = min(node.busy_until for node in self.nodes)
            ready_tasks = [t for t in pending if t.arrival_time <= current_time]

            if not ready_tasks:
                current_time = min(t.arrival_time for t in pending)
                ready_tasks = [t for t in pending if t.arrival_time <= current_time]

            selected_task = min(
                ready_tasks,
                key=lambda t: (t.abs_deadline, t.arrival_time, t.id)
            )

            best_node = min(
                self.nodes,
                key=lambda n: self.evaluate_allocation(selected_task, n)[1]
            )

            _, finish, missed, energy, response = self.evaluate_allocation(
                selected_task, best_node
            )

            self._commit(
                selected_task, best_node, finish, missed, energy, response
            )

            pending.remove(selected_task)

        return self.compute_metrics(tasks)


class RMScheduler(Scheduler):

    def __init__(self, nodes):
        super().__init__("RM", nodes)

    def _set_priorities(self, tasks):
        for t in tasks:
            if t.period > 0:
                t.priority = 10000.0 / t.period
            else:
                t.priority = 0.0

    def schedule(self, tasks):
        self.reset()
        self._set_priorities(tasks)
        pending = tasks.copy()

        while pending:
            current_time = min(node.busy_until for node in self.nodes)
            ready_tasks = [t for t in pending if t.arrival_time <= current_time]

            if not ready_tasks:
                current_time = min(t.arrival_time for t in pending)
                ready_tasks = [t for t in pending if t.arrival_time <= current_time]

            selected_task = max(
                ready_tasks,
                key=lambda t: (t.priority, -t.arrival_time, -t.id)
            )

            best_node = min(
                self.nodes,
                key=lambda n: self.evaluate_allocation(selected_task, n)[1]
            )

            _, finish, missed, energy, response = self.evaluate_allocation(
                selected_task, best_node
            )

            self._commit(
                selected_task, best_node, finish, missed, energy, response
            )

            pending.remove(selected_task)

        return self.compute_metrics(tasks)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  SEDM —  Slack-based Energy-aware Deadline Mapping Heuristic
# ══════════════════════════════════════════════════════════════════════════════

class SEDMScheduler(Scheduler):
    """
    SEDM_Heuristic
    ==================

    Slack-based Energy-aware Deadline Mapping

    Strategy
    --------
    1. Build a ready queue from arrived tasks.

    2. Select the ready task with the smallest dynamic slack:

           slack =
               abs_deadline
               - current_time
               - best_case_execution_time

    3. Evaluate every node.

    4. Keep only deadline-feasible nodes.

    5. Among feasible nodes, minimise a weighted cost:

           cost =
               w_energy * normalized_energy
             + w_finish * normalized_finish_time

    6. If no node can satisfy the deadline,
       choose the node with the earliest finish time.
    """

    def __init__(
        self,
        nodes,
        w_energy: float = SEDM_W_ENERGY,
        w_finish: float = SEDM_W_FINISH
    ):
        super().__init__("SEDM_Heuristic", nodes)

        self.w_energy = w_energy
        self.w_finish = w_finish

    def _dynamic_slack(
        self,
        task,
        current_time
    ):
        """
        Estimate urgency using the fastest node available.
        """

        fastest_speed = max(
            node.speed
            for node in self.nodes
        )

        best_exec = task.exec_time / fastest_speed

        return (
            task.abs_deadline
            - current_time
            - best_exec
        )

    def _normalize(
        self,
        value,
        minimum,
        maximum
    ):
        """
        Stable min-max normalization.
        """

        if abs(maximum - minimum) < 1e-9:
            return 0.0

        return (
            value - minimum
        ) / (
            maximum - minimum
        )

    def schedule(self, tasks):

        self.reset()

        pending = tasks.copy()

        while pending:

            # --------------------------------------------------
            # Advance simulation time
            # --------------------------------------------------

            current_time = min(
                node.busy_until
                for node in self.nodes
            )

            ready_tasks = [
                t for t in pending
                if t.arrival_time <= current_time
            ]

            if not ready_tasks:

                current_time = min(
                    t.arrival_time
                    for t in pending
                )

                ready_tasks = [
                    t for t in pending
                    if t.arrival_time <= current_time
                ]

            # --------------------------------------------------
            # Select most urgent task
            # --------------------------------------------------

            selected_task = min(
                ready_tasks,
                key=lambda t: (
                    self._dynamic_slack(
                        t,
                        current_time
                    ),
                    t.abs_deadline,
                    t.id
                )
            )

            # --------------------------------------------------
            # Evaluate all candidate nodes
            # --------------------------------------------------

            feasible = []
            all_options = []

            for node in self.nodes:

                (
                    _,
                    finish,
                    missed,
                    energy,
                    response
                ) = self.evaluate_allocation(
                    selected_task,
                    node
                )

                option = {
                    "node": node,
                    "finish": finish,
                    "missed": missed,
                    "energy": energy,
                    "response": response
                }

                all_options.append(option)

                if not missed:
                    feasible.append(option)

            # --------------------------------------------------
            # Choose node
            # --------------------------------------------------

            if feasible:

                energies = [
                    x["energy"]
                    for x in feasible
                ]

                finishes = [
                    x["finish"]
                    for x in feasible
                ]

                min_energy = min(energies)
                max_energy = max(energies)

                min_finish = min(finishes)
                max_finish = max(finishes)

                for option in feasible:

                    norm_energy = self._normalize(
                        option["energy"],
                        min_energy,
                        max_energy
                    )

                    norm_finish = self._normalize(
                        option["finish"],
                        min_finish,
                        max_finish
                    )

                    option["cost"] = (
                        self.w_energy * norm_energy
                        + self.w_finish * norm_finish
                    )

                best_option = min(
                    feasible,
                    key=lambda x: (
                        x["cost"],
                        x["finish"],
                        x["node"].id
                    )
                )

            else:

                # overload fallback:
                # minimise lateness

                best_option = min(
                    all_options,
                    key=lambda x: (
                        x["finish"],
                        x["node"].id
                    )
                )

            # --------------------------------------------------
            # Commit allocation
            # --------------------------------------------------

            self._commit(
                selected_task,
                best_option["node"],
                best_option["finish"],
                best_option["missed"],
                best_option["energy"],
                best_option["response"]
            )

            pending.remove(selected_task)

        return self.compute_metrics(tasks)
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  RL  —  Q-Learning Node Selector
# ══════════════════════════════════════════════════════════════════════════════
 
class RLScheduler(Scheduler):
    """
    RL_Q_Learning

    Q-Learning based Task Offloading Scheduler
    for Heterogeneous Edge Computing.

    State
    -----
    (
        urgency_bin,
        avg_load_bin,
        task_type_bin
    )

    Action
    ------
    node_id

    Reward
    ------
    + deadline satisfaction
    - deadline violation
    - energy consumption
    - response time
    - lateness severity
    """

    def __init__(
        self,
        nodes,
        n_episodes=RL_N_EPISODES,
        alpha=RL_ALPHA,
        gamma=RL_GAMMA,
        eps0=RL_EPS_START,
        eps_min=RL_EPS_MIN,
        eps_decay=RL_EPS_DECAY
    ):
        super().__init__("RL_Q_Learning", nodes)

        self.n_nodes = len(nodes)
        self.n_ep = n_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.initial_eps = eps0
        self.eps = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # State dimensions: urgency(5) × avg_load(5) × least_loaded_node(n_nodes) × task_type(3) × action(n_nodes)
        self.Q = np.zeros((5, 5, self.n_nodes, 3, self.n_nodes))

    # ==================================================
    # State Encoding
    # ==================================================

    def _task_type_bin(self, task):

        if task.type == "periodic":
            return 0

        if task.type == "sporadic":
            return 1

        return 2

    def _state(self, task, current_time):
        fastest_speed = max(n.speed for n in self.nodes)
        best_exec = task.exec_time / fastest_speed

        slack = task.abs_deadline - current_time - best_exec
        ratio = slack / (task.deadline + 1e-9)

        if ratio < 0.10:
            urgency = 0
        elif ratio < 0.25:
            urgency = 1
        elif ratio < 0.50:
            urgency = 2
        elif ratio < 1.00:
            urgency = 3
        else:
            urgency = 4

        # 1. محاسبه بار کلی سیستم (برای درک Overload)
        avg_load = np.mean([max(0.0, n.busy_until - current_time) for n in self.nodes])
        load_bin = min(4, int(avg_load / 20))

        # 2. پیدا کردن خلوت‌ترین نود (برای جلوگیری از گره‌خوردگی ترافیک)
        least_loaded_node_id = min(
            self.nodes, 
            key=lambda n: max(0.0, n.busy_until - current_time)
        ).id

        task_type = self._task_type_bin(task)

        return (
            urgency,
            load_bin,               # اضافه شد
            least_loaded_node_id,   # حفظ شد
            task_type
        )

    # ==================================================
    # Reward Function
    # ==================================================

    def _reward(self, task, finish_time, missed, energy, response_time):
        reward = 0.0

        if missed:
            lateness = max(0.0, finish_time - task.abs_deadline)
            # جریمه سنگین‌تر برای از دست دادن ددلاین در اضافه‌بار
            reward -= (150.0 + 1.0 * lateness)
        else:
            reward += 50.0

        # تنظیم ضریب انرژی تا ایجنت تشویق شود مثل SEDM رفتار کند
        reward -= 0.8 * energy
        
        # جریمه اندک برای کند بودن پاسخ
        reward -= 0.02 * response_time

        return reward

    # ==================================================
    # Training
    # ==================================================

    def train(self, tasks):

        self.eps = self.initial_eps

        for _ in range(self.n_ep):

            self.reset()

            ordered = sorted(
                tasks,
                key=lambda t: t.arrival_time
            )

            for idx, task in enumerate(ordered):

                current_time = min(
                    n.busy_until
                    for n in self.nodes
                )

                current_time = max(
                    current_time,
                    task.arrival_time
                )

                state = self._state(
                    task,
                    current_time
                )

                # ε-greedy

                if random.random() < self.eps:

                    action = random.randint(
                        0,
                        self.n_nodes - 1
                    )

                else:

                    action = int(
                        np.argmax(
                            self.Q[state]
                        )
                    )

                node = self.nodes[action]

                (
                    _,
                    finish,
                    missed,
                    energy,
                    response
                ) = self.evaluate_allocation(
                    task,
                    node
                )

                # simulated environment update

                node.busy_until = finish
                node.energy_used += energy

                reward = self._reward(
                    task,
                    finish,
                    missed,
                    energy,
                    response
                )

                # next state

                if idx < len(ordered) - 1:

                    next_task = ordered[idx + 1]

                    next_time = min(
                        n.busy_until
                        for n in self.nodes
                    )

                    next_time = max(
                        next_time,
                        next_task.arrival_time
                    )

                    next_state = self._state(
                        next_task,
                        next_time
                    )

                    target = (
                        reward
                        + self.gamma
                        * np.max(
                            self.Q[next_state]
                        )
                    )

                else:

                    target = reward

                self.Q[state][action] += (
                    self.alpha
                    * (
                        target
                        - self.Q[state][action]
                    )
                )

            self.eps = max(
                self.eps_min,
                self.eps * self.eps_decay
            )

    # ==================================================
    # Evaluation
    # ==================================================

    def schedule(self, tasks):

        self.reset()

        ordered = sorted(
            tasks,
            key=lambda t: t.arrival_time
        )

        for task in ordered:

            current_time = min(
                n.busy_until
                for n in self.nodes
            )

            current_time = max(
                current_time,
                task.arrival_time
            )

            state = self._state(
                task,
                current_time
            )

            # --- Action Masking ---
            feasible_actions = []
            
            for i, node in enumerate(self.nodes):
                _, _, missed, _, _ = self.evaluate_allocation(task, node)
                if not missed:
                    feasible_actions.append(i)

            if feasible_actions:
                action = max(
                    feasible_actions,
                    key=lambda a: self.Q[state][a]
                )
            else:
                action = int(
                    np.argmax(
                        self.Q[state]
                    )
                )

            node = self.nodes[action]

            (
                _,
                finish,
                missed,
                energy,
                response
            ) = self.evaluate_allocation(
                task,
                node
            )

            self._commit(
                task,
                node,
                finish,
                missed,
                energy,
                response
            )

        return self.compute_metrics(tasks)