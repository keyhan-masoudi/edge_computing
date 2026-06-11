import numpy as np

from core import EdgeNode
from task import Task


class Scheduler:
    """
    Base class for edge real-time task schedulers.

    Parameters
    ----------
    name  : human-readable algorithm name (used in output tables)
    nodes : list of EdgeNode objects (owned by this scheduler instance)
    """

    def __init__(self, name: str, nodes: list):
        self.name  = name
        self.nodes = nodes

    # ── State management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset every node's runtime state (busy_until, energy_used)."""
        for node in self.nodes:
            node.reset()

    # ── Core allocation model ────────────────────────────────────────────────

    def evaluate_allocation(self,
                            task: Task,
                            node: EdgeNode,
                            current_node_time: float = None
                            ) -> tuple:
        """
        Compute the outcome of assigning *task* to *node* WITHOUT
        committing any state changes.  Used by all schedulers for
        look-ahead / scoring.

        Parameters
        ----------
        task              : the Task to evaluate
        node              : the EdgeNode candidate
        current_node_time : override for node.busy_until (used in RL training)

        Returns
        -------
        (actual_exec, finish_time, missed, energy_J, response_time)  — all floats
        """
        if current_node_time is None:
            current_node_time = node.busy_until

        start       = (max(current_node_time, task.arrival_time)
                       + task.comm_latency + node.comm_delay)
        actual_exec = task.exec_time / node.speed
        finish      = start + actual_exec
        missed      = finish > task.abs_deadline

        wait_time   = max(0.0, task.arrival_time - current_node_time)
        energy      = (node.p_idle   * wait_time   / 1000.0
                     + node.p_active * actual_exec / 1000.0)

        response_time = finish - task.arrival_time
        return actual_exec, finish, missed, energy, response_time

    # ── State commit ─────────────────────────────────────────────────────────

    def _commit(self,
                task:          Task,
                node:          EdgeNode,
                finish:        float,
                missed:        bool,
                energy:        float,
                response_time: float) -> None:
        """
        Persist the results of an allocation decision to both the
        task object and the node object.
        """
        actual_exec        = task.exec_time / node.speed
        task.assigned_node = node.id
        task.start_time    = finish - actual_exec
        task.finish_time   = finish
        task.missed        = missed
        node.busy_until    = finish
        node.energy_used  += energy

    # ── Metric computation ───────────────────────────────────────────────────

    def compute_metrics(self, tasks: list) -> dict:
        """
        Compute the four evaluation KPIs defined in the question:
          Miss Ratio (%)       — fraction of tasks that missed their deadline
          Avg Response (ms)    — mean (finish_time − arrival_time)
          Energy (J)           — total energy consumed by all nodes
          Utilization (%)      — fraction of time nodes were busy
        """
        done = [t for t in tasks if t.finish_time is not None]
        if not done:
            return {}

        missed       = [t for t in done if t.missed]
        miss_ratio   = len(missed) / len(done)
        avg_response = float(np.mean([t.finish_time - t.arrival_time
                                      for t in done]))
        total_energy = sum(n.energy_used for n in self.nodes)
        total_time   = max(t.finish_time for t in done)
        busy_sum     = sum(n.busy_until  for n in self.nodes)
        utilization  = (busy_sum / (len(self.nodes) * total_time)
                        if total_time > 0 else 0.0)

        return {
            "algorithm":        self.name,
            "total_tasks":      len(done),
            "missed_deadlines": len(missed),
            "Miss Ratio (%)":   round(miss_ratio * 100, 2),
            "Avg Response (ms)":round(avg_response, 3),
            "Energy (J)":       round(total_energy, 4),
            "Utilization (%)":  round(min(utilization, 1.0) * 100, 2),
        }

    # ── Abstract interface ───────────────────────────────────────────────────

    def schedule(self, tasks: list) -> dict:
        
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement schedule()")