class Task:
    """
    One real-time task instance (periodic, sporadic, or aperiodic).

    The class-level counter `_counter` is reset by the workload generator
    at the start of each scenario so task IDs are always 1-based and
    reproducible within a run.
    """

    _counter: int = 0   # global id counter — reset via Task.reset_counter()

    def __init__(self,
                 task_type:    str,
                 exec_time:    float,
                 period:       float,
                 deadline:     float,
                 energy_cost:  float,
                 comm_latency: float,
                 arrival_time: float):
        Task._counter += 1
        self.id           = Task._counter

        # ── Static attributes (set at creation) ─────────────────────────────
        self.type         = task_type        # 'periodic' | 'sporadic' | 'aperiodic'
        self.exec_time    = exec_time        # ms — WCET
        self.period       = period           # ms — 0 means no period
        self.deadline     = deadline         # ms — relative deadline
        self.energy_cost  = energy_cost      # mJ — base estimate
        self.comm_latency = comm_latency     # ms — task-side offload cost
        self.arrival_time = arrival_time     # ms — release time
        self.abs_deadline = arrival_time + deadline   # ms — absolute

        # ── Priority (filled by RM scheduler) ───────────────────────────────
        self.priority: int = 0

        # ── Result fields (filled after scheduling) ──────────────────────────
        self.assigned_node = None    # node id
        self.start_time    = None    # ms
        self.finish_time   = None    # ms
        self.missed        = False   # True if finish_time > abs_deadline

    # ── Class-level helpers ──────────────────────────────────────────────────

    @classmethod
    def reset_counter(cls) -> None:
        """Call before generating a new task set for reproducible IDs."""
        cls._counter = 0

    # ── Dunder ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Task(id={self.id}, {self.type[:3]}, "
                f"exec={self.exec_time:.1f}ms, "
                f"abs_dl={self.abs_deadline:.1f}ms)")