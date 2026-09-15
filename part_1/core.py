from config import NODE_SPECS


class EdgeNode:
    """
    One heterogeneous edge node / core.

    Attributes
    ----------
    id          : unique integer identifier
    speed       : relative MIPS factor  (1.0 = baseline)
    p_active    : Watts consumed while executing a task
    p_idle      : Watts consumed while idle / waiting
    comm_delay  : extra ms added when a task is offloaded to this node
    busy_until  : simulation time (ms) when this node next becomes free
    energy_used : cumulative energy (J) consumed during this run
    """

    def __init__(self, node_id: int, speed: float,
                 p_active: float, p_idle: float, comm_delay: float):
        self.id          = node_id
        self.speed       = speed
        self.p_active    = p_active
        self.p_idle      = p_idle
        self.comm_delay  = comm_delay
        self.busy_until  = 0.0
        self.energy_used = 0.0

    def reset(self) -> None:
        self.busy_until  = 0.0
        self.energy_used = 0.0

    def __repr__(self) -> str:
        return (f"EdgeNode(id={self.id}, speed={self.speed}x, "
                f"p_active={self.p_active}W, comm_delay={self.comm_delay}ms)")


# ── Factory ──────────────────────────────────────────────────────────────────

def build_platform() -> list:
    """
    Construct the heterogeneous edge platform defined in config.NODE_SPECS.
    Returns a fresh list of EdgeNode objects every time it is called,
    so each scheduler gets an independent copy of the hardware state.
    """
    return [EdgeNode(*spec) for spec in NODE_SPECS]