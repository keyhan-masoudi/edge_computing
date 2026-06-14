RANDOM_SEED = 42

# ── Simulation window ────────────────────────────────────────────────────────
HORIZON = 800          # ms — total simulation duration

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"

# ── Platform — heterogeneous edge nodes ─────────────────────────────────────
# Each entry: (node_id, speed, p_active_W, p_idle_W, comm_delay_ms)
NODE_SPECS = [
    (0, 6.0, 40, 6, 1.0),   # Very Fast
    (1, 3.0, 20, 3, 2.0),   # Medium
    (2, 1.0, 8,  1, 6.0),   # Slow Green
    (3, 5.0, 35, 5, 15.0),  # Fast but huge network delay
]

# (n_periodic_templates, n_sporadic, n_aperiodic)
LOAD_PARAMS = {
    "low_load":    (4,   10,   12),
    "medium_load": (8,  20,  22),
    "overload":    (11, 40,  50),
}

# ── Periodic task templates — (exec_time_ms, period_ms, deadline_ms) ─────────
PERIODIC_TEMPLATES = [
    (3, 20, 15),
    (5, 20, 15),
    (8, 30, 20),
    (4, 15, 12),
    (12, 50, 36),
    (6, 25, 18),
    (10, 40, 28),
    (7, 35, 25),
    (9, 45, 32),
    (5, 22, 15),
    (11, 55, 35),
]

# ── SEDM heuristic weights ──────────────────────────────────────────────────
SEDM_W_FINISH = 0.65
SEDM_W_ENERGY = 0.35

# ── RL Q-Learning hyper-parameters ───────────────────────────────────────────
RL_N_EPISODES = 2000
RL_ALPHA = 0.10
RL_GAMMA = 0.95
RL_EPS_START = 1.0
RL_EPS_MIN = 0.05
RL_EPS_DECAY = 0.99

# ── Plot colours per algorithm ───────────────────────────────────────────────
ALG_COLORS = {
    "EDF":             "#2196F3",
    "RM":              "#FF5722",
    "SEDM_Heuristic":  "#4CAF50",
    "RL_Q-Learning":   "#9C27B0",
}

TASK_COLORS = {
    "periodic":  "#2196F3",
    "sporadic":  "#FF9800",
    "aperiodic": "#4CAF50",
}