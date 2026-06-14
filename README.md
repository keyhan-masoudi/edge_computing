### q4 
implements a comprehensive scheduling simulator designed for real-time task management across a heterogeneous Edge Computing platform. The core objective is to evaluate how different mapping and offloading strategies handle a mix of periodic, sporadic, and aperiodic tasks under varying network workloads (low, medium, and overload).

The simulator evaluates and compares four distinct scheduling algorithms:

Rate Monotonic (RM): A static priority scheduling algorithm based on task periods.

Earliest Deadline First (EDF): A dynamic priority algorithm that prioritizes tasks with the closest absolute deadlines.

Slack-Time Heuristic: A custom, context-aware mapping strategy that balances the load dynamically based on available slack time.

Machine Learning (Random Forest): A predictive, data-driven model used to select the most optimal edge node for task execution.

Performance is benchmarked against critical system metrics, including Deadline Miss Ratio, Average Response Time, Energy Consumption, and overall Resource Utilization.

### q6 
implements a lightweight, context-aware decision engine for Industrial Edge Computing, designed to monitor and process Acoustic Emission (AE) signals from critical machinery. It simulates a complete edge architecture pipeline: from signal processing and feature extraction to lightweight machine learning classification.

The core of the system is an intelligent decision engine that dynamically determines the optimal processing destination (Local, Edge, or Cloud) based on real-time network conditions and latency requirements. Furthermore, it integrates a Zero Trust security module and Data Passports to strictly enforce data privacy, transfer constraints, and secure access control in sensitive industrial environments.
