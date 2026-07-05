# Assumptions and Scope Limitations

This document lists the simplifying assumptions made in ggSwarm and their
implications for real-world deployment. These assumptions were necessary to
complete the project within a strict deadline while working full-time, allowing
focus on the core swarming algorithms and multi-agent coordination rather than
low-level systems engineering.

## 1. Perfect State Knowledge

**Assumption:** Each drone knows its exact position, velocity, and orientation
in world coordinates at every timestep.

**Reality:** Real drones rely on noisy sensors (IMU, GPS, UWB, optical flow,
motion capture) with latency, drift, and occasional dropouts. Position
estimates have uncertainty of 1-10cm (motion capture) to 1-5m (GPS).

**Impact:** The GNN policy, MINCO filter, and CBF safety shield all operate on
exact state. In deployment, a state estimator (Extended Kalman Filter or
visual-inertial odometry) would be needed, and the CBF would need to account
for estimation uncertainty (Robust CBFs).

## 2. Perfect Communication

**Assumption:** All drones in a swarm group share state instantaneously with
zero latency, zero packet loss, and infinite bandwidth.

**Reality:** Real swarm communication uses WiFi, Bluetooth, or radio with
latency (5-50ms), packet loss (1-10%), and bandwidth limits. Drones may
not hear each other at long range.

**Impact:** The K-nearest neighbor observations assume exact relative positions
of neighbors are always available. In reality, stale or missing neighbor data
would degrade formation quality. The GNN's attention mechanism could
potentially be robust to intermittent data, but this is untested.

## 3. No Low-Level Flight Controller

**Assumption:** The RL policy directly outputs thrust and moment commands
that are applied as forces/torques to the rigid body. There is no PID
controller, rate controller, or attitude controller between the policy
and the physics.

**Reality:** Real quadrotors use cascaded PID controllers (position → velocity
→ attitude → rate → motor PWM). The RL policy would typically output
position or velocity setpoints to an onboard flight controller, not raw
thrust/moments.

**Impact:** The policy learned its own implicit flight controller through RL
training. This is more fragile than a tuned PID stack but demonstrates that
end-to-end learned control is viable. For deployment, the policy output
would be mapped to setpoints for the onboard controller.

## 4. Centralized Slot Assignment

**Assumption:** Formation slot positions are computed centrally by the
environment. At episode reset, greedy nearest-slot matching assigns each
drone to the closest available slot. On SwarmRaft dropout, the environment
recomputes formation offsets for N-1 alive agents and reassigns slots.

**Reality:** In a truly decentralized swarm, drones would negotiate slot
assignments through a consensus protocol (auction-based allocation,
Hungarian algorithm, or Raft consensus). Each drone would independently
express a preference for a slot, and conflicts would be resolved through
multi-round bidding or distributed voting.

**Impact:** The current system is spatially aware (nearest-slot, not
index-based) but centralized — the environment acts as an omniscient
coordinator. The GNN learns spatial coordination through message passing,
but slot allocation decisions are not part of the learned policy.

**Why full decentralization is hard:** The CTDE architecture uses a shared
policy with single-pass inference. Multi-round consensus (auction, Raft
voting) requires iterative communication loops between agents, which
breaks the vectorized batch inference model. In Isaac Lab with 4096
parallel environments, running iterative consensus per swarm group would
destroy training throughput.

**Future work path (semi-decentralized):**

1. Add all slot positions to observations (drones can see available options)
2. Extend action space with slot preference logits (drones express choice)
3. Environment resolves conflicts using preferences as tie-breakers
4. GNN learns emergent spatial reasoning ("I'm on the right, prefer rightmost slot")

This would give drones agency in slot selection while keeping conflict
resolution centralized — practical within CTDE constraints. Tracked as
[ggSwarm Live backlog § E1](../../ggswarm_live/archive/backlog_detailed.md#e1-semi-decentralized-slot-allocation).

## 5. Homogeneous Agents

**Assumption:** All drones are identical Crazyflie 2.x models with the same
mass, thrust capacity, and dynamics. The same policy is shared across all
drones (CTDE).

**Reality:** Real swarms may have heterogeneous agents with different
capabilities, battery levels, and payload weights.

**Impact:** The shared policy and symmetric training mean every drone
behaves identically. Heterogeneous swarms would require agent-specific
observations or policy conditioning.

## 6. No Wind or Aerodynamic Interactions

**Assumption:** There is no wind disturbance, and drones do not create
downwash or aerodynamic interference on nearby drones.

**Reality:** Quadrotor downwash significantly affects nearby drones,
especially in tight formations. Wind gusts create unpredictable
disturbances.

**Impact:** The 0.5m target spacing partially mitigates downwash effects,
but real deployment would need larger spacing or downwash-aware control.

## 7. Flat Terrain

**Assumption:** The environment is a flat plane with no terrain variation.
Obstacles (Phase 4) are static cylinders at known positions.

**Reality:** Real environments have complex 3D terrain, moving obstacles,
and unknown layouts requiring SLAM and path planning.

**Impact:** The CBF obstacle avoidance treats obstacles as known fixed
positions. Dynamic or unknown obstacles would require perception and
online mapping.

## 8. Simulation-to-Real Gap

**Assumption:** Isaac Sim physics (PhysX 5) accurately represents real
Crazyflie dynamics.

**Reality:** Sim-to-real transfer is a known challenge. Motor response,
battery voltage sag, sensor noise, and actuator delays are not modeled.

**Impact:** Direct policy transfer to real hardware would likely fail
without domain randomization or sim-to-real fine-tuning.

---

## Scope Justification

These assumptions were made deliberately to focus the capstone project on
the **core research contribution: decentralized swarm coordination via
GNN + MINCO + CBF**. The project demonstrates:

- Graph Neural Network spatial reasoning for formation control
- Minimum-jerk trajectory optimization for smooth flight
- Control Barrier Function safety guarantees
- Fault-tolerant consensus (SwarmRaft agent dropout recovery)
- Scalable architecture (train 8, deploy 20+)

Each assumption listed above represents a well-studied engineering problem
with known solutions. Removing these assumptions would extend the project
scope significantly but would not change the fundamental swarm coordination
architecture.

---

## See Also

- [Architecture](architecture.md)
- [Proposal](../project/proposal.md)
