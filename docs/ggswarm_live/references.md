# References and Ecosystem Watch

<!-- markdownlint-disable MD033 -->
<!-- MD033 disabled: inline <a id="..."></a> anchors are intentional so
     inline citations in other docs (e.g. [Shi 2022] in vision.md)
     can link to individual papers, not just section headers. -->

*Living document. Bibliography and ecosystem watch for ggSwarm v1
(capstone) and ongoing post-capstone work. Grows over time. Remove items
only if they become irrelevant or inaccurate.*

This file is the project-wide citation and ecosystem index. For capstone
thesis citations, see also [proposal.md § 13](proposal.md#13-references).
For informal learning resources, see [concepts.md § 14](../concepts.md).

---

## 1. Academic References

Citation style matches [proposal.md § 13](proposal.md#13-references):
APA / Harvard author-year with DOI or arxiv URL on the following line.
Inline citation tags like `[Shi 2022]` in other docs link to the
anchors defined below.

### 1.1 Formation control and multi-agent swarms

<a id="preiss2017"></a>
Preiss, J. A., Hönig, W., Sukhatme, G. S., & Ayanian, N. (2017).
Crazyswarm: A large nano-quadcopter swarm. In *Proc. IEEE International
Conference on Robotics and Automation (ICRA)*, pp. 3299–3304.
<https://doi.org/10.1109/ICRA.2017.7989376>

<a id="shi2022"></a>
Shi, G., Hönig, W., Shi, X., Yue, Y., & Chung, S.-J. (2022).
Neural-Swarm2: Planning and control of heterogeneous multirotor swarms
using learned interactions. *IEEE Transactions on Robotics, 38*(2),
1063–1079.
<https://doi.org/10.1109/TRO.2021.3098436>

See also the five swarm formation papers cited in
[proposal.md § 13](proposal.md#13-references) (Bu 2024, Jiang 2022,
Ma 2024, Wang 2024, Xia 2024).

### 1.2 Graph neural networks

<a id="brody2022"></a>
Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention
networks? In *Proc. International Conference on Learning Representations
(ICLR)*.
<https://arxiv.org/abs/2105.14491>

### 1.3 Reinforcement learning

<a id="schulman2017"></a>
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
Proximal policy optimization algorithms. *arXiv preprint*.
<https://arxiv.org/abs/1707.06347>

### 1.4 Aerodynamics and sim-to-real

<a id="faessler2018"></a>
Faessler, M., Franchi, A., & Scaramuzza, D. (2018). Differential flatness
of quadrotor dynamics subject to rotor drag for accurate tracking of
high-speed trajectories. *IEEE Robotics and Automation Letters, 3*(2),
620–626.
<https://doi.org/10.1109/LRA.2017.2776353>

<a id="bauersfeld2024"></a>
Bauersfeld, L., & Scaramuzza, D. (2024). Range, endurance, and optimal
speed estimates for multicopters, and related aerodynamic effects.
*IEEE Robotics and Automation Letters*. (Cited from
<https://rpg.ifi.uzh.ch/docs/RAL24_Bauersfeld.pdf>, referenced by the
limshoonkit PX4 Isaac bridge as the template for future downwash
modeling.)

See also [Shi 2022](#shi2022) for learned inter-drone interaction forces.

### 1.5 Simulators and surveys

<a id="dimmig2024"></a>
Dimmig, C. A., Silano, G., McGuire, K., Gabellieri, C., Hönig, W., Moore,
J., & Kobilarov, M. (2024). Survey of simulators for aerial robots: An
overview and in-depth systematic comparisons. *IEEE Robotics and
Automation Magazine*.
<https://doi.org/10.1109/MRA.2024.3433171>

<a id="makoviychuk2021"></a>
Makoviychuk, V., Wawrzyniak, L., Guo, Y., Lu, M., Storey, K., Macklin,
M., Hoeller, D., Rudin, N., Allshire, A., Handa, A., & State, G. (2021).
Isaac Gym: High performance GPU-based physics simulation for robot
learning. *arXiv preprint*.
<https://arxiv.org/abs/2108.10470>

### 1.6 Consensus and distributed control

<a id="ongaro2014"></a>
Ongaro, D., & Ousterhout, J. (2014). In search of an understandable
consensus algorithm. In *Proc. USENIX Annual Technical Conference*,
pp. 305–319.

<a id="bertsekas1988"></a>
Bertsekas, D. P. (1988). The auction algorithm: A distributed relaxation
method for the assignment problem. *Annals of Operations Research,
14*(1), 105–123.
<https://doi.org/10.1007/BF02186476>

### 1.7 Trajectory optimization and safety

<a id="wang2022"></a>
Wang, Z., Zhou, X., Xu, C., & Gao, F. (2022). Geometrically constrained
trajectory optimization for multicopters. *IEEE Transactions on
Robotics, 38*(5), 3259–3278.
<https://doi.org/10.1109/TRO.2022.3160022>

<a id="ames2019"></a>
Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., &
Tabuada, P. (2019). Control barrier functions: Theory and applications.
In *Proc. European Control Conference (ECC)*, pp. 3420–3431.
<https://doi.org/10.23919/ECC.2019.8796030>

### 1.8 Simulator-focused papers

<a id="panerati2021"></a>
Panerati, J., Zheng, H., Zhou, S., Xu, J., Prorok, A., & Schoellig, A. P.
(2021). Learning to fly: A gym environment with PyBullet physics for
reinforcement learning of multi-agent quadcopter control. In *Proc.
IEEE/RSJ International Conference on Intelligent Robots and Systems
(IROS)*, pp. 7512–7519.
<https://doi.org/10.1109/IROS51168.2021.9635857>

Note: gym-pybullet-drones is the reference implementation of analytic
drone-to-drone downwash in a PyBullet-based RL environment. See § 2.

<a id="kulkarni2023"></a>
Kulkarni, M., Forgaard, T. J. L., & Alexis, K. (2023). Aerial Gym:
Isaac Gym simulator for aerial robots. *arXiv preprint*.
<https://arxiv.org/abs/2305.16510>

<a id="jacinto2024"></a>
Jacinto, M., Pinto, J., Patrikar, J., Keller, J., Cunha, R., Scherer, S.,
& Pascoal, A. (2024). Pegasus Simulator: An Isaac Sim framework for
multiple aerial vehicles simulation. In *Proc. International Conference
on Unmanned Aircraft Systems (ICUAS)*, pp. 917–922.
<https://doi.org/10.1109/ICUAS60882.2024.10556959>

<a id="folk2023"></a>
Folk, S., Paulos, J., & Kumar, V. (2023). RotorPy: A Python-based
multirotor simulator with aerodynamics for education and research.
*arXiv preprint*.
<https://arxiv.org/abs/2306.04485>

<a id="huang2023"></a>
Huang, Z., Batra, S., Chen, T., Krupnik, R., Kumar, T., Molchanov, A.,
Petrenko, A., Preiss, J. A., Ren, Z., Shim, D., Zhang, Z., Sukhatme,
G. S., & Yadkori, Y. A. (2023). QuadSwarm: A modular multi-quadrotor
simulator for deep reinforcement learning with direct thrust control.
*arXiv preprint*.
<https://arxiv.org/abs/2306.09537>

<a id="foehn2022"></a>
Foehn, P., Kaufmann, E., Romero, A., Penicka, R., Sun, S., Bauersfeld,
L., Laengle, T., Loquercio, A., & Scaramuzza, D. (2022). Agilicious:
Open-source and open-hardware agile quadrotor for vision-based flight.
*Science Robotics, 7*(67).
<https://doi.org/10.1126/scirobotics.abl6259>

<a id="song2021"></a>
Song, Y., Naji, S., Kaufmann, E., Loquercio, A., & Scaramuzza, D. (2021).
Flightmare: A flexible quadrotor simulator. In *Proc. Conference on
Robot Learning (CoRL)*, pp. 1147–1157.
<https://proceedings.mlr.press/v155/song21a.html>

<a id="shah2018"></a>
Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2018). AirSim: High-fidelity
visual and physical simulation for autonomous vehicles. In *Field and
Service Robotics*, pp. 621–635.
<https://doi.org/10.1007/978-3-319-67361-5_40>

<a id="llanes2024"></a>
Llanes, C., Kakish, Z., Williams, K., & Coogan, S. (2024). CrazySim: A
software-in-the-loop simulator for the Crazyflie nano quadrotor. In
*Proc. IEEE International Conference on Robotics and Automation (ICRA)*,
pp. 12248–12254.
<https://doi.org/10.1109/ICRA57147.2024.10610906>

---

## 2. Simulator Ecosystem

Reference implementations and ecosystems adjacent to ggSwarm. Each entry
notes the license, maintenance posture at time of writing, and which v2
phase(s) it is relevant to.

### Isaac Sim / Isaac Lab

NVIDIA's PhysX-based photorealistic simulator and its Python learning
layer. **Current ggSwarm stack.** Proprietary license (BSD-3 for Isaac
Lab), actively maintained.

- <https://developer.nvidia.com/isaac-sim>
- <https://isaac-sim.github.io/IsaacLab/>

### Pegasus Simulator (Jacinto 2024)

Isaac Sim extension that ships multi-vehicle support, PX4 integration,
ROS 2, and additional sensors (magnetometer, GPS, barometer) out of the
box. **Flagged for Phase 10 evaluation before custom PX4 integration work
starts.** BSD-3, active.

- <https://github.com/PegasusSimulator/PegasusSimulator>
- See [Jacinto 2024](#jacinto2024).

**Risk:** single-thesis project. Sustainability uncertain over multi-year
horizons. See Dimmig 2024 Academia-vs-Industry discussion.

### Aerial Gym (Kulkarni 2023)

Isaac Gym extension that demonstrates thousands of multirotors simulated
in parallel on GPU. **Relevant to Phase 3 (sim scale)** as the throughput
demonstration that removes simulator bandwidth from the bottleneck list.
BSD-3, active.

- <https://github.com/ntnu-arl/aerial_gym_simulator>
- See [Kulkarni 2023](#kulkarni2023).

### gym-pybullet-drones (Panerati 2021)

PyBullet-based multi-quadrotor simulator. **Reference implementation of
analytic drone-to-drone downwash** plus ground effect and drag. Not an
Isaac-based stack, but the downwash force model can be ported.
**Relevant to Phase 1 (shared-scene training with aerodynamic coupling).**
MIT, active.

- <https://github.com/utiasDSL/gym-pybullet-drones>
- See [Panerati 2021](#panerati2021).

### RotorPy (Folk 2023)

Python-based single-multirotor simulator with full 6-DoF dynamics,
aerodynamic wrenches, actuator dynamics, sensors, and wind models.
Validated against real Crazyflie agile maneuvers. **Relevant to
Phases 10–13 (hardware bring-up + decentralized hardware)** as a
high-fidelity Crazyflie reference. MIT, active.

- <https://github.com/spencerfolk/rotorpy>
- See [Folk 2023](#folk2023).

### QuadSwarm (Huang 2023)

Multi-quadrotor simulator for deep RL with explicit zero-shot sim-to-real
transfer demonstrations on Crazyflie. **Relevant to Phases 10–13
(hardware bring-up + decentralized hardware)** as a proven DRL-to-hardware
pipeline. MIT, active.

- <https://github.com/Zhehui-Huang/quad-swarm-rl>
- See [Huang 2023](#huang2023).

### Crazyswarm2 and CrazySim (Llanes 2024)

Crazyswarm2 is the ROS 2 framework for controlling real Crazyflie
swarms. CrazySim is the Gazebo-based SITL companion for the same stack.
**Phases 10–13 tooling.** MIT, active.

- <https://github.com/IMRCLab/crazyswarm2>
- <https://github.com/gtfactslab/CrazySim>
- See [Llanes 2024](#llanes2024).

### Flightmare (Song 2021)

Unity-based photorealistic quadrotor simulator. Less multi-agent focused
than the above but useful for perception-heavy work. MIT, maintenance has
slowed.

- <https://github.com/uzh-rpg/flightmare>
- See [Song 2021](#song2021).

### uosm.isaac.px4_bridge

Omniverse extension that bridges PX4 SITL to Isaac Sim with dynamically
tunable vehicle aerodynamics (rotor drag, airframe drag, moment
constants). Alternative to Pegasus for PX4 integration. Notes `Forcefield
Wind` and downwash as future-work items. MIT.

- <https://github.com/limshoonkit/uosm.isaac.px4_bridge>

---

## 3. Hardware Ecosystem

Factual platform snapshot for v2 phase hardware decisions. Not an
exhaustive catalog; focus is on candidates already named in the v2 plan.

| Platform | Class | Role | Typical cost (unit, 2025) |
| :--- | :--- | :--- | :--- |
| Crazyflie 2.1 | Micro, indoor | Phases 10–14 baseline | $200 to $350 |
| Holybro X500 v2 | Medium, outdoor | Phase 15 outdoor candidate | $500 to $1500 |
| ModalAI Seeker | Medium, outdoor + VIO | Phase 15 outdoor + Phase 16 onboard | $2000 to $5000 |
| ModalAI VOXL2 (as carrier) | Compute carrier | Phase 16 onboard compute | $1500 to $3000 |
| Jetson Orin Nano (carrier) | Compute | Phase 16 onboard compute | $500 to $1000 |

Vendor references:

- Bitcraze Crazyflie 2.1: <https://www.bitcraze.io/products/crazyflie-2-1/>
- Holybro X500 v2: <https://holybro.com/products/x500-v2-kits>
- ModalAI Seeker: <https://www.modalai.com/products/seeker>
- ModalAI VOXL2: <https://www.modalai.com/products/voxl-2>
- NVIDIA Jetson Orin Nano: <https://developer.nvidia.com/embedded/jetson-orin-nano>

---

## 4. Flight Stacks and Middleware

Open-source flight stacks, wire protocols, and middleware. Relevant to
v2 Phase 15+ when real airframes larger than Crazyflie enter the picture.
Phases 10–14 on Crazyflie use a different stack (CRTP over Crazyswarm2,
see § 2) and bypass this layer entirely.

### PX4 Autopilot (chosen)

Open-source flight stack for multirotor, fixed-wing, VTOL, and rover
platforms. Runs on Pixhawk-family flight controllers plus ModalAI
VOXL2. Modular codebase, strong research ecosystem, native integration
with Pegasus Simulator and `uosm.isaac.px4_bridge` (see § 2). BSD-3,
actively maintained.

**Chosen as the v2 flight stack from Phase 15 onward.** Rationale:
dominant in research multi-vehicle work, Isaac Sim tooling targets it
first (Pegasus is PX4-only), Agilicious and most cited academic
quadrotor work runs on it.

- <https://px4.io/>
- <https://docs.px4.io/>

### ArduPilot (alternative, not chosen)

Older open-source flight stack, broader platform coverage (copter,
plane, rover, boat, sub). More production-proven, larger community.
Listed for completeness; v2 is standardizing on PX4 instead.

- <https://ardupilot.org/>

### MAVLink

Lightweight message-marshalling protocol for micro air vehicles. Wire
format used by both PX4 and ArduPilot to talk to companion computers,
ground stations (QGroundControl, Mission Planner), and telemetry radios.
You consume it via a library (`mavlink`, `pymavlink`, or via MAVROS
below), not by implementing it.

- <https://mavlink.io/>

### ROS 2

Robot Operating System 2. Middleware on the companion computer that
hosts the GATv2 policy, perception, decentralized algorithms (peer
ranging, auction, gossip), and any other application logic. DDS-based
pub/sub. Humble is the LTS as of 2025.

- <https://docs.ros.org/>

### MAVROS

Classic bridge between MAVLink and ROS. Subscribes to MAVLink over
serial/UDP, republishes as ROS topics (and vice versa). Mature but
adds a translation hop.

- <https://github.com/mavlink/mavros>

### uXRCE-DDS (PX4 native ROS 2 bridge)

PX4's newer native DDS integration. No MAVLink translation for onboard
traffic: PX4 publishes uORB topics directly as DDS/ROS 2 topics via the
uXRCE-DDS agent. Lower latency, less overhead than MAVROS. This is what
the `uosm.isaac.px4_bridge` extension uses internally.

- <https://docs.px4.io/main/en/middleware/uxrce_dds.html>

### QGroundControl

Cross-platform ground station for PX4 and ArduPilot. Mission planning,
parameter tuning, firmware flashing, telemetry visualization. Not in the
flight-critical path; used during bring-up and tuning.

- <http://qgroundcontrol.com/>

---

## See Also

- [proposal.md § 13 References](../capstone/project/proposal.md#13-references)
- [concepts.md § 14 Further reading](../capstone/concepts.md)
- [vision.md § 11](vision.md)
