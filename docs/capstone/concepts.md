# Concepts Reference

A scannable glossary + explainer for the ML / RL / GNN topics this
project uses. Scoped to what actually appears in the codebase — not a
general ML textbook. Cross-referenced to the files where each concept
lives.

For a depth-ordered learning path, read top-to-bottom. For lookup, use
the table of contents.

## Contents

1. [Neural networks — the function](#1-neural-networks-the-function)
2. [Training — gradient descent + backprop](#2-training-gradient-descent-backprop)
3. [Tensors in PyTorch](#3-tensors-in-pytorch)
4. [Supervised learning vs reinforcement learning](#4-supervised-learning-vs-reinforcement-learning)
5. [RL formalism — states, actions, rewards, policy, value](#5-rl-formalism-states-actions-rewards-policy-value)
6. [Policy gradient + advantage](#6-policy-gradient-advantage)
7. [PPO — Proximal Policy Optimization](#7-ppo-proximal-policy-optimization)
8. [Actor-critic architecture](#8-actor-critic-architecture)
9. [Graph neural networks — nodes, edges, message passing](#9-graph-neural-networks-nodes-edges-message-passing)
10. [GATv2 attention](#10-gatv2-attention)
11. [CTDE — centralized training, decentralized execution](#11-ctde-centralized-training-decentralized-execution)
12. [Curriculum learning + reward engineering](#12-curriculum-learning-reward-engineering)
13. [The stack: PyTorch → SKRL → Isaac Lab → PhysX](#13-the-stack-pytorch-skrl-isaac-lab-physx)
14. [Further reading](#14-further-reading)

---

## 1. Neural networks — the function

A neural net is a parameterized function $y = f(x; \theta)$ where $x$ is
input, $y$ is output, and $\theta$ is the vector of **weights** (learned
numbers).

Built from stacked **linear layers** separated by **nonlinearities**:

```text
x  →  Linear₁  →  ReLU  →  Linear₂  →  ReLU  →  Linear₃  →  y
```

Each `Linear` is `y = Wx + b` — matrix multiply + bias vector. Without
the nonlinearity, stacking collapses to one linear map (useless). With
it, the network can approximate nearly any function (universal
approximation).

**In this repo:** see `nn.Sequential(nn.Linear(...), nn.ELU(), ...)` in
[gnn_policy.py](../source/ggswarm/ggswarm/gnn_policy.py).

## 2. Training — gradient descent + backprop

Given a scalar **loss** $L(\theta)$ ("how wrong am I?"), update weights
in the direction that reduces it:

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L$$

- $\eta$ = **learning rate** (step size)
- $\nabla_\theta L$ = gradient of loss w.r.t. weights

**Backpropagation** = chain rule applied across the computation graph.
PyTorch `autograd` tracks every op and computes gradients for free —
you just call `loss.backward()`.

**Adam** is the optimizer of choice — it adapts the step size per weight
from recent gradient history. Default in SKRL's PPO config.

## 3. Tensors in PyTorch

A **tensor** in PyTorch = an n-dimensional array on a device
(usually GPU). Not the same as physics tensors — no transformation-law
guarantees, just shape + dtype + memory.

Key attributes:

- `.shape` — e.g. `[num_envs, 12]`
- `.dtype` — `float32`, `bool`, `long`, etc.
- `.device` — `cuda:0` or `cpu`. Ops require all operands on the same
  device.
- `requires_grad` — if true, autograd tracks ops for backprop.

**Convention in this repo:** every env-owned tensor carries a shape
comment on first access, e.g. `# shape: [num_envs, 3]`. See
[tensor_contracts.md](design/tensor_contracts.md).

## 4. Supervised learning vs reinforcement learning

**Supervised**: you have labeled `(x, y_true)` pairs. Loss is "distance
between prediction and truth." Example: cat/dog photo classifier.

**Reinforcement learning (this project)**: no labels. You have:

- An **environment** (Isaac Sim + your env code)
- A **reward signal** (scalar per step, designed by you)
- An **agent** (policy network) that picks actions

The agent learns by trial — trying actions, observing rewards, and
pushing weights toward behaviors that earned good rewards.

## 5. RL formalism — states, actions, rewards, policy, value

The canonical loop:

```text
state s →  [policy π(a | s; θ)]  → action a
                                    ↓
                              [environment]
                                    ↓
                       next state s', reward r, done flag
```

Vocab:

- **Trajectory** — one sequence $(s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
- **Return** — $G = r_0 + \gamma r_1 + \gamma^2 r_2 + \ldots$,
  discounted by $\gamma \in [0, 1)$
- **Policy** $\pi(a | s; \theta)$ — neural net that outputs an action
  distribution. For continuous control (this project) it's a Gaussian:
  mean from the net, learned log-std parameter.
- **Value function** $V(s)$ — learned estimate of expected return from
  state $s$. See `self.value_head` in
  [gnn_policy.py](../source/ggswarm/ggswarm/gnn_policy.py).
- **Goal**: find $\theta$ maximizing $\mathbb{E}[G]$.

## 6. Policy gradient + advantage

The policy gradient theorem says:

$$\nabla_\theta \mathbb{E}[G] \approx \mathbb{E}\!\left[\, G \cdot \nabla_\theta \log \pi(a | s)\, \right]$$

In words: **"nudge the policy to make the actions you took more likely,
weighted by how much return you got."** This lets you train without
labels.

Problem: raw $G$ has huge variance. **Fix**: use the **advantage**
$A(s, a) = G - V(s)$ instead — "how much better was this action than
average?" Much more stable. This is why the architecture has a value
head alongside the policy head.

**GAE** (Generalized Advantage Estimation) is the smoothed version used
in PPO. A $\lambda \in [0, 1]$ parameter trades bias against variance.

## 7. PPO — Proximal Policy Optimization

Problem with vanilla policy gradient: one big update can move the
policy far enough that the next rollout is garbage, and training
collapses.

**PPO's fix**: clip the update. Let $r(\theta) = \pi_\theta(a|s) /
\pi_{\theta_{\text{old}}}(a|s)$ be the ratio between new and old
policy. PPO optimizes:

$$L = \mathbb{E}\!\left[\, \min\!\bigl(r \cdot A,\; \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A\bigr)\, \right]$$

with $\epsilon \approx 0.2$ typically. Effect: the policy can only
change by a bounded ratio per update. "Proximal" = "stay close to the
old policy."

**Training iteration** structure (what SKRL does each outer step):

1. **Rollout** — run current policy for N steps × num_envs. Collect
   `(s, a, r, done, log_prob, value)`.
2. **Compute advantages** via GAE using the value network.
3. **PPO update** — shuffle into mini-batches, do ~5 gradient epochs
   over the data with the clipped objective.
4. Discard the rollout. Collect new data with updated weights. Repeat.

See [agents/skrl_ppo_cfg.yaml](../agents/skrl_ppo_cfg.yaml) for the
actual hyperparameters used.

## 8. Actor-critic architecture

"Actor-critic" = one network outputs both:

- **Actor head** — the policy distribution (action mean + log-std)
- **Critic head** — the value estimate $V(s)$

They usually share a feature trunk and branch at the end. In
[gnn_policy.py](../source/ggswarm/ggswarm/gnn_policy.py), the trunk is
node encoder + GNN layers; the heads are `policy_head` (actor) and
`value_head` (critic). The critic trains to predict returns (MSE
regression); the actor trains via clipped policy gradient.

## 9. Graph neural networks — nodes, edges, message passing

A GNN operates on a **graph** `(V, E)`:

- **Nodes** `V` carry feature vectors — for us, each drone is a node,
  features are its local state (12D).
- **Edges** `E` connect pairs of nodes — for us, K-nearest-neighbor
  edges computed per timestep in the env.

**Message passing** = for each layer, each node aggregates information
from its neighbors:

```text
for each node j:
    message_ij = f(node_i, node_j, edge_ij)    for each neighbor i
    h_j_new    = aggregate({message_ij for i in neighbors(j)})
```

Stacking K layers gives **K-hop reach** — node j is indirectly influenced
by neighbors-of-neighbors after K rounds.

**Why it matters for swarms**: a drone's action should depend on where
its neighbors are. An MLP on a flat obs has to encode every pairwise
relationship manually; a GNN does it structurally via the graph. It's
also permutation-equivariant — reorder the drones and the answer is the
same, which fits physical symmetry.

## 10. GATv2 attention

GATv2 = Graph Attention Network v2 (Brody et al., 2022). The specific
GNN layer used in this project.

Given node features $h_i$, $h_j$ on an edge $i \to j$, GATv2 computes:

```text
message_ij = W · h_i                              # linear sender transform
score_ij   = a · LeakyReLU(W · [h_i ‖ h_j])       # attention logit
α_ij       = softmax over neighbors-of-j (score_ij)
h_j_new    = Σ_i α_ij · message_ij
```

**"Attention"** = learned weights $\alpha_{ij}$ controlling how much
each neighbor's message matters for this node's update. A drone can
learn "pay more attention to the closest neighbor on a collision
path."

**Multi-head** = several independent $(W, a)$ pairs run in parallel,
concatenated. Each head can specialize.

**Edge features** (optional): passing `edge_attr` with `edge_dim=N`
lets attention condition on per-edge info (e.g., relative position).
This project doesn't currently use them — tracked as
[ggSwarm Live backlog § A1](../ggswarm_live/backlog.md#a1-gatv2-edge-features).

## 11. CTDE — centralized training, decentralized execution

**Centralized training**: during training, a single shared policy sees
data from all agents at once. Cheap to train, one gradient path.

**Decentralized execution**: at deploy time, each drone independently
runs a local copy of the same policy using only its own observation
and its neighbors (via the graph). No central controller required.

This project is single-agent PPO with a shared policy over drones — not
MAPPO. Works because the task is symmetric (all drones identical,
formation reward is group-level). See
[assumptions.md § 4](design/assumptions.md) for the caveats and the
semi-decentralized future-work path.

## 12. Curriculum learning + reward engineering

**Reward hacking** is the central RL pitfall: the policy optimizes
*exactly* what you put in the reward, which is often not what you
meant. Classic example: reward "stay airborne" without penalizing
drift → drone hovers beautifully in the wrong place.

**Reward shaping** = adding auxiliary terms (small, dense signals) that
guide gradient descent through the reward landscape. Phase 2 iterated
on this heavily — see
[changelog.md](status/changelog.md) entries tagged `reward`.

**Curriculum learning** = start with an easy task, add difficulty as
the policy succeeds. Phase 2 trained hover first, then formation, then
obstacles in Phase 4. Without curriculum, the gradient signal is often
too sparse early on and training stalls.

## 13. The stack: PyTorch → SKRL → Isaac Lab → PhysX

| Layer | Role | Where it appears |
| :--- | :--- | :--- |
| **PyTorch** | tensor ops, autograd, neural net layers | every `torch.` call |
| **PyTorch Geometric** | GNN layers (`GATv2Conv`) | `gnn_policy.py` |
| **SKRL** | RL framework; implements PPO, trainer loop, memory | `scripts/skrl/train.py`, `agents/skrl_ppo_cfg.yaml` |
| **PPO** | algorithm (math recipe, implemented by SKRL) | — |
| **DirectRLEnv** | Isaac Lab base class for RL envs | `ggswarm_env.py` |
| **Isaac Lab** | robotics framework on Isaac Sim | all asset/sensor/actuator cfg |
| **Isaac Sim** | NVIDIA GPU simulator | underneath everything |
| **PhysX** | rigid-body physics engine | under Isaac Sim |

See [architecture.md](design/architecture.md) for the full layered
diagram of the project's control stack (L0 physics → L4 policy).

## 14. Further reading

Rank-ordered by usefulness for where this project is:

1. **Spinning Up in Deep RL** (OpenAI) —
   <https://spinningup.openai.com> — best introduction to PPO and
   policy gradients with clean math + code.
2. **Karpathy, "A Recipe for Training Neural Networks"** —
   <http://karpathy.github.io/2019/04/25/recipe/> — pragmatic
   debugging playbook.
3. **Sutton & Barto, *Reinforcement Learning: An Introduction*** —
   canonical textbook, readable.
4. **Brody, Alon, Yahav, "How Attentive Are Graph Attention
   Networks?" (GATv2 paper)** — <https://arxiv.org/abs/2105.14491> —
   motivation for the specific layer this project uses.
5. **The 3Blue1Brown neural network series** — YouTube, visual
   intuition for backprop + gradient descent.

Most valuable habit: pull up TensorBoard on a past run in
`logs/skrl/ggswarm/` and stare at reward, episode length, and
policy-std curves side by side. Intuition for "what does a healthy
training run look like" is worth more than any single reading.

---

## See Also

- [Architecture](design/architecture.md)
- [Tensor Shape Contracts](design/tensor_contracts.md)
- [Assumptions](design/assumptions.md)
- [ggSwarm Live program](../ggswarm_live/README.md) — successor to the deferred Phase 7 plan
