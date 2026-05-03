# Phase 8: Onboard Inference Profiling + Distillation (sim only)

**Status:** Planned. Sim phase. Pulled forward from the old "Phase 6
onboard" framing under the *exhaust sim before hardware* principle. Can
proceed in parallel with Phases 3, 4, 5, 6, 7, 9.

**New capability:** the policy is profiled and distilled to fit the
≥ 50 Hz onboard inference budget on the target compute platform (Jetson
Orin Nano-class), entirely in sim and target-platform emulation, before
any onboard-hardware integration.

## Scope

1. **Inference latency profiling.** GATv2 + 5-layer GNSC under varying N
   (3, 5, 8, 16) and varying K (KNN observation cap), measured on a real
   Jetson Orin Nano dev kit (or QEMU emulation if hardware unavailable).
2. **Policy distillation.** Teacher = Phase 1c GATv2; student = lightweight
   MLP / smaller GAT. Measure performance gap vs. teacher across the
   Phase 4 shape library and Phase 7 obstacle scenes.
3. **ONNX export pipeline.** Reproducible export of the production
   checkpoint to ONNX / TorchScript with exact-output verification against
   the PyTorch reference.
4. **CBF QP solve-time budget.** Phase 7 documents the sim QP cost; Phase 8
   confirms it fits the onboard budget after distillation, or recommends a
   differentiable-barrier alternative.
5. **Memory + power profiling.** Document the Jetson Orin Nano power draw
   per inference rate; informs the Phase 16 airframe BOM decisions.

## Inputs from prior phase

- Phase 1c GATv2 + MAPPO checkpoint (teacher)
- Phase 7 CBF QP solve-time bound
- Jetson Orin Nano dev kit (or emulation environment)

## Sim methodology

- Distillation training in sim using the existing eval harness; teacher
  rollouts as supervised target for student-policy regression.
- Per-platform inference benchmarking: PyTorch CUDA on host, ONNX
  Runtime on the Jetson dev kit, TensorRT-optimized export as the
  production target.
- Power / memory measurements via the Jetson dev kit's onboard sensors;
  emulation path documented as an interim fallback only.

## Milestone artifact

Distillation report: teacher vs. student performance on shape library and
obstacle scenes; latency / power / memory measurements per platform.
Production-ready ONNX export of the student policy. Video recorded with
`--video_prefix p8-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Student vs. teacher formation error gap | ≤ 1.20× teacher |
| Student inference latency per drone (Jetson Orin Nano, N=8) | ≤ 20 ms (≥ 50 Hz) |
| Student inference latency per drone (Jetson Orin Nano, N=16) | ≤ 30 ms |
| ONNX export bit-exactness vs. PyTorch reference | Bit-exact within 1e-5 |
| Jetson Orin Nano power draw at 50 Hz inference | Documented; flag if > 10 W |
| Memory footprint of student model | ≤ 200 MB resident |

## FAA evidence produced

Indirect — establishes that onboard inference is feasible within the
hardware budget claimed by the safety case. Phase 16 provides the
on-airframe validation.

## Risks

- Distillation gap too large → stop distilling; ship the GATv2 teacher
  with a tight K cap and revisit distillation post-show.
- Jetson Orin Nano dev kit unavailable → QEMU emulation or AWS Graviton
  proxy; document as approximation.
- ONNX export breaks GATv2-specific ops → custom op or fallback to
  TorchScript; document the path.
- CBF QP solve-time + inference + perception (Phase 17) jointly miss the
  budget → revisit composition; consider perception at lower rate than
  control.

## Decline list

- **Onboard-airframe deployment in Phase 8** — declined; Phase 16
  (post-show vision).
- **Real perception integration** — declined; Phase 17.
- **Multi-platform distillation** — declined; Phase 9 covers cross-platform
  DR; Phase 8 stays single-platform (Jetson Orin Nano-class).

## See Also

- [Phase 1 Shared-Scene Sim](phase1_shared_scene_sim.md)
- [Phase 7 Obstacle-Aware Formation Control](phase7_obstacle_sim.md)
- [Phase 16 Onboard Compute Hardware](phase16_onboard_hw.md) — post-show
- [Vision § Phase 8](../vision.md)
