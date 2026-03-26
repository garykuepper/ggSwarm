# Phase 6: Delivery

**Timeline:** Apr 22 -- Apr 24 (Week 16)  |  **Gate:** Final -- Capstone Festival presentation and all submissions

## 1. Goals

| ID | Goal | Success Criteria |
| :--- | :--- | :--- |
| P6.1 | Capstone Festival presentation delivered | Live demo or recorded video running during poster session |
| P6.2 | Portfolio and Learning Journal submitted | Per-course submission requirements met by end of day Apr 24 |
| P6.3 | Repository reproducible from scratch | Reviewer can train, eval, play with `scripts/run.py` using only the README |
| P6.4 | Final documentation complete | All phase design docs, changelog, and architecture doc up to date |

**Hard deadline: Apr 24, 2026.** All work must be complete and submitted by end of day.

## 2. Tasks

No new code. Three-day sprint converting Phase 5 results into polished deliverables.

**Day 1 (Apr 22): Documentation and repo polish** -- update `README.md` to final state,
finalize changelog entries for Phases 5--6, remove or archive debug scripts to
`scripts/dev/`, run reproducibility verification (clean install, pull checkpoint, pytest,
eval, play), tag `v1.0.0-capstone`.

**Day 2 (Apr 23): Presentation materials** -- build poster or slide deck (10 slides:
title, problem, GNSC architecture, Phase 1--2, Phase 3, Phase 4, Phase 5 showcase,
results vs. O1--O4, lessons learned, future work). Export demo video to final format
with QR code. Prepare live demo machine.

**Day 3 (Apr 24): Capstone Festival** -- present poster and run live demo. Submit Portfolio and Learning Journal. Archive final GCS state.

**Deliverables checklist:**
- Academic: Learning Journal, portfolio entry, Testing Report.
- Festival: poster/slides, HD demo video (<= 3 min), live demo if hardware permits.
- Repository: final README, `DEPLOYMENT_SUMMARY.md`, verified `requirements.txt`, git tag `v1.0.0-capstone`.

## 3. Design Integration

Phase 6 introduces no architectural changes. It packages the completed GNSC stack for presentation and submission.

Documentation to review and finalize:

| File | Action |
| :--- | :--- |
| `docs/design/architecture.md` | Verify all phases reflected; update Key Files table |
| `docs/status/changelog.md` | Add Phase 5 and Phase 6 entries |
| `docs/status/weekly_updates.md` | Add final week entry with project outcome |
| `docs/testing_report.md` | Review for completeness |
| `DEPLOYMENT_SUMMARY.md` | Update with final GCS checkpoint URI and metrics |
| All `phase*.md` design docs | Confirm objectives tables reflect actual results |

Reproducibility verification before tagging:

```bash
pip install -r requirements.txt
python scripts/cloud/pull_results_from_gcs.py --family marl --latest 1
pytest tests/unit/ -q
python scripts/run.py phase3 eval --num_episodes 5
python scripts/run.py phase3 play --checkpoint <path>
```

Cross-references: `docs/project/proposal.md` section 7 (timeline).

## 4. Results

Phase 6 has not started.

---

## See Also

- `docs/project/proposal.md` -- project proposal with timeline and milestones
- `docs/phases/phase5_showcase_prep.md` -- Phase 5: showcase preparation
