# Phase 14d: First Paid Booking

**Status:** Planned. Depends on Phase 14c. The program's headline
milestone.

**New capability:** revenue. A paid drone show flown to completion for an
external client.

## Scope

1. **Booking acquisition.** Sales / outreach / quote process documented;
   first booking confirmed in writing with insurance certificate
   acknowledged.
2. **Site survey + pre-flight authorization.** LAANC if required;
   landowner permission; spectator-area separation per the safety case;
   weather window selected.
3. **Show flight + recovery.** Full show flown end-to-end; post-show
   recovery + telemetry archived.
4. **Client deliverables.** Recorded show video for the client; invoice
   sent + paid.
5. **Public artifact.** Recorded show video + social-media posts; repo
   release tagged `v3.0.0-first-show`; `status/changelog.md` entry
   recording the date.

## Inputs from prior phase

- Phase 14c reliability bar met
- § 107.35 waiver in force
- Insurance policy in force
- Booking confirmed in writing

## Methodology

- Booking process documented separately (commercial-ops repo / private
  notes — not in this phase doc).
- Site survey checklist + go/no-go decision criteria documented.
- Show-day RPIC + visible-observer roles per architecture.md § 6 safety
  case.
- Post-show debrief + telemetry archive standard within 48 hours.

## Milestone artifact

- Invoice paid by external client
- Recorded show video (client-deliverable copy + portfolio copy)
- Repo release `v3.0.0-first-show`
- Public social-media posts announcing the milestone
- Booking ledger entry (private repo)

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Show completion rate (this booking) | 1.0 (no failed show on a paid booking) |
| Crash rate (this booking) | 0 |
| Client satisfaction (subjective + paid invoice) | Paid + positive review |
| Time from booking confirmed → show flown | Documented per booking |
| Repo release + public artifact published | Within 7 days post-show |

## FAA evidence produced

The show itself becomes case-study evidence for any future waiver
amendments or insurance renewals. Telemetry archive is the
load-bearing artifact.

## Risks

- Crash on a paid show → severe consequence (lost booking, insurance
  claim, reputational damage). Mitigation: Phase 14c reliability bar must
  be met before this sub-phase is entered; weather go/no-go criteria
  enforced strictly.
- Weather no-go on show day → reschedule clauses in booking contract;
  weather contingency communication with client.
- Equipment failure pre-show → spare-airframe stock; pre-show test flight
  on the day-of, with go/no-go criteria.
- Booking acquisition slower than engineering predicts → continue
  Phase 14c rehearsals + content cadence (Phase 14b) until first booking
  lands.

## Decline list

- **Multiple bookings in Phase 14d** — declined; this sub-phase closes on
  the *first* booking. Subsequent bookings are post-Phase-14, in the
  ongoing operational cadence.
- **Show scaling beyond Phase 14c-validated drone count** — declined;
  Phase 14d uses the validated envelope. Scaling is a post-Phase-14
  conversation.
- **Showplace expansion (multiple cities)** — declined; first booking is
  local. Geographic expansion is post-Phase-14.

## See Also

- [Phase 14c Multi-Drone Rehearsals](phase14c_multidrone_rehearsals.md)
- [Phase 14 parent index](phase14_drone_show.md)
- [Architecture § 6 Part 107.35 alignment](../architecture.md#6-faa-part-10735-waiver-alignment)
- Post-show phases: [15](phase15_outdoor_hw.md) / [16](phase16_onboard_hw.md) / [17](phase17_obstacle_hw.md) / [18](phase18_multiplatform_hw.md)
