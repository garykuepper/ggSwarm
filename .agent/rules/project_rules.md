# Project Rules: ggSwarm

## 1. Architecture Maintenance (MANDATORY)
The file `docs/architecture.md` is the source of truth for the system design.

- **RULE:** Any change to the environment structure, message passing logic, or coordination layers MUST be accompanied by an update to `docs/architecture.md`.
- **RATIONALE:** Ensures scalability and maintainability of the decentralized control logic.

## 2. Status Reporting (MANDATORY)

- **RULE:** A weekly status summary must be added to `docs/status/weekly_updates.md` every Monday.
- **RULE:** Major phase transitions and technical milestones must be logged in `docs/status/changelog.md`.
- **RATIONALE:** Provides transparency for project stakeholders and website updates.

## 3. Python Coding Standards
- Follow the global Python standards defined in the user settings (PEP 8, Snake_case for filenames, Type Hinting).
- Explicitly documented public interfaces in all module-level components.
