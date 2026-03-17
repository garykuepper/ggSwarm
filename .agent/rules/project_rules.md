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
- **RULE:** Always auto-resolve common linting issues like "line too long" (wrap at 88-100 chars) and "unused imports" (remove them) as they occur.

## 4. Documentation Standards

- **RULE:** Use professional, clear language in all `.md` files.
- **RULE:** Always auto-resolve any problems indicated by `markdownlint` to maintain formatting consistency.
- **RATIONALE:** Ensures high-quality, professional-grade documentation for the project.

## 5. Technical Terminology (Spelling Exceptions)

- **RULE:** The following terms are correct technical jargon for this project and should NOT be flagged as spelling errors:
  - **Frameworks/Tools:** Isaac Lab, `isaaclab`, `isaacsim`, `conda`, `PyPI`, `Py`.
  - **Project Specific:** `ggSwarm`, `ggswarm`, `Crazyflie`.
  - **Technical Shorthand:** `cfg`, `envs`, `quat`, `lin_vel`, `ang_vel`, `pos_w`, `rel_pos`, `multirotor`, `multirotors`.
  - **User Context:** `gkuep`.
- **RATIONALE:** Prevents false positives in spelling checks for common robotics and simulation terminology.
