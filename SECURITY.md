# Security policy

## Supported version

Security fixes currently target the latest release on the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or include private memory data, database files, or API keys in a report.

Use GitHub's private vulnerability reporting for this repository when available. If that option is unavailable, contact the repository owner privately through their GitHub profile and share only the minimum information needed to reproduce the issue.

Include the affected version, impact, reproduction steps, and any suggested mitigation. You can expect an acknowledgement after the report has been reviewed; disclosure timing will be coordinated based on severity and fix availability.

## Secrets and data

- Keep `.env` files and API keys outside version control.
- Store production databases in a protected application data directory.
- Treat transcripts, profiles, and facts as potentially sensitive data.
- Test backup, restoration, retention, and deletion behavior for your deployment.
