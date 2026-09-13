# Security and data handling

Do not commit credentials, private data or production runtime state. In particular, this repository must never contain:

- `.env` files, API keys, tokens or passwords;
- Firebase/service-account/admin SDK JSON files;
- user uploads, databases, sessions, job folders or logs;
- unreviewed serialized models (`.pkl`, `.joblib`) or derived artifacts;
- data without a documented owner, licence/permission and approved release route.

Run the staging check before opening a pull request or creating a release:

```bash
python scripts/preflight_release.py
python scripts/preflight_release.py --strict-public-release
```

The strict command is expected to fail until the approved licence, completed citation and verified paper-data manifest are present. If any credential was ever pushed to a remote or shared outside the authorized team, revoke/rotate it through the relevant provider and remove it from repository history before publishing. Do not paste a secret into an issue, pull request or chat to diagnose it.
