# Security policy

This is a single-user **edge** control plane — by default it has **no auth** and binds on the LAN.
Do not expose it to the public internet without protection.

- **Auth:** set `ANIMA_API_KEY` to require `Authorization: Bearer <key>` (or `x-api-key`) on the
  inference + control endpoints. Put it behind a reverse proxy / VPN for any non-LAN use.
- **Secrets:** `HF_TOKEN` and any API key live in `~/thor-serve/.env` (perms 600, not git-tracked).
  Never commit tokens. Rotate immediately if one leaks.
- **Reboot endpoint** (`/api/reboot`) reboots the host — only meaningful on the LAN/single-user box.

## Reporting a vulnerability
Email **ilessio@aiflowlabs.io** (do not open a public issue for sensitive reports). We'll acknowledge
within a few days.
