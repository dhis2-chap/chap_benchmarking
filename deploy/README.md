# chap-core deployment on the benchmarking server

Copies of the files that run chap-core on the benchmarking server. The live
copies are in `/data/chap-deploy` on the server; keep the two in sync.

- `compose.yml`: chap-core's `compose.yml` with the build context pointing at the
  chap-core checkout in `/data/chap-core`, port 8000 bound to localhost only, and
  `models/benchmark.yaml` bind-mounted into the seed directory.
- `models/benchmark.yaml`: extra models seeded on top of chap-core's
  `config/configured_models/default.yaml`. Every version is a pinned commit sha.
  To benchmark a new revision, add a new version label and restart chap.

Secrets live in `/data/chap-deploy/.env` on the server (`POSTGRES_USER`,
`POSTGRES_PASSWORD`, `POSTGRES_DB`, `CHAP_API_TOKEN`) and are not in git.

Upgrade chap-core:

```bash
cd /data/chap-core && git pull
cd /data/chap-deploy && docker compose build && docker compose up -d
```
