## What & why
<!-- one or two lines -->

## Checklist
- [ ] `cd ui && make lint && make test` green (CI gates these)
- [ ] If perf/compat changed: a one-line **measurement** included (house style — report what helped *and* didn't)
- [ ] If a new arch/model: verified it serves/quantizes on `sm_110a`, added to `SUPPORTED_ARCHS`/`QUANTIZABLE_ARCHS` + `docs/MODELS.md`
- [ ] No secrets/tokens committed
