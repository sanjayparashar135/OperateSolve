# Deployment and Rollback Procedures

**Document type:** operations  
**Scope:** Northstar Commerce production (`northstar-prod`)  
**Change window:** Tuesday–Thursday 15:00–19:00 UTC preferred  
**Last reviewed:** 2026-09-10

## Deployment model

- GitOps via Argo CD applications per service (`oms-prod`, `pga-prod`, `irs-prod`, `sho-prod`, `cis-prod`, `nh-prod`, `catalog-prod`).
- Images are promoted from `northstar-staging` after smoke tests pass.
- Database migrations run as Argo PreSync Jobs and must be backward compatible with the previously deployed app version.

## Standard deploy checklist

1. Confirm no open P1/P2 incidents on the target service.
2. Announce in `#northstar-changes` with change ticket id and rollback image tag.
3. Sync Argo CD application. Watch rollout: `kubectl rollout status deployment/<svc> -n northstar-prod`.
4. Run smoke tests: healthz, one synthetic create-order in staging-like prod canary namespace `northstar-canary`.
5. Watch error rate and p95 latency for 20 minutes before closing the change.

## Service-specific notes

### OMS

- Feature flags that change payment or inventory sequencing require a dual write rehearsal in staging.
- Never deploy OMS pool size increases without checking `(replicas × pool) < 0.7 × max_connections`.

### PGA

- Coordinate with AcmePay if certificate or merchant id changes.
- Keep the previous adapter version available for instant rollback; AcmePay contracts are versioned.

### IRS

- Avoid deploys during flash-sale starts. Lock contention masks regressions.

## Rollback procedure

1. In Argo CD, retarget the application to the previous known-good image tag recorded in the change ticket.
2. If a migration is not backward compatible, **stop** and engage DBA; do not roll forward blindly.
3. For OMS/PGA/IRS, prefer rollback over hotfix when create-order success rate drops more than 2% absolute after a deploy.
4. After rollback, verify:
   - Service healthz
   - Synthetic canary create-order
   - No growth in Kafka DLQ depth
5. Write a short incident note even if customer impact was limited.

## Freeze policy

Hard freeze begins 48 hours before major retail events listed in the Northstar calendar. Emergency fixes require director approval and a named rollback owner online for the duration.

## Related documents

- OMS database connection exhaustion runbook
- Operational support guidelines
- Northstar Commerce architecture overview
