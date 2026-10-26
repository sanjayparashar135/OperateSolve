# INC-2407 Postmortem: OMS Checkout Outage from Connection Exhaustion

**Document type:** incident  
**Incident ID:** INC-2407  
**Severity:** P1  
**Date:** 2026-06-18  
**Duration:** 47 minutes  
**Services affected:** Order Management API, Notification Hub (downstream), storefront checkout

## Summary

Checkout create-order failed for most customers between 14:12 and 14:59 UTC. The proximate cause was Postgres connection exhaustion on `oms_primary` after a new finance reporting job held idle transactions open. Payment and inventory systems remained healthy.

## Timeline (UTC)

| Time | Event |
| --- | --- |
| 13:40 | Finance team enabled nightly-style reconciliation job `fin-oms-recon` in production by mistake. |
| 14:05 | Job opened 80 sessions against `oms_primary` and left them `idle in transaction`. |
| 14:12 | Alert `OMSPostgresConnectionsHigh` fired. Create-order 503 rate crossed 5%. |
| 14:18 | Incident commander declared P1. OMS replica scale-up was attempted and made pool wait worse. |
| 14:31 | On-call terminated idle reporting sessions and disabled the job. |
| 14:41 | Pool wait returned to baseline. Rollout restart of OMS completed. |
| 14:59 | Error rate sustained under 0.2%. Incident mitigated. |

## Root cause

The reconciliation job used the shared `oms_app` database role and did not set a statement or idle-in-transaction timeout. Combined with OMS pool settings, total connections exceeded safe headroom on `oms_primary`.

## Contributing factors

- No separate read-only reporting role with connection limits.
- Runbook at the time recommended scaling OMS pods first, which amplified pressure.
- Feature flag for emergency pool reduction existed but was not documented in the primary runbook.

## Customer impact

- Approximately 12,400 failed checkout attempts.
- 310 orders entered `PENDING_PAYMENT` and required support cancellation.
- No incorrect charges; PGA authorizations for failed creates were not captured.

## Corrective actions

1. Moved reporting workloads to `oms_reporting` role with `connection limit 10` and mandatory timeouts.
2. Updated the OMS database connection exhaustion runbook to forbid scaling during pool saturation.
3. Added alert `OMSIdleInTransaction` at 5 minutes.
4. Required change tickets for enabling finance jobs in production.

## Lessons learned

Database connection budgets must account for batch jobs, not only application pools. Recovery steps that increase concurrency can deepen an exhaustion incident.

## Related documents

- OMS database connection exhaustion runbook
- Northstar Commerce architecture overview
- Operational support guidelines
