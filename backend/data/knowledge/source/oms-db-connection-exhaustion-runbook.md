# Runbook: OMS Database Connection Exhaustion

**Document type:** runbook  
**Service:** Order Management API (OMS)  
**Severity default:** P1 when create-order error rate > 5%  
**Last drill:** 2026-08-02

## Symptoms

- OMS returns HTTP 503 with code `oms_overloaded`.
- Grafana panel `oms_db_pool_wait_ms` rises above 200ms p95.
- Alert `OMSPostgresConnectionsHigh` fires when active connections exceed 90% of `max_connections` on `oms_primary`.
- Storefront checkout failures concentrate on create-order; GET order may still succeed.

## Immediate triage (first 5 minutes)

1. Confirm scope: check OMS 5xx rate and AcmePay/IRS error rates. If only OMS DB pool is saturated, stay on this runbook.
2. Open Datadog dashboard **OMS Production** and note `pg_stat_activity` count, waiting queries, and pod restart count.
3. Do **not** scale OMS replicas yet if the database `max_connections` is already near the ceiling; more pods will worsen exhaustion.

## Recovery procedure

### Step 1 — Identify blockers

On the OMS primary (read-only session as `oms_ro`):

```sql
SELECT pid, usename, state, wait_event_type, query_start, left(query, 120) AS query
FROM pg_stat_activity
WHERE datname = 'oms_primary'
ORDER BY query_start ASC;
```

Look for long-running `idle in transaction` sessions and lock waits on `orders` or `order_lines`.

### Step 2 — Free safe connections

- Terminate abandoned `idle in transaction` sessions older than 10 minutes owned by `oms_app` after approving with the incident commander.
- Reduce OMS HikariCP `maximumPoolSize` from 20 to 12 via the `oms-pool-emergency` feature flag if blockers cannot be cleared quickly.

### Step 3 — Drain and heal application pods

1. Disable new deploys in Argo CD application `oms-prod`.
2. Roll OMS Deployment with `kubectl rollout restart deployment/oms -n northstar-prod` only after pool wait drops below 50ms.
3. Verify create-order success rate recovers above 99% for 10 minutes.

### Step 4 — Customer recovery

- Orders stuck in `PENDING_PAYMENT` for more than 15 minutes should be cancelled with the support cancellation playbook so IRS holds are released.
- Do not manually insert rows into `orders`; use OMS cancel/create APIs.

## Root causes seen in drills

- A reporting job opened transactions without closing them (INC-2407).
- Connection pool size multiplied by replica count exceeded Postgres `max_connections`.
- A slow Catalog dependency held OMS request threads, which held DB connections.

## Prevention

- Keep `(replicas × maximumPoolSize) < 0.7 × max_connections`.
- Page on `idle in transaction` older than 5 minutes.
- Require load-test sign-off when changing pool settings.

## Related documents

- INC-2407 postmortem
- Operational support guidelines
- Deployment and rollback procedures
