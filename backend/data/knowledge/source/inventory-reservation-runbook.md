# Runbook: Inventory Reservation Failures

**Document type:** runbook  
**Service:** Inventory Reservation Service (IRS)  
**Depends on:** Order Management API, Shipping Orchestrator  
**Last reviewed:** 2026-07-22

## When to use this runbook

Use when OMS returns `409 inventory_unavailable` at elevated rates, or when IRS alerts on reservation latency / lock contention.

## Reservation model

- Soft hold: 10 minutes during cart checkout preview (`POST /v1/holds`).
- Hard reserve: created by OMS after payment authorization (`POST /v1/reservations`).
- Release: on cancel, payment void, or TTL expiry for soft holds.

SKU stock is stored in `irs_primary.sku_stock` with row-level locks during hard reserve.

## Symptoms and checks

1. Spike in OMS code `inventory_unavailable` without AcmePay errors.
2. IRS metric `irs_lock_wait_ms` elevated on popular SKUs.
3. Kafka lag on `inventory.released` delaying restocks after cancellations.

## Recovery steps

### Hot SKU lock contention

1. Identify SKUs with the highest lock waits from the IRS dashboard.
2. Temporarily lower storefront purchase limits for those SKUs via Catalog merchandising flag.
3. If a stuck reservation worker holds locks, restart `irs-worker` Deployment after confirming no in-flight hard reserves older than 2 minutes.

### Stale hard reservations

Orders cancelled in OMS should emit release events. If IRS still shows active reservations:

1. Confirm OMS cancel completed and `order.cancelled` was published.
2. Replay the consumer from the last committed offset only for the affected `reservation_id`s using the IRS admin tool.
3. Never decrement `sku_stock` with a manual SQL update in production.

### Regional read replica lag

IRS has no multi-region writer. If `us-east-1` IRS is unavailable, OMS should keep orders in `PENDING_INVENTORY` and page the platform team. Do not fail over writers to `us-west-2`.

## Handoffs

- Payment already authorized but inventory blocked: follow support guidelines to void auth or complete reservation after stock returns.
- Shipment created against a released reservation is a Sev-1 data issue; stop SHO consumers and escalate.

## Related documents

- Northstar Commerce architecture overview
- Order Management API specification
- Operational support guidelines
