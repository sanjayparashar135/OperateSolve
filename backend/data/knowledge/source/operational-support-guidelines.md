# Operational Support Guidelines

**Document type:** support  
**Audience:** L1/L2 support and on-call engineers  
**Platform:** Northstar Commerce  
**Last reviewed:** 2026-09-20

## Principles

1. Prefer API-driven remediation over database edits.
2. Preserve payment and inventory consistency; never "fix" an order status in SQL.
3. Escalate early when blast radius spans OMS + PGA + IRS.

## Severity guide

| Severity | Example | Response |
| --- | --- | --- |
| P1 | Checkout down for >5% of traffic | Incident channel, commander, exec update every 15m |
| P2 | Single SKU reservation failures, regional latency | On-call + service owner |
| P3 | Elevated soft payment declines | Support monitors; engineering next business day |
| P4 | Cosmetic notification delay | Ticket queue |

## Common support plays

### Customer says payment was charged but no order

1. Look up AcmePay `payment_auth_id` in PGA audit.
2. Search OMS by customer id for matching create attempts.
3. If auth exists without OMS `CONFIRMED` order, void/cancel auth with finance playbook and apologize; do not manually create an order row.

### Order stuck in `PENDING_INVENTORY`

1. Check IRS reservation status.
2. If stock returned, ask OMS support tool to retry reservation.
3. If payment auth is near expiry, void and ask the customer to reorder.

### OMS 503 `oms_overloaded`

Follow the OMS database connection exhaustion runbook. Do not restart all pods as the first step.

### Notification missing

Check Notification Hub outbox for `order.confirmed`. Re-drive from NH admin if the event is present and send status is `failed_transient`.

## Access and tooling

- Support UI scopes: `oms.support.read`, `oms.support.cancel`, `nh.redrive`.
- Production DB break-glass requires incident commander approval and is audited.
- Secrets and customer PII must not be pasted into public Slack channels.

## Handoff template

When escalating to engineering, include: incident id, customer/order ids, error codes, first/last seen timestamps, dashboards checked, and actions already taken.

## Related documents

- Order Management API specification
- OMS database connection exhaustion runbook
- Payment Gateway Adapter troubleshooting guide
- Inventory reservation runbook
