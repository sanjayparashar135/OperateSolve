# Payment Gateway Adapter Troubleshooting Guide

**Document type:** runbook  
**Service:** Payment Gateway Adapter (PGA)  
**Upstream:** AcmePay  
**Last reviewed:** 2026-09-01

## Role in checkout

OMS calls PGA to authorize card payments before inventory hard reservation. PGA does not store card numbers; it forwards tokens from the storefront tokenization SDK to AcmePay.

## Common failure modes

### AcmePay timeout (502 `payment_upstream_error`)

- Symptom: OMS create-order latency spikes near 2.5–5s; PGA logs `acme_timeout`.
- Check AcmePay status page and PGA metric `pga_acme_latency_ms`.
- Action: If AcmePay is degraded, enable OMS feature flag `checkout_payments_maintenance` to show a friendly storefront banner. Do not bypass authorization.

### Elevated declines (not an outage)

- Hard declines (`insufficient_funds`, `stolen_card`) are customer-side.
- Soft declines (`try_again_later`) may be retried once by PGA automatically.
- Support must not ask engineering to "force approve" an authorization.

### Duplicate authorization risk

- If OMS retries create-order without an Idempotency-Key, PGA may create two AcmePay auths.
- Recovery: void the orphaned auth via AcmePay console using `payment_auth_id` from PGA audit topic `payment.authorized`.

## Health checks

1. `GET https://pga.northstar.example/healthz` must return `ok` and show circuit state `closed`.
2. Synthetic canary `pga-canary` authorizes $1.00 every minute in the AcmePay sandbox merchant.
3. Compare OMS `payment_upstream_error` rate with PGA `acme_5xx` rate; divergence suggests OMS→PGA networking issues.

## Circuit breaker

PGA opens its AcmePay circuit after 20 failures in 60 seconds. While open, PGA returns `503 payment_circuit_open` without calling AcmePay. The circuit half-opens after 90 seconds.

## Coordination with OMS

- When PGA is unhealthy, inventory should not be reserved. OMS already sequences payment before IRS.
- After mitigation, reprocess `payment.authorized` DLQ only with finance approval.

## Related documents

- Order Management API specification
- Operational support guidelines
- INC-2511 AcmePay partial outage notes (archived)
