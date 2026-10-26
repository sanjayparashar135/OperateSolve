# Order Management API Specification

**Document type:** api_spec  
**Service:** Order Management API (OMS)  
**Base URL (prod):** `https://oms.northstar.example/v1`  
**Auth:** Bearer session token issued by Customer Identity Service

## Overview

OMS is the system of record for customer orders in Northstar Commerce. Clients must not write order state directly to the database.

## Endpoints

### Create order

`POST /v1/orders`

Request body:

```json
{
  "customer_id": "cus_123",
  "currency": "USD",
  "lines": [
    {"sku": "SKU-1001", "quantity": 2}
  ],
  "shipping_address_id": "addr_9"
}
```

Success response `201`:

```json
{
  "order_id": "ord_7f3a",
  "status": "CONFIRMED",
  "payment_auth_id": "pay_auth_22",
  "reservation_id": "res_88"
}
```

Common error codes:

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_lines` | Empty lines or unknown SKU |
| 401 | `unauthorized` | Missing or expired CIS token |
| 409 | `inventory_unavailable` | IRS could not hard-reserve |
| 502 | `payment_upstream_error` | AcmePay failure after retry |
| 503 | `oms_overloaded` | Connection pool or circuit open |

### Get order

`GET /v1/orders/{order_id}`

Returns status, line items, payment auth id, reservation id, and shipment ids when present.

### Cancel order

`POST /v1/orders/{order_id}/cancel`

Allowed when status is `CONFIRMED` or `PENDING_INVENTORY` and SHO has not marked the shipment as `LABEL_CREATED`. Cancellation releases the IRS reservation and voids the PGA authorization when still open.

### List orders for customer

`GET /v1/customers/{customer_id}/orders?limit=20`

Requires the caller to own the customer id or hold the `oms.support.read` scope.

## Idempotency

Create and cancel accept header `Idempotency-Key`. Keys are retained for 24 hours in Redis `oms-idempotency`. Replays return the original response without a second payment authorization.

## Rate limits

- Storefront clients: 30 create-order requests per minute per customer.
- Internal support tools: 120 reads per minute per operator.

## Dependency timeouts

| Dependency | Timeout | Retry |
| --- | --- | --- |
| CIS session validate | 300ms | none |
| Catalog price | 200ms | none |
| PGA authorize | 2500ms | 1 |
| IRS reserve | 800ms | 1 |

## Support notes

When create-order returns `503 oms_overloaded`, check the OMS database connection pool and the runbook for connection exhaustion before restarting pods.
