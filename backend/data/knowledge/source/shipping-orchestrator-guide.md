# Shipping Orchestrator Operations Guide

**Document type:** operations  
**Service:** Shipping Orchestrator (SHO)  
**Consumes:** `order.confirmed`, `order.cancelled`  
**Last reviewed:** 2026-08-28

## Purpose

SHO plans carrier selection, creates shipping labels, and publishes tracking ids back to OMS. It must never ship an order that OMS has cancelled.

## Happy path

1. Consume `order.confirmed` with `reservation_id` and ship-to address.
2. Select carrier via rate shop against FastParcel and RedRoute sandboxes/prod APIs.
3. Create label; store `shipment_id` in `sho_primary`.
4. Call OMS `POST /v1/orders/{order_id}/shipments` to attach tracking.
5. Publish `shipment.created` for Notification Hub.

## Failure handling

### Kafka consumer lag

- Alert `SHOConsumerLag` at 2 minutes for `order.confirmed`.
- Scale `sho-consumer` horizontally only if CPU is the constraint; if carrier APIs are slow, reduce concurrency instead.

### Carrier API outage

- Enable feature flag `sho_carrier_fastparcel_only` or `sho_carrier_redroute_only` to avoid the unhealthy carrier.
- Orders remain `CONFIRMED` in OMS without tracking until labels succeed; this is preferred over inventing tracking numbers.

### Cancel race

If `order.cancelled` arrives after label creation:

1. Void the label with the carrier API.
2. Mark shipment `VOIDED` in SHO.
3. Confirm IRS release already happened via OMS cancel.

## Runbook: missing tracking after confirmation

1. Verify OMS order status is `CONFIRMED`.
2. Check SHO for a shipment row; if missing, inspect consumer lag and DLQ.
3. Re-drive the specific `order.confirmed` event from the admin tool after verifying no duplicate shipment exists.
4. Never insert tracking into OMS without a SHO `shipment_id`.

## Dependencies

| Dependency | Timeout | Notes |
| --- | --- | --- |
| FastParcel label API | 3s | One retry |
| RedRoute label API | 3s | One retry |
| OMS attach shipment | 500ms | Required for customer visibility |

## Related documents

- Northstar Commerce architecture overview
- Deployment and rollback procedures
- Operational support guidelines
