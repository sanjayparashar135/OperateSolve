# Northstar Commerce Architecture Overview

**Document type:** architecture  
**System:** Northstar Commerce Platform  
**Owner:** Platform Architecture  
**Last reviewed:** 2026-09-15

## Purpose

Northstar Commerce is a fictional enterprise retail platform that accepts customer orders, authorizes payments, reserves inventory, and schedules shipments. This document describes the production service topology used by operators during incidents.

## Core services

| Service | Responsibility | Primary datastore |
| --- | --- | --- |
| Customer Identity Service (CIS) | Authentication, sessions, account profile | Postgres `cis_primary` |
| Catalog Service | Product metadata and pricing reads | Redis cache + Postgres `catalog_ro` |
| Order Management API (OMS) | Order create, update, cancel, status | Postgres `oms_primary` |
| Payment Gateway Adapter (PGA) | Card authorization and capture via AcmePay | No durable store; audit in Kafka |
| Inventory Reservation Service (IRS) | Soft and hard stock holds | Postgres `irs_primary` |
| Shipping Orchestrator (SHO) | Carrier selection and label creation | Postgres `sho_primary` |
| Notification Hub (NH) | Email/SMS/push for order events | DynamoDB `nh_outbox` |

## Synchronous request path (happy path)

1. The storefront calls **OMS** `POST /v1/orders`.
2. OMS validates the session with **CIS**.
3. OMS prices line items via **Catalog**.
4. OMS requests payment authorization through **PGA**.
5. On approval, OMS asks **IRS** to hard-reserve inventory.
6. OMS persists the order as `CONFIRMED` and publishes `order.confirmed` to Kafka.
7. **SHO** consumes the event and creates a shipment plan.
8. **NH** sends the customer confirmation.

## Asynchronous dependencies

- Kafka cluster `northstar-prod` topics: `order.confirmed`, `order.cancelled`, `payment.captured`, `inventory.released`.
- Schema Registry enforces Avro compatibility for all order events.
- Dead-letter queues exist for each consumer group under `*.dlq`.

## Failure domains

- **us-east-1** is primary. **us-west-2** hosts warm standby read replicas for OMS and CIS only.
- PGA calls AcmePay over the public internet with a 2.5s timeout and one retry.
- IRS is single-region; a regional outage blocks hard reservation and forces OMS into `PENDING_INVENTORY`.

## Operational SLOs

- OMS p95 create-order latency: 400ms excluding AcmePay.
- End-to-end confirmation (order create → NH send): 8s p95.
- Error budget for OMS 5xx: 0.1% per rolling 30 days.

## Related documents

- Order Management API specification
- OMS database connection exhaustion runbook
- Deployment and rollback procedures
