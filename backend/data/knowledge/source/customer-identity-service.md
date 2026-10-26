# Customer Identity Service Overview

**Document type:** architecture  
**Service:** Customer Identity Service (CIS)  
**Last reviewed:** 2026-09-05

## Role

CIS issues session tokens used by the storefront and validated by OMS, Catalog, and support tools. It is on the synchronous checkout path: every create-order validates the bearer token.

## Token model

- Access tokens are JWTs signed with rotating keys from CIS JWKS.
- Lifetime: 30 minutes. Refresh tokens: 30 days, rotating.
- Scopes relevant to operations: `oms.checkout`, `oms.support.read`, `oms.support.cancel`.

## Validation path for OMS

OMS calls CIS `POST /v1/sessions/validate` unless local JWKS cache hits. Cache TTL is 5 minutes. If CIS is unreachable and the JWT signature verifies against cached JWKS, OMS may accept the token when feature flag `oms_cis_offline_jwks` is enabled.

## Incidents involving CIS

### Elevated 401 on checkout

1. Check CIS error rate and JWKS publish time.
2. Confirm storefront is not shipping an expired client id.
3. If JWKS rotation failed mid-publish, roll back the CIS key set using the deployment rollback procedures.

### Session store latency

CIS sessions live in Postgres `cis_primary` with Redis session cache. Redis miss storms can raise OMS create-order latency even when payments are healthy. Scale CIS cache replicas before scaling OMS.

## Support boundaries

- Support may revoke sessions for a customer id via CIS admin.
- Support must not mint privileged scopes without identity team approval.
- Password reset flows are owned by CIS; OMS tickets that mention login failures should be reclassified.

## Related documents

- Order Management API specification
- Northstar Commerce architecture overview
- Deployment and rollback procedures
