---
doc_id: knowledge_base
title: Platform Overview & Service Tiers
---

# Platform Overview

The platform is a general-purpose developer API service. It exposes
REST and GraphQL endpoints, issues API keys per workspace, and
stores data in one of three regions. This document covers service
tiers, rate limits, data regions, and account setup.

## Service tiers

| Tier | API calls/day | Rate limit | Price |
| --- | --- | --- | --- |
| Free | 1,000 | 10 requests/second | $0 |
| Pro | 100,000 | 100 requests/second | $49/month |
| Enterprise | Unlimited | 1,000 requests/second | Custom (starts at $999/month) |

## Rate limits

Rate limits are enforced per API key, not per workspace. The Free tier
allows 10 requests/second with a burst of up to 20. The Pro tier allows
100 requests/second with a burst of up to 200. The Enterprise tier
allows 1,000 requests/second with a burst of up to 2,000. Requests
that exceed the rate limit receive HTTP 429.

## Data regions

Data can be stored in three regions:

- US (us-east-1)
- EU (eu-west-1)
- APAC (ap-southeast-1)

The region is selected at workspace creation and cannot be changed
afterward. All data for a workspace stays in the selected region.

## Account setup

New accounts start on the Free tier. Email verification is required
before any API key is issued. Two-factor authentication (2FA) is
optional on the Pro tier and required on the Enterprise tier. The
Free tier does not offer 2FA. Password requirements: minimum 12
characters, at least one number and one symbol.
