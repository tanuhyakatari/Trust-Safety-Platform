# Trust & Safety Platform — Fraud, Review Manipulation & Counterfeit Defense

**Track:** Trust & Safety / Marketplace Integrity
**Core AI focus:** Multi-agent synergy + vision-language counterfeit detection + behavioral anomaly scoring

## The Problem

Multi-vendor marketplaces lose 3–5% of GMV every year to three coordinated attack vectors:

1. **Return & COD abuse** — empty-box returns, swapped counterfeits, uncollectible cash-on-delivery orders
2. **Fake review rings** — astroturfed seller ratings from coordinated or AI-generated reviews
3. **Counterfeit listings** — unauthorized or unsafe products (including uncertified cosmetics) published under legitimate-looking listings

Static rule-based defenses (IP blocks, transaction caps) can't adapt fast enough, and manual review queues are swamped — leading to reviewer fatigue, delayed fulfillment, and genuine buyers wrongly blocked.

## Our Solution

A **three-agent Trust & Safety platform** where each agent specializes in one attack vector, coordinated by a central orchestrator with a deterministic, human-readable audit trail.

| Agent | Job | Detects |
|---|---|---|
| **Risk Scoring Agent** | Real-time checkout risk scoring | COD-refusal probability, return-fraud probability, using device fingerprint, IP velocity, payment-method flags |
| **Authenticity & Integrity Agent** | Pre-publish listing screening | Counterfeit products via image/logo comparison, text, and price-vs-MSRP deviation |
| **Review Moderation Agent** | Post-review screening | Coordinated fake-review rings and AI-generated review text |

All three feed into an **orchestration layer** that logs every decision (score, reason, action) to a deterministic audit trail, and routes borderline cases to a human review queue instead of a blind auto-action.

## Who Uses This

This is an **internal admin tool**, not a customer or seller-facing app. Customers and sellers interact with the marketplace normally (checkout, listing upload, posting reviews) — the three agents work silently in the background. The only human-facing surface is a **Trust & Safety admin dashboard**, used by the platform's internal moderation team to monitor flagged activity, review edge cases, and audit past decisions.

## Architecture

```
Customer checkout ──┐
Seller new listing ─┼──► [Risk Agent] [Authenticity Agent] [Review Agent]
Customer review ────┘              │
                                    ▼
                    Orchestrator + Audit Trail (SQLite log)
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                            ▼
                Auto action                 Human review queue
             (approve / block)          (T&S team via dashboard)
```

## Datasets Used

| Agent | Dataset | Source |
|---|---|---|
| Risk Scoring Agent | IEEE-CIS Fraud Detection Dataset | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) — 590,540 real-world transactions |
| Review Moderation Agent | Amazon Fake Reviews Dataset / OpSpam Corpus | [Kaggle](https://www.kaggle.com/datasets/thearijitdas/fake-reviews-dataset) |
| Authenticity Agent | Counterfeit Product & Logo Detection Dataset (INNV Luxury Fashion) | [Hugging Face](https://huggingface.co/datasets/haemin8777/innv-luxury-fashion-dataset-fraud-detection) |

## Guardrails (Enterprise Security — highest-weighted scoring dimension)

- **Latency:** checkout risk decisions return in under 250ms — achieved by using lightweight models (gradient-boosted trees), not LLMs, on the hot path
- **Data sovereignty:** all customer PII and transaction records stay within local (India) region assumptions in design — no data sent to external LLM APIs for the risk-scoring hot path
- **Fairness:** false-positive rate is tracked separately for small/new sellers vs. established sellers, and surfaced on the dashboard, to catch unfair targeting
- **Explainability:** every automated decision is logged with a human-readable reason string, not just a score

## Tech Stack

- **Interface:** Gradio (single app — combines simulation UI + admin dashboard + tabs)
- **Risk Scoring Agent:** LightGBM / XGBoost trained on IEEE-CIS sample
- **Authenticity Agent:** Pretrained CLIP (image embeddings) + rule-based price-vs-MSRP check
- **Review Moderation Agent:** Pretrained text classifier (toxicity/fake-review model from Hugging Face) + timestamp/account-age pattern rules
- **Orchestration & Audit Log:** Python + SQLite
- **Language:** Python throughout

## Success Metrics (Target)

- Cut return fraud / empty-box claims / uncollectible COD losses by 35%
- >96% precision on automated listing holds/counterfeit flags
- Genuine customers wrongly blocked kept below 0.1%
- Deterministic, human-readable audit trail for every automated action

## Setup Instructions

_(To be completed in v2 once code is finalized)_

```bash
git clone <repo-url>
cd trust-safety-platform
pip install -r requirements.txt
python app.py
```
