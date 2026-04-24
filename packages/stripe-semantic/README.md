# stripe-semantic

Semantic model and proven query patterns for Stripe's core tables. Covers MRR, active subscriptions, gross charges, net revenue, and customer LTV. Amounts are stored in cents in Stripe's schema; all metric definitions here handle the conversion to dollars.

## Install

```bash
talkdb registry install ./packages/stripe-semantic
```

or, once the package is published to a registry:

```bash
talkdb registry install stripe-semantic
```

## What it covers

- **5 metrics:** `mrr`, `active_subscriptions`, `gross_charges`, `net_revenue`, `customer_ltv`
- **5 tables:** `customers`, `subscriptions`, `charges`, `refunds`, `invoices`
- **Join rules** for each of the common Stripe table relationships
- **~6 proven example queries** seeded into the retriever

## After install

TalkDB's hybrid retriever will surface these definitions when a user asks Stripe-related questions. Example:

> User: "What is our MRR?"
>
> TalkDB: `SELECT SUM(plan_amount) / 100.0 AS mrr FROM subscriptions WHERE status = 'active'`

## License

MIT.
