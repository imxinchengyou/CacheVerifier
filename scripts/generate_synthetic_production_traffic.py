"""Generate a synthetic dataset in the commercial product's production
feedback schema — (query, candidate_answer, was_correct) records — for
smoke-testing cacheverifier-service outside this repo.

This is NOT real user traffic; there is no real-traffic source available
here. It's a hand-authored set of ~20 customer-support intents across six
domains (subscription, billing, orders, account/security, shipping, product
support), each rendered as several paraphrased queries and paired with either
its own correct answer or a plausible wrong one (same-domain neighbor most of
the time, cross-domain distractor otherwise), to mimic what a real semantic
cache's hit/miss log looks like.

Usage:
    python scripts/generate_synthetic_production_traffic.py
"""

import json
import random
from pathlib import Path

SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "production_traffic_test.json"

INTENTS = [
    {
        "id": "subscription_cancel",
        "answer": "Go to Settings > Billing > Cancel subscription.",
        "queries": [
            "how do I cancel my subscription",
            "I want to cancel my subscription",
            "cancel my subscription please",
            "how do I cancel my plan",
            "I'd like to cancelled my subscription",  # casual/typo realism
        ],
    },
    {
        "id": "subscription_pause",
        "answer": "Go to Settings > Billing > Pause subscription for a month.",
        "queries": [
            "how do I pause my subscription",
            "can I pause my subscription for a bit",
            "I want to pause my plan temporarily",
            "how do I put my subscription on hold",
        ],
    },
    {
        "id": "subscription_renew",
        "answer": "Go to Settings > Billing > Renew subscription now.",
        "queries": [
            "how do I renew my subscription",
            "I want to renew my plan early",
            "how do I renew before it expires",
        ],
    },
    {
        "id": "subscription_upgrade",
        "answer": "Go to Settings > Billing > Upgrade to Pro plan.",
        "queries": [
            "how do I upgrade my plan",
            "I want to upgrade to Pro",
            "how do I get the Pro plan",
        ],
    },
    {
        "id": "subscription_downgrade",
        "answer": "Go to Settings > Billing > Downgrade to Free plan.",
        "queries": [
            "how do I downgrade my plan",
            "I want to downgrade to the free plan",
            "how do I go back to the free tier",
        ],
    },
    {
        "id": "billing_update_email",
        "answer": "Go to Account > Profile > Update billing email.",
        "queries": [
            "how do I update my billing email",
            "I need to change my billing email address",
            "how do I change where invoices are sent",
        ],
    },
    {
        "id": "billing_update_payment",
        "answer": "Go to Account > Billing > Update payment method.",
        "queries": [
            "how do I update my payment method",
            "I need to change my credit card on file",
            "how do I add a new card",
        ],
    },
    {
        "id": "billing_get_invoice",
        "answer": "Go to Account > Billing > Download latest invoice.",
        "queries": [
            "how do I get my invoice",
            "I need a copy of last month's invoice",
            "where can I download my receipt",
        ],
    },
    {
        "id": "billing_dispute_charge",
        "answer": "Go to Support > Billing Dispute > File a chargeback request.",
        "queries": [
            "how do I dispute a charge",
            "I was charged twice, how do I dispute it",
            "how do I file a chargeback",
        ],
    },
    {
        "id": "orders_refund",
        "answer": "Go to Orders > Return item > Request refund.",
        "queries": [
            "how do I get a refund for my order",
            "I want a refund for this item",
            "how do I request my money back",
        ],
    },
    {
        "id": "orders_return",
        "answer": "Go to Orders > Return item > Print return label.",
        "queries": [
            "how do I return an item",
            "I need to print a return label",
            "how do I send this item back",
        ],
    },
    {
        "id": "orders_exchange",
        "answer": "Go to Orders > Return item > Exchange for a different size.",
        "queries": [
            "how do I exchange this for a different size",
            "I want to exchange my order for another color",
            "can I swap this item for a different one",
        ],
    },
    {
        "id": "orders_track",
        "answer": "Go to Orders > Track shipment > View live tracking.",
        "queries": [
            "how do I track my order",
            "where is my package right now",
            "I want to see my shipment status",
        ],
    },
    {
        "id": "orders_cancel_order",
        "answer": "Go to Orders > Cancel order > Confirm cancellation.",
        "queries": [
            "how do I cancel my order",
            "I want to cancel an order I just placed",
            "can I still cancel my order before it ships",
        ],
    },
    {
        "id": "account_reset_password",
        "answer": "Go to Account > Security > Reset password.",
        "queries": [
            "how do I reset my password",
            "I forgot my password",
            "how do I change my password",
        ],
    },
    {
        "id": "account_enable_2fa",
        "answer": "Go to Account > Security > Enable two-factor authentication.",
        "queries": [
            "how do I enable two-factor authentication",
            "I want to turn on 2FA",
            "how do I add extra login security",
        ],
    },
    {
        "id": "account_delete",
        "answer": "Go to Account > Settings > Delete account permanently.",
        "queries": [
            "how do I delete my account",
            "I want to permanently close my account",
            "how do I remove my account entirely",
        ],
    },
    {
        "id": "account_update_email",
        "answer": "Go to Account > Profile > Update login email.",
        "queries": [
            "how do I update my login email",
            "I need to change the email on my account",
            "how do I change my account email address",
        ],
    },
    {
        "id": "shipping_change_address",
        "answer": "Go to Orders > Shipping > Update delivery address.",
        "queries": [
            "how do I change my shipping address",
            "I need to update where my order is delivered",
            "how do I fix my delivery address before it ships",
        ],
    },
    {
        "id": "shipping_missed_delivery",
        "answer": "Go to Orders > Track shipment > Report missed delivery.",
        "queries": [
            "my package never arrived, what do I do",
            "how do I report a missed delivery",
            "the courier says delivered but I never got it",
        ],
    },
    {
        "id": "product_warranty",
        "answer": "Go to Support > Warranty > File a warranty claim.",
        "queries": [
            "how do I file a warranty claim",
            "my item broke, is it still under warranty",
            "how do I get warranty support",
        ],
    },
    {
        "id": "product_defective",
        "answer": "Go to Orders > Return item > Report defective item.",
        "queries": [
            "how do I report a defective item",
            "I received a broken product",
            "this item arrived damaged, what do I do",
        ],
    },
]

DOMAIN_OF = {
    "subscription_cancel": "subscription",
    "subscription_pause": "subscription",
    "subscription_renew": "subscription",
    "subscription_upgrade": "subscription",
    "subscription_downgrade": "subscription",
    "billing_update_email": "billing",
    "billing_update_payment": "billing",
    "billing_get_invoice": "billing",
    "billing_dispute_charge": "billing",
    "orders_refund": "orders",
    "orders_return": "orders",
    "orders_exchange": "orders",
    "orders_track": "orders",
    "orders_cancel_order": "orders",
    "account_reset_password": "account",
    "account_enable_2fa": "account",
    "account_delete": "account",
    "account_update_email": "account",
    "shipping_change_address": "shipping",
    "shipping_missed_delivery": "shipping",
    "product_warranty": "product",
    "product_defective": "product",
}

TARGET_TOTAL = 150
CORRECT_RATE = 0.70
SAME_DOMAIN_WRONG_RATE = 0.70  # when wrong, prefer a same-domain neighbor (harder / more realistic miss)


def build_records(rng: random.Random) -> list[dict]:
    by_id = {intent["id"]: intent for intent in INTENTS}
    domains = {}
    for intent in INTENTS:
        domains.setdefault(DOMAIN_OF[intent["id"]], []).append(intent["id"])

    occurrences_per_intent = TARGET_TOTAL // len(INTENTS)
    remainder = TARGET_TOTAL - occurrences_per_intent * len(INTENTS)

    records = []
    for idx, intent in enumerate(INTENTS):
        n = occurrences_per_intent + (1 if idx < remainder else 0)
        domain = DOMAIN_OF[intent["id"]]
        same_domain_others = [i for i in domains[domain] if i != intent["id"]]
        cross_domain_others = [i for i in by_id if DOMAIN_OF[i] != domain]

        for _ in range(n):
            query = rng.choice(intent["queries"])
            if rng.random() < CORRECT_RATE:
                answer = intent["answer"]
                was_correct = True
            else:
                if same_domain_others and rng.random() < SAME_DOMAIN_WRONG_RATE:
                    wrong_id = rng.choice(same_domain_others)
                else:
                    wrong_id = rng.choice(cross_domain_others)
                answer = by_id[wrong_id]["answer"]
                was_correct = False
            records.append({"query": query, "candidate_answer": answer, "was_correct": was_correct})

    rng.shuffle(records)
    return records


def main() -> None:
    rng = random.Random(SEED)
    records = build_records(rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    n_correct = sum(1 for r in records if r["was_correct"])
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")
    print(f"  correct: {n_correct} ({n_correct / len(records):.1%})")
    print(f"  incorrect: {len(records) - n_correct} ({1 - n_correct / len(records):.1%})")


if __name__ == "__main__":
    main()
