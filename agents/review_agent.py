import joblib
import time
from datetime import datetime, timedelta

model = joblib.load("models/review_model.pkl")
vectorizer = joblib.load("models/review_vectorizer.pkl")

def score_review_text(review_text: str):
    """
    Scores a single review's text for fake/genuine probability.
    """
    vec = vectorizer.transform([review_text])
    prob_fake = model.predict_proba(vec)[0][0]  # index 0 = fake (CG)
    return round(float(prob_fake), 4)

def check_review_ring(seller_reviews: list):
    """
    Checks a batch of reviews for the same seller for coordinated-ring patterns.
    Each item in seller_reviews should be a dict:
      { "text": str, "account_age_days": int, "timestamp": "YYYY-MM-DD HH:MM:SS" }
    Returns a ring-risk assessment for the whole batch.
    """
    start = time.time()

    if not seller_reviews:
        return {"decision": "approve", "reason": "No reviews to check", "latency_ms": 0}

    timestamps = [datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") for r in seller_reviews]
    new_accounts = [r for r in seller_reviews if r["account_age_days"] < 7]

    time_span = max(timestamps) - min(timestamps)
    burst_flag = len(seller_reviews) >= 5 and time_span <= timedelta(hours=1)
    new_account_flag = len(new_accounts) / len(seller_reviews) >= 0.6

    fake_scores = [score_review_text(r["text"]) for r in seller_reviews]
    avg_fake_score = sum(fake_scores) / len(fake_scores)
    text_flag = avg_fake_score >= 0.5

    latency_ms = (time.time() - start) * 1000

    flags_triggered = sum([burst_flag, new_account_flag, text_flag])

    if flags_triggered >= 2:
        decision = "block"
        reason = f"Coordinated ring suspected — burst:{burst_flag}, new-accounts:{new_account_flag} ({len(new_accounts)}/{len(seller_reviews)}), avg fake-text score:{avg_fake_score:.2f}"
    elif flags_triggered == 1:
        decision = "human_review"
        reason = f"One risk signal triggered — burst:{burst_flag}, new-accounts:{new_account_flag}, avg fake-text score:{avg_fake_score:.2f}"
    else:
        decision = "approve"
        reason = f"No ring pattern detected — avg fake-text score:{avg_fake_score:.2f}"

    return {
        "num_reviews": len(seller_reviews),
        "avg_fake_score": round(avg_fake_score, 4),
        "burst_flag": burst_flag,
        "new_account_flag": new_account_flag,
        "decision": decision,
        "reason": reason,
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    # Test 1: a single genuine-sounding review
    print("Single review test:", score_review_text("The battery life is decent but the case scratches easily after a week of use."))

    # Test 2: simulate a fake review ring (5 reviews, new accounts, all within 1 hour)
    fake_ring = [
        {"text": "Amazing product, best purchase ever, five stars highly recommend to everyone!", "account_age_days": 2, "timestamp": "2026-08-08 10:00:00"},
        {"text": "This is the best product I have ever bought, five stars, highly recommend!", "account_age_days": 1, "timestamp": "2026-08-08 10:15:00"},
        {"text": "Excellent quality, amazing value, five stars, highly recommend to all!", "account_age_days": 3, "timestamp": "2026-08-08 10:30:00"},
        {"text": "Best purchase of my life, five stars, highly recommend this to everyone!", "account_age_days": 1, "timestamp": "2026-08-08 10:45:00"},
        {"text": "Amazing quality and value, five stars, highly recommend to all buyers!", "account_age_days": 2, "timestamp": "2026-08-08 10:50:00"},
    ]
    print("Ring test:", check_review_ring(fake_ring))
