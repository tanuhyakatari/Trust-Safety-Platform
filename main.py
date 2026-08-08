from fastapi import FastAPI, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import shutil
import random
from datetime import datetime, timedelta

from orchestrator import run_risk_check, run_authenticity_check, run_review_check
import audit_log
import agents.authenticity_agent as auth_agent
import glob

app = FastAPI()

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/risk")
def api_risk(
    mode: str = Form(...),
    amount: float = Form(None),
    payment_method: str = Form("prepaid"),
):
    if mode == "random":
        row = pd.read_csv("data/ieee_fraud_sample.csv").sample(1).iloc[0].to_dict()
        result = run_risk_check(row, seller_type="established")
        return result

    # Manual test mode: blend the trained model's base signal with an
    # explainable amount/payment-method rule, since manual input only
    # supplies 1-2 of the 380+ features the full model expects.
    row = {"TransactionAmt": amount or 100.0}
    base_result = run_risk_check(row, seller_type="established")
    base_prob = base_result["fraud_probability"]

    amt = amount or 100.0
    rule_bump = 0.0
    reasons = []
    if amt > 30000:
        rule_bump += 0.45
        reasons.append(f"high transaction amount (Rs.{amt:,.0f})")
    elif amt > 10000:
        rule_bump += 0.2
        reasons.append(f"elevated transaction amount (Rs.{amt:,.0f})")
    if payment_method == "cod" and amt > 5000:
        rule_bump += 0.25
        reasons.append("high-value COD order")

    adjusted_prob = min(base_prob + rule_bump, 0.99)

    if adjusted_prob >= 0.7:
        decision = "block"
    elif adjusted_prob >= 0.3:
        decision = "human_review"
    else:
        decision = "approve"

    reason_text = f"Base model score {base_prob:.2f}"
    if reasons:
        reason_text += " + " + ", ".join(reasons)
    reason_text += f" -> adjusted score {adjusted_prob:.2f}"

    return {
        "fraud_probability": round(adjusted_prob, 4),
        "decision": decision,
        "reason": reason_text,
        "latency_ms": base_result["latency_ms"],
    }

@app.post("/api/authenticity")
async def api_authenticity(
    listed_price: float = Form(...),
    msrp: float = Form(...),
    seller_type: str = Form("established"),
    image: UploadFile = None,
):
    with open("temp_upload.jpg", "wb") as f:
        shutil.copyfileobj(image.file, f)
    result = run_authenticity_check("temp_upload.jpg", listed_price, msrp, seller_type=seller_type)
    return result

@app.post("/api/review")
def api_review(seller_type: str = Form("established"), scenario: str = Form("ring")):
    fake_ring = [
        {"text": "Amazing product, best purchase ever, five stars highly recommend!", "account_age_days": 2, "timestamp": "2026-08-08 10:00:00"},
        {"text": "This is the best product I have ever bought, five stars!", "account_age_days": 1, "timestamp": "2026-08-08 10:15:00"},
        {"text": "Excellent quality, amazing value, five stars!", "account_age_days": 3, "timestamp": "2026-08-08 10:30:00"},
        {"text": "Best purchase of my life, five stars, highly recommend!", "account_age_days": 1, "timestamp": "2026-08-08 10:45:00"},
        {"text": "Amazing quality and value, five stars to all buyers!", "account_age_days": 2, "timestamp": "2026-08-08 10:50:00"},
    ]
    normal_batch = [
        {"text": "Good quality, arrived a bit late but works as described.", "account_age_days": 240, "timestamp": "2026-07-01 09:12:00"},
        {"text": "Not bad for the price, would buy again honestly.", "account_age_days": 512, "timestamp": "2026-07-15 14:20:00"},
        {"text": "Decent product, packaging could have been better.", "account_age_days": 88, "timestamp": "2026-07-20 18:40:00"},
    ]
    batch = fake_ring if scenario == "ring" else normal_batch
    result = run_review_check(batch, seller_type=seller_type)
    return result

@app.get("/api/queue")
def api_queue():
    rows = audit_log.get_all_logs()
    return [
        {"id": r[0], "timestamp": r[1], "agent": r[2], "input": r[3], "decision": r[4], "reason": r[5], "latency_ms": r[6]}
        for r in rows
    ]

@app.get("/api/fairness")
def api_fairness():
    rows = audit_log.get_fairness_stats()
    return [{"seller_type": r[0], "decision": r[1], "count": r[2]} for r in rows]

CUSTOMER_NAMES = ["Ananya R.", "Rohan K.", "Priya S.", "Vikram M.", "Sneha T.", "Arjun P.", "Divya N.", "Karthik V."]
CITIES = ["Bengaluru", "Vijayawada", "Hyderabad", "Chennai", "Mumbai", "Pune"]
PRODUCTS_RISK = ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Laptop Bag", "Air Fryer", "Running Shoes"]
PRODUCT_LABELS = ["Chanel Timeless Bag", "Chanel Boy Bag", "Chanel Gabrielle Bag", "Chanel Chance Perfume"]
REVIEWER_NAMES = ["user_rk92", "shopperpriya", "deal_hunter_88", "newbuyer21", "reviewqueen"]
normal_reviews_pool = ["Good quality, arrived a bit late but works as described.", "Not bad for the price, would buy again.", "Decent product, packaging could be better."]
ring_reviews_pool = ["Amazing product, best purchase ever, five stars highly recommend!", "This is the best product I have ever bought, five stars!", "Excellent quality, amazing value, five stars!", "Best purchase of my life, five stars, highly recommend!", "Amazing quality and value, five stars to all buyers!"]

def make_review_batch(pool, is_ring):
    base = datetime(2026, 8, 8, 10, 0, 0)
    return [{"text": random.choice(pool), "account_age_days": random.randint(1,5) if is_ring else random.randint(60,600), "timestamp": (base + timedelta(minutes=random.randint(0,15) if is_ring else random.randint(0,4000))).strftime("%Y-%m-%d %H:%M:%S")} for _ in range(5)]

@app.get("/api/live-event")
def api_live_event():
    event_type = random.choice(["risk", "authenticity", "review"])
    seller_type = random.choice(["small", "established"])
    customer, city = random.choice(CUSTOMER_NAMES), random.choice(CITIES)

    if event_type == "risk":
        row = pd.read_csv("data/ieee_fraud_sample.csv").sample(1).iloc[0].to_dict()
        product, amount = random.choice(PRODUCTS_RISK), round(random.uniform(400, 65000), 0)
        row["TransactionAmt"] = amount
        result = run_risk_check(row, seller_type=seller_type)
        headline = f"{customer} in {city} ordered {product} - Rs.{amount:,.0f} ({'COD' if amount < 5000 else 'Prepaid'})"
        agent = "Risk Scoring"
    elif event_type == "authenticity":
        img_path = random.choice(glob.glob("static/sample_listings/*.jpg"))
        product = random.choice(PRODUCT_LABELS)
        msrp = random.choice([1500, 2500, 3200, 4500])
        price = round(msrp * random.choice([0.85, 0.9, 0.15, 0.1]), 0)
        result = run_authenticity_check(img_path, price, msrp, seller_type=seller_type)
        headline = f"New listing '{product}' - Rs.{price:,.0f} (MSRP Rs.{msrp:,.0f})"
        agent = "Authenticity"
    else:
        is_ring = random.random() < 0.5
        batch = make_review_batch(ring_reviews_pool if is_ring else normal_reviews_pool, is_ring)
        result = run_review_check(batch, seller_type=seller_type)
        reviewers = ", ".join(random.sample(REVIEWER_NAMES, 3))
        headline = f"5 new reviews on a listing (incl. {reviewers}...)"
        agent = "Review Moderation"

    return {"agent": agent, "headline": headline, **result}

@app.get("/api/authenticity-sample")
def api_authenticity_sample(scenario: str = "genuine"):
    img_path = random.choice(glob.glob("static/sample_listings/*.jpg"))
    msrp = 1500
    price = msrp * 0.85 if scenario == "genuine" else msrp * 0.08
    result = run_authenticity_check(img_path, price, msrp, seller_type="established")
    return result
