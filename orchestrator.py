import sys
sys.path.append("agents")

from agents.risk_agent import score_transaction
from agents.authenticity_agent import check_listing
from agents.review_agent import check_review_ring
import audit_log

def run_risk_check(transaction_dict: dict, seller_type: str = "established"):
    result = score_transaction(transaction_dict)
    audit_log.log_decision(
        agent="risk_scoring",
        input_summary=f"transaction seller_type:{seller_type}",
        decision=result["decision"],
        reason=result["reason"],
        latency_ms=result["latency_ms"]
    )
    return result

def run_authenticity_check(image_path: str, listed_price: float, msrp: float, seller_type: str = "established"):
    result = check_listing(image_path, listed_price, msrp)
    audit_log.log_decision(
        agent="authenticity",
        input_summary=f"listing:{image_path} price:{listed_price} msrp:{msrp} seller_type:{seller_type}",
        decision=result["decision"],
        reason=result["reason"],
        latency_ms=result["latency_ms"]
    )
    return result

def run_review_check(seller_reviews: list, seller_type: str = "established"):
    result = check_review_ring(seller_reviews)
    audit_log.log_decision(
        agent="review_moderation",
        input_summary=f"num_reviews:{len(seller_reviews)} seller_type:{seller_type}",
        decision=result["decision"],
        reason=result["reason"],
        latency_ms=result["latency_ms"]
    )
    return result
