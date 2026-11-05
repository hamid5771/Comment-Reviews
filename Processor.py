import json
import math
import re
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RAW_GOOGLE_PATH = Path("data/raw/google_reviews.jsonl")
RAW_TA_PATH = Path("data/raw/tripadvisor_reviews.jsonl")
PROCESSED_OUTPUT_PATH = Path("data/processed/analyzed_reviews.jsonl")

ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

HAS_RESTAURANT_MENTION = re.compile(
    r"\b(breakfast|restaurant|food|dinner|eatery|pub|drink|drinks|beer|beers|"
    r"menu|starter|starters|main|mains|waiter|waitress|chef|kitchen|table|tables|"
    r"buffet|pastry|pastries|fruit|coffee|tea|toast|bacon|sausage|eggs|meal|meals|"
    r"bar|dining|order|orders|ordered|portion|portions|serve|served|serving|"
    r"walkins|walk-ins|cook|cooked|cereal|juice|beefeater|beef eater)\b",
    re.IGNORECASE,
)


def calculate_credibility_weight(contributions: int, raw_rating: float, is_local_guide: bool = False) -> float:
    effective_c = max(contributions, 15 if is_local_guide else 1)
    if effective_c <= 1 and raw_rating <= 2.0:
        return 0.10
    if effective_c < 10:
        return 1.00
    boosted = 1.0 + 0.15 * math.log(effective_c)
    return round(min(1.50, boosted), 3)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"[a-z0-9]{3,}", text_a.lower()))
    words_b = set(re.findall(r"[a-z0-9]{3,}", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def mark_cross_platform_duplicates(records: list[dict]) -> list[dict]:
    for rec in records:
        rec["is_cross_platform_duplicate"] = False
        rec["duplicate_of"] = None

    for i, rec_a in enumerate(records):
        if rec_a["is_cross_platform_duplicate"]:
            continue
        for j in range(i + 1, len(records)):
            rec_b = records[j]
            if rec_a["source"] == rec_b["source"] or rec_b["is_cross_platform_duplicate"]:
                continue

            auth_a = re.sub(r"[^a-z]", "", rec_a.get("author", "").lower())[:4]
            auth_b = re.sub(r"[^a-z]", "", rec_b.get("author", "").lower())[:4]
            sim = jaccard_similarity(rec_a.get("raw_text", ""), rec_b.get("raw_text", ""))

            if sim >= 0.45 or (auth_a and auth_a == auth_b and sim >= 0.20):
                rec_b["is_cross_platform_duplicate"] = True
                rec_b["duplicate_of"] = rec_a["review_id"]

    return records


class WholeReviewABSAEngine:
    def __init__(self, model_name: str = ABSA_MODEL_NAME):
        print(f"Loading DeBERTa-v3 ABSA model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_aspect(self, full_text: str, aspect_prompt: str) -> dict:
        encoded = self.tokenizer(
            full_text,
            aspect_prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        p_neg, p_neu, p_pos = probs[0], probs[1], probs[2]
        polarity = p_pos - p_neg
        star_score = 3.0 + 2.0 * polarity

        return {
            "polarity": round(polarity, 3),
            "star_score": round(star_score, 2),
            "confidence": {
                "neg": round(p_neg, 3),
                "neu": round(p_neu, 3),
                "pos": round(p_pos, 3),
            },
        }


def load_raw_reviews() -> list[dict]:
    records = []
    for path in (RAW_GOOGLE_PATH, RAW_TA_PATH):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def run_pipeline():
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_reviews()
    if not raw_records:
        print("No raw reviews found in data/raw/. Run the scrapers first!")
        return

    print(f"Loaded {len(raw_records)} total raw reviews from Google and TripAdvisor.")
    records = mark_cross_platform_duplicates(raw_records)

    engine = None  # Ratings without text do not need an NLP model.

    processed_records = []
    with open(PROCESSED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            full_text = (rec.get("raw_text") or "").strip()
            hotel_score = rest_score = hotel_conf = rest_conf = None
            if full_text:
                if engine is None:
                    engine = WholeReviewABSAEngine()
                hotel_res = engine.score_aspect(full_text, "hotel room, staff, and stay")
                hotel_score = hotel_res["star_score"]
                hotel_conf = hotel_res["confidence"]

                if HAS_RESTAURANT_MENTION.search(full_text):
                    rest_res = engine.score_aspect(full_text, "restaurant, breakfast, food, and drinks")
                    rest_score = rest_res["star_score"]
                    rest_conf = rest_res["confidence"]

            weight = calculate_credibility_weight(
                contributions=rec.get("author_contributions", 1),
                raw_rating=rec.get("raw_rating", 5.0),
                is_local_guide=rec.get("is_local_guide", False),
            )

            enriched = {
                **rec,
                "has_text": bool(full_text),
                "credibility_weight": weight,
                "hotel_score": hotel_score,
                "restaurant_score": rest_score,
                "absa_details": {
                    "hotel_confidence": hotel_conf,
                    "restaurant_confidence": rest_conf,
                },
                "processed_at": datetime.now().isoformat(),
            }
            processed_records.append(enriched)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    unique_recs = [r for r in processed_records if not r["is_cross_platform_duplicate"]]
    dups_count = len(processed_records) - len(unique_recs)

    raw_avg = sum(r["raw_rating"] for r in unique_recs) / len(unique_recs)
    weighted_avg = sum(r["raw_rating"] * r["credibility_weight"] for r in unique_recs) / sum(
        r["credibility_weight"] for r in unique_recs
    )

    hotel_vals = [r for r in unique_recs if r["hotel_score"] is not None]
    rest_vals = [r for r in unique_recs if r["restaurant_score"] is not None]

    hotel_weighted = (sum(r["hotel_score"] * r["credibility_weight"] for r in hotel_vals) / sum(
        r["credibility_weight"] for r in hotel_vals
    ) if hotel_vals else None)
    rest_weighted = (
        sum(r["restaurant_score"] * r["credibility_weight"] for r in rest_vals)
        / sum(r["credibility_weight"] for r in rest_vals)
        if rest_vals
        else None
    )

    print("\n" + "=" * 68)
    print(" 🌙 PREMIER INN EVESHAM — WHOLE-REVIEW DeBERTa-v3 ABSA SUMMARY")
    print("=" * 68)
    print(f"Total Reviews Ingested       : {len(processed_records)} ({dups_count} cross-platform duplicate flagged)")
    print(f"Unique Stays Analyzed        : {len(unique_recs)}")
    print("-" * 68)
    print(f"1. Raw Unfiltered Rating     : {raw_avg:.2f} / 5.00 ★")
    print(f"2. Credibility-Weighted (W_i): {weighted_avg:.2f} / 5.00 ★  (Delta: {weighted_avg - raw_avg:+.2f} ★)")
    hotel_display = f"{hotel_weighted:.2f} / 5.00" if hotel_weighted is not None else "N/A"
    rest_display = f"{rest_weighted:.2f} / 5.00" if rest_weighted is not None else "N/A"
    print(f"3. Core HOTEL Performance    : {hotel_display}  (across {len(hotel_vals)} reviews)")
    print(f"4. RESTAURANT Performance    : {rest_display}  (across {len(rest_vals)} reviews)")
    print("=" * 68)

    print("\n--- SAMPLE MIXED-REVIEW DECOUPLING CHECK ---")
    for r in unique_recs:
        if r["restaurant_score"] is not None:
            print(
                f"• {r['author'][:18]:<18} | Raw: {r['raw_rating']}★ | "
                f"W_i: {r['credibility_weight']:<4} | "
                f"HOTEL: {r['hotel_score']:<4}★ | "
                f"RESTAURANT: {r['restaurant_score']:<4}★"
            )
    print(f"\nSaved enriched dataset to: {PROCESSED_OUTPUT_PATH}\n")


if __name__ == "__main__":
    run_pipeline()
