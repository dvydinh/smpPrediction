import argparse
import logging
import json
from vcgm.inference.predictor import SmpPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=False)
    args = parser.parse_args()

    pred = SmpPredictor()
    result = pred.predict(args.date)
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2))
    if result["status"] != "failed":
        for c, p in zip(result["cycles"], result["predictions"]):
            print(f"  {c}: {p:>8.2f}")

if __name__ == "__main__":
    main()
