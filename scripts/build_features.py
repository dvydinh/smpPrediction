import logging
from vcgm.features import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    df = pipeline.build_design_matrix(save=True)
    print(f"Done: {df.shape}")

if __name__ == "__main__":
    main()
