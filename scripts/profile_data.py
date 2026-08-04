import logging
from vcgm.data import loader, alignment, profiler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    smp = loader.load_smp()
    master = alignment.build_master_timeline()
    smp_aligned = alignment.align_to_timeline(smp, master)
    prof = profiler.profile_smp(smp_aligned["smp_system_price"])
    for k, v in prof.items():
        print(f"{k}: {v}")
    gaps = profiler.find_gaps(smp_aligned.index, smp_aligned["smp_system_price"].isna())
    print(f"\n{len(gaps)} gaps" if gaps else "\nNo gaps")

if __name__ == "__main__":
    main()
