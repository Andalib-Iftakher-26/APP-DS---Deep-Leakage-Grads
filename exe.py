import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Measure the average execution time of main.py."
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="number of executions to measure"
    )
    parser.add_argument(
        "--index", type=int, default=25, help="CIFAR-100 image index passed to main.py"
    )
    parser.add_argument(
        "--image", default="", help="optional custom image path passed to main.py"
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    main_script = Path(__file__).with_name("main.py")
    command = [sys.executable, str(main_script), "--index", str(args.index)]
    if args.image:
        command.extend(["--image", args.image])

    durations = []
    for run_number in range(1, args.runs + 1):
        start = time.perf_counter()
        subprocess.run(command, check=True)
        duration = time.perf_counter() - start
        durations.append(duration)
        print(f"Run {run_number}: {duration:.2f} seconds")

    average = sum(durations) / len(durations)
    print(f"Average execution time: {average:.2f} seconds")


if __name__ == "__main__":
    main()
