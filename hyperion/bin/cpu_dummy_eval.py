import os
import time
import argparse
import socket

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()

    print(f"Running dummy job for trigger={args.trigger} and target={args.target}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Working dir: {os.getcwd()}")

    # Simulate some workload
    time.sleep(5)

    print("Finished dummy evaluation.")

if __name__ == "__main__":
    main()
