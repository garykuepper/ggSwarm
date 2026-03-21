#!/usr/bin/env python3
"""SSH tunnel to GCE VM and open TensorBoard in a local browser."""

import argparse
import subprocess
import sys
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training via TensorBoard on remote VM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_training.py
  python monitor_training.py --family hover
  python monitor_training.py --port 6007 --instance isaacsim
        """,
    )
    parser.add_argument(
        "--family",
        choices=["marl", "hover"],
        default="marl",
        help="Training family (default: marl)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Local port to forward to (default: 6006)",
    )
    parser.add_argument(
        "--zone",
        default="us-central1-a",
        help="GCP zone (default: us-central1-a)",
    )
    parser.add_argument(
        "--instance",
        default="isaacsim",
        help="GCE instance name (default: isaacsim)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 40)
    print("  TensorBoard Remote Monitor Setup")
    print("=" * 40 + "\n")

    print(f"Opening SSH tunnel to {args.instance} ({args.zone}) on port {args.port}...")
    print(f"Family: {args.family}\n")

    print("IMPORTANT:")
    print("  1. Keep this terminal open while monitoring TensorBoard")
    print(f"  2. Your browser will open to http://localhost:{args.port}")
    print("  3. Press Ctrl+C here to close the tunnel when done\n")

    time.sleep(1)

    # Open the SSH tunnel
    print("Starting tunnel...")
    tunnel_cmd = [
        "gcloud",
        "compute",
        "ssh",
        args.instance,
        f"--zone={args.zone}",
        "--",
        "-N",
        f"-L",
        f"{args.port}:127.0.0.1:6006",
    ]

    try:
        # Start tunnel in the foreground (will be interrupted by Ctrl+C)
        tunnel_process = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=sys.platform == "win32",
        )
        print(f"Tunnel started (PID: {tunnel_process.pid})\n")

        # Give the tunnel a moment to establish
        time.sleep(2)

        # Open the browser
        url = f"http://localhost:{args.port}"
        print(f"Opening browser to {url}...")
        webbrowser.open(url)

        print(f"\nTensorBoard is loading at {url}")
        print("Keep this terminal open. Press Ctrl+C to stop the tunnel.\n")

        # Wait for the tunnel process
        tunnel_process.wait()

    except FileNotFoundError:
        print("Error: gcloud command not found. Install Google Cloud SDK.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nClosing tunnel...")
        tunnel_process.terminate()
        tunnel_process.wait(timeout=5)
        print("Tunnel closed.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
