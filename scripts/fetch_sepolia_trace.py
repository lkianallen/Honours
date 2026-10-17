"""Download the last N Sepolia blocks via Alchemy and save them where load_alchemy_blocks() expects. Needs network access and `pip install requests`; run it locally."""
from pathlib import Path

import os 
import json
import time
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

# Set ALCHEMY_URL in your environment; never hard-code the key.
# export ALCHEMY_URL="https://eth-sepolia.g.alchemy.com/v2/<your-key>"
ALCHEMY_URL = os.environ.get("ALCHEMY_URL", "")

NUM_BLOCKS = 50          # how many recent blocks to pull
OUTPUT_PATH = REPO_ROOT / "sepolia_trace.json"
REQUEST_DELAY_SECONDS = 0.1  # be polite to the free tier rate limit


def rpc_call(method: str, params: list) -> dict:
    response = requests.post(
        ALCHEMY_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"RPC error calling {method}: {payload['error']}")
    return payload["result"]


def main():
    latest_hex = rpc_call("eth_blockNumber", [])
    latest = int(latest_hex, 16)
    print(f"Latest Sepolia block: {latest} ({latest_hex})")

    start = latest - NUM_BLOCKS + 1
    print(f"Pulling blocks {start}..{latest} ({NUM_BLOCKS} blocks)")

    blocks = []
    for block_num in range(start, latest + 1):
        block_hex = hex(block_num)
        # `True` = include full transaction objects, not just their hashes
        block = rpc_call("eth_getBlockByNumber", [block_hex, True])
        if block is None:
            print(f"  block {block_num}: not found, skipping")
            continue

        num_tx = len(block.get("transactions", []))
        print(f"  block {block_num}: {num_tx} transactions")
        blocks.append(block)

        time.sleep(REQUEST_DELAY_SECONDS)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(blocks, f)

    total_tx = sum(len(b.get("transactions", [])) for b in blocks)
    print(f"\nSaved {len(blocks)} blocks ({total_tx} total transactions) to {OUTPUT_PATH}")
    print("Load it with: sim.data_sources.load_transactions_from_alchemy(OUTPUT_PATH, node_ids=[...])")


if __name__ == "__main__":
    main()
