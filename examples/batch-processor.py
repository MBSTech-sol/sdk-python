#!/usr/bin/env python3
"""
MBS SDK Example — Batch Processor

Reads prompts from a CSV file, processes them in parallel against the
MBS server, and writes results to an output CSV.

Usage:
    pip install mbs-python
    python batch-processor.py prompts.csv --output results.csv --concurrency 4

CSV format (input):
    prompt
    "Summarize quantum computing"
    "Explain neural networks"
    ...
"""

import asyncio
import argparse
import csv
import sys
import time
from pathlib import Path

from mbs import MbsClient, ChatCompletionRequest, ChatMessage


async def main():
    parser = argparse.ArgumentParser(description="MBS Batch Processor")
    parser.add_argument("input_csv", help="Input CSV with a 'prompt' column")
    parser.add_argument("--output", "-o", default="results.csv", help="Output CSV path")
    parser.add_argument("--url", default="http://127.0.0.1:3030", help="MBS server URL")
    parser.add_argument("--key", default=None, help="API key (optional)")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--model", default=None, help="Model to use")
    args = parser.parse_args()

    # Read prompts
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    prompts: list[str] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if prompt:
                prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts from {input_path}")

    # Build requests
    requests = [
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=p)],
            model=args.model,
            max_tokens=args.max_tokens,
        )
        for p in prompts
    ]

    # Process batch
    async with MbsClient(base_url=args.url, api_key=args.key) as client:
        start = time.perf_counter()
        summary = await client.batch_chat(requests, concurrency=args.concurrency)
        elapsed = time.perf_counter() - start

    # Write results
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response", "status", "tokens"])
        for i, result in enumerate(summary.results):
            prompt = prompts[i] if i < len(prompts) else ""
            if result.ok and result.value:
                val = result.value
                text = val.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = val.get("usage", {}).get("total_tokens", 0)
                writer.writerow([prompt, text, "ok", tokens])
            else:
                writer.writerow([prompt, result.error or "unknown error", "error", 0])

    print(f"\nResults written to {args.output}")
    print(f"  Succeeded: {summary.succeeded}/{len(summary.results)}")
    print(f"  Failed:    {summary.failed}/{len(summary.results)}")
    print(f"  Time:      {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
