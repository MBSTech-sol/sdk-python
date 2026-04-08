#!/usr/bin/env python3
"""
MBS SDK Example — Agent CLI

Runs an autonomous AI agent that can use MCP tools to accomplish
multi-step tasks via the MBS ReAct agent API.

Usage:
    pip install mbs-python
    python agent-cli.py "Find all TODO comments in the current project"
    python agent-cli.py --list-tools
"""

import asyncio
import argparse
import json

from mbs import MbsClient, AgentRunRequest


async def main():
    parser = argparse.ArgumentParser(description="MBS Agent CLI")
    parser.add_argument("task", nargs="?", help="Task for the agent to perform")
    parser.add_argument("--url", default="http://127.0.0.1:3030", help="MBS server URL")
    parser.add_argument("--key", default=None, help="API key (optional)")
    parser.add_argument("--list-tools", action="store_true", help="List available MCP tools")
    parser.add_argument("--max-steps", type=int, default=10, help="Max agent steps")
    parser.add_argument("--model", default=None, help="Model to use")
    args = parser.parse_args()

    async with MbsClient(base_url=args.url, api_key=args.key) as client:
        if args.list_tools:
            tools_resp = await client.list_tools()
            print(f"Available MCP tools ({len(tools_resp.tools)}):\n")
            for tool in tools_resp.tools:
                params = json.dumps(tool.parameters, indent=2) if tool.parameters else "none"
                print(f"  {tool.name}")
                print(f"    {tool.description}")
                print(f"    Parameters: {params}\n")
            return

        if not args.task:
            parser.print_help()
            return

        print(f"Agent task: {args.task}")
        print(f"Max steps: {args.max_steps}\n")
        print("=" * 60)

        req = AgentRunRequest(
            task=args.task,
            max_steps=args.max_steps,
            model=args.model,
        )

        resp = await client.run_agent(req)

        print(f"\nStatus:   {resp.status}")
        print(f"Steps:    {resp.steps_taken}")
        print(f"Duration: {resp.duration_ms / 1000:.1f}s" if hasattr(resp, 'duration_ms') and resp.duration_ms else "")

        if resp.tool_calls:
            print(f"\nTool calls ({len(resp.tool_calls)}):")
            for tc in resp.tool_calls:
                name = tc.get("name", "unknown") if isinstance(tc, dict) else str(tc)
                print(f"  - {name}")

        print(f"\n{'=' * 60}")
        print(f"Result:\n{resp.result}")


if __name__ == "__main__":
    asyncio.run(main())
