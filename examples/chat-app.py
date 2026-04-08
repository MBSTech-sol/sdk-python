#!/usr/bin/env python3
"""
MBS SDK Example — Interactive Chat App

A simple terminal chatbot that connects to a local MBS Workbench server
and maintains a multi-turn conversation.

Usage:
    pip install mbs-python
    python chat-app.py [--url http://127.0.0.1:3030] [--key YOUR_KEY]
"""

import asyncio
import argparse
from mbs import MbsClient, ChatMessage


async def main():
    parser = argparse.ArgumentParser(description="MBS Chat App")
    parser.add_argument("--url", default="http://127.0.0.1:3030", help="MBS server URL")
    parser.add_argument("--key", default=None, help="API key (optional)")
    parser.add_argument("--model", default=None, help="Model to use")
    args = parser.parse_args()

    async with MbsClient(base_url=args.url, api_key=args.key) as client:
        # Check server is reachable
        models = await client.models()
        print(f"Connected to MBS server at {args.url}")
        print(f"Available models: {', '.join(m.id for m in models.data)}")
        print("Type 'quit' to exit, 'clear' to reset conversation.\n")

        history: list[ChatMessage] = [
            ChatMessage(role="system", content="You are a helpful assistant.")
        ]

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                history = [history[0]]  # Keep system prompt
                print("Conversation cleared.\n")
                continue

            history.append(ChatMessage(role="user", content=user_input))

            resp = await client.chat(
                messages=history,
                model=args.model,
                temperature=0.7,
                max_tokens=1024,
            )

            assistant_msg = resp.choices[0].message.content
            history.append(ChatMessage(role="assistant", content=assistant_msg))
            print(f"Assistant: {assistant_msg}\n")


if __name__ == "__main__":
    asyncio.run(main())
