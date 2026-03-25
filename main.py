import argparse
import logging

from voice_app.config import load_config
from voice_app.history import save_history
from voice_app.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice LLM — Full Duplex with VAD. Talk to GPT-4o mini like a phone call."
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Clear conversation history and start a fresh session.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging (thread events, interim transcripts).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    # Keep third-party libraries quieter unless --debug.
    if not args.debug:
        for noisy in ("httpx", "httpcore", "openai", "websockets"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load_config(args.config)

    if args.new:
        save_history(config.history.file, [])
        print("Started a new conversation.")

    orchestrator = Orchestrator(config, debug=args.debug)
    orchestrator.run()


if __name__ == "__main__":
    main()
