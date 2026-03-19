#!/usr/bin/env python3
"""
Quasar - Radio Astronomy AI Assistant
Main entry point

Usage:
    python quasar.py cli          Launch interactive CLI
    python quasar.py test         Run system tests
    python quasar.py query "..."  Run a single query
"""

import sys
import os

# --- Python 3.13 'cgi' module shim for pyvo ---
import sys as _sys
if "cgi" not in _sys.modules:
    import types
    import email.message
    cgi = types.ModuleType("cgi")
    def parse_header(line):
        m = email.message.Message()
        m['content-type'] = line
        return m.get_content_type(), m.get_params() or {}
    cgi.parse_header = parse_header
    _sys.modules["cgi"] = cgi
# ---------------------------------------------

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "cli":
        from core.cli import QuasarCLI
        cli = QuasarCLI()
        cli.cmdloop()

    elif command == "test":
        # Run the test suite
        test_path = os.path.join(os.path.dirname(__file__), "tests", "test_quasar.py")
        if os.path.exists(test_path):
            import runpy
            runpy.run_path(test_path, run_name="__main__")
        else:
            print(f"Test file not found: {test_path}")
            sys.exit(1)

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python quasar.py query \"your question here\"")
            sys.exit(1)

        from dotenv import load_dotenv
        load_dotenv()

        query = " ".join(sys.argv[2:])
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
            sys.exit(1)

        from core.agent import QuasarAgent, AgentConfig
        config = AgentConfig(api_key=api_key)
        agent = QuasarAgent(config)
        result = agent.process_query(query)

        if isinstance(result, tuple):
            _data, response_text, _intent = result
        else:
            response_text = str(result)

        print(response_text)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
