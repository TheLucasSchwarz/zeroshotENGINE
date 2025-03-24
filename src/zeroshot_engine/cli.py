"""CLI interface for the zeroshot_engine package."""

import sys
from zeroshot_engine.demo.demo_runner import run_demo_classification
from zeroshot_engine.functions.visualization import display_label_flowchart
from zeroshot_engine.functions.utils import get_demo_stop_conditions


def show_help():
    """Show help information."""
    print("zeroshot_engine CLI")
    print("Available commands:")
    print("  demo      - Run the interactive demo")
    print("  quickdemo - Run a quick demo with default settings")
    print("  flowchart - Display the label dependency flowchart")
    print("  --help    - Show this help message")


def main():
    """Main entry point for the CLI."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo_classification(interactive=True)
            return
        elif sys.argv[1] == "quickdemo":
            run_demo_classification(interactive=False)
            return
        elif sys.argv[1] == "flowchart":
            # New command to display the label dependency flowchart
            valid_keys = ["political", "presentation", "attack", "target"]
            stop_conditions = get_demo_stop_conditions()
            label_codes = {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}
            display_label_flowchart(valid_keys, stop_conditions, label_codes)
            return
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_help()
            return

    # Default behavior if no arguments or unrecognized command
    show_help()


if __name__ == "__main__":
    main()
