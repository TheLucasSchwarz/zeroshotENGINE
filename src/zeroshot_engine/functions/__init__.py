"""Functions module for the zeroshot_engine package."""

# Export names without necessarily importing them immediately
__all__ = [
    "initialize_model",
    "iterative_double_zeroshot_classification",
    "set_zeroshot_parameters",
    "setup_openai_api_key",
    "display_label_flowchart",
    "get_demo_prompt_structure",
    "get_demo_stop_conditions",
    "setup_demo_model",
    "get_demo_text_selection",
    "parse_dependencies",
    "print_stop_conditions",
]

try:
    from .base import initialize_model
    from .izsc import (
        set_zeroshot_parameters,
        single_iterative_zeroshot_classification,
        iterative_zeroshot_classification,
        apply_iterative_zeroshot_classification,
        parallel_iterative_zeroshot_classification,
    )
    from .openai import setup_openai_api_key
    from .visualization import display_label_flowchart
    from .utils import (
        get_demo_prompt_structure,
        get_demo_stop_conditions,
        setup_demo_model,
        get_demo_text_selection,
        parse_dependencies,
        print_stop_conditions,
    )
except ImportError:
    pass
