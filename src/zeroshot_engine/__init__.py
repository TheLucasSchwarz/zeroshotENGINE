"""zeroshot_engine package."""

# Package version
__version__ = "0.2.0"

# Package metadata
__author__ = "Lucas Schwarz"
__email__ = "luc.schwarz@posteo.de"

# Import important functions for easy access
from zeroshot_engine.functions.izsc import (
    set_zeroshot_parameters,
    single_iterative_zeroshot_classification,
    iterative_zeroshot_classification,
    apply_iterative_zeroshot_classification,
    parallel_iterative_zeroshot_classification,
)

from zeroshot_engine.functions.base import (
    initialize_model,
    generate_prompt,
    get_prompt_id,
    request_to_model,
    classification_step,
    ensure_numeric,
)

from zeroshot_engine.functions.utils import (
    get_demo_prompt_structure,
    get_demo_stop_conditions,
    get_demo_text_selection,
    setup_demo_model,
    print_stop_conditions,
    parse_dependencies,
)

from zeroshot_engine.functions.ollama import (
    check_ollama_gpu_support,
    check_ollama_installation,
    check_system_requirements,
    install_ollama,
    update_ollama,
    start_ollama_service,
    test_model_compatibility,
    get_model_size_from_ollama,
    estimate_model_size,
    download_model_with_progress,
    check_ollama_updates,
    setup_ollama,
)

from zeroshot_engine.demo.demo_runner import run_demo_classification


from zeroshot_engine.functions.openai import setup_openai_api_key

from zeroshot_engine.functions.visualization import display_label_flowchart


from zeroshot_engine.cli import main
