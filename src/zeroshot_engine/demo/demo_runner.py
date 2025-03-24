"""Demo module for the zeroshot_engine package."""

import os
import time

# Use these imports (absolute imports are preferred in a package)
from zeroshot_engine.functions.idzsc import (
    iterative_double_zeroshot_classification,
    set_zeroshot_parameters,
)

from zeroshot_engine.functions.utils import (
    get_demo_prompt_structure,
    get_demo_stop_conditions,
    setup_demo_model,
    get_demo_text_selection,
)

# Set environment variables for GPU acceleration if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_CUDA"] = "1"


def set_classification_parameters(
    model_family, client, model_name, prompts_df, stop_condition
):
    """Set up the classification parameters for both naive and with definition approaches."""
    parameters_naive = set_zeroshot_parameters(
        model_family=model_family,
        client=client,
        model=model_name,
        prompt_build=prompts_df,
        prompt_ids_list=[
            "P1_political_naive",
            "P2_presentation_naive",
            "P3_attack_naive",
            "P4_target_naive",
        ],
        prompt_id_col="Prompt-ID",
        prompt_block_cols=[
            "Block_A_Introduction",
            "Block_B_History",
            "Block_C_Definition",
            "Block_D_Task",
            "Block_E_Structure",
            "Block_F_Output",
        ],
        valid_keys=["political", "presentation", "attack", "target"],
        label_codes={"present": 1, "absent": 0, "non-coded": 8, "empty-list": []},
        stop_conditions=stop_condition,
        output_types={
            "political": "numeric",
            "presentation": "numeric",
            "attack": "numeric",
            "target": "list",
        },
        validate=True,
        combining_strategies={
            "numeric": "optimistic",
            "list": "union",
        },
        max_retries=2,
        feedback=False,
        print_prompts=False,
        debug=False,
    )

    parameters_with_definitions = set_zeroshot_parameters(
        model_family=model_family,
        client=client,
        model=model_name,
        prompt_build=prompts_df,
        prompt_ids_list=[
            "P1_political_with_definition",
            "P2_presentation_with_definition",
            "P3_attack_with_definition",
            "P4_target_with_definition",
        ],
        prompt_id_col="Prompt-ID",
        prompt_block_cols=[
            "Block_A_Introduction",
            "Block_B_History",
            "Block_C_Definition",
            "Block_D_Task",
            "Block_E_Structure",
            "Block_F_Output",
        ],
        valid_keys=["political", "presentation", "attack", "target"],
        label_codes={"present": 1, "absent": 0, "non-coded": 8, "empty-list": []},
        stop_conditions=stop_condition,
        output_types={
            "political": "numeric",
            "presentation": "numeric",
            "attack": "numeric",
            "target": "list",
        },
        validate=True,
        combining_strategies={
            "numeric": "optimistic",
            "list": "union",
        },
        max_retries=2,
        feedback=False,
        print_prompts=False,
        debug=False,
    )

    return parameters_naive, parameters_with_definitions


def run_classification(text, parameters, context, description):
    """Run the classification and time it."""
    print(f"\n\n🧮 {description} started...")
    start_time = time.time()

    result = iterative_double_zeroshot_classification(
        text=text,
        parameter=parameters,
        context=context,
    )

    # Calculate and display execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  {description} completed in {elapsed_time:.2f} seconds\n")

    return result


def run_demo_classification(interactive=True):
    """
    Main function to run the demo classification.

    Args:
        interactive: If True, run in interactive mode with user prompts.
                    If False, run with default settings.
    """
    print("🚀 Starting zeroshot_engine Demo")

    # Get the prompt structure
    print("📋 Creating prompt structure...")
    prompts_df = get_demo_prompt_structure()

    # Get the stop conditions
    stop_condition = get_demo_stop_conditions()

    # Set up the model
    client, model_name, model_family = setup_demo_model(interactive)

    # Get text and context
    text, context = get_demo_text_selection(interactive)

    # Set classification parameters
    print("\n\n⚙️  Configuring classification parameters...")
    parameters_naive, parameters_with_definitions = set_classification_parameters(
        model_family, client, model_name, prompts_df, stop_condition
    )

    # Print the text to analyze
    print("\n\n📝 Text to analyze:")
    print("-------------------------")
    print(text)
    print("-------------------------")

    # Determine which classifications to run based on text source
    if not interactive:
        # Run both classification methods
        result_naive = run_classification(
            text,
            parameters_naive,
            context,
            "Classification (Naive without Definition)",
        )

        result_with_definitions = run_classification(
            text,
            parameters_with_definitions,
            context,
            "Classification (with Definitions)",
        )

        # Display results for both methods
        print("\n\n📊 Classification Results: (Naive without Definition)")
        print("-------------------------")
        print(result_naive)

        print("\n\n📊 Classification Results: (With Definition)")
        print("-------------------------")
        print(result_with_definitions)
    else:  # User-provided text
        # Only run with definitions
        result = run_classification(
            text,
            parameters_with_definitions,
            context,
            "Classification",
        )

        # Display results
        print("\n\n📊 Classification Results:")
        print("-------------------------")
        print(result)

    print("\n✅ Demo completed!")
    return True
