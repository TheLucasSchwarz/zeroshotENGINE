"""Demo module for the zeroshot_engine package."""

import os
import time

# Use these imports (absolute imports are preferred in a package)
from zeroshot_engine.functions.izsc import (
    set_zeroshot_parameters,
    iterative_zeroshot_classification,
)

from zeroshot_engine.functions.utils import (
    get_demo_prompt_structure,
    get_demo_stop_conditions,
    setup_demo_model,
    get_demo_text_selection,
    ask_to_display_label_structure,
)

# Set environment variables for GPU acceleration if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_CUDA"] = "1"


def set_classification_parameters(
    model_family, client, model_name, prompts_df, stop_condition
):
    """Set up the classification parameters for both naive and with definition approaches."""

    valid_keys = ["political", "presentation", "attack", "target"]
    label_values = {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}

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
        valid_keys=valid_keys,
        label_codes=label_values,
        stop_conditions=stop_condition,
        output_types={
            "political": "numeric",
            "presentation": "numeric",
            "attack": "numeric",
            "target": "list",
        },
        double_shot=True,
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
        valid_keys=valid_keys,
        label_codes=label_values,
        stop_conditions=stop_condition,
        output_types={
            "political": "numeric",
            "presentation": "numeric",
            "attack": "numeric",
            "target": "list",
        },
        validate=True,
        double_shot=True,
        combining_strategies={
            "numeric": "optimistic",
            "list": "union",
        },
        max_retries=2,
        feedback=False,
        print_prompts=False,
        debug=False,
    )

    return (parameters_naive, parameters_with_definitions, valid_keys, label_values)


def run_classification(text, parameters, context, description):
    """Run the classification and time it."""
    print(f"\n\n🧮 {description} started...")
    start_time = time.time()

    result = iterative_zeroshot_classification(
        text=text,
        parameter=parameters,
        context=context,
    )

    # Calculate and display execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"⏱️  {description} completed in {elapsed_time:.2f} seconds\n")

    return result


def display_combined_results(result, label_values):
    """
    Display a reorganized view of the classification results.

    The output shows:
      - Final combined predictions for each label along with their corresponding codes.
      - The individual predictions from two runs (with the same prompt setup).
      - The methods used to combine these predictions.
      - Total number of validation conflicts.

    Args:
        result (pandas.Series): The classification result.
        label_values (dict): Mapping of label codes, e.g. {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}
    """
    # Define labels to display
    final_labels = ["political", "presentation", "attack", "target"]

    # Create an inverted mapping for non-list label codes
    code_to_label = {}
    for key, value in label_values.items():
        if not isinstance(value, list):
            code_to_label[value] = key
        else:
            # Assume list values are represented as a string for display purposes
            code_to_label[str(value)] = key

    print("\n📊 Final Classification Codes:")
    print("-------------------------")
    for label in final_labels:
        final_val = result.get(label, "N/A")
        if final_val != "N/A":
            # For lists compare via string conversion
            if isinstance(final_val, list):
                meaning = code_to_label.get(str(final_val), "")
            else:
                meaning = code_to_label.get(final_val, "")
            if meaning:
                print(f"{label:25}: {final_val} ({meaning})")
            else:
                print(f"{label:25}: {final_val}")
        else:
            print(f"{label:25}: {final_val}")

    print(
        "\nCombined from individual predictions (two runs with the same prompt setup):"
    )
    print("-------------------------")
    for label in final_labels:
        pred1 = result.get(label + "_pred1", "N/A")
        pred2 = result.get(label + "_pred2", "N/A")
        if pred1 != "N/A":
            if isinstance(pred1, list):
                meaning1 = code_to_label.get(str(pred1), "")
            else:
                meaning1 = code_to_label.get(pred1, "")
        else:
            meaning1 = ""
        if pred2 != "N/A":
            if isinstance(pred2, list):
                meaning2 = code_to_label.get(str(pred2), "")
            else:
                meaning2 = code_to_label.get(pred2, "")
        else:
            meaning2 = ""
        if meaning1:
            print(f"{label + '_pred1':25}: {pred1} ({meaning1})")
        else:
            print(f"{label + '_pred1':25}: {pred1}")
        if meaning2:
            print(f"{label + '_pred2':25}: {pred2} ({meaning2})")
        else:
            print(f"{label + '_pred2':25}: {pred2}")

    print("\nBy these methods:")
    print("-------------------------")
    for label in final_labels:
        method_val = result.get(label + "_method", "N/A")
        print(f"{label + '_method':25}: {method_val}")

    print("\nValidation Conflicts:")
    print("-------------------------")
    conflict = result.get("validation_conflict", "N/A")
    print(f"{'validation_conflict':25}: {conflict}")


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
    parameters_naive, parameters_with_definitions, labels, label_values = (
        set_classification_parameters(
            model_family, client, model_name, prompts_df, stop_condition
        )
    )

    # Print the text to analyze
    print("\n\n📝 Text to analyze:")
    print("-------------------------")
    print(text)
    print("-------------------------")

    # Ask the user if they want to display the hierarchical structure
    ask_to_display_label_structure(labels, label_values, stop_condition)

    # Determine which classifications to run based on text source
    if not interactive:  # the quickdemo
        # Run both classification methods
        result_naive = run_classification(
            text,
            parameters_naive,
            context,
            "Classification (Naive without Definition)",
        )

        # Display results
        display_combined_results(result_naive, label_values)

    else:  # the interactive demo with user provided text
        result = run_classification(
            text,
            parameters_with_definitions,
            context,
            "Classification",
        )

        # Display results
        display_combined_results(result, label_values)

    print(
        "\n✅ Demo completed! To learn more about the package look into the documentation under https://github.com/TheLucasSchwarz/zeroshotENGINE."
    )
    return True
