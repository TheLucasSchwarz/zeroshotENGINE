import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from .base import classification_step, ensure_numeric, generate_prompt, get_prompt_id


@dataclass
class CombinedPrediction:
    pred1: any
    pred2: any
    method: str


def set_zeroshot_parameters(
    model_family: str = "openai",
    client: any = None,
    model: str = "gpt-4o-mini",
    prompt_build: pd.DataFrame = None,
    prompt_ids_list: list[str] = None,
    prompt_id_col: str = "Prompt-ID",
    prompt_block_cols: list[str] = None,
    valid_keys: list[str] = None,
    label_codes: dict[str, any] = None,
    stop_conditions: dict[int, dict[str, any]] = None,
    output_types: dict[str, str] = None,
    double_shot: bool = False,
    combining_strategies: dict[str, str] = None,
    validate: bool = False,
    max_retries: int = 1,
    feedback: bool = False,
    print_prompts: bool = False,
    debug: bool = False,
    temperature: float = None,
) -> dict[str, any]:
    """
    Creates and validates parameter dictionary for iterative double zero-shot classification.

    This function builds a parameter dictionary with all needed settings for the
    iterative_double_zeroshot_classification function, performs validation on the inputs,
    and applies sensible defaults where appropriate.

    Args:
        model_family (str, optional): The family of the model. This determines how the response is handled.
            - For OpenAI/OpenRouter: "openai", "openrouter", "custom".
            - For standard Ollama models: "ollama_llm", "llama", "phi", "gemma", "mistral", "qwen".
            - For Ollama reasoning models: "ollama_reasoning_llm", "deepseek".
            Defaults to "openai".
        client (any, optional): The initialized client object for model interaction. Defaults to None.
        model (str, optional): The specific model to use. Defaults to "gpt-4o-mini".
        prompt_build (pd.DataFrame, optional): DataFrame containing the prompt components. Required.
        prompt_ids_list (list[str], optional): list of prompt IDs to use for classification. Required.
        prompt_id_col (str, optional): Column name in prompt_build containing the prompt IDs. Defaults to "Prompt-ID".
        prompt_block_cols (list[str], optional): Column names in prompt_build containing prompt blocks. Required.
        valid_keys (list[str], optional): list of valid classification keys/labels that will be processed sequentially for each text. The order of this list determines the processing sequence - each key is processed in the order it appears in this list. For example, if valid_keys=["political", "attack", "target"], the system will first classify the text for "political" content, then for "attack" content, and finally identify any "target". This sequential processing allows for conditional logic through stop_conditions - if an earlier classification step returns a specific value (e.g., "political" is absent), later steps (like "attack" or "target") can be skipped to improve efficiency. Each key in this list should correspond to a classification output in your prompt templates. Required.
        label_codes (dict[str, any], optional): dictionary mapping label names to their code values.
            Defaults to {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}.
        stop_conditions (dict[int, dict[str, any]], optional): Controls conditional classification flow.
            Each key represents a step index (0-based) where conditions should be checked.
            The value is a dictionary with "condition" (threshold value) and "blocked_keys" (list of keys to skip).
            If the classification result for the key at the given step is less than or equal to the condition value,
            subsequent classifications for all keys in "blocked_keys" will be skipped, saving processing time.
            For example, {0: {"condition": 0, "blocked_keys": ["attack", "target"]}} means that if the first key
            (e.g., "political") returns 0 (absent), then "attack" and "target" keys won't be classified.
            Defaults to {}.
        output_types (dict[str, str], optional): Output type for each key ('numeric' or 'list').
            Defaults to all keys as 'numeric' except 'target' as 'list' if present.
        double_shot (bool, optional): Execute each classification step twice and receive two predictions for each step. These two predictions can be combined and validated. Defaults to False.
        combining_strategies (dict[str, str], optional): Strategies for combining predictions.
            Defaults to {"numeric": "optimistic", "list": "intersection"}.
        validate (bool, optional): Whether to validate and combine multiple predictions. Defaults to False.
        max_retries (int, optional): Maximum retries for failed classification steps. Defaults to 1.
        feedback (bool, optional): Whether to print classification progress. Defaults to False.
        print_prompts (bool, optional): Whether to print the generated prompts. Defaults to False.
        debug (bool, optional): Whether to print raw model responses. Defaults to False.
        temperature (float, optional): Temperature parameter for model sampling. When None, uses model defaults.
            Set to 0 for deterministic output, higher values (e.g., 0.7-1.0) for more randomness.

    Returns:
        dict[str, any]: Complete parameter dictionary for classification.

    Raises:
        ValueError: If required parameters are missing or invalid.
        TypeError: If parameters are of incorrect types.

    ### Example Prompt Structure:

    The `prompt_build` DataFrame should have columns for prompt IDs and prompt blocks.
    The variable `text` is automatically available through the "text"-parameter of iterative_(double)_zeroshot_classification() functions.
    Here's an example of how the DataFrame might be structured:

    | Prompt-ID     | Block_A_Introduction  | Block_B_History | Block_C_Task | Block_D_Structure | Block_E_Output |
    |---------------|-------------------------------------------|-------------------------------------------|-----------------------------------|--------------------------------------|--------------------------------------|
    | P1_political | You are an expert in political communication and your task is to classify a text. | empty | Determine whether  the following text is political: {text} | Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political. | Do not hallucinate and do not provide any explanation for your decision. |
    | P2_presentation | You are an expert in political communication and your task is to classify a text. | The text we will show you was already classified as political in a previous classification task. | Determine whether the following text contains a political presentation: {text} | Mark the identified category presentation in a dictionary with key "presentation" and value "1" if you identify any sentence or hashtag in the text as presentation, and value "0" if you cannot identify any presentation within the text. | Do not hallucinate and do not provide any explanation for  your decision. |
    | P3_attack | You are an expert in political communication and your task is to classify a text. | The text we will show you was already classified as political in a previous classification task. | Determine whether the following text contains a political attack: {text} | Mark the identified category attack in a dictionary with key "attack" and value "1" if you identify any sentence or hashtag in the text as attack, and value "0" if you cannot identify any attack within the text. |
    | P4_target | You are an expert in political communication and your task is to classify a text. | The text we will show you was already classified as a political attack in a previous classification task. | Please identify the target or targets of the attack in the following text: {text} |Write all identified targets of this attack in a dictionary with key "target" and  a list as value "["target1", "target2", …]". If you cannot identify a target, give back an empty python list element. | Do not hallucinate and do not provide any explanation for  your decision. |

    In this example:
    - `prompt_id_col` would be "Prompt-ID"
    - `prompt_block_cols` would be ["Block_A_Introduction", "Block_B_History", "Block_C_Task", "Block_D_Structure", "Block_E_Output"]
    - `prompt_ids_list` might be ["P1_political", "P2_presentation", "P3_attack", "P4_target"]
    - `valid_keys` would be ["political", "presentation", "attack", "target"]
    - any block containing only the string "empty" (like Block_B_History in the P1_political row) will be excluded from the final prompt, allowing for flexible prompt construction with optional sections.
    - stop_condition must be defined as {
        0: {
            "condition": 0,
            "blocked_keys": [
                "presentation",
                "attack",
                "target",
            ],
        },
        2: {
            "condition": 0,
            "blocked_keys": ["target"],
        },
    }

    ### Using Context with f-strings:

    The prompt blocks support Python f-string syntax for dynamic content insertion.
    The variable `text` is automatically available, and additional context variables
    can be provided through the `context` parameter.

    Example of prompt blocks with context variables:

    | Prompt-ID     | Block_A_Introduction_with_Context | Block_B_Task | Block_C_Structure | Block_D_Output |
    |---------------|-------------------------------------------|-----------------------------------|--------------------------------------|--------------------------------------|
    | P1_political | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} from {author}.  | Determine whether  the following text is political: {text} | Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political. | Do not hallucinate and do not provide any explanation for  your decision. |

    To use a context variables, provide a context dictionary:

    ```python
    # When calling the classification function directly
    result = iterative_double_zeroshot_classification(
        text="Sample text to analyze",
        parameter=parameters,
        context={
            "lang": "English",
            "author": "John Doe",
            "platform": "Twitter",
            "date": "2023-05-15"
        }
    )

    # Or when using with DataFrame apply
    apply_iterative_double_zeroshot_classification(
        data=df,
        parameter=parameters,
        context=["lang", "author", "platform", "date"]  # These column names from df will be used.
        # They should be stored in the same row as the corresponding "text"-string.
    )
    ```
    The variable `text` has to be always available in one block of the prompt.

    The function will use these components to construct complete prompts for classification.

    Examples:
    >>> # Full hierarchical example with all parameters and specialized settings
    >>> parameters = set_zeroshot_parameters(
    ...     model_family="ollama_llm",
    ...     client=client,
    ...     model="phi4:latest",
    ...     prompt_build=prompts_df,
    ...     prompt_ids_list=["P1_political", "P2_presentation", "P3_attack", "P4_target"],
    ...     prompt_id_col="Prompt-ID",
    ...     prompt_block_cols=["Block_A_Introduction", "Block_B_History", "Block_C_Task", "Block_D_Structure", "Block_E_Output"]
    ...     valid_keys=["political", "presentation", "attack", "target"],
    ...     label_codes={"present": 1, "absent": 0, "non-coded": 8, "empty-list": []},
    ...     stop_conditions=stop_condition,
    ...     output_types={
    ...         "political": "numeric",
    ...         "presentation": "numeric",
    ...         "attack": "numeric",
    ...         "target": "list",
    ...     },
            validate=True,
    ...     combining_strategies={
    ...         "numeric": "optimistic",
    ...         "list": "union",
    ...     },
    ...     max_retries=2,
    ...     feedback=True,
    ...     temperature=0.0,
    ... )
    >>>

    >>> # With context variables
    >>> result = iterative_double_zeroshot_classification(
    ...     text="Sample text to analyze",
    ...     parameter=parameters_with_context_prompts,
    ...     context={
    ...         "lang": "English",
    ...         "author": "John Doe",
    ...         "platform": "Twitter",
    ...         "date": "2023-05-15"
    ...     }
    ... )
    """
    # Check required parameters
    if prompt_build is None:
        raise ValueError("prompt_build is required")

    if prompt_ids_list is None or not prompt_ids_list:
        raise ValueError("prompt_ids_list is required and cannot be empty")

    if prompt_block_cols is None or not prompt_block_cols:
        raise ValueError("prompt_block_cols is required and cannot be empty")

    if valid_keys is None or not valid_keys:
        raise ValueError("valid_keys is required and cannot be empty")

    # Set default label codes if not provided
    if label_codes is None:
        label_codes = {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}

    # Set default stop conditions if not provided
    if stop_conditions is None:
        stop_conditions = {}

    # Validate label codes
    if not isinstance(label_codes, dict):
        raise TypeError("label_codes must be a dictionary")

    required_codes = ["present", "absent", "non-coded"]
    if "empty-list" not in label_codes and any(key == "target" for key in valid_keys):
        raise ValueError(
            "label_codes must contain 'empty-list' when 'target' is in valid_keys"
        )

    for code in required_codes:
        if code not in label_codes:
            raise ValueError(f"label_codes must contain '{code}'")

    # Set default output types if not provided
    if output_types is None:
        output_types = {key: "numeric" for key in valid_keys}
        # Set target to list type if it's in valid_keys
        if "target" in valid_keys:
            output_types["target"] = "list"

    # Validate output types
    for key, type_val in output_types.items():
        if key not in valid_keys:
            raise ValueError(
                f"Output type specified for '{key}', but not in valid_keys"
            )
        if type_val not in ["numeric", "list"]:
            raise ValueError(
                f"Output type for '{key}' must be 'numeric' or 'list', got '{type_val}'"
            )

    # Set default combining strategies if not provided
    if combining_strategies is None:
        combining_strategies = {"numeric": "optimistic", "list": "union"}

    if double_shot is None:
        double_shot = False

    if validate is None:
        validate = False

    # Validate combining strategies
    valid_numeric_strategies = ["conservative", "optimistic", "probabilistic"]
    valid_list_strategies = ["union", "intersection", "first", "second"]

    if "numeric" not in combining_strategies:
        raise ValueError("combining_strategies must contain 'numeric'")
    if combining_strategies["numeric"] not in valid_numeric_strategies:
        raise ValueError(f"Numeric strategy must be one of {valid_numeric_strategies}")

    if "list" not in combining_strategies:
        raise ValueError("combining_strategies must contain 'list'")
    if combining_strategies["list"] not in valid_list_strategies:
        raise ValueError(f"list strategy must be one of {valid_list_strategies}")

    # Construct and return the parameter dictionary
    parameters = {
        "model_family": model_family,
        "client": client,
        "model": model,
        "prompt_build": prompt_build,
        "prompt_ids_list": prompt_ids_list,
        "prompt_id_col": prompt_id_col,
        "prompt_block_cols": prompt_block_cols,
        "valid_keys": valid_keys,
        "label_codes": label_codes,
        "stop_conditions": stop_conditions,
        "max_retries": max_retries,
        "feedback": feedback,
        "print_prompts": print_prompts,
        "debug": debug,
        "double_shot": double_shot,
        "validate": validate,
        "output_types": output_types,
        "combining_strategies": combining_strategies,
        "temperature": temperature,
    }

    return parameters


def single_iterative_zeroshot_classification(
    prompt_build: pd.DataFrame,
    prompt_ids_list: list[str],
    prompt_id_col: str,
    prompt_block_cols: list[str],
    valid_keys: list[str],
    label_codes: dict[str, any],
    text: str,
    stop_conditions: dict[str, any],
    max_retries: int = 2,
    feedback: bool = False,
    print_prompts: bool = False,
    model: str = "gpt-4o-mini",
    model_family: str = "openai",
    client: any = None,
    context: dict[str, any] = None,
    temperature: int = None,
    debug: bool = False,
) -> dict:
    """
    Classifies a single given text in multiple steps based on the provided parameters and returns the resulting classifications.

    Args:
        prompt_build (pd.DataFrame): The DataFrame containing the prompt parts.
        prompt_ids_list (list[str]): list of specific prompt IDs to classify.
        prompt_id_col (str): The prompt ID column name.
        prompt_block_cols (list[str]): Names of the prompt block columns.
        valid_keys (list[str]): list of valid keys to check in the response.
        label_codes (dict[str, any]): dictionary containing label codes.
        text (str): The text to classify.
        stop_conditions (dict[str, any]): dictionary defining the stop conditions.
        max_retries (int, optional): Maximum number of retries for classification steps. Defaults to 2.
        feedback (bool, optional): Whether to print messages during classification. Defaults to False.
        print_prompts (bool, optional): Whether to print prompts (only if feedback is True). Defaults to False.
        debug (bool): Whether to print raw model responses for debugging.
        model (str, optional): Model to use for classification. Defaults to "gpt-4o-mini".
        model_family (str, optional): Family of models used. Defaults to "openai".
        client (any, optional): The client object to interact with the API.
        context (Optional[dict[str, any]], optional): Additional context for the prompt.
        temperature (Optional[int], optional): Optional temperature parameter.


    Returns:
        dict: A dictionary containing the step-by-step classifications.
    """
    print_messages = print if feedback else lambda *args, **kwargs: None
    classifications = {}
    non_coded = label_codes["non-coded"]
    empty_list = label_codes["empty-list"]
    skip_indices = set()

    print_messages(text)

    for i, current_prompt_id in enumerate(prompt_ids_list):
        if i in skip_indices:
            key_to_skip = valid_keys[i]
            classifications[key_to_skip] = (
                empty_list if key_to_skip == "target" else non_coded
            )
            continue

        prompt_id_row = get_prompt_id(
            prompt_build,
            prompt_id_col,
            current_prompt_id,
        )
        prompt = generate_prompt(
            prompt_id_row, prompt_id_col, prompt_block_cols, text, context
        )
        if print_prompts:
            print_messages(prompt)
        current_key = valid_keys[i]
        classification = classification_step(
            model,
            model_family,
            client,
            prompt,
            valid_keys,
            current_key,
            print_messages,
            max_retries,
            temperature,
            debug=debug,
        )

        if isinstance(classification, dict):
            if current_key in classification and not isinstance(
                classification[current_key], list
            ):
                classification[current_key] = ensure_numeric(
                    classification[current_key]
                )

            classifications.update(classification)

        print_messages(
            f"Step {i}: Classified '{valid_keys[i]}' as {classification.get(valid_keys[i], 'unknown')}."
        )

        if i in stop_conditions:
            stop_condition = stop_conditions[i]
            condition_value = stop_condition["condition"]
            stop_keys = stop_condition["blocked_keys"]
            current_stop_key = valid_keys[i]

            if classification.get(current_stop_key) == condition_value:
                print_messages(
                    f"Blocking {stop_keys} due to stop condition at step {i} ({valid_keys[i]})."
                )
                for stopped_key in stop_keys:
                    classifications[stopped_key] = (
                        empty_list if stopped_key == "target" else non_coded
                    )
                skip_these = [valid_keys.index(k) for k in stop_keys if k in valid_keys]
                skip_indices.update(skip_these)

    if print_messages:
        print_messages(classifications)
    return classifications


def iterative_zeroshot_classification(
    parameter: dict,
    text: str,
    context: dict[str, any] = None,
    double_shot: bool = False,
    validate: bool = None,
    strategy: str = None,
) -> pd.Series:
    """
    Performs a single or double classification on the given text using an iterative zero-shot approach and optionally validates the results.

    Args:
        parameter (dict): dictionary of parameters for classification (see set_parameters function).
        text (str): The text to classify.
        context (Optional[dict[str, any]], optional): Additional context to include in the prompt. Defaults to None.
        double_shot (bool, optional): Execute each classification step twice and receive two predictions for each step. These two predictions can be combined and validated. Defaults to False.
        validate (bool, optional): Whether to validate the classification results. Defaults to False.
        strategy (Optional[str], optional): Strategy for validation ("conservative", "optimistic", or "probabilistic").
            Defaults to the value in parameter or "conservative".

    Returns:
        pd.Series: A Pandas Series containing combined predictions and (if validated) final results.
    """
    valid_keys = parameter["valid_keys"]
    validate = parameter["validate"]
    double_shot = parameter["double_shot"]

    # Add a new parameter to specify output types
    output_types = parameter.get("output_types", {})
    # Default all keys to "numeric" unless specified
    for key in parameter["valid_keys"]:
        if key not in output_types:
            output_types[key] = "numeric"

    # Add combining strategies for different output types
    combining_strategies = parameter.get("combining_strategies", {})
    # Default strategies
    default_combining_strategies = {
        "numeric": strategy or "conservative",
        "list": "union",
    }
    for type_name, default_strategy in default_combining_strategies.items():
        if type_name not in combining_strategies:
            combining_strategies[type_name] = default_strategy

    strategy = (
        strategy if strategy is not None else parameter.get("strategy", "conservative")
    )
    stop_condition = parameter.get("stop_conditions", {})

    # Create a clean copy of parameters for the single classification
    request_params = {"text": text, **parameter}

    # Remove parameters that_single_iterative_zeroshot_classification doesn't accept
    request_params.pop("strategy", None)
    request_params.pop("validate", None)
    request_params.pop("output_types", None)
    request_params.pop("combining_strategies", None)
    request_params.pop("double_shot", None)
    request_params.pop("promptIdsList", None)

    if context is None:
        context = {}
    request_params["context"] = context

    if double_shot:
        prediction1 = single_iterative_zeroshot_classification(**request_params)
        prediction2 = single_iterative_zeroshot_classification(**request_params)

        for key in valid_keys:
            if key not in prediction1:
                prediction1[key] = None
            if key not in prediction2:
                prediction2[key] = None

        prediction1_series = pd.Series(prediction1).rename(lambda x: f"{x}_pred1")
        prediction2_series = pd.Series(prediction2).rename(lambda x: f"{x}_pred2")
        combined_prediction_series = pd.concat([prediction1_series, prediction2_series])

        valid_numeric_values = {
            parameter["label_codes"]["present"],
            parameter["label_codes"]["absent"],
        }

        if validate:
            print("in validation")
            final_predictions = {}
            chosen_methods = {}
            validation_conflict = 0
            combined_prediction_series.replace(
                parameter["label_codes"]["non-coded"],
                parameter["label_codes"]["absent"],
                inplace=True,
            )

            # We check for every key, if the two predictions are identical
            for key in valid_keys:
                raw_pred1 = combined_prediction_series[f"{key}_pred1"]
                raw_pred2 = combined_prediction_series[f"{key}_pred2"]

                if output_types[key] == "list":
                    # Handle list-based outputs
                    if isinstance(raw_pred1, list) and isinstance(raw_pred2, list):
                        if combining_strategies["list"] == "union":
                            # Combine unique items from both lists
                            final_predictions[key] = list(set(raw_pred1 + raw_pred2))
                            chosen_methods[key] = "union_lists"
                        elif combining_strategies["list"] == "intersection":
                            # Keep only items in both lists
                            final_predictions[key] = list(set(raw_pred1) & set(raw_pred2))
                            chosen_methods[key] = "intersection_lists"
                        elif combining_strategies["list"] == "first":
                            final_predictions[key] = raw_pred1
                            chosen_methods[key] = "first_list"
                        elif combining_strategies["list"] == "second":
                            final_predictions[key] = raw_pred2
                            chosen_methods[key] = "second_list"
                        else:
                            # Default to union
                            final_predictions[key] = list(set(raw_pred1 + raw_pred2))
                            chosen_methods[key] = "default_union_lists"
                    elif isinstance(raw_pred1, list):
                        final_predictions[key] = raw_pred1
                        chosen_methods[key] = "only_first_list"
                    elif isinstance(raw_pred2, list):
                        final_predictions[key] = raw_pred2
                        chosen_methods[key] = "only_second_list"
                    else:
                        final_predictions[key] = parameter["label_codes"]["empty-list"]
                        chosen_methods[key] = "empty_list_default"

                else:
                    pred1 = (
                        raw_pred1[0]
                        if isinstance(raw_pred1, list) and raw_pred1
                        else raw_pred1
                    )
                    pred2 = (
                        raw_pred2[0]
                        if isinstance(raw_pred2, list) and raw_pred2
                        else raw_pred2
                    )

                    if (
                        pred1 == pred2
                        and not isinstance(pred1, list)
                        and not isinstance(pred2, list)
                        and pred1 in valid_numeric_values
                        and pred2 in valid_numeric_values
                    ):
                        final_predictions[key] = pred1
                        chosen_methods[key] = "identical"

                    # Second case: predictions differ but both are valid
                    elif (
                        pred1 != pred2
                        and not isinstance(pred1, list)
                        and not isinstance(pred2, list)
                        and pred1 in valid_numeric_values
                        and pred2 in valid_numeric_values
                    ):
                        validation_conflict = 1
                        # Use the specific numeric strategy from combining_strategies
                        numeric_strategy = combining_strategies["numeric"]

                        if numeric_strategy == "conservative":
                            final_predictions[key] = parameter["label_codes"]["absent"]
                            chosen_methods[key] = "conservative"
                        elif numeric_strategy == "optimistic":
                            final_predictions[key] = parameter["label_codes"]["present"]
                            chosen_methods[key] = "optimistic"
                        elif numeric_strategy == "probabilistic":
                            final_predictions[key] = random.choice([pred1, pred2])
                            chosen_methods[key] = "probabilistic"
                        else:
                            raise ValueError(f"Unknown strategy: {numeric_strategy}")

                    # Third case: only pred1 is valid
                    elif (
                        not isinstance(pred1, list)
                        and pred1 in valid_numeric_values
                        and (isinstance(pred2, list) or pred2 not in valid_numeric_values)
                    ):
                        final_predictions[key] = pred1
                        chosen_methods[key] = "pred1_only"

                    # Fourth case: only pred2 is valid
                    elif (
                        (isinstance(pred1, list) or pred1 not in valid_numeric_values)
                        and not isinstance(pred2, list)
                        and pred2 in valid_numeric_values
                    ):
                        final_predictions[key] = pred2
                        chosen_methods[key] = "pred2_only"

                    # Default case: neither prediction is valid
                    else:
                        final_predictions[key] = parameter["label_codes"]["absent"]
                        chosen_methods[key] = "default"

            for key, condition in stop_condition.items():
                if final_predictions.get(key) == condition["condition"]:
                    for blocked_key in condition["blocked_keys"]:
                        final_predictions[blocked_key] = parameter["label_codes"]["absent"]

            final_predictions_series = pd.Series(final_predictions)
            chosen_methods_series = pd.Series(chosen_methods).rename(
                lambda x: f"{x}_method"
            )
            validation_conflict_series = pd.Series(
                {"validation_conflict": validation_conflict}
            )

            return pd.concat(
                [
                    combined_prediction_series,
                    final_predictions_series,
                    chosen_methods_series,
                    validation_conflict_series,
                ]
            )
        return combined_prediction_series
    else:
        prediction = single_iterative_zeroshot_classification(**request_params)
        for key in valid_keys:
            if key not in prediction:
                prediction[key] = None

        combined_prediction_series = pd.Series(prediction)
    return combined_prediction_series

def apply_iterative_zeroshot_classification(
    data: pd.DataFrame,
    parameter: dict,
    context: list[str] = None,
) -> pd.DataFrame:
    """
    Applies the iterative zero-shot classifier to each row of the input DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the text and additional columns.
        parameter (dict): dictionary of parameters required by the classifier.
        context (Optional[list[str]], optional): list of column names to include in classification context.
            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame with the original data and the classification results appended.
    """

    def process_row(row) -> pd.DataFrame:
        text = row["text"]
        context_dict = (
            {col: row[col] for col in context if col in row} if context else {}
        )
        combined_prediction_series = iterative_zeroshot_classification(
            parameter=parameter, text=text, context=context_dict
        )
        return pd.DataFrame([combined_prediction_series])

    results = data.apply(process_row, axis=1)
    return pd.concat(
        [data.reset_index(drop=True), pd.concat(results.values, ignore_index=True)],
        axis=1,
    )


def parallel_iterative_zeroshot_classification(
    data: pd.DataFrame,
    parameter: dict,
    context: list[str] = None,
    num_workers: int = 4,
) -> pd.DataFrame:
    """
    Applies the classifier across data chunks using multi-threading for improved performance.

    Args:
        data (pd.DataFrame): The input DataFrame to classify.
        parameter (dict): dictionary of parameters for classification.
        context (Optional[list[str]], optional): list of column names to be used for context.
            Defaults to None.
        num_workers (int, optional): Number of worker threads. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame with classification results combined from all threads.
    """

    def apply_classifier_wrapper(
        data_chunk: pd.DataFrame, parameter: dict, context: list[str] = None
    ) -> pd.DataFrame:
        return apply_iterative_zeroshot_classification(
            data_chunk, parameter, context
        )

    apply_classifier_wrapper_partial = partial(
        apply_classifier_wrapper, parameter=parameter, context=context
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            executor.map(
                apply_classifier_wrapper_partial,
                [
                    data_chunk
                    for _, data_chunk in data.groupby(
                        np.arange(len(data)) // num_workers
                    )
                ],
            )
        )
    return pd.concat(results)
