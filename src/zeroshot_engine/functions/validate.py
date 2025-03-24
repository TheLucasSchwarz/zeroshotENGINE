import pandas as pd


def validate_double_zeroshot_predictions(
    dataframe: pd.DataFrame, params: dict[str, any], recode_non_coded: bool
) -> pd.DataFrame:
    """
    Validate double zeroshot predictions for a given DataFrame.

    This function processes prediction columns in the input DataFrame using
    double zeroshot strategies ('conservative', 'optimistic', 'probabilistic').
    It first optionally updates "non-coded" labels to "absent". For each valid key,
    the function compares paired predictions from two columns and, based on the selected
    strategy and predefined label codes, determines the final prediction and method used.
    Additionally, it applies stop conditions defined in the parameters to adjust predictions.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame containing prediction columns.
        params (dict): A dictionary containing:
            - "valid_keys" (list): Keys to identify prediction columns.
            - "label_codes" (dict): Mappings for label values e.g., "non-coded", "absent", "present".
            - "stop_conditions" (dict): Conditions for adjusting final predictions.
        recode_non_coded (bool): Flag to indicate if "non-coded" labels should be recoded to "absent".

    Returns:
        pd.DataFrame: The updated DataFrame with new prediction columns for each strategy.
    """
    valid_keys = params["valid_keys"]
    label_codes = params["label_codes"]
    stop_conditions = params["stop_conditions"]

    # Replace all occurrences of "non-coded" with "absent"
    if recode_non_coded:
        dataframe.replace(label_codes["non-coded"], label_codes["absent"], inplace=True)

    final_predictions = {}
    chosen_methods = {}
    strategies = ["conservative", "optimistic", "probabilistic"]

    for strategy in strategies:
        for key in valid_keys:
            prediction1 = dataframe[f"{key}_pred1"]
            prediction2 = dataframe[f"{key}_pred2"]
            key_strategy = f"{key}_{strategy}"
            final_predictions[key_strategy] = []
            chosen_methods[key_strategy] = []
            for pred1, pred2 in zip(prediction1, prediction2):
                if pred1 == pred2:
                    final_predictions[key_strategy].append(pred1)
                    chosen_methods[key_strategy].append("identical")
                else:
                    valid_numeric_values = {
                        label_codes["absent"],
                        label_codes["present"],
                    }
                    if pred1 in valid_numeric_values and pred2 in valid_numeric_values:
                        if strategy == "conservative":
                            final_predictions[key_strategy].append(
                                label_codes["absent"]
                            )
                            chosen_methods[key_strategy].append("conservative")
                        elif strategy == "optimistic":
                            final_predictions[key_strategy].append(
                                label_codes["present"]
                            )
                            chosen_methods[key_strategy].append("optimistic")
                        elif strategy == "probabilistic":
                            import random

                            final_predictions[key_strategy].append(
                                random.choice([pred1, pred2])
                            )
                            chosen_methods[key_strategy].append("probabilistic")
                    elif (
                        pred1 in valid_numeric_values
                        and pred2 not in valid_numeric_values
                    ):
                        final_predictions[key_strategy].append(pred1)
                        chosen_methods[key_strategy].append("numeric_choice")
                    elif (
                        pred2 in valid_numeric_values
                        and pred1 not in valid_numeric_values
                    ):
                        final_predictions[key_strategy].append(pred2)
                        chosen_methods[key_strategy].append("numeric_choice")
                    else:
                        final_predictions[key_strategy].append(label_codes["absent"])
                        chosen_methods[key_strategy].append("string_fallback")

    for key, condition in stop_conditions.items():
        for strategy in strategies:
            key_strategy = f"{key}_{strategy}"
            if key_strategy in final_predictions:
                final_predictions[key_strategy] = [
                    label_codes["non-coded"] if pred == condition["condition"] else pred
                    for pred in final_predictions[key_strategy]
                ]

    for key, predictions in final_predictions.items():
        dataframe[key] = predictions

    return dataframe
