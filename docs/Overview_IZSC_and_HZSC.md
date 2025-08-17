# Understanding Iterative and Hierarchical Double Zero-Shot Classification in zeroshotENGINE
This document provides an in-depth explanation of the Iterative Zero-Shot Classification (IZSC) and Hierarchical Zero-Shot Classification (HZSC) processes as implemented in the zeroshotENGINE, focusing on the iterative_zeroshot_classification function.

## Overview of IZSC and HZSC
Both IDZSC and HDZSC are classification approaches designed to leverage the power of large language models (LLMs) for complex classification tasks (multi-label, multi-class, single-class) without requiring task-specific training data. While IDZSC provides a robust foundation for sequential multi-step classification, HDZSC extends this framework by incorporating hierarchical relationships between categories to improve accuracy and efficiency.

## Theoretical Background
These zero-shot classification approaches were developed to address the challenges of classification in contexts requiring high reliability and methodological rigor. The approaches integrate best practices from previous research to maximize the effectiveness of zero-shot classification with LLMs for scientific purposes.

The underlying framework was initially developed in research on candidates' negative campaigning (Schwarz, 2025), with HDZSC specifically designed to handle hierarchical multi-label data structures where certain classifications depend on previous ones. Both approaches have already undergone promising evaluation and validation in academic research settings.

IZSC provides a flexible architecture for sequential classification tasks, while HZSC adds the capability to model explicit dependencies between classification categories, allowing for more efficient and accurate analysis of complex hierarchical classification schemes.

### Key Features and Methodology

* **Step-by-Step Analysis**: Both IDZSC and HDZSC operate through a multi-prompt approach rather than relying on a single comprehensive prompt. Each prompt focuses on specific criteria for classification.

* **Label Structure**: The system supports both sequential (IZSC) and hierarchical (HZSC) classification structures. HZSC takes into account explicit dependencies between categories.Later classification steps may depend on the results of earlier ones. IZSC allows for more general sequential multi-step classification.

* **Model Flexibility**: Any generative LLM can be integrated, whether self-hosted or accessed via an API.

* **Sequential Processing**: Classification steps can be executed either sequentially (IZSC) or following a hierarchical dependency structure (HZSC). Both approaches are fully supported by the system.

* **Customizable Prompt Building Blocks**: The system allows for prompt customization through building blocks, enabling easy iterative refinement to optimize performance for each category.

* **Validation Options**: The system offers you to do every classification twice (double shot) leveraging the probabilistic power of LLMs. Alternatively, use the single shot with optional temperature control for determnistic findings. Users can choose either method depending on their research requirements.

### Validation Approaches

Both IZSC and HZSC offer flexible validation strategies through temperature control when interacting with the language model:

* **Deterministic Validation (Zero Temperature):**
  * Setting temperature to 0 produces deterministic results from the language model
  * Each query with the same prompt and text will yield identical responses
  * In this scenario each prompt is send for each unit of analysis once
  * This approach provides consistency but may not capture model uncertainty
  * Useful for reproducible research and when exact replicability is required
  
* **Probabilistic Validation (Non-Zero Temperature):**
  * Using default or higher temperature settings (e.g., 0.7) introduces controlled randomness
  * This enables the "double shot validation" approach where the same text and prompt are processed twice
  * Consistency between the two responses indicates higher confidence in the classification
  * Differences between responses highlight uncertainty or ambiguity in the classification task
  
* **Comparison and Resolution:**
  * By comparing the results of two probabilistic runs, researchers can:
    * Identify high-confidence classifications (consistent across runs)
    * Detect ambiguous or borderline cases (different across runs)
    * Apply resolution strategies to handle discrepancies:
      * Conservative strategy: Only accept classifications consistent across runs
      * Optimistic strategy: Accept a positive classification if it appears in any run
      * Probabilistic strategy: Randomly select between conflicting classifications
      * Human verification: Flag inconsistent results for manual review (following Heseltine & Clemm v. Hohenburg, 2024)

This dual approach to validation allows researchers to balance between deterministic reproducibility and capturing model uncertainty, making HZSC adaptable to different research requirements and classification tasks.


## The `iterative_zeroshot_classification` Function

This function is the core of the IZSC and HZSC implementation in `zeroshot_engine`. It takes a text input and a set of parameters to perform iterative zero-shot classification using a multi-prompting approach.

### Function Signature

```python
iterative__zeroshot_classification(text, parameter, context)
```

```python
parallel_iterative_zeroshot_classification(text, parameter, context, num_workers)
```

### Parameters

*   **`text` (str):** The input text to be classified. This is the primary content that the model will analyze.
*   **`parameter` (dict):** A dictionary containing the configuration parameters for the classification task. This includes:
    *   `model_family` (str): Specifies the family of the language model being used (e.g., "llama").
    *   `client` (object): An instance of the client used to interact with the language model (e.g., `OllamaClient`).
    *   `model` (str): The name or identifier of the specific language model being used (e.g., "llama2:13b").
    *   `prompt_build` (pd.DataFrame): A DataFrame containing the prompt templates and instructions.
    *   `prompt_ids_list` (list): A list of prompt IDs from `prompt_build` to be used in the classification process.
    *   `prompt_id_col` (str): The name of the column in `prompt_build` that contains the prompt IDs.
    *   `prompt_block_cols` (list): A list of column names in `prompt_build` that contain the different blocks of the prompt (e.g., introduction, definition, task).
    *   `valid_keys` (list): A list of valid keys or labels for the classification task (e.g., ["political", "presentation", "attack", "target"]).
    *   `label_codes` (dict): A dictionary mapping labels to numerical codes or other elements like strings or lists (e.g., `{"present": 1, "absent": 0}`).
    *   `stop_conditions` (dict): Conditions that determine when the iterative process should stop.
    *   `output_types` (dict): Specifies the expected output type for each label (e.g., `{"political": "numeric", "target": "list"}).
    *   `double_shot` (bool): Execute each classification step twice and receive two predictions for each step. These two predictions can be combined and validated. Defaults to False.
    *   `validate` (bool): Whether to validate the output against the expected types.
    *   `combining_strategies` (dict): Strategies for combining multiple predictions (e.g., `{"numeric": "optimistic", "list": "union"}).
    *   `max_retries` (int): The maximum number of retries for generating a prediction.
    *   `feedback` (bool): Whether to incorporate feedback into the iterative process.
    *   `print_prompts` (bool): Whether to print the prompts sent to the language model.
    *   `debug` (bool): Whether to enable debugging mode.
*   **`context` (str):** Additional context or information that can help the model make more accurate predictions. This could include background information, relevant facts, or specific instructions.

### Inner workings

The `iterative_zeroshot_classification` function orchestrates the following steps:

1.  **Prompt Building:**
    *   The function uses the `prompt_build` DataFrame and the specified `prompt_ids_list` to construct prompts.
    *   Each prompt consists of blocks defined in `prompt_block_cols` (e.g., "Block\_A\_Introduction", "Block\_C\_Definition").
    *   These blocks are filled with instructions, definitions, and task descriptions to guide the LLM.
    *   Every final prompt is than combined with the text and optional context of the current unit of analysis.

2.  **LLM Interaction:**
    *   The prompts are sent to the LLM via the specified `client` and `model`.
    *   The LLM generates predictions based on the input text and the provided prompts.

3.  **Output Parsing and Validation:**
    *   The function parses the LLM's output to extract the predicted labels and their corresponding values.
    *   It validates the output against the expected `output_types` (e.g., numeric, list) and uses `label_codes` to convert the predictions into a standardized format.

4.  **Combining Strategies:**
    *   If the classifier generates divergent predictions when processing a single unit task (using the “double classification per step” feature), the function combines these predictions using the specified `combining_strategies`.

