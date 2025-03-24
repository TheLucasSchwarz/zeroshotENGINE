# Understanding Iterative and Hierarchical Double Zero-Shot Classification in zeroshotENGINE
This document provides an in-depth explanation of the Iterative Double Zero-Shot Classification (IDZSC) and Hierarchical Double Zero-Shot Classification (HDZSC) processes as implemented in the zeroshotENGINE, focusing on the iterative_double_zeroshot_classification function.

## Overview of IDZSC and HDZSC
Both IDZSC and HDZSC are classification approaches designed to leverage the power of large language models (LLMs) for complex classification tasks (multi-label, multi-class, single-class) without requiring task-specific training data. While IDZSC provides a robust foundation for sequential multi-step classification, HDZSC extends this framework by incorporating hierarchical relationships between categories to improve accuracy and efficiency.

## Theoretical Background
These zero-shot classification approaches were developed to address the challenges of classification in contexts requiring high reliability and methodological rigor. The approaches integrate best practices from previous research to maximize the effectiveness of zero-shot classification with LLMs for scientific purposes.

The underlying framework was initially developed in research on candidates' negative campaigning (Schwarz, 2025), with HDZSC specifically designed to handle hierarchical multi-label data structures where certain classifications depend on previous ones. Both approaches have already undergone promising evaluation and validation in academic research settings.

IDZSC provides a flexible architecture for sequential classification tasks, while HDZSC adds the capability to model explicit dependencies between classification categories, allowing for more efficient and accurate analysis of complex hierarchical classification schemes.

### Key Features and Methodology

* **Step-by-Step Analysis**: Both IDZSC and HDZSC operate through a multi-prompt approach rather than relying on a single comprehensive prompt. Each prompt focuses on specific criteria for classification.

* **Label Structure**: The system supports both sequential (IDZSC) and hierarchical (HDZSC) classification structures. HDZSC takes into account explicit dependencies between categories.Later classification steps may depend on the results of earlier ones. IDZSC allows for more general sequential multi-step classification.

* **Model Flexibility**: Any generative LLM can be integrated, whether self-hosted or accessed via an API.

* **Sequential Processing**: Classification steps can be executed either sequentially (IDZSC) or following a hierarchical dependency structure (HDZSC). Both approaches are fully supported by the system.

* **Customizable Prompt Building Blocks**: The system allows for prompt customization through building blocks, enabling easy iterative refinement to optimize performance for each category.

* **Validation Options**: The system offers both deterministic (zero temperature) and probabilistic (double validation) approaches. Users can choose either method depending on their research requirements.

### Validation Approaches

Both IDZSC and HDZSC offer flexible validation strategies through temperature control when interacting with the language model:

* **Deterministic Validation (Zero Temperature):**
  * Setting temperature to 0 produces deterministic results from the language model
  * Each query with the same prompt and text will yield identical responses
  * In this scenario each prompt is send for each unit of analysis once
  * This approach provides consistency but may not capture model uncertainty
  * Useful for reproducible research and when exact replicability is required
  
* **Probabilistic Validation (Non-Zero Temperature):**
  * Using default or higher temperature settings (e.g., 0.7) introduces controlled randomness
  * This enables the "double validation" approach where the same text and prompt are processed twice
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
      * Human verification: Flag inconsistent results for manual review (following Heseltine, 2024)

This dual approach to validation allows researchers to balance between deterministic reproducibility and capturing model uncertainty, making HDZSC adaptable to different research requirements and classification tasks.


## The `iterative_double_zeroshot_classification` Function

This function is the core of the IDZSC and HDZSC implementation in `zeroshot_engine`. It takes a text input and a set of parameters to perform iterative zero-shot classification using a multi-prompting approach.

### Function Signature

```python
iterative_double_zeroshot_classification(text, parameter, context)
```

```python
parallel_iterative_double_zeroshot_classification(text, parameter, context, num_workers)
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
    *   `validate` (bool): Whether to validate the output against the expected types.
    *   `combining_strategies` (dict): Strategies for combining multiple predictions (e.g., `{"numeric": "optimistic", "list": "union"}).
    *   `max_retries` (int): The maximum number of retries for generating a prediction.
    *   `feedback` (bool): Whether to incorporate feedback into the iterative process.
    *   `print_prompts` (bool): Whether to print the prompts sent to the language model.
    *   `debug` (bool): Whether to enable debugging mode.
*   **`context` (str):** Additional context or information that can help the model make more accurate predictions. This could include background information, relevant facts, or specific instructions.

### Inner workings

The `iterative_double_zeroshot_classification` function orchestrates the following steps:

1.  **Prompt Building:**
    *   The function uses the `prompt_build` DataFrame and the specified `prompt_ids_list` to construct prompts.
    *   Each prompt consists of blocks defined in `prompt_block_cols` (e.g., "Block\_A\_Introduction", "Block\_C\_Definition").
    *   These blocks are filled with instructions, definitions, and task descriptions to guide the LLM.

2.  **LLM Interaction:**
    *   The prompts are sent to the LLM via the specified `client` and `model`.
    *   The LLM generates predictions based on the input text and the provided prompts.

3.  **Output Parsing and Validation:**
    *   The function parses the LLM's output to extract the predicted labels and their corresponding values.
    *   It validates the output against the expected `output_types` (e.g., numeric, list) and uses `label_codes` to convert the predictions into a standardized format.

4.  **Combining Strategies:**
    *   If multiple prompts are used or if the iterative process generates multiple predictions, the function combines these predictions using the specified `combining_strategies`.
*   **`set_zeroshot_parameters`:** This function is used to configure the `parameter` dictionary that is passed to `iterative_double_zeroshot_classification`. It sets up the prompts, defines the valid labels, and specifies the output types.
*   **`get_demo_prompt_structure`:** This function provides example prompt structures for demonstration purposes.
*   **`setup_demo_model`:** This function sets up the LLM client and model that will be used for classification.

## Demo Logic Showcase - Simple Negative Campaigning Classifier 

The demo prompts used in `zeroshot_engine` are designed to showcase the capabilities of HDZSC with different prompt structures. The `get_demo_prompt_structure` function creates a DataFrame containing these prompts. Here's how the prompts are structured:

### Example Prompt Structure

The `prompt_build` DataFrame should have columns for prompt IDs and prompt blocks. The variable `text` is automatically available through the "text"-parameter of iterative_(double)_zeroshot_classification() functions.

Here's an example of how the DataFrame might be structured:
| Prompt-ID | Block_A_Introduction | Block_B_History | Block_C_Definition | Block_D_Task | Block_E_Structure | Block_F_Output |
|-------|----------|----------|----------|--------------------|-----------------|-----------------------|
| P1_political_naive | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | empty | empty | Let us think step by step. Please determine whether the following text is political: "{text}" | Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political. | Do not hallucinate and do not provide any explanation for your decision. |
| P2_presentation_naive | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as political in a previous classification task. | empty | Let us think step by step. Please determine whether the following text contains a political presentation: "{text}" | Mark the identified category presentation in a dictionary with key "presentation" and value "1" if you identify any sentence or hashtag in the text as presentation, and value "0" if you cannot identify any presentation within the text. | Do not hallucinate and do not provide any explanation for your decision. |
| P3_attack_naive | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as political in a previous classification task. | empty | Let us think step by step. Please determine whether the following text contains a political attack: "{text}" | Mark the identified category attack in a dictionary with key "attack" and value "1" if you identify any sentence or hashtag in the text as attack, and value "0" if you cannot identify any attack within the text. | Do not hallucinate and do not provide any explanation for your decision. |
| P4_target_naive | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as a political attack in a previous classification task. | empty | Let us think step by step. Please identify the target or targets of the attack in the following text: "{text}" | Write all identified targets of this attack in a dictionary with key "target" and a list as value ["target1", "target2", ...]. If you cannot identify a target, give back an empty python list element. | Do not hallucinate and do not provide any explanation for your decision. |
| P1_political_with_definition | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | empty | Political texts contain information about political developments, political actors or political topics, be they on an international, national or local level. This includes references to federal organizations, branches of government political programs but also criticism of political actors & content. This also includes content on financial institutions and national economic developments, but not references to individual share prices or companies. | Let us think step by step. Please etermine whether the following text is political: "{text}" | Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political. | Do not hallucinate and do not provide any explanation for your decision. |
| P2_presentation_with_definition | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as political in a previous classification task. | A presentation is characterized by the emphazise and praise of one's own political ideas, positions, opinions, achievements, work or plans or the political ideas, positions, opinions, achievements, work or plans of political allies (members of one's own party) in a neutral or positive tone. The decisive factor for a presentation is not the tonality, but the fact, that the candidate talks about a topic without attacking an opponent to highlight his view of the world. It is not an exclusion criterion for a presentation in a text that an opponent is attacked or criticized in the same text. | Let us think step by step. Please etermine whether the following text contains a political presentation: "{text}" | Mark the identified category presentation in a dictionary with key "presentation" and value "1" if you identify any sentence or hashtag in the text as presentation, and value "0" if you cannot identify any presentation within the text. | Do not hallucinate and do not provide any explanation for your decision. |
| P3_attack_with_definition | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as political in a previous classification task. | A attack is defined by criticism of the political opponent with a generally critical or negative, but not necessarily unobjective tone recognizable in the text. It is not an exclusion criterion for an attack label if the same text speaks positively about one's own plans, work, position or achievements (self-presentation). A candidate or party, but also the entire opposition, government or other organizations (e.g. NGOs, movements, central banks, etc.) as well as broad abstract groups (e.g. "conservatives", "progressives", "the extreme left") can be named as opponents/targets of the attack. An attack can adress political positions, achievements, plans or work of the critizied opponents. | Let us think step by step. Please etermine whether the following text contains a political attack: "{text}" | Mark the identified category attack in a dictionary with key "attack" and value "1" if you identify any sentence or hashtag in the text as attack, and value "0" if you cannot identify any attack within the text. | Do not hallucinate and do not provide any explanation for your decision. |
| P4_target_with_definition | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}. | The text we will show you was already classified as a political attack in a previous classification task. | Depending on the context, the target of an attack would be usually a specific political person or party. Sometimes a group such as "the government", "the opposition" or "GroKo" or even abstract political groups such as "the left-wing forces" or "the authoritarians", "the populists" can also be the target of an attack. Combinations of groups, e.g. a candidate and his party or several candidates or several parties, can also be the target of an attack. In the context of this texts, the target could also be mentioned via the "@" followed by the twitter handle like "@target". But be cautious, because these texts often start with a longer list of multiple @mentions, which must not be the actual target of the attach but only accounts that are part of the ongoing conversation. It is also important, that political topics cannot be the target. Also, sometimes the person or party attacked has a hashtag before his name like "#name". | Let us think step by step. Please identify the target or targets of the attack in the following text: "{text}" | Write all identified targets of this attack in a dictionary with key "target" and a list as value ["target1", "target2", ...]. If you cannot identify a target, give back an empty python list element. | Do not hallucinate and do not provide any explanation for your decision. |

In this example:
- `prompt_id_col` would be "Prompt-ID"
- `prompt_block_cols` would be ["Block_A_Introduction",	"Block_B_History",	"Block_C_Definition", "Block_D_Task",	"Block_E_Structure", "Block_F_Output"]
- `prompt_ids_list` might be ["P1_political_naive", "P2_presentation_naive", "P3_attack_naive", "P4_target_naive", "P1_political_with_defintion"]
- `valid_keys` would be ["political", "presentation", "attack", "target"]
- Any block containing only the string "empty" (like Block_B_History in the P1_political row) will be excluded from the final prompt, allowing for flexible prompt construction with optional sections.
- `stop_condition` must be defined as:

```python
stop_condition = {
    0: { # Check after fist label in valid_keys (Index 0: political), if this label was classified as 0 (absent), stop directly.
        "condition": 0,
        "blocked_keys": [
            "presentation",
            "attack",
            "target",
        ],
    },
    2: { # Check after third label  in valid_keys (Index 2: attack), if this label was classified as 0 (absent), stop directly.
        "condition": 0,
        "blocked_keys": ["target"],
    },
}
```

### Using optional Context with f-string notation

The prompt blocks support Python f-string syntax for dynamic content insertion.
The variable `text` is automatically available, and additional context variables
can be provided through the `context` parameter.

Example of prompt blocks with context variables:

| Prompt-ID | Block_A_Introduction_with_Context | Block_B_Task | Block_C_Structure | Block_D_Output |
|-----------|-----------------------------------|--------------|-------------------|----------------|
| P1_political | You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} from {author}. | Determine whether the following text is political: {text} | Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political. | Do not hallucinate and do not provide any explanation for your decision. |

To use context variables, provide a context dictionary:

```python
# When calling the classification function directly for a single text
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
    data=df, # Contains texts-strings to classify in "text" column with context information in context columns (see below).
    parameter=parameters,
    context=["lang", "author", "platform", "date"]  # These column names from df will be used.
    # They should be stored in the same row as the corresponding "text"-string.
)

# Or in parallelization for faster processing
parallel_iterative_double_zeroshot_classification(
    data=df, # Provide the texts-strings to classify in "text" column with context information in context columns (see below).
    parameter=parameters,
    context=["lang", "author", "platform", "date"],  # These column names from df will be used.
    numworkers=4 # Define the number of parallel workers according to your system.
)
```

#### Example-DataFrame structure of **df** that would be passed to this function as **data**:

| text | lang | author | platform | date |
|------|------|--------|----------|------|
| "We need to invest more in renewable energy to combat climate change. The opposition continues to ignore scientific evidence." | English | Jane Smith | Twitter | 2023-06-15 |
| "Today I announced our new healthcare plan that will benefit all citizens. Together we can build a better future." | English | John Doe | Facebook | 2023-07-22 |
| "Our economic policies have created thousands of jobs. Unlike @OppositionParty who only raised taxes during their term." | English | Sarah Johnson | Twitter | 2023-08-10 |

> **Note:** The variable `text` has to be always available in one block of the prompt via docstring in brackets {text}. See prompt examples above.

The function will use these components to construct complete prompts for classification.

### Example Code: iterative_double_zeroshot_classification()

```python
# Full hierarchical example with all parameters and specialized settings
parameters = set_zeroshot_parameters(
    model_family="phi",
    client=client,
    model="phi4:latest",
    prompt_build=prompts_df,
    prompt_ids_list=["P1_political_naive", "P2_presentation_naive", "P3_attack_naive", "P4_target_naive"]
    prompt_id_col="Prompt-ID",
    prompt_block_cols=["Block_A_Introduction", "Block_B_History", "Block_C_Definition", "Block_D_Task", "Block_E_Structure", "Block_F_Output"]
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
    feedback=True,
)

# With context variables
result = iterative_double_zeroshot_classification(
    text="Sample text to analyze",
    parameter=parameters_with_context_prompts,
    context={
        "lang": "English",
        "author": "John Doe",
        "platform": "Twitter",
        "date": "2023-05-15"
    }
)
```

### Example Code: parallel_iterative_double_zeroshot_classification()

```python
# Full hierarchical example with all parameters and specialized settings
# Create the example DataFrame
import pandas as pd

df = pd.DataFrame({
    "text": [
        "We need to invest more in renewable energy to combat climate change. The opposition continues to ignore scientific evidence.",
        "Today I announced our new healthcare plan that will benefit all citizens. Together we can build a better future.",
        "Our economic policies have created thousands of jobs. Unlike @OppositionParty who only raised taxes during their term."
    ],
    "lang": ["English", "English", "English"],
    "author": ["Jane Smith", "John Doe", "Sarah Johnson"],
    "platform": ["Twitter", "Facebook", "Twitter"],
    "date": ["2023-06-15", "2023-07-22", "2023-08-10"],
    "party": ["Green Party", "Democrats", "Republicans"]
})

# Full hierarchical example with all parameters and specialized settings
parameters = set_zeroshot_parameters(
    model_family="phi",
    client=client,
    model="phi4:latest",
    prompt_build=prompts_df,
    prompt_ids_list=["P1_political_naive", "P2_presentation_naive", "P3_attack_naive", "P4_target_naive"],
    prompt_id_col="Prompt-ID",
    prompt_block_cols=["Block_A_Introduction", "Block_B_History", "Block_C_Definition", "Block_D_Task", "Block_E_Structure", "Block_F_Output"],
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
    feedback=True,
)

# Process the entire DataFrame in parallel
results_df = parallel_iterative_double_zeroshot_classification(
    data=df,  # Pass the example DataFrame with text and context columns
    parameter=parameters,
    context=["lang", "author", "platform", "date", "party"],  # Use all context columns from df
    num_workers=2  # Adjust based on your system's capabilities
)

# The returned results_df will have all original columns plus classification results
print(results_df.head())
```
