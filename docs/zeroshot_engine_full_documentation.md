# zeroshot-engine Full Documentation

Welcome to the complete documentation for `zeroshot-engine`. This guide provides all the information you need to use this package effectively for your zero-shot classification tasks.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Basic Usage](#basic-usage)
3.  [APIs and Models](#apis-and-models)
    *   [Supported APIs](#supported-apis)
    *   [Initializing a Model](#initializing-a-model)
4.  [Core Concepts](#core-concepts)
    *   [Iterative Zero-Shot Classification](#iterative-zero-shot-classification)
    *   [Prompt Engineering](#prompt-engineering)
    *   [Hierarchical Classification](#hierarchical-classification)
5.  [Main Functions](#main-functions)
    *   [`set_zeroshot_parameters`](#set_zeroshot_parameters)
    *   [`iterative_zeroshot_classification`](#iterative_zeroshot_classification)
    *   [`parallel_iterative_zeroshot_classification`](#parallel_iterative_zeroshot_classification)
6.  [Utility Functions](#utility-functions)
    *   [Demo Helpers](#demo-helpers)
    *   [Visualization](#visualization)
    *   [API Key Management](#api-key-management)
    *   [Ollama Management](#ollama-management)
7.  [Command-Line Interface (CLI)](#command-line-interface-cli)

---

## Introduction

`zeroshot-engine` is a powerful Python package designed for flexible and robust zero-shot text classification using Large Language Models (LLMs). It allows you to define complex, multi-step classification workflows, use various local and remote LLMs, and process large datasets efficiently.

Key features include:
*   **Iterative Classification**: Break down complex classification tasks into a series of simpler steps.
*   **Hierarchical Logic**: Define conditional paths (e.g., if a text is not "political", skip "attack" classification) to save time and cost.
*   **Multi-API Support**: Seamlessly switch between APIs like OpenAI, OpenRouter, and local models via Ollama.
*   **Parallel Processing**: Classify large pandas DataFrames in parallel to speed up your workflow.
*   **Prompt Templating**: Easily build and manage complex prompts from a structured format.
*   **Validation and Retries**: Improve reliability with built-in validation for dual-prediction setups and automatic retries.

---

## Getting Started

### Installation

You can install the latest version of the `zeroshot-engine` directly from pip.

```bash
pip install zeroshot-engine
```

This will install the package and all its dependencies listed in `pyproject.toml`.

### Basic Usage

Here's a simple example of how to classify a single piece of text. This example uses a local model via Ollama.

```python
import pandas as pd
from zeroshot_engine import (
    initialize_model,
    set_zeroshot_parameters,
    iterative_zeroshot_classification,
    get_demo_prompt_structure,
)

# 1. Initialize the model you want to use
# This example uses a local model via Ollama. Make sure Ollama is running.
client = initialize_model(api="ollama", model="gemma:2b")

# 2. Load a prompt structure (using a demo helper for this example)
prompts_df = get_demo_prompt_structure()

# 3. Define your classification labels and prompt mapping
labels = ["political", "presentation", "attack"]
prompt_ids = ["P1_political_naive", "P2_presentation_naive", "P3_attack_naive"]

# 4. Set the classification parameters
parameters = set_zeroshot_parameters(
    client=client,
    model_family="ollama_llm", # Use 'ollama_llm' for langchain-ollama
    model="gemma:2b",
    prompt_build=prompts_df,
    prompt_ids_list=prompt_ids,
    prompt_block_cols=[
        "Block_A_Introduction", "Block_B_History", "Block_C_Definition", 
        "Block_D_Task", "Block_E_Structure", "Block_F_Output"
    ],
    valid_keys=labels,
)

# 5. The text you want to classify
text_to_classify = "The new bill proposes significant changes to our healthcare system."

# 6. Perform the classification
result = iterative_zeroshot_classification(
    text=text_to_classify,
    parameter=parameters,
    context={"lang": "English", "author": "Politician X", "platform": "web", "date": "2023-10-27", "party": "N/A"}
)

print(result)
```

---

## APIs and Models

### Supported APIs

`zeroshot-engine` supports multiple providers for LLMs:

*   **Ollama**: For running open-source models locally on your own machine (CPU or GPU).
*   **OpenAI**: For accessing models like GPT-4o, GPT-4, and GPT-3.5-turbo. Requires an API key.
*   **OpenRouter**: A service that provides access to a wide variety of models from different providers through a single API. Requires an API key.
*   **Custom (OPENAI CLIENT COMPATIBLE)**: For accessing models like e.g. Gemini directly via Google AI Studio Endpoints from the OpenAI Python Client.

### Finding and Using Supported Models

To use a model with `zeroshot-engine`, you need to know its name and the API that provides it. Here’s how to find the models available for each supported service:

#### Ollama Models
Ollama supports a wide range of open-source models that you can run locally.

*   **How to find models**: You can browse the full list of available models in the [Ollama Library](https://ollama.com/library).
*   **How to use**:
    1.  First, pull the model to your local machine using the Ollama CLI. For example, to download Gemma 2B, run:
        ```bash
        ollama pull gemma:2b
        ```
    2.  Then, use the model name in the `initialize_model` function:
        ```python
        client = initialize_model(api="ollama", model="gemma:2b")
        ```

#### OpenRouter Models
OpenRouter provides access to a vast collection of models from different developers, including new and experimental ones.

*   **How to find models**: The list of supported models is available on the [OpenRouter Models page](https://openrouter.ai/models).
*   **How to use**:
    1.  Find the "Model Name" on their page (e.g., `google/gemma-7b-it`).
    2.  Use this name in the `initialize_model` function:
        ```python
        client = initialize_model(api="openrouter", model="google/gemma-7b-it")
        ```

#### OpenAI Models
OpenAI offers its own state-of-the-art models.

*   **How to find models**: The official list of models is available in the [OpenAI API documentation](https://platform.openai.com/docs/models).
*   **How to use**:
    1.  Choose a model ID from the documentation (e.g., `gpt-4o-mini`).
    2.  Use this ID in the `initialize_model` function:
        ```python
        client = initialize_model(api="openai", model="gpt-4o-mini")
        ```

#### Custom Models
OpenAI offers its own state-of-the-art models.

*   **How to find models**: The official list of models from the correspondong providers documentation (e.g. Google Gemini AI Studio Documentation).
*   **How to use**:
    1.  Choose a model ID from the corresponding documentation (e.g., `gemini-2.5-flash`).
    2.  Use this ID in the `initialize_model` function with the custom Base URL and the API Key Name (named by you):
        ```python
        initialize_model(api="custom", model="gemini-2.5-flash", base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key_name="GEMINI_API_KEY")
        ```

### Initializing a Model

The `initialize_model` function is your entry point for setting up an LLM. It handles API connections, key management, and returns a ready-to-use client.

`initialize_model(api: str, model: str, base_url: str = None, api_key_name: str = None) -> any`

*   **`api` (str)**: The API to use. Can be `"ollama"`, `"openai"`, `"openrouter"`, or `"custom"`.
*   **`model` (str)**: The name of the model to use (e.g., `"gemma:2b"`, `"gpt-4o-mini"`, `"google/gemma-7b-it"`).
*   **`base_url` (str, optional)**: A custom base URL for the API. This is used to override the default for `"openrouter"` or is **required** for `"custom"`.
*   **`api_key_name` (str, optional)**: The name of the environment variable holding your API key (e.g., `"GEMINI_API_KEY"`). This is specifically for use with `api="custom"`.

#### Using a Custom OpenAI-Compatible API
The `api="custom"` option provides the flexibility to connect to any API that is compatible with OpenAI's client. This is useful for services like Google's Gemini, or other self-hosted or private models.

To use it, you must provide:
1.  `base_url`: The root URL of the custom API.
2.  `api_key_name`: The name of the environment variable where your API key is stored (e.g., `"GEMINI_API_KEY"`).

If the specified API key is not found in your environment or `.env` file, the package will securely prompt you to enter it for the session or save it for future use.

**Examples:**

```python
# Initialize a local model with Ollama
# Make sure Ollama is installed and the model is downloaded (e.g., `ollama pull gemma:2b`)
ollama_client = initialize_model(api="ollama", model="gemma:2b")

# Initialize an OpenAI model
# Requires OPENAI_API_KEY to be set as an environment variable
openai_client = initialize_model(api="openai", model="gpt-4o-mini")

# Initialize a model via OpenRouter using its default URL
# Requires OPENROUTER_API_KEY to be set as an environment variable
openrouter_client = initialize_model(api="openrouter", model="google/gemma-7b-it")

# Initialize a model via OpenRouter with a custom base URL (e.g., for a proxy)
custom_router_client = initialize_model(
    api="openrouter", 
    model="google/gemma-7b-it", 
    base_url="https://my-custom-proxy.com/api/v1"
)

# Initialize a model using a completely custom OpenAI-compatible API (e.g., Gemini)
# This will prompt for the "GEMINI_API_KEY" if it's not found in the environment.
gemini_client = initialize_model(
    api="custom",
    model="gemini-2.5-flash", # The model name as expected by the custom API
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # Example URL for the custom API
    api_key_name="GEMINI_API_KEY" # The environment variable to look for
)
```

---

## Core Concepts

### Iterative Zero-Shot Classification

Instead of asking an LLM to perform a complex classification in one go, `zeroshot-engine` breaks it down into a sequence of simpler steps. For each step, a targeted prompt is generated to classify one specific label. This improves accuracy and allows for more control over the classification logic.

### Prompt Engineering

Prompts are constructed dynamically from a `pandas.DataFrame` (the `prompt_build` parameter). This DataFrame holds different parts of your prompts, which are assembled at runtime.

*   **`prompt_build`**: A DataFrame where each row corresponds to a specific prompt template.
*   **`prompt_block_cols`**: A list of column names in `prompt_build` that contain the text blocks for your prompt. These blocks are concatenated to form the final prompt.
*   **`prompt_ids_list`**: A list of IDs that specifies which rows from `prompt_build` to use and in what order.
*   **Context Variables**: You can use f-string-like placeholders (e.g., `{text}`, `{lang}`) in your prompt blocks. The `{text}` is filled automatically, and you can pass other variables via the `context` dictionary.

### Hierarchical Classification

You can define `stop_conditions` to create a classification hierarchy. This allows you to skip certain classification steps based on the results of previous ones, which is useful for saving computational resources and API costs.

For example: "Only classify for 'attack' if the text has already been classified as 'political'."

This is configured via the `stop_conditions` parameter in `set_zeroshot_parameters`.

---

## Main Functions

### `initialize_model`

**Title**: Initialize a Model Client for a Given API

**Description**

This function is the entry point for setting up a connection to a Large Language Model. It handles the specific initialization logic for different supported APIs (Ollama, OpenAI, OpenRouter) and returns a client object that can be used for making requests. For local models via Ollama, it also triggers the setup process if needed.

**Usage**

```python
initialize_model(
    api: str,
    model: str
) -> any
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `api` | `str` | (required) | The API provider to use. Supported values are `"ollama"`, `"openai"`, `"openrouter"`, and `"custom"`. |
| `model` | `str` | (required) | The name of the model to initialize. This should be a valid model name for the specified API (e.g., `"gemma:2b"` for Ollama, `"gpt-4o-mini"` for OpenAI). |
| `base_url` | `str` | `None` | A custom base URL for API requests. If provided with `api="openrouter"`, it overrides the default OpenRouter URL. It is **required** for `api="custom"`. |
| `api_key_name` | `str` | `None` | The name of the environment variable for your custom API key (e.g., `"GEMINI_API_KEY"`). Used when `api="custom"`. |

**Details**

*   **Ollama (`api="ollama"`)**: This will check for a local Ollama installation. If Ollama is not running or the specified model is not available, it will trigger the interactive `setup_ollama` process to guide the user through installation and model download. It returns a `langchain_ollama.OllamaLLM` client.
*   **OpenAI (`api="openai"`)**: This will initialize the official OpenAI client. It requires the `OPENAI_API_KEY` to be available as an environment variable. The `setup_openai_api_key` helper will be used to configure this.
*   **OpenRouter (`api="openrouter"`)**: This initializes an OpenAI-compatible client configured to use the OpenRouter API. It defaults to `https://openrouter.ai/api/v1`, but you can override this with the `base_url` parameter. It requires the `OPENROUTER_API_KEY` to be available as an environment variable.
*   **Custom (`api="custom"`)**: This allows you to connect to any OpenAI-compatible API endpoint. You **must** provide the `base_url`. You can also provide an `api_key_name` to specify which environment variable holds your API key. If `api_key_name` is not provided, it defaults to `"CUSTOM_API_KEY"`. The `setup_custom_api_key` helper will be used to configure it.

The function caches initialized model clients to avoid re-initializing the same model multiple times within a session, which can save time and resources.

**Value**

Returns an initialized client object specific to the chosen API. This object should be passed to the `client` argument of `set_zeroshot_parameters`.

### `set_zeroshot_parameters`

**Title**: Set and Validate Parameters for Zero-Shot Classification

**Description**

This function is the primary configuration utility for `zeroshot-engine`. It creates a validated parameter dictionary that governs the behavior of the classification functions (`iterative_zeroshot_classification`, `parallel_iterative_zeroshot_classification`, etc.). It ensures all necessary settings are provided, applies sensible defaults, and performs validation checks on the inputs.

**Usage**

```python
set_zeroshot_parameters(
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
)
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_family` | `str` | `"openai"` | The family of the model being used. This determines how the model's response is processed. Supported families include: <br> - `"openai"`, `"openrouter"`, `"custom"` for standard API-based models. <br> - `"ollama_llm"` (Legacy Implementation: `"llama"`, `"phi"`, `"gemma"`, `"mistral"`, `"qwen"`) for standard Ollama models. <br> - `"ollama_reasoning_llm"` (Legacy Implementation: `"deepseek"`) for Ollama models that use a reasoning/thought process before outputting the final JSON. |
| `client` | `any` | `None` | The initialized model client object returned by `initialize_model`. **Required**. |
| `model` | `str` | `"gpt-4o-mini"` | The specific name of the model to use for classification (e.g., `"gpt-4o-mini"`, `"google/gemma-7b-it"`, `"gemma:2b"`). |
| `prompt_build` | `pd.DataFrame` | `None` | A DataFrame containing the components of the prompts. Each row represents a prompt template, and columns contain different blocks of text. **Required**. |
| `prompt_ids_list` | `list[str]` | `None` | An ordered list of prompt IDs (from `prompt_id_col`) specifying which prompts to use and in what sequence for the iterative classification. **Required**. |
| `prompt_id_col` | `str` | `"Prompt-ID"` | The name of the column in `prompt_build` that contains the unique prompt identifiers. |
| `prompt_block_cols` | `list[str]` | `None` | A list of column names in `prompt_build` that contain the text blocks to be assembled into the final prompt. **Required**. |
| `valid_keys` | `list[str]` | `None` | An ordered list of the classification labels (keys) that the model is expected to return. The order must correspond to `prompt_ids_list`. **Required**. |
| `label_codes` | `dict` | See Details | A dictionary mapping human-readable label outcomes to their coded values (e.g., `{"present": 1, "absent": 0}`). |
| `stop_conditions` | `dict` | `None` | A dictionary defining the hierarchical classification logic. It maps a step index to a condition that, if met, will cause subsequent specified steps to be skipped. |
| `output_types` | `dict` | `None` | A dictionary specifying the expected output type for each label (e.g., `{"political": "numeric", "target": "list"}`). Defaults to `'numeric'` for all keys. |
| `double_shot` | `bool` | `False` | If `True`, each classification step is performed twice to generate two independent predictions. This is a prerequisite for using the `validate` feature. |
| `combining_strategies` | `dict` | See Details | A dictionary defining the strategy for combining predictions when `validate=True`. Separate strategies can be set for `'numeric'` and `'list'` output types. |
| `validate` | `bool` | `False` | If `True` (and `double_shot=True`), the two predictions for each step are combined into a single, more reliable result based on the `combining_strategies`. |
| `max_retries` | `int` | `1` | The maximum number of times to retry a classification step if the model returns an invalid or improperly formatted response. |
| `feedback` | `bool` | `False` | If `True`, prints progress messages and classification results to the console during execution. |
| `print_prompts` | `bool` | `False` | If `True`, prints the full, formatted prompt that is sent to the model for each classification step. Useful for debugging prompts. |
| `debug` | `bool` | `False` | If `True`, prints the raw, unparsed response from the model. Useful for debugging model behavior and response format issues. |
| `temperature` | `float` | `None` | The sampling temperature for the model. A lower value (e.g., `0.0`) makes the output more deterministic, while a higher value increases randomness. If `None`, the model's default temperature is used. |

**Details**

*   **`label_codes`**: If not provided, defaults to `{"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}`.
*   **`stop_conditions`**: The key of the dictionary is the zero-based index of the step in the `valid_keys` list. The value is a dictionary containing a `condition` (the classification result to check for, e.g., `0`) and `blocked_keys` (a list of labels to skip if the condition is met).
*   **`combining_strategies`**: If not provided, defaults to `{"numeric": "optimistic", "list": "union"}`.
    *   Valid numeric strategies: `"conservative"` (prefers `absent`), `"optimistic"` (prefers `present`), `"probabilistic"` (random choice).
    *   Valid list strategies: `"union"`, `"intersection"`, `"first"`, `"second"`.

**Value**

Returns a `dict` containing all the validated and processed parameters. This dictionary is intended to be passed directly to the classification functions (`iterative_zeroshot_classification`, etc.).

**Raises**

*   `ValueError`: If required parameters are missing or have invalid values.
*   `TypeError`: If parameters are of an incorrect type.

### `iterative_zeroshot_classification`

**Title**: Perform Iterative Zero-Shot Classification on a Single Text

**Description**

This function performs a multi-step zero-shot classification on a single input text. It iterates through the labels defined in the `parameter` dictionary, generating a specific prompt for each step and invoking the LLM. It handles hierarchical logic (stop conditions) and can perform validation by running each step twice.

**Usage**

```python
iterative_zeroshot_classification(
    parameter: dict,
    text: str,
    context: dict[str, any] = None,
) -> pd.Series
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `parameter` | `dict` | (required) | The parameter dictionary generated by `set_zeroshot_parameters`. It contains all settings for the classification run. |
| `text` | `str` | (required) | The input text string to be classified. |
| `context` | `dict` | `None` | A dictionary of context variables (e.g., `{"lang": "English", "author": "John Doe"}`) that can be dynamically inserted into the prompts using f-string-like placeholders. |

**Details**

The function orchestrates the entire iterative process for a single piece of text. If `double_shot` is enabled in the `parameter` object, it calls the internal `single_iterative_zeroshot_classification` function twice. If `validate` is also enabled, it then combines the two sets of predictions based on the specified `combining_strategies` to produce a final, validated result.

For a detailed walkthrough, see the [Getting Started Tutorial](./Tutorial_Get_Started.md).

**Value**

Returns a `pandas.Series` containing the classification results.
*   If `validate=False`, the series contains one column for each label in `valid_keys`.
*   If `validate=True`, the series will contain additional columns:
    *   `<label>_pred1` and `<label>_pred2`: The raw results from the two prediction runs.
    *   `<label>`: The final, validated result.
    *   `<label>_method`: The method used to arrive at the final result (e.g., `"identical"`, `"conservative"`).
    *   `validation_conflict`: A flag (`1` or `0`) indicating if there was a disagreement between the two predictions that required a validation strategy to resolve.

### `parallel_iterative_zeroshot_classification`

**Title**: Perform Parallel Iterative Zero-Shot Classification on a DataFrame

**Description**

This function applies the iterative zero-shot classification process to an entire pandas DataFrame in parallel. It splits the DataFrame into chunks and processes them concurrently across multiple worker threads, making it highly efficient for classifying large datasets.

**Usage**

```python
parallel_iterative_zeroshot_classification(
    data: pd.DataFrame,
    parameter: dict,
    context: list[str] = None,
    num_workers: int = 4,
) -> pd.DataFrame
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `pd.DataFrame` | (required) | The input DataFrame. It must contain a column named `"text"` with the texts to be classified. |
| `parameter` | `dict` | (required) | The parameter dictionary from `set_zeroshot_parameters`. |
| `context` | `list[str]` | `None` | A list of column names from the `data` DataFrame. The values from these columns for each row will be passed as context variables to the prompts. |
| `num_workers` | `int` | `4` | The number of parallel worker threads to use for processing the DataFrame. Adjust this based on your system's capabilities. |

**Details**

This function is a wrapper around `apply_iterative_zeroshot_classification` that leverages Python's `concurrent.futures.ThreadPoolExecutor` to parallelize the workload. It is the recommended way to process datasets with more than a few dozen entries.

For a detailed walkthrough on processing DataFrames, see the [Getting Started Tutorial](./Tutorial_Get_Started.md).

**Value**

Returns a new `pandas.DataFrame` containing all the columns from the original `data` DataFrame, with the classification result columns (as described in the return value of `iterative_zeroshot_classification`) appended.

---

## Utility Functions

### Demo Helpers
These functions provide pre-configured data for running demos and tests.
*   `get_demo_prompt_structure()`: Returns a DataFrame with sample prompts.
*   `get_demo_stop_conditions()`: Returns a sample `stop_conditions` dictionary.
*   `get_demo_text_selection()`: Provides a sample text for classification.

### Visualization

#### `display_label_flowchart`

**Title**: Display the Hierarchical Label Dependency Flowchart

**Description**

This function generates and prints an ASCII-art flowchart to the console, visualizing the hierarchical structure and dependencies of your classification labels as defined by the `stop_conditions`. This is an essential tool for debugging your classification logic and ensuring the conditional flow is correct.

**Usage**

```python
display_label_flowchart(
    valid_keys: list[str],
    stop_conditions: dict,
    label_codes: dict
)
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `valid_keys` | `list[str]` | (required) | The ordered list of classification labels. |
| `stop_conditions` | `dict` | (required) | The dictionary defining the hierarchical stop conditions. |
| `label_codes` | `dict` | (required) | The dictionary mapping label outcomes to their coded values. |

**Value**

This function does not return a value. It prints the flowchart directly to standard output.

### API Key Management

#### `setup_openai_api_key`

**Title**: Set Up OpenAI API Key

**Description**

A convenience function to manage the OpenAI API key. It checks for the key in environment variables and a local `.env` file. If not found, it securely prompts the user to enter the key and offers to save it to a `.env` file for future use.

**Usage**

```python
setup_openai_api_key()
```

**Arguments**: None.

**Details**

The function prioritizes security by using `getpass` to hide key input and provides clear information about how the key is stored and used. It will create or update a `.env` file in the current working directory.

**Value**: None.

#### `setup_openrouter_api_key`

**Title**: Set Up OpenRouter API Key

**Description**

Similar to `setup_openai_api_key`, this function manages the OpenRouter API key, checking environment variables and `.env` files, and prompting the user if the key is not found.

**Usage**

```python
setup_openrouter_api_key()
```

**Arguments**: None.

**Value**: None.

### Ollama Management

#### `setup_ollama`

**Title**: Comprehensive Ollama and Local Model Setup

**Description**

This is an all-in-one utility for managing a local Ollama setup. It can check if Ollama is installed, prompt for installation or updates, start the Ollama service, and download a specified model if it's not already available locally. It also checks for GPU support.

**Usage**

```python
setup_ollama(model: str) -> bool
```

**Arguments**

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `str` | (required) | The name of the local model to set up (e.g., `"gemma:2b"`). |

**Details**

This function automates many of the tedious steps involved in setting up a local LLM environment. It interacts with the user via the console to confirm actions like installation or model downloads.

**Value**

Returns `True` if the setup was successful and the model is ready to be used. Returns `False` if any part of the setup fails or is cancelled by the user.

#### Other Ollama Helpers

The package also includes several lower-level helper functions used by `setup_ollama`:

*   `check_ollama_installation()`: Checks if the `ollama` command is available.
*   `install_ollama(os_type)`: Attempts to run the Ollama installation script.
*   `update_ollama(os_type)`: Attempts to run the Ollama update script.
*   `download_model_with_progress(model)`: Downloads a model from the Ollama registry with a progress bar.
*   `check_ollama_gpu_support()`: Inspects the system and Ollama service to determine if GPU acceleration is active.

---

## Command-Line Interface (CLI)
The package also includes a command-line interface for demo purposes.

`zeroshot-engine`

You can use this to run the demo of the engine directly from your terminal. Use the `--help` flag for more information.

```bash
zeroshot-engine --help
```
This will show the available commands and options.

