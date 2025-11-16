import json

import re

import pandas as pd

from .custom import setup_custom_api_key
from .ollama import check_ollama_updates, setup_ollama, update_ollama
from .openai import setup_openai_api_key
from .openrouter import setup_openrouter_api_key


# Function to get a specific prompt by Prompt-ID
def get_prompt_id(
    prompt_build: pd.DataFrame,
    prompt_id_col: str,
    current_prompt_id: str,
) -> pd.Series:
    """
    Retrieves a specific prompt string by Prompt-ID.

    Args:
        prompt_build (pd.DataFrame): The DataFrame containing the data.
        prompt_id_col (str): The name of the prompt-id column.
        current_prompt_id (str): The specific Prompt-ID to retrieve.

    Returns:
        pd.Series: The row corresponding to the specific Prompt-ID.
    """
    return prompt_build[prompt_build[prompt_id_col] == current_prompt_id].iloc[0]


def generate_prompt(
    prompt_id_row: pd.DataFrame,
    prompt_id_col: str,
    prompt_block_cols: list[str],
    text: str,
    context: dict[str, any] = None,
) -> str:
    """
    Generates a prompt string from different columns of DataFrame row and additional text and context information.

    Args:
        prompt_id_row (pd.Series): A pandas DataFrame row.
        prompt_id_col (str): The name of the prompt-id column.
        prompt_block_cols (list): A list of the names of the block columns.
        text (str): The text to include in the prompt.
        context (dict): A dictionary containing context information to be included in the prompt.

    Returns:
        str: A string containing the full prompt.
    """
    if context is None:
        context = {}
    prompt_id_row[prompt_id_col]
    blocks = prompt_id_row[prompt_block_cols].dropna().values.tolist()
    blocks = [b for b in blocks if b != "empty"]
    prompt_text = " ".join(blocks)
    context["text"] = text
    return prompt_text.format(**context)


_LOADED_MODELS = {}


def initialize_model(api: str, model: str, base_url: str = None, api_key_name: str = None) -> any:
    """
    Initializes the specified model based on the API.

    Args:
        api (str): The API to use (e.g., "openai", "ollama", "openrouter", "custom").
        model (str): The name of the model to initialize.
        base_url (str, optional): A custom base URL for API requests.
                                Defaults to None. Used by "openrouter" and "custom" APIs.
        api_key_name (str, optional): The name of the environment variable for the API key.
                                      Used by the "custom" API.

    Returns:
        any: The initialized model client.
    """
    cache_key = f"{api}:{model}:{base_url}:{api_key_name}"

    # Check if model is already loaded and cached
    if cache_key in _LOADED_MODELS:
        print(f"Using cached {model} model")
        return _LOADED_MODELS[cache_key]

    if api == "openai":
        import os

        from openai import OpenAI

        setup_openai_api_key()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"{model} connection set up successfully!")
        return client

    if api == "openrouter":
        import os
        from openai import OpenAI

        setup_openrouter_api_key()

        # Use the provided base_url or default to OpenRouter's URL
        final_base_url = base_url if base_url else "https://openrouter.ai/api/v1"

        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=final_base_url
        )
        print(f"{model} connection to OpenRouter (via OpenAI Client) set up successfully!")
        # Cache the client
        _LOADED_MODELS[cache_key] = client
        return client

    if api == "custom":
        import os
        from openai import OpenAI

        if not api_key_name:
            api_key_name = "CUSTOM_API_KEY"

        setup_custom_api_key(api_key_name)
        api_key = os.getenv(api_key_name)

        if not base_url:
            raise ValueError("A 'base_url' must be provided for the 'custom' API.")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print(f"{model} connection to custom API at {base_url} set up successfully!")
        # Cache the client
        _LOADED_MODELS[cache_key] = client
        return client

    if api == "ollama":
        import os
        import platform
        import sys

        # Check for Ollama updates first
        is_updated = check_ollama_updates()
        if not is_updated:
            update_choice = input(
                "\nWould you like to update Ollama now? (yes/no): "
            ).lower()
            if update_choice in ("yes", "y"):
                success = update_ollama(platform.system())
                if not success:
                    print("Unable to update automatically. Please update manually.")
                    print("Continuing with current version...")

        # Set up Ollama and model
        setup_success = setup_ollama(model)
        if not setup_success:
            print("Failed to set up Ollama properly.")
            sys.exit(1)

        # Initialize the model client
        os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
        from langchain_ollama import OllamaLLM

        try:
            client = OllamaLLM(model=model)
            print(f"{model} loaded successfully!")
            # Cache the model
            _LOADED_MODELS[cache_key] = client
            return client
        except Exception as e:
            # Check if error indicates version incompatibility
            error_msg = str(e)
            if (
                "not supported by your version" in error_msg
                or "unsupported model" in error_msg
            ):
                print("\n⚠️ Your version of Ollama doesn't support this model.")
                update_choice = input(
                    "Update Ollama to support this model? (yes/no): "
                ).lower()

                if update_choice in ("yes", "y"):
                    import platform

                    success = update_ollama(platform.system())
                    if success:
                        print("Trying to load model again after update...")
                        client = OllamaLLM(model=model)
                        print(f"{model} loaded successfully!")
                        return client

            print(f"Error loading model: {e}")
            print(
                "This might be due to version incompatibility. Consider updating Ollama."
            )
            sys.exit(1)

    # Handle unsupported API
    print(f"Unsupported API: {api}")
    sys.exit(1)


def request_to_model(
    model: str = "gpt-4o-mini",
    model_family: str = "openai",
    client: any = None,
    prompt: str = "",
    temperature: int = None,
    debug: bool = False,
) -> dict:
    """
    Sends a request to the specified model and returns the response as a dictionary.

    Args:
        model (str): The model to use for the request.
        model_family (str): The family of the model. This determines how the response is handled.
                            - For OpenAI/OpenRouter: "openai", "openrouter", "custom".
                            - For standard Ollama models: "ollama_llm", "llama", "phi", "gemma", "mistral", "qwen".
                            - For Ollama reasoning models: "ollama_reasoning_llm", "deepseek".
        client (any): The client object to interact with the API.
        prompt (str): The prompt to send to the model.
        temperature (int, optional): The temperature to use for the model. Defaults to None.
        debug (bool, optional): Whether to print the raw model response. Defaults to False.

    Returns:
        dict: A dictionary containing the model's response.
    """
    if model_family in ["openai"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        raw_response = response.choices[0].message.content
        if debug:
            print("\n" + "=" * 80)
            print("RAW MODEL RESPONSE:")
            print("-" * 80)
            print(raw_response)
            print("=" * 80 + "\n")
        response_dict = json.loads(raw_response)

    if model_family in ["openrouter", "custom"]:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        raw_response = response.choices[0].message.content
        if debug:
            print("\n" + "=" * 80)
            print("RAW MODEL RESPONSE:")
            print("-" * 80)
            print(raw_response)
            print("=" * 80 + "\n")
        pattern = r"\{(.*?)\}"
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            cleaned_response_dict_string = match.group(0)
            response_dict = json.loads(cleaned_response_dict_string)
        else:
            response_no_dict = "no dict possible"
            response_dict = json.loads(response_no_dict)

    if model_family in ["ollama_llm", "llama", "phi", "gemma", "mistral", "qwen"]:
        response = client.invoke(prompt)
        if debug:
            print("\n" + "=" * 80)
            print("RAW MODEL RESPONSE:")
            print("-" * 80)
            print(response)
            print("=" * 80 + "\n")

        pattern = r"\{(.*?)\}"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            cleaned_response_dict_string = match.group(0)
            response_dict = json.loads(cleaned_response_dict_string)
        else:
            response_no_dict = "no dict possible"
            response_dict = json.loads(response_no_dict)

    if model_family in ["ollama_reasoning_llm", "deepseek"]:
        response = client.invoke(prompt)
        if debug:
            print("\n" + "=" * 80)
            print("RAW MODEL RESPONSE:")
            print("-" * 80)
            print(response)
            print("=" * 80 + "\n")

        cleaned_response = re.sub(
            r"<think>.*?</think>\n?", "", response, flags=re.DOTALL
        )
        pattern = r"\{(.*?)\}"
        match = re.search(pattern, cleaned_response, re.DOTALL)
        if match:
            cleaned_response_dict_string = match.group(0)
            response_dict = json.loads(cleaned_response_dict_string)
        else:
            response_no_dict = "no dict possible"
            response_dict = json.loads(response_no_dict)

    return response_dict


def classification_step(
    model: str = "gpt-4o-mini",
    model_family: str = "openai",
    client: any = None,
    prompt: str = "",
    valid_keys: list[str] = None,
    current_key: str = "",
    print_messages: bool = False,
    max_retries: int = 2,
    temperature: int = None,
    debug: bool = False,
) -> dict:
    """
    Classifies the text using the model and the given prompt.

    Args:
        model (str): The model to use for classification.
        client (any): The client object to interact with the API.
        prompt (str): The prompt to classify.
        valid_keys (list): A list of valid keys to check in the response.
        current_key (str): The currently classified label.
        print_messages (bool): Whether to print messages during classification.
        max_retries (int): The maximum number of retries for classification steps.
        temperature (int, optional): The temperature to use for the model. Defaults to None.
        debug (bool, optional): Whether to print the raw model response. Defaults to False.

    Returns:
        dict: A dictionary containing the classifications or an error message.
    """
    if valid_keys is None:
        valid_keys = []

    retries = max_retries
    while retries > 0:
        try:
            response_dict = request_to_model(
                model, model_family, client, prompt, temperature, debug
            )
            print_messages(response_dict)
            if any(key in response_dict for key in valid_keys):
                return response_dict
            return {current_key: "error with response dict"}
        except json.JSONDecodeError as e:
            print_messages(f"JSONDecodeError: {e}")
            retries -= 1

    return {current_key: "jsonError"}


# Add these helper functions at the top
def ensure_numeric(value):
    """Convert string numbers to integers if possible."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value
