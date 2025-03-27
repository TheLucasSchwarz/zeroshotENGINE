import pandas as pd
from .base import initialize_model
from .openai import setup_openai_api_key
from .visualization import display_label_flowchart


def get_demo_prompt_structure():
    """Create and return the prompt structure DataFrame."""
    # Create a simple prompt structure DataFrame
    return pd.DataFrame(
        {
            "Prompt-ID": [
                "P1_political_naive",
                "P2_presentation_naive",
                "P3_attack_naive",
                "P4_target_naive",
                "P1_political_with_definition",
                "P2_presentation_with_definition",
                "P3_attack_with_definition",
                "P4_target_with_definition",
            ],
            "Block_A_Introduction": [
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
                "You are an expert in political communication and your task is to classify a text in {lang}. The platform it was posted is {platform} on {date} by {author} from the party {party}.",
            ],
            "Block_B_History": [
                "empty",
                "The text we will show you was already classified as political in a previous classification task.",
                "The text we will show you was already classified as political in a previous classification task.",
                "The text we will show you was already classified as a political attack in a previous classification task.",
                "empty",
                "The text we will show you was already classified as political in a previous classification task.",
                "The text we will show you was already classified as political in a previous classification task.",
                "The text we will show you was already classified as a political attack in a previous classification task.",
            ],
            "Block_C_Definition": [
                "empty",
                "empty",
                "empty",
                "empty",
                "Political texts contain information about political developments, political actors or political topics, be they on an international, national or local level. This includes references to federal organizations, branches of government political programs but also criticism of political actors & content. This also includes content on financial institutions and national economic developments, but not references to individual share prices or companies.",
                "A presentation is characterized by the emphazise and praise of one's own political ideas, positions, opinions, achievements, work or plans or the political ideas, positions, opinions, achievements, work or plans of political allies (members of one's own party) in a neutral or positive tone. The decisive factor for a presentation is not the tonality, but the fact, that the candidate talks about a topic without attacking an opponent to highlight his view of the world. It is not an exclusion criterion for a presentation in a text that an opponent is attacked or criticized in the same text.",
                'A attack is defined by criticism of the political opponent with a generally critical or negative, but not necessarily unobjective tone recognizable in the text. It is not an exclusion criterion for an attack label if the same text speaks positively about one\'s own plans, work, position or achievements (self-presentation). A candidate or party, but also the entire opposition, government or other organizations (e.g. NGOs, movements, central banks, etc.) as well as broad abstract groups (e.g. "conservatives", "progressives", "the extreme left") can be named as opponents/targets of the attack. An attack can adress political positions, achievements, plans or work of the critizied opponents.',
                'Depending on the context, the target of an attack would be usually a specific political person or party. Sometimes a group such as "the government", "the opposition" or "GroKo" or even abstract political groups such as "the left-wing forces" or "the authoritarians", "the populists" can also be the target of an attack. Combinations of groups, e.g. a candidate and his party or several candidates or several parties, can also be the target of an attack. In the context of this texts, the target could also be mentioned via the "@" followed by the twitter handle like "@target". But be cautious, because these texts often start with a longer list of multiple @mentions, which must not be the actual target of the attach but only accounts that are part of the ongoing conversation. It is also important, that political topics cannot be the target. Also, sometimes the person or party attacked has a hashtag before his name like "#name".',
            ],
            "Block_D_Task": [
                'Let us think step by step. Please determine whether the following text is political: "{text}"',
                'Let us think step by step. Please determine whether the following text contains a political presentation: "{text}"',
                'Let us think step by step. Please determine whether the following text contains a political attack: "{text}"',
                'Let us think step by step. Please identify the target or targets of the attack in the following text: "{text}"',
                'Let us think step by step. Please etermine whether the following text is political: "{text}"',
                'Let us think step by step. Please etermine whether the following text contains a political presentation: "{text}"',
                'Let us think step by step. Please etermine whether the following text contains a political attack: "{text}"',
                'Let us think step by step. Please identify the target or targets of the attack in the following text: "{text}"',
            ],
            "Block_E_Structure": [
                'Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political.',
                'Mark the identified category presentation in a dictionary with key "presentation" and value "1" if you identify any sentence or hashtag in the text as presentation, and value "0" if you cannot identify any presentation within the text.',
                'Mark the identified category attack in a dictionary with key "attack" and value "1" if you identify any sentence or hashtag in the text as attack, and value "0" if you cannot identify any attack within the text.',
                'Write all identified targets of this attack in a dictionary with key "target" and a list as value ["target1", "target2", …]. If you cannot identify a target, give back an empty python list element.',
                'Mark the identified category political in a dictionary with key "political" and value "1" if you identify any sentence or hashtag in the text as political, and value "0" if you identify the text as non-political.',
                'Mark the identified category presentation in a dictionary with key "presentation" and value "1" if you identify any sentence or hashtag in the text as presentation, and value "0" if you cannot identify any presentation within the text.',
                'Mark the identified category attack in a dictionary with key "attack" and value "1" if you identify any sentence or hashtag in the text as attack, and value "0" if you cannot identify any attack within the text.',
                'Write all identified targets of this attack in a dictionary with key "target" and a list as value ["target1", "target2", …]. If you cannot identify a target, give back an empty python list element.',
            ],
            "Block_F_Output": [
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
                "Do not hallucinate and do not provide any explanation for your decision.",
            ],
        }
    )


def get_demo_stop_conditions():
    """Get the stop conditions for the classification."""
    return {
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


def setup_demo_model(interactive=True):
    """
    Set up the model for classification.

    Args:
        interactive: If True, prompt the user for model choice. If False, default to Ollama.

    Returns:
        tuple: (client, model_name, model_family)
    """
    if not interactive:
        # Default to gemma2 for non-interactive mode
        print("Initializing default model 'gemma2:9b' locally...")
        client = initialize_model("ollama", "gemma2:9b")
        return client, "gemma2:9b", "gemma"

    # Ask user which model to use
    print("\n\n🤖 Model Selection")
    print("-------------------------")
    print(
        "1. Run gemma2:2b locally (basic model: very weak, but small and fast (suitable for low-end PCs) - requires Ollama)"
    )
    print(
        "2. Run gemma2:9b locally (intermediate model: better accuracy than gemma2:2b, but slower - requires Ollama)"
    )
    print(
        "3. Run phi-4:latest locally (advanced model: usually high quality, but significantly slower - requires Ollama)"
    )
    print(
        "4. Use GPT-4o-mini via OpenAI API (premium model: best quality, but requires an API key and incurs costs - requires OPENAI API KEY)"
    )
    print("5. Stop Demo")
    print("-------------------------")

    while True:
        choice = input("\nEnter your choice (1, 2, 3, 4, or 5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            break
        print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")

    # Initialize the model based on user choice
    if choice == "1":
        print("\n\n🤖 Model Initialization")
        print("-------------------------")
        print("Initializing model 'gemma2:2b' locally...")
        print("This model is very weak but small and fast, suitable for low-end PCs.\n")
        client = initialize_model("ollama", "gemma2:2b")
        model_name = "gemma2:2b"
        model_family = "gemma"
    elif choice == "2":
        print("\n\n🤖 Model Initialization")
        print("-------------------------")
        print("Initializing model 'gemma2:9b' locally...")
        print("This model has better accuracy than gemma2:2b but runs slower.\n")
        client = initialize_model("ollama", "gemma2:9b")
        model_name = "gemma2:9b"
        model_family = "gemma"
    elif choice == "3":
        print("\n\n🤖 Model Initialization")
        print("-------------------------")
        print("Initializing model 'phi4:latest' locally...")
        print(
            "This model provides usually high quality but is significantly slower than gemma2:9b.\n"
        )
        client = initialize_model("ollama", "phi4:latest")
        model_name = "phi4:latest"
        model_family = "phi"
    elif choice == "4":
        # OpenAI API setup
        print("\n\n🔑 OpenAI API setup")
        print("--------------------------------")
        setup_openai_api_key()

        # Continue with OpenAI initialization
        print("🤖 Initializing model 'gpt-4o-mini' via API...")
        print(
            "This model provides the best quality but requires an API key and incurs costs.\n"
        )
        client = initialize_model("openai", "gpt-4o-mini")
        model_name = "gpt-4o-mini"
        model_family = "openai"
    else:
        # Stop Demo
        print("\n🛑 Demo stopped by user.")
        exit()

    print("-------------------------")
    return client, model_name, model_family


def ask_to_display_label_structure(labels, label_values, stop_condition):
    """
    Ask the user if they want to display the hierarchical structure of the labels.

    Args:
        labels (list): List of labels to classify.
        label_values (dict): Possible values the labels can receive.
        stop_condition (dict): Stop conditions for the hierarchical structure.

    Returns:
        None
    """
    print("\n🔍 Display Label Hierarchical Structure")
    print("----------------------------------------")
    print(
        "Would you like to display the hierarchical structure of the labels before the classification run?"
    )
    print("1. Yes, show me the label structure.")
    print("2. No, proceed directly to the classification.")
    print("----------------------------------------")

    while True:
        user_choice = input("Enter your choice (1 or 2): ").strip()
        if user_choice in ["1", "2"]:
            break
        print("❌ Invalid choice. Please enter 1 or 2.")

    if user_choice == "1":
        print("\n📊 Displaying the hierarchical structure of the labels:")
        display_label_flowchart(
            valid_keys=labels, stop_conditions=stop_condition, label_codes=label_values
        )
        print("\n✅ Hierarchical structure displayed (scroll up to see it).")

        # Ask the user if they want to proceed
        print("\n➡️  Do you want to proceed?")
        print("1. Yes")
        print("2. No (Exit)")
        print("----------------------------------------")

        while True:
            proceed_choice = input("Enter your choice (1 or 2): ").strip()
            if proceed_choice in ["1", "2"]:
                break
            print("❌ Invalid choice. Please enter 1 or 2.")

        if proceed_choice == "2":
            print("\n🛑 Demo stopped by user.")
            exit()
        else:
            print("\n✅ Proceeding with the next step.")
    else:
        print("\n✅ Skipping the display of the hierarchical structure.")


def get_demo_text_selection(interactive=True):
    """
    Get the text and context for classification.

    Args:
        interactive: If True, prompt user for text choice. If False, use default text.

    Returns:
        tuple: (text, context_naive, context_with_definitions)
    """
    default_text = (
        "@This federal government is doing great harm to the industrial location."
    )
    default_context = {
        "lang": "English",
        "author": "Reinhard Houben",
        "party": "FDP",
        "platform": "Twitter",
        "date": "2021-09-03",
    }

    if not interactive:
        return default_text, default_context

    # Ask user which text to use
    print("\n📝 Text Selection")
    print("-------------------------")
    print(
        "1. Use default German text (translated) and context from German Federal Election 2021"
    )
    print("2. Enter your own text (without context information needed).")
    print("3. Stop Demo")
    print("-------------------------")

    while True:
        text_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if text_choice in ["1", "2", "3"]:
            break
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

    if text_choice == "1":
        # Return default text and context
        return default_text, default_context
    elif text_choice == "2":
        # User provides their own text
        user_text = input("\nEnter the text you want to analyze: ").strip()

        # For user-provided text, we use minimal context
        minimal_context = {
            "lang": "unknown",
            "author": "unknown",
            "party": "unknown",
            "platform": "unknown",
            "date": "unknown",
        }
        return user_text, minimal_context
    else:
        # Stop Demo
        print("\n🛑 Demo stopped by user.")
        exit()


def print_stop_conditions(
    stop_conditions: dict[int, dict[str, any]], valid_keys: list[str]
):
    """
    Print a human-readable explanation of the stop conditions.

    Args:
        stop_conditions: dictionary of stop conditions
        valid_keys: list of valid keys in order
    """
    # Create a mapping from condition index to label name
    key_to_label = {i: label for i, label in enumerate(valid_keys)}

    # Track which conditions we've already printed to avoid duplicates
    printed_conditions = set()

    for condition_key, condition_data in stop_conditions.items():
        condition_index = condition_data.get("condition")
        if condition_index is not None and 0 <= condition_index < len(valid_keys):
            label = key_to_label.get(condition_index, f"Label {condition_index}")
            blocked_keys = condition_data.get("blocked_keys", [])

            # Create a unique key for this condition to avoid duplicates
            condition_key = (label, tuple(sorted(blocked_keys)))

            if condition_key in printed_conditions:
                continue

            printed_conditions.add(condition_key)

            # Show different messages based on the expected condition value
            print(f"• If {label} = 0 (negative), the following steps are skipped:")
            for blocked in blocked_keys:
                print(f"  - {blocked}")
            print()


def parse_dependencies(
    stop_conditions: dict[int, dict[str, any]], valid_keys: list[str]
) -> dict[str, list[str]]:
    """
    Parse the stop conditions into a dependencies dictionary.

    Args:
        stop_conditions: dictionary of stop conditions
        valid_keys: list of valid keys in order

    Returns:
        dictionary mapping each label to labels that depend on it
    """
    # Initialize dependencies with all valid keys
    dependencies = {}
    for key in valid_keys:
        dependencies[key] = {"blocked_if_zero": []}

    # Update dependencies based on stop conditions
    for condition_key, condition_data in stop_conditions.items():
        # Map condition_key to the corresponding label if in range
        if 0 <= condition_data.get("condition", -1) < len(valid_keys):
            condition_label = valid_keys[condition_data.get("condition")]
            dependencies[condition_label]["blocked_if_zero"] = condition_data.get(
                "blocked_keys", []
            )

    return dependencies
