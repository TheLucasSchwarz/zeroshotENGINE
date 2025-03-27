# Tutorial: Get started with the zeroshot-engine
Here's a minimal example using the demo functions from your `zeroshot_engine` package. This example will demonstrate how to set up and run a zero-shot classification using the provided functions.

We have two examples:

* **Single Text Classification-Tutorial**: This example is good for setting up all the parameters.

* **Parallel Processing of a DataFrame**: This example processes a DataFrame with 10 example texts in parallel, which is the typical scenario for scientific investigation (with more texts obviously).

# Step-by-Step Tutorial

## Single-Text-Scenario

### 1. Set Up a clean Python Environment
Ensure you have the necessary dependencies installed and to prevent conflicts, install a fresh virtual environment. If you haven't already, create and activate a virtual environment, then install the required packages:

```bash
pip install virtualenv
```

#### On Linux/Mac
```bash
python3 -m venv test_venv
source test_venv/bin/activate
pip install zeroshot-engine
```

#### On Windows
```bash
python3 -m venv test_venv
test_venv\Scripts\activate
pip install zeroshot-engine
```

### 2. Create the Tutorial Script
Create a new Python script, `tutorial_example.py`, in your project directory and copy the following code into it:

```python
import os
import time

from zeroshot_engine import (
    initialize_model,
    iterative_double_zeroshot_classification,
    set_zeroshot_parameters,
    get_demo_prompt_structure,
    get_demo_stop_conditions,
    display_label_flowchart,
)

# Set environment variables (if you have CUDA and want to use a local LLM)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_CUDA"] = "1"

# Load the demo prompts (you can download the demo prompt table as xlsx under https://github.com/TheLucasSchwarz/zeroshotENGINE/blob/master/docs/prompt_structure.xlsx)
prompts_df = get_demo_prompt_structure()

# Define all relevant columns from the demo prompts table
prompt_blocks_columns = [
    "Block_A_Introduction",
    "Block_B_History",
    "Block_C_Definition",
    "Block_D_Task",
    "Block_E_Structure",
    "Block_F_Output",
]

# Define all labels we want to classify
labels = ["political", "presentation", "attack", "target"]

# Define the corresponding row of the prompt table for each label
label_prompt_ids = [
    "P1_political_naive",
    "P2_presentation_naive",
    "P3_attack_naive",
    "P4_target_naive",
]

# Define the possible values the labels can receive
label_values = {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}

# Define the possible output_types each specific label can have
output_type_labels = {
    "political": "numeric",
    "presentation": "numeric",
    "attack": "numeric",
    "target": "list",
}

# Define the mismatch strategy per output_type
combining_strategy_output = {
    "numeric": "optimistic",
    "list": "union",
}

# Get the stop condition from the demo project
stop_condition = get_demo_stop_conditions()

# Display the defined hierarchical structure
display_label_flowchart(
    valid_keys=labels, stop_conditions=stop_condition, label_codes=label_values
)

# Choose the model you want to use.
client = initialize_model("ollama", "gemma2:2b")
# client = initialize_model("openai", "gpt-4o-mini")

# Set zero-shot parameters# Full hierarchical example with all parameters and specialized settings
parameters = set_zeroshot_parameters(
    model_family="gemma",  # alternatively: openai
    client=client,
    model="gemma2:2b",  # alternatively: gpt-4o-mini
    prompt_build=prompts_df,
    prompt_ids_list=label_prompt_ids,
    prompt_id_col="Prompt-ID",
    prompt_block_cols=prompt_blocks_columns,
    valid_keys=labels,
    label_codes=label_values,
    stop_conditions=stop_condition,
    output_types=output_type_labels,
    validate=True,
    combining_strategies=combining_strategy_output,
    max_retries=2,
    feedback=True,
)

# Example text for classification (the politician and party this example is chosen from is was chosen at random only for demonstration purposes)
text = "'Für uns ist klar: Staatsschulden sind eine Gefahr für die Wirtschafts- und Währungsunion'. 'Das Schuldenmachen zur neuen Staatsphilosophie zu verklären' wird mit uns als #FDP nicht gehen, werter @OlafScholz! #Regierungserklaerung  #Bundestag  @fdpbt"

# Perform classification
start_time = time.time()  # Record start time

result = iterative_double_zeroshot_classification(
    text=text,
    parameter=parameters,
    context={
        "lang": "German",
        "author": "Gerald Ullrich",
        "platform": "Twitter",
        "date": "2021-05-15",
        "party": "FDP",
    },
)

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Calculate elapsed time

print(f"Time taken: {elapsed_time:.2f} seconds")
print(result)
```

### 3. Run the Tutorial Script
Run the script to see the output:

```bash
python tutorial_example.py
```

## Explanation
* Environment Setup: The script sets up the environment variables and adds the project root to the Python path.
* Demo Data: It uses demo functions to get the prompt structure and stop conditions.
Model Initialization: The setup_demo_model function initializes the model.
* Parameter Setup: The set_zeroshot_parameters function sets up the parameters for zero-shot classification.
* Classification: The iterative_double_zeroshot_classification function performs the classification on the example text.
* Output: The script prints the time taken and the classification result.

This minimal example demonstrates how to use the demo functions to set up and run a zero-shot classification with your `zeroshot_engine` package.



## Parallel-Text-Scenario
Here's a second subvariant that demonstrates how to process an entire DataFrame in parallel using the `parallel_iterative_double_zeroshot_classification` function.

### 0. If not already done, set up the environment:

```bash
python3 -m venv test_venv
source test_venv/bin/activate
pip install zeroshot-engine
```


### 1. Create the Parallel Processing Script
Create a new Python script, `parallel_example.py`, in your project directory and copy the following code into it:

```python
import os
import pandas as pd

# Set environment variables (if you have CUDA and want to use a local LLM)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_CUDA"] = "1"

from zeroshot_engine import (
    initialize_model,
    parallel_iterative_double_zeroshot_classification,
    set_zeroshot_parameters,
    get_demo_prompt_structure,
    display_label_flowchart,
)

# Load the demo prompts (you can download the demo prompt table as xlsx under https://github.com/TheLucasSchwarz/zeroshotENGINE/blob/master/docs/prompt_structure.xlsx)
prompts_df = get_demo_prompt_structure()

# Define all relevant columns from the demo prompts table
prompt_blocks_columns = [
    "Block_A_Introduction",
    "Block_B_History",
    "Block_C_Definition",
    "Block_D_Task",
    "Block_E_Structure",
    "Block_F_Output",
]

# Define all labels we want to classify
labels = ["political", "presentation", "attack", "target"]

# Define the corresponding row of the prompt table for each label
label_prompt_ids = [
    "P1_political_with_definition",
    "P2_presentation_with_definition",
    "P3_attack_with_definition",
    "P4_target_with_definition",
]

# Define the possible values the labels can receive
label_values = {"present": 1, "absent": 0, "non-coded": 8, "empty-list": []}

# Define the possible output_types each specific label can have
output_type_labels = {
    "political": "numeric",
    "presentation": "numeric",
    "attack": "numeric",
    "target": "list",
}

# Define the mismatch strategy per output_type
combining_strategy_output = {
    "numeric": "optimistic",
    "list": "union",
}

# Get the stop condition from the demo project
stop_condition = {
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

# Display the defined hierarchical structure
display_label_flowchart(
    valid_keys=labels, stop_conditions=stop_condition, label_codes=label_values
)

# Choose the model you want to use
client = initialize_model(api="openai", model="gpt-4o-mini")

# Set zero-shot parameters# Full hierarchical example with all parameters and specialized settings
parameters = set_zeroshot_parameters(
    model_family="openai",  # alternatively: gemma
    client=client,
    model="gpt-4o-mini",  # alternatively: gemma2:9b
    prompt_build=prompts_df,
    prompt_ids_list=label_prompt_ids,
    prompt_id_col="Prompt-ID",
    prompt_block_cols=prompt_blocks_columns,
    valid_keys=labels,
    label_codes=label_values,
    stop_conditions=stop_condition,
    output_types=output_type_labels,
    validate=True,
    combining_strategies=combining_strategy_output,
    max_retries=2,
    feedback=False,
)

# Create the example DataFrame (generated by an LLM, no actual posts.)
df = pd.DataFrame(
    {
        "text": [
            "We need to invest more in renewable energy to combat climate change. The government continues to ignore scientific evidence.",
            "Today I announced our new healthcare plan that will benefit all citizens. Together we can build a better future.",
            "Our economic policies have created thousands of jobs. Unlike @OppositionParty who only raised taxes during their term.",
            "Yesterday I was swimming on the beach.",
            "Our infrastructure plan will create jobs and improve our roads and bridges. It's time to invest in our future.",
            "The opposition's stance on healthcare is dangerous and irresponsible. We need to protect our citizens.",
            "Climate change is the biggest threat to our planet. We must act now to reduce emissions and invest in renewable energy.",
            "Our tax cuts have benefited millions of families. The democrats wants to raise taxes and hurt our economy.",
            "We need to support our military and veterans. They have sacrificed so much for our country.",
            "The opposition's plan to defund the police is reckless. We need to ensure the safety of our communities.",
        ],
        "lang": ["English"] * 10,
        "author": [
            "Jane Smith",
            "John Doe",
            "Sarah Johnson",
            "Michael Brown",
            "Emily Davis",
            "David Wilson",
            "Laura Martinez",
            "James Anderson",
            "Patricia Thomas",
            "Robert Jackson",
        ],
        "platform": [
            "Twitter",
            "Facebook",
            "Twitter",
            "Twitter",
            "Facebook",
            "Twitter",
            "Facebook",
            "Twitter",
            "Facebook",
            "Twitter",
        ],
        "date": [
            "2023-06-15",
            "2023-07-22",
            "2023-08-10",
            "2023-09-01",
            "2023-09-15",
            "2023-10-05",
            "2023-10-20",
            "2023-11-01",
            "2023-11-15",
            "2023-12-01",
        ],
        "party": [
            "Green Party",
            "Democrats",
            "Republicans",
            "Democrats",
            "Republicans",
            "Green Party",
            "Democrats",
            "Republicans",
            "Green Party",
            "Democrats",
        ],
    }
)

# Process the entire DataFrame in parallel
results_df = parallel_iterative_double_zeroshot_classification(
    data=df,  # Pass the example DataFrame with text and context columns
    parameter=parameters,
    context=[
        "lang",
        "author",
        "platform",
        "date",
        "party",
    ],  # Use all context columns from df
    num_workers=4,  # Adjust based on your system's capabilities
)

# The returned results_df will have all original columns plus classification results
selected_columns = ["text"] + labels
print(results_df[selected_columns].head(n=10))
```

### 2. Run the Parallel Processing Script
Run the script to see the output:

```bash
python parallel_example.py
```
### Explanation
* Environment Setup: The script sets up the environment variables and adds the project root to the Python path.
* Demo Data: It uses demo functions to get the prompt structure and stop conditions.
* Model Initialization: The initialize_model function initializes the model.
* Parameter Setup: The set_zeroshot_parameters function sets up the parameters for zero-shot classification.
* Parallel Classification: The parallel_iterative_double_zeroshot_classification function processes the entire DataFrame in parallel.
* Output: The script prints the first few rows of the resulting DataFrame, which includes the original columns plus the classification results.

This second subvariant demonstrates how to process an entire DataFrame in parallel using the parallel_iterative_double_zeroshot_classification function with your `zeroshot_engine package`.
