
## Demo Logic Showcase - Simple Negative Campaigning Classifier 

The demo prompts used in `zeroshot_engine` are designed to showcase the capabilities of HDZSC with different prompt structures. The `get_demo_prompt_structure` function creates a DataFrame containing these prompts. Here's how the prompts are structured:

### The Hierarchical Flowchart
The following chart was created using `display_label_flowchart(valid_keys, stop_conditions, label_codes)`

```
==============================================================
          ZEROSHOTENGINE LABEL DEPENDENCY FLOWCHART           
==============================================================

 [POLITICAL]
 ├─ if political = 1:
 │   [PRESENTATION]
 │   [ATTACK]
 │   ├─ if attack = 1:
 │   │   [TARGET]
 │   │   │
 │   │   ▼
 │   │   STOP
 │   └─ if attack = 0:
 │       → Skip: target
 │       STOP
 └─ if political = 0:
     → Skip: presentation, attack, target
     STOP

--------------------------------------------------------------
                 STOP CONDITIONS EXPLANATION                  
--------------------------------------------------------------
  If political = 0 (absent), the following steps are skipped:
    - presentation
    - attack
    - target

  If attack = 0 (absent), the following steps are skipped:
    - target

--------------------------------------------------------------
                            LEGEND                            
--------------------------------------------------------------
 - 1 (present): Proceeds to the next classification step
 - 0 (absent): Skips one or more subsequent classifications

 LABEL CODES 
    present: 1
    absent: 0
    non-coded: 8
    empty-list: []

--------------------------------------------------------------
```

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
result = iterative_zeroshot_classification(
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
apply_iterative_zeroshot_classification(
    data=df, # Contains texts-strings to classify in "text" column with context information in context columns (see below).
    parameter=parameters,
    context=["lang", "author", "platform", "date"]  # These column names from df will be used.
    # They should be stored in the same row as the corresponding "text"-string.
)

# Or in parallelization for faster processing
parallel_iterative_zeroshot_classification(
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
    double_shot=True,
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

# Full hierarchical example with all parameters and specialized settings, but with no double shot
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
    double_shot=False,
    validate=False,
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
