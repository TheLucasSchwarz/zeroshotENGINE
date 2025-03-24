
### Model Recommendations and Performance

#### Benchmark Disclaimer

The benchmark results presented in this document are derived from an ongoing research project examining negative campaigning in political communication. This research is being developed as part of an unpublished manuscript:

> Schwarz, L. (2025). “The User Made Me Do It! Dynamic Negative Campaigning in the Digital Sphere: Evidence from Candidates' Twitter Communication during the 2021 German Federal Election Year”. *Unpublished manuscript*, Johannes Gutenberg University, Mainz, Germany.

I am currently working on this manuscript, which analyzes political communication patterns during election campaigns, with particular focus on negative campaigning strategies. The hierarchical classification system implemented in zeroshotENGINE was originally developed for and tested extensively in this research context.

**Important limitations to consider:**

* The current evaluation is based on a benchmark dataset of 500 randomly selected tweets from political candidates during the 2021 German Federal Election
* This sample size, while informative, is relatively small for comprehensive model evaluation
* I plan to expand this benchmark to 1000-1500 tweets in the near future to strengthen the reliability of our findings
* This document will be updated with these expanded results once available
* Performance metrics may change with the larger benchmark dataset

The current benchmark dataset includes tweets classified across multiple dimensions of political communication including presentation strategies, attack presence, and target identification. This multi-label, hierarchical dataset provides an initial testing ground for comparing different LLM implementations and prompt engineering approaches, but findings should be considered preliminary until validated with the expanded dataset.

The early results demonstrate the potential practical application of zeroshotENGINE in real-world research scenarios and suggest the effectiveness of smaller, well-tuned models like GPT-4o mini for specialized classification tasks when provided with optimized prompts. However, further validation is required before drawing definitive conclusions about model performance hierarchies.

#### External Validation
In addition to our internal benchmark dataset, we have externally validated these results using training data from Nai & Petkevic (2022) focused on US Primary election tweets. This cross-validation with an independent dataset yielded similar performance values across models, reinforcing the reliability of our findings despite the relatively small size of our primary benchmark dataset. The consistency between our German election data results and the US election data validation provides additional confidence in the model's cross-cultural and cross-contextual applicability for political communication classification tasks.

This external validation represents an important step in establishing the generalizability of both our hierarchical classification approach and the relative performance differences between models, even as we work to expand our primary benchmark dataset to 1000-1500 tweets in the near future.

#### Model Recommendations

Based on extensive testing in a Negative Campaiging hierarchical classification tasks, the following models have shown strong performance:

* **Top Performance**: 
  * GPT-4o mini (via API)
    * Provides the best balance of accuracy and cost
    * Surprisingly outperformed the larger GPT-4o model in our benchmark tests
    * Recommended for production use cases or research requiring high reliability
    * Note: Performance metrics shown are with optimized prompts after iterative prompt engineering

* **Strong Alternatives**: 
  * GPT-4o (via API)
    * Despite being a larger model, performed slightly worse than GPT-4o mini on our benchmarks
    * Higher cost with no accuracy benefit for this specific classification task
  * Self-hosted via Google Colab on A100 GPU:
    * Phi-4
    * Gemma2 9B
  * These models perform moderately well but show some accuracy reduction compared to GPT-4o mini

#### Benchmark Performance

The following metrics are based on a benchmark dataset of 500 tweets analyzed for negative campaigning across 6 classification categories. Cost and emissions are extrapolated for a scenario of 250,000 tweets:

| Model | Prompt | Company | Access | Size | Weighted Area under ROC Curve | Macro Area under ROC Curve | Weighted F1 Score | Macro F1 Score | Est. Cost (250K tweets) | Est. Inference Time in Hours (250.000 tweets) | Est. Inference CO₂ Emissions* (250K tweets) |
|-------|----------|----------|----------|--------------------|-----------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| GPT 4o mini | naive | OpenAI | API | ~ 8 B | 0.76 | 0.78 | 0.71 | 0.73 | not estimated | not estimated | not estimated |
| GPT 4o mini | optimized on GPT 4o mini | OpenAI | API | ~ 8 B | 0.88 | 0.85 | 0.84 | 0.74 | ~ 105 € | 96.9 h | 24.81 kg CO₂e |
| GPT 4o | optimized on GPT 4o mini | OpenAI | API | ? B | 0.85 | 0.80 | 0.81 | 0.70 | ~ 3655 € | 101.7 h | ? |
| gemma2 (9B) | optimized on GPT 4o mini | Google | Self-hosted | 9 B | 0.87 | 0.85 | 0.80 | 0.65 | ~ 100 € | 108.2 h | 26.4 kg CO₂e |
| mistral (7B) | optimized on GPT 4o mini | Mistral AI | Self-hosted | 7 B | 0.81 | 0.75 | 0.77 | 0.62 | ~ 50  € | 53.6 h | 13.08 kg CO₂e |
| Llama3.1 (8B) | optimized on GPT 4o mini | Meta | Self-hosted | 8 B | 0.82 | 0.82 | 0.78 | 0.64 | ~ 72 € | 77.6 h | 18.93 kg CO₂e |
| phi4 (14B) | optimized on GPT 4o mini | Microsoft | Self-hosted | 14 B | 0.86 | 0.86 | 0.81 | 0.69 | ~ 121 € | 130.1 h | 31.74 kg CO₂e |
| Deepseek R1 (8B) | optimized on GPT 4o mini | Deepseek | Self-hosted | 8 B | 0.81 | 0.76 | 0.77 | 0.62 | ~ 1361 € | 1464.4 h | 357.31 kg CO₂e |

*CO₂ estimates based on Lacoste et al. (2019): https://mlco2.github.io/impact/?#compute  
*Comparison: A typical gasoline car produces approximately 19.7 kg CO₂ per 100 km (based on 7.1 liter consumption) (UBA, ifeu)

> **Important Note**: Performance is highly dependent on prompt engineering and model selection and the task. The results for GPT-4o mini reflect extensive prompt optimization and refinement for the specific negative campaigning task. zeroshotENGINE provides a structured framework for implementing hierarchical classification but does not guarantee specific performance levels. Careful prompt design, including clear task instructions and comprehensive category definitions, significantly impacts classification accuracy. We recommend conducting smaller validation studies with different prompt variations before scaling to large datasets.

#### GPT 4o mini detailed evaluation (optimized)

| Label | Area under ROC Curve | F1 Score (Absence of Dimension) | F1 Score  (Presence of Dimension) | Conditional Presence-F1 (attack=1) | Mismatch Rate | Support |
|-------|----------|----------|----------|--------------------|-----------------|-----------------------|
| political | 0.92 | 0.86 | 0.95 |  | 0.01 | 379 |
| presentation | 0.82 | 0.87 | 0.79 |  | 0.03 | 211 |
| attack | 0.88 | 0.9 | 0.77 |  | 0.02 | 130 |
| policyAttack | 0.9 | 0.92 | 0.74 | 0.9 | 0.02 | 91 |
| personalAttack | 0.72 | 0.97 | 0.5 | 0.58 | 0.01 | 34 |
| harshLang | 0.88 | 0.98 | 0.68 | 0.82 | 0.01 | 31 |
| target | - | - | - | - | - | - |

*Note: Average metrics from 8 runs with identical parameters and prompts (GPT 4o mini)


### References

Schwarz, L. (2025). “The User Made Me Do It! Dynamic Negative Campaigning in the Digital Sphere: Evidence from Candidates' Twitter Communication during the 2021 German Federal Election Year”. *Unpublished manuscript*, Johannes Gutenberg University, Mainz, Germany.

Petkevic, Vladislav, and Alessandro Nai. 2022. “Political Attacks in 280 Characters or Less: A New Tool for the Automated Classification of Campaign Negativity on Social Media.” *American Politics Research* 50 (3): 279–302. https://doi.org/10.1177/1532673X211055676.
