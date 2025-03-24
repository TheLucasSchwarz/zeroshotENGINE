import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_auc_score


def detailedLabelEvaluationPrint(benchmarkData, IDZSCData, sample_weight=None):
    # Define the categories and their corresponding columns
    categories = [
        "political",
        "presentation",
        "attack",
        "policyAttack",
        "personalAttack",
        "harshLang",
    ]

    # Initialize a list to store the results
    results = []

    # Calculate classification report for each category and variant
    for category in categories:
        y_true = benchmarkData[category]
        y_pred_conservative = IDZSCData[f"{category}_conservative"]
        y_pred_optimistic = IDZSCData[f"{category}_optimistic"]
        y_pred_probabilistic = IDZSCData[f"{category}_probabilistic"]

        report_conservative = classification_report(
            y_true,
            y_pred_conservative,
            output_dict=True,
            zero_division=0,
            sample_weight=sample_weight,
        )
        report_optimistic = classification_report(
            y_true,
            y_pred_optimistic,
            output_dict=True,
            zero_division=0,
            sample_weight=sample_weight,
        )
        report_probabilistic = classification_report(
            y_true,
            y_pred_probabilistic,
            output_dict=True,
            zero_division=0,
            sample_weight=sample_weight,
        )

        print(f"Category: {category} - Conservative")
        print(
            classification_report(
                y_true,
                y_pred_conservative,
                zero_division=0,
                sample_weight=sample_weight,
            )
        )
        print(f"Category: {category} - Optimistic")
        print(
            classification_report(
                y_true, y_pred_optimistic, zero_division=0, sample_weight=sample_weight
            )
        )
        print(f"Category: {category} - Probabilistic")
        print(
            classification_report(
                y_true,
                y_pred_probabilistic,
                zero_division=0,
                sample_weight=sample_weight,
            )
        )

        for label in report_conservative:
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                results.append(
                    {
                        "Category": category,
                        "Variant": "Conservative",
                        "Label": label,
                        "Precision": report_conservative[label]["precision"],
                        "Recall": report_conservative[label]["recall"],
                        "F1 Score": report_conservative[label]["f1-score"],
                        "Support": report_conservative[label]["support"],
                    }
                )
                results.append(
                    {
                        "Category": category,
                        "Variant": "Optimistic",
                        "Label": label,
                        "Precision": report_optimistic[label]["precision"],
                        "Recall": report_optimistic[label]["recall"],
                        "F1 Score": report_optimistic[label]["f1-score"],
                        "Support": report_optimistic[label]["support"],
                    }
                )
                results.append(
                    {
                        "Category": category,
                        "Variant": "Probabilistic",
                        "Label": label,
                        "Precision": report_probabilistic[label]["precision"],
                        "Recall": report_probabilistic[label]["recall"],
                        "F1 Score": report_probabilistic[label]["f1-score"],
                        "Support": report_probabilistic[label]["support"],
                    }
                )

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(results)

    return results_df


# Function to calculate metrics
def calculateMetrics(y_true, y_pred, categories):
    metrics = {}
    for i, category in enumerate(categories):
        support = int((y_true[:, i] == 1).sum())
        f1_absence = f1_score(y_true[:, i], y_pred[:, i], pos_label=0, average="binary")
        f1_presence = f1_score(
            y_true[:, i], y_pred[:, i], pos_label=1, average="binary"
        )
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        metrics[category] = {
            "F1 Score (Absence of Dimension)": f1_absence,
            "F1 Score (Presence of Dimension)": f1_presence,
            "Area under ROC Curve": auc,
            "Support": support,
        }

    # Calculate weighted and macro averages
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    roc_auc_weighted = roc_auc_score(y_true, y_pred, average="weighted")
    roc_auc_macro = roc_auc_score(y_true, y_pred, average="macro")
    hamming = hamming_loss(y_true, y_pred)

    metrics["overall"] = {
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "roc_auc_macro": roc_auc_macro,
        "hamming_loss": hamming,
    }

    return metrics


# Function to calculate and print metrics for all strategies and tweet versions
def generalEvaluation(tweetsHuman1, tweetsHuman2, tweetsIDZSC, categories):
    strategies = ["conservative", "optimistic", "probabilistic"]
    tweet_versions = [tweetsHuman1, tweetsHuman2]

    all_metrics = []  # Liste um alle Metriken zu speichern

    for strategy in strategies:
        IDZSCcolumns_with_strategy = [
            f"{category}_{strategy}" for category in categories
        ]
        print(IDZSCcolumns_with_strategy)
        tweetsIDZSC_subset = tweetsIDZSC[IDZSCcolumns_with_strategy]

        for i, tweetsHuman in enumerate(tweet_versions, start=1):
            tweetsHuman_subset = tweetsHuman[categories]

            y_true = np.array(tweetsHuman_subset)
            y_pred = np.array(tweetsIDZSC_subset)

            metrics = calculateMetrics(y_true, y_pred, categories)

            print(f"Metrics for strategy: {strategy}, tweetsHuman version: {i}")
            for category, metric in metrics.items():
                if category == "overall":
                    print("\nOverall Metrics:")
                    print(f"  Weighted F1 Score: {metric['f1_weighted']:.2f}")
                    print(f"  Macro F1 Score: {metric['f1_macro']:.2f}")
                    print(
                        f"  Weighted Area under ROC Curve: {metric['roc_auc_weighted']:.2f}"
                    )
                    print(
                        f"  Macro Area under ROC Curve: {metric['roc_auc_macro']:.2f}"
                    )
                    print(f"  Hamming Loss: {metric['hamming_loss']:.2f}")
                else:
                    print(f"\nCategory: {category}")
                    for metric_name, value in metric.items():
                        print(f"  {metric_name}: {value:.2f}")
            print()

            # Metriken in einem Dictionary speichern
            for category, metric in metrics.items():
                if category == "overall":
                    all_metrics.append(
                        {
                            "strategy": strategy,
                            "tweetsHuman_version": i,
                            "category": "overall",
                            "metric": "Weighted F1 Score",
                            "value": metric["f1_weighted"],
                        }
                    )
                    all_metrics.append(
                        {
                            "strategy": strategy,
                            "tweetsHuman_version": i,
                            "category": "overall",
                            "metric": "Macro F1 Score",
                            "value": metric["f1_macro"],
                        }
                    )
                    all_metrics.append(
                        {
                            "strategy": strategy,
                            "tweetsHuman_version": i,
                            "category": "overall",
                            "metric": "Weighted Area under ROC Curve",
                            "value": metric["roc_auc_weighted"],
                        }
                    )
                    all_metrics.append(
                        {
                            "strategy": strategy,
                            "tweetsHuman_version": i,
                            "category": "overall",
                            "metric": "Macro Area under ROC Curve",
                            "value": metric["roc_auc_macro"],
                        }
                    )
                    all_metrics.append(
                        {
                            "strategy": strategy,
                            "tweetsHuman_version": i,
                            "category": "overall",
                            "metric": "Hamming Loss",
                            "value": metric["hamming_loss"],
                        }
                    )
                else:
                    for metric_name, value in metric.items():
                        all_metrics.append(
                            {
                                "strategy": strategy,
                                "tweetsHuman_version": i,
                                "category": category,
                                "metric": metric_name,
                                "value": value,
                            }
                        )

    # DataFrame aus den Metriken erstellen
    df = pd.DataFrame(all_metrics)

    # DataFrame in eine Excel-Datei exportieren
    df.to_excel("metriken3.xlsx", index=False)
    print("Metriken wurden in 'metriken3.xlsx' gespeichert.")


def calculateMismatchRate(pred1, pred2):
    mismatches = np.sum(pred1 != pred2)
    total = len(pred1)
    mismatch_rate = mismatches / total
    return mismatch_rate


def calculateConditionalF1s(
    df_computerCoded, tweetsHuman, categories, hierarchichalConditions, strategy
):
    results = []
    y_true = tweetsHuman[categories].values
    y_pred = df_computerCoded[[f"{cat}_{strategy}" for cat in categories]].values

    for cond_key, cond_info in hierarchichalConditions.items():
        parent_cat = categories[cond_key]
        sub_cats = cond_info["evalKeys"]

        idx_cond = y_true[:, cond_key] == cond_info["condition"]
        y_true_filtered = y_true[idx_cond, :]
        y_pred_filtered = y_pred[idx_cond, :]

        for sc in sub_cats:
            sc_idx = categories.index(sc)
            f1_cond = f1_score(
                y_true_filtered[:, sc_idx], y_pred_filtered[:, sc_idx], average="binary"
            )
            results.append(
                {
                    "category": sc,
                    "metric": f"Conditional F1 ({parent_cat}=1)",
                    "value": f1_cond,
                }
            )
    return results


def comparativeEvaluation(
    tweetsHuman,
    validated_pairs,
    categories,
    variant,
    benchmark_keys,
    metrics_dir,
    model_names,
):
    strategies = ["conservative", "optimistic", "probabilistic"]
    all_metrics = []

    tweetsHuman_subset = tweetsHuman[categories].values

    for idx, (file_name, df_computerCoded) in enumerate(validated_pairs, start=1):
        # Extract key from the filename
        key = file_name.replace("Validated_NegBTW2021_", "").replace(".xlsx", "")
        label = benchmark_keys.get(key, key)

        # Determine model name from benchmark label using model_names list
        model = label
        for m in model_names:
            if m in label:
                model = m
                break

        for strategy in strategies:
            IDZSCcolumns_with_strategy = [f"{cat}_{strategy}" for cat in categories]

            # Force columns to integer
            df_computerCoded[IDZSCcolumns_with_strategy] = (
                df_computerCoded[IDZSCcolumns_with_strategy]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )

            tweetsIDZSC_subset = df_computerCoded[IDZSCcolumns_with_strategy].values

            metrics = calculateMetrics(
                tweetsHuman_subset, tweetsIDZSC_subset, categories
            )

            # Calculate mismatch rate between pred1 and pred2 for each category
            mismatch_rates = {}
            for category in categories:
                pred1 = df_computerCoded[f"{category}_pred1"]
                pred2 = df_computerCoded[f"{category}_pred2"]
                mismatch_rate = calculateMismatchRate(pred1, pred2)
                mismatch_rates[category] = mismatch_rate

            # Add overall mismatch rate
            overall_mismatch_rate = np.mean(list(mismatch_rates.values()))
            mismatch_rates["overall"] = overall_mismatch_rate

            conditionalF1_scores = calculateConditionalF1s(
                df_computerCoded,
                tweetsHuman,
                categories,
                hierarchichalConditions,
                strategy,
            )

            for category, metric_vals in metrics.items():
                if category == "overall":
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Weighted F1 Score",
                            "value": metric_vals["f1_weighted"],
                            "model": model,
                        }
                    )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Macro F1 Score",
                            "value": metric_vals["f1_macro"],
                            "model": model,
                        }
                    )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Weighted Area under ROC Curve",
                            "value": metric_vals["roc_auc_weighted"],
                            "model": model,
                        }
                    )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Macro Area under ROC Curve",
                            "value": metric_vals["roc_auc_macro"],
                            "model": model,
                        }
                    )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Hamming Loss",
                            "value": metric_vals["hamming_loss"],
                            "model": model,
                        }
                    )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": "overall",
                            "metric": "Mismatch Rate",
                            "value": mismatch_rates["overall"],
                            "model": model,
                        }
                    )
                else:
                    for metric_name, val in metric_vals.items():
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "category": category,
                                "metric": metric_name,
                                "value": val,
                                "model": model,
                            }
                        )
                    all_metrics.append(
                        {
                            "benchmark_label": label,
                            "dataset": idx,
                            "strategy": strategy,
                            "category": category,
                            "metric": "Mismatch Rate",
                            "value": mismatch_rates[category],
                            "model": model,
                        }
                    )

                    for cond_item in conditionalF1_scores:
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "category": cond_item["category"],
                                "metric": cond_item["metric"],
                                "value": cond_item["value"],
                                "model": model,
                            }
                        )

    df = pd.DataFrame(all_metrics)
    # Save the validated DataFrame to a new Excel file
    metrics_file_path = os.path.join(metrics_dir, f"compMetrics{variant}.xlsx")
    df.to_excel(metrics_file_path, index=False)
    print(f"Metriken wurden in 'compMetrics{variant}.xlsx' gespeichert.")


def comparativeEvaluationByParty(
    tweetsHuman,
    validated_pairs,
    categories,
    variant,
    benchmark_keys,
    metrics_dir,
    model_names,
):
    all_metrics = []
    for idx, (file_name, df_computerCoded) in enumerate(validated_pairs, start=1):
        key = file_name.replace("Validated_NegBTW2021_", "").replace(".xlsx", "")
        label = benchmark_keys.get(key, key)

        # Determine model name from benchmark label
        model = label
        for m in model_names:
            if m in label:
                model = m
                break

        for strategy in ["conservative", "optimistic", "probabilistic"]:
            IDZSCcolumns_with_strategy = [f"{cat}_{strategy}" for cat in categories]

            # Force columns to integer
            df_computerCoded[IDZSCcolumns_with_strategy] = (
                df_computerCoded[IDZSCcolumns_with_strategy]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .astype(int)
            )

            for party in df_computerCoded["party"].unique():
                # Subset computer coded data and human data by Party
                df_subset = df_computerCoded[df_computerCoded["party"] == party]
                tweetsHuman_subset = tweetsHuman[tweetsHuman["party"] == party]

                # Skip if one of the subsets is empty
                if df_subset.empty or tweetsHuman_subset.empty:
                    continue

                # Calculate the total count of positive human labels (assumes positive is coded as 1)
                positive_count = (tweetsHuman_subset[categories] == 1).sum().sum()
                # Append the Positive Human Label Count metric
                all_metrics.append(
                    {
                        "benchmark_label": label,
                        "dataset": idx,
                        "strategy": strategy,
                        "Party": party,
                        "category": "overall",
                        "metric": "Support",
                        "value": positive_count,
                        "model": model,
                    }
                )

                tweetsHuman_arr = tweetsHuman_subset[categories].values
                tweetsIDZSC_arr = df_subset[IDZSCcolumns_with_strategy].values

                metrics = calculateMetrics(tweetsHuman_arr, tweetsIDZSC_arr, categories)

                mismatch_rates = {}
                for category in categories:
                    pred1 = df_subset[f"{category}_pred1"]
                    pred2 = df_subset[f"{category}_pred2"]
                    mismatch_rate = calculateMismatchRate(pred1, pred2)
                    mismatch_rates[category] = mismatch_rate
                overall_mismatch_rate = np.mean(list(mismatch_rates.values()))
                mismatch_rates["overall"] = overall_mismatch_rate

                conditionalF1_scores = calculateConditionalF1s(
                    df_subset,
                    tweetsHuman_subset,
                    categories,
                    hierarchichalConditions,
                    strategy,
                )

                for category, metric_vals in metrics.items():
                    if category == "overall":
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Weighted F1 Score",
                                "value": metric_vals["f1_weighted"],
                                "model": model,
                            }
                        )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Macro F1 Score",
                                "value": metric_vals["f1_macro"],
                                "model": model,
                            }
                        )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Weighted Area under ROC Curve",
                                "value": metric_vals["roc_auc_weighted"],
                                "model": model,
                            }
                        )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Macro Area under ROC Curve",
                                "value": metric_vals["roc_auc_macro"],
                                "model": model,
                            }
                        )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Hamming Loss",
                                "value": metric_vals["hamming_loss"],
                                "model": model,
                            }
                        )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": "overall",
                                "metric": "Mismatch Rate",
                                "value": mismatch_rates["overall"],
                                "model": model,
                            }
                        )
                    else:
                        for metric_name, val in metric_vals.items():
                            all_metrics.append(
                                {
                                    "benchmark_label": label,
                                    "dataset": idx,
                                    "strategy": strategy,
                                    "Party": party,
                                    "category": category,
                                    "metric": metric_name,
                                    "value": val,
                                    "model": model,
                                }
                            )
                        all_metrics.append(
                            {
                                "benchmark_label": label,
                                "dataset": idx,
                                "strategy": strategy,
                                "Party": party,
                                "category": category,
                                "metric": "Mismatch Rate",
                                "value": mismatch_rates[category],
                                "model": model,
                            }
                        )
                        for cond_item in conditionalF1_scores:
                            all_metrics.append(
                                {
                                    "benchmark_label": label,
                                    "dataset": idx,
                                    "strategy": strategy,
                                    "Party": party,
                                    "category": cond_item["category"],
                                    "metric": cond_item["metric"],
                                    "value": cond_item["value"],
                                    "model": model,
                                }
                            )

    df_out = pd.DataFrame(all_metrics)
    metrics_file_path = os.path.join(metrics_dir, f"compMetricsByParty{variant}.xlsx")
    df_out.to_excel(metrics_file_path, index=False)
    print(f"Metriken wurden in 'compMetricsByParty{variant}.xlsx' gespeichert.")
