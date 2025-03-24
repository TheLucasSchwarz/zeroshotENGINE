def display_label_flowchart(valid_keys, stop_conditions, label_codes):
    """
    Display a simplified flowchart of labels in ASCII tree format with enhanced formatting.

    Args:
        valid_keys (list): List of keys to be included in the flowchart
        stop_conditions (dict): Dictionary defining when to stop traversing branches
        label_codes (dict): Dictionary mapping descriptors to coded values
    """
    # Check if valid_keys is empty
    if not valid_keys:
        print("No valid keys provided.")
        return

    # Print header
    border_head = "=" * 62
    border = "-" * 62
    print(f"\n{border_head}")
    print(f"{' ZEROSHOTENGINE LABEL DEPENDENCY FLOWCHART ':^62}")
    print(f"{border_head}")
    print("")

    # Get branching points from stop_conditions
    branch_points = set(stop_conditions.keys())

    # Print the root node
    root_key = valid_keys[0]
    root_idx = 0
    print(f" [{root_key.upper()}]")

    # Process positive branch (if root = 1)
    print(f" ├─ if {root_key} = {label_codes['present']}:")

    def process_nodes(start_idx, indent, remaining_keys, is_last=False):
        """
        Recursively process nodes in the flowchart

        Args:
            start_idx: Index to start processing from
            indent: Current indentation string
            remaining_keys: Keys that are still valid on this path
            is_last: Whether this is the last branch at this level
        """
        if start_idx >= len(valid_keys) or not remaining_keys:
            # We've reached the end of this branch
            print(f"{indent}│")
            print(f"{indent}▼")
            print(f"{indent}STOP")
            return

        current_key = valid_keys[start_idx]

        # Skip this key if it's not in remaining_keys
        if current_key not in remaining_keys:
            process_nodes(start_idx + 1, indent, remaining_keys, is_last)
            return

        # Print the current node
        print(f"{indent}[{current_key.upper()}]")

        # Check if this is a branch point
        if start_idx in branch_points:
            # Present case (value = 1)
            branch_char = "└─ " if is_last else "├─ "
            print(f"{indent}{branch_char}if {current_key} = {label_codes['present']}:")

            # Determine next indent for present branch
            next_indent = indent + ("    " if is_last else "│   ")

            # Find keys blocked in the absent case - these are the ones that continue in present case
            next_keys = remaining_keys.copy()

            # Process the present branch recursively
            process_nodes(start_idx + 1, next_indent, next_keys)

            # Absent case (value = 0)
            print(f"{indent}└─ if {current_key} = {label_codes['absent']}:")

            # Find blocked keys for this branch point
            blocked_keys = []
            for condition_id, condition_data in stop_conditions.items():
                if (
                    condition_id == start_idx
                    and condition_data["condition"] == label_codes["absent"]
                ):
                    # Filter blocked_keys to only include those that are still in remaining_keys
                    blocked_keys = [
                        k for k in condition_data["blocked_keys"] if k in remaining_keys
                    ]
                    break

            # Show skipped keys
            if blocked_keys:
                skip_text = ", ".join(blocked_keys)
                print(f"{indent}    → Skip: {skip_text}")

                # Add note about potential override conditions
                for blocked_key in blocked_keys:
                    override_conditions = []
                    for other_id, other_data in stop_conditions.items():
                        if (
                            other_id != start_idx
                            and other_data["condition"] == label_codes["present"]
                            and blocked_key not in other_data.get("blocked_keys", [])
                        ):
                            # This condition might override the skipping
                            if other_id < len(valid_keys):
                                override_conditions.append(
                                    f"{valid_keys[other_id]} = {label_codes['present']}"
                                )

                    if override_conditions:
                        override_text = " or ".join(override_conditions)
                        print(
                            f"{indent}    (Note: {blocked_key} may be included if {override_text})"
                        )

                # Remove blocked keys from remaining_keys for the absent branch
                next_keys = [k for k in remaining_keys if k not in blocked_keys]

                # If all remaining keys are blocked, just stop here
                if len(next_keys) <= 1 or all(
                    k not in next_keys for k in valid_keys[start_idx + 1 :]
                ):
                    print(f"{indent}    STOP")
                else:
                    # Find the next valid index to process
                    next_idx = start_idx + 1
                    while (
                        next_idx < len(valid_keys)
                        and valid_keys[next_idx] not in next_keys
                    ):
                        next_idx += 1

                    # Process the absent branch recursively starting from next valid key
                    if next_idx < len(valid_keys):
                        process_nodes(next_idx, indent + "    ", next_keys)
                    else:
                        print(f"{indent}    STOP")
            else:
                # No blocked keys, continue normally
                process_nodes(start_idx + 1, indent + "    ", remaining_keys)

        else:
            # Not a branch point, continue to next node
            process_nodes(start_idx + 1, indent, remaining_keys)

    # Start processing from the first node after root
    process_nodes(1, " │   ", valid_keys)

    # Process negative branch for the root (if root = 0)
    print(f" └─ if {root_key} = {label_codes['absent']}:")

    # Find blocked keys for root = 0
    blocked_keys = []
    for condition_id, condition_data in stop_conditions.items():
        if condition_id == 0 and condition_data["condition"] == label_codes["absent"]:
            blocked_keys = condition_data["blocked_keys"]
            break

    if blocked_keys:
        skip_text = ", ".join(blocked_keys)
        print(f"     → Skip: {skip_text}")

    print("     STOP")
    print("")
    print(f"{border}")

    # Print stop conditions explanation
    print(f"{' STOP CONDITIONS EXPLANATION ':^62}")
    print(f"{border}")
    for condition_id, condition_data in stop_conditions.items():
        if condition_id < len(valid_keys):
            key_name = valid_keys[condition_id]
            value = condition_data["condition"]
            # Find the actual label name for this value instead of hardcoding
            value_text = next(
                (k for k, v in label_codes.items() if v == value), str(value)
            )
            blocked_keys = condition_data["blocked_keys"]

            if blocked_keys:
                blocked_text = "\n    - " + "\n    - ".join(blocked_keys)
                print(
                    f"  If {key_name} = {value} ({value_text}), the following steps are skipped:{blocked_text}"
                )
                print("")
    print(f"{border}")

    # Print legend
    print(f"{' LEGEND ':^62}")
    print(f"{border}")
    print(
        f" - {label_codes['present']} ({next((k for k, v in label_codes.items() if v == label_codes['present']), 'present')}): Proceeds to the next classification step"
    )
    print(
        f" - {label_codes['absent']} ({next((k for k, v in label_codes.items() if v == label_codes['absent']), 'absent')}): Skips one or more subsequent classifications"
    )
    print("")
    print(f"{' LABEL CODES '}")

    # Display label codes
    for key, value in label_codes.items():
        print(f"    {key}: {value}")
    print("")
    print(f"{border}\n")
