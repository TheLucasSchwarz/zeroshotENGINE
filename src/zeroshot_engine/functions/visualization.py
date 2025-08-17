def visualize_graphical_flowchart(valid_keys, stop_conditions, label_codes, output_file='flowchart.png'):
    """
    Create a universal graphical visualization of the flowchart using Graphviz.

    Args:
        valid_keys (list): List of keys to be included in the flowchart
        stop_conditions (dict): Dictionary defining when to stop traversing branches
        label_codes (dict): Dictionary mapping descriptors to coded values
        output_file (str): Output filename for the image
    """
    try:
        import graphviz
    except ImportError:
        print("Graphviz not installed. Please run: pip install graphviz")
        return

    # Initialize a directed graph
    dot = graphviz.Digraph(comment='Zeroshot Engine Label Flowchart')
    dot.attr(rankdir='TB', size='8,11', dpi='300')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial')

    # Add title
    dot.attr(label='ZEROSHOTENGINE LABEL DEPENDENCY FLOWCHART', fontsize='20', labelloc='t')

    # Create nodes for all keys
    node_ids = {}
    for i, key in enumerate(valid_keys):
        node_id = f"node_{i}"
        node_ids[key] = node_id
        dot.node(node_id, key.upper())

    # Process dependencies and stop conditions
    for i, key in enumerate(valid_keys):
        # Check for stop conditions
        if i in stop_conditions:
            condition = stop_conditions[i]
            if condition["condition"] == label_codes["absent"]:
                # Create STOP node for this branch
                stop_id = f"stop_{i}"
                dot.node(stop_id, "STOP", shape='ellipse', style='filled', fillcolor='lightgray')

                # Connect current node to STOP node for "absent" case
                dot.edge(node_ids[key], stop_id, label=f"if {key} = {label_codes['absent']}", color='red')

                # Add note about skipped keys
                if "blocked_keys" in condition and condition["blocked_keys"]:
                    skipped = ", ".join(condition["blocked_keys"])
                    note_id = f"note_{i}"
                    dot.node(note_id, f"Skip: {skipped}", shape='note', style='filled', fillcolor='lightyellow')
                    dot.edge(node_ids[key], note_id, style='dashed', arrowhead='none', color='gray')

        # Add logical dependencies
        if i < len(valid_keys) - 1:
            next_key = valid_keys[i + 1]
            if next_key not in stop_conditions.get(i, {}).get("blocked_keys", []):
                dot.edge(node_ids[key], node_ids[next_key], label=f"if {key} = {label_codes['present']}", color='blue')

    # Handle parallel branches for independent labels
    for i, key in enumerate(valid_keys):
        if i in stop_conditions:
            condition = stop_conditions[i]
            if condition["condition"] == label_codes["absent"]:
                # Parallel branches for independent labels
                for blocked_key in condition["blocked_keys"]:
                    if blocked_key in valid_keys and blocked_key in node_ids:  # Ensure blocked_key exists in valid_keys and node_ids
                        dot.edge(node_ids[key], node_ids[blocked_key], label=f"if {key} = {label_codes['absent']}", color='blue')

    # Add explanation notes for all stop conditions
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='STOP CONDITIONS EXPLANATION', style='filled', color='lightgray')
        c.attr('node', shape='plaintext', style='filled', fillcolor='white')

        for idx, condition in stop_conditions.items():
            if idx < len(valid_keys):
                key_name = valid_keys[idx]
                value_text = "absent" if condition["condition"] == 0 else "present"
                if "blocked_keys" in condition and condition["blocked_keys"]:
                    blocked_text = ", ".join(condition["blocked_keys"])
                    legend_id = f"legend_{idx}"
                    c.node(legend_id, f"If {key_name} = {condition['condition']} ({value_text}),\nskip: {blocked_text}")

    # Render the graph to a file
    try:
        dot.render(output_file.split('.')[0], format=output_file.split('.')[-1], cleanup=True)
        print(f"Flowchart saved as {output_file}")
    except Exception as e:
        print(f"Error rendering flowchart: {e}")
        # Try to save as PDF if the requested format fails
        try:
            dot.render('flowchart', format='pdf', cleanup=True)
            print("Flowchart saved as flowchart.pdf")
        except:
            print("Failed to save flowchart.")

    return dot

# Add graphical visualization option to your existing function
def display_label_flowchart(valid_keys, stop_conditions, label_codes, graphical=False):
    """
    Display a flowchart of labels.

    Args:
        valid_keys (list): List of keys to be included in the flowchart
        stop_conditions (dict): Dictionary defining when to stop traversing branches
        label_codes (dict): Dictionary mapping descriptors to coded values
        graphical (bool): Whether to generate a graphical output instead of ASCII
    """
    if graphical:
        return visualize_graphical_flowchart(valid_keys, stop_conditions, label_codes)

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
