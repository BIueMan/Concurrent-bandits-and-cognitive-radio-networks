
def group_values_by_delta(values, delta):
    """
    Groups values that are within delta range of each group's current average.
    
    Args:
        values: A list of values (does not need to be sorted)
        delta: The maximum allowed difference to be in the same group
        
    Returns:
        A dictionary with group averages as keys and lists of indices as values,
        sorted by group average (keys)
    """
    if not values:
        return {}
    
    # Initialize dictionaries
    groups = {}  # (group_loc, num_elements)
    result = {}  # (group_loc, [list of idx])
    
    # Process each value
    for idx, value in enumerate(values):
        assigned = False
        
        # Check if the value fits in any existing group
        for group_loc in list(groups.keys()):
            if abs(value - group_loc) <= delta:
                # Add to this group
                num_elements = groups[group_loc]
                # Update group location (weighted average)
                new_group_loc = ((group_loc * num_elements) + value) / (num_elements + 1)
                
                # Update the groups dictionary
                groups.pop(group_loc)  # Remove old group location
                groups[new_group_loc] = num_elements + 1  # Add updated group location
                
                # Update the result dictionary
                result[new_group_loc] = result.pop(group_loc) + [idx]
                
                assigned = True
                break
        
        # If not assigned to any existing group, create a new group
        if not assigned:
            groups[value] = 1
            result[value] = [idx]
    
    # Sort the result by group averages (keys)
    sorted_result = dict(sorted(result.items(), reverse=True))
    
    return sorted_result  
def main():
    # Example 1: Simple array with clear groupings
    values1 = [100, 98, 96, 85, 84, 82, 70, 65, 63]
    delta1 = 5
    result1 = group_values_by_delta(values1, delta1)
    print("Example 1:")
    print(f"Values: {values1}")
    print(f"Delta: {delta1}")
    print("Groups:")
    for group_loc, indices in result1.items():
        group_values = [values1[idx] for idx in indices]
        print(f"  Group average: {group_loc:.2f}, Values: {group_values}, Indices: {indices}")
    print()
    
    # Example 2: Array with tighter grouping
    values2 = [50, 48, 47, 46, 40, 39, 38, 30, 29, 28]
    delta2 = 2
    result2 = group_values_by_delta(values2, delta2)
    print("Example 2:")
    print(f"Values: {values2}")
    print(f"Delta: {delta2}")
    print("Groups:")
    for group_loc, indices in result2.items():
        group_values = [values2[idx] for idx in indices]
        print(f"  Group average: {group_loc:.2f}, Values: {group_values}, Indices: {indices}")
    print()
    
    # Example 3: Array where all elements could be in the same group
    values3 = [20, 19, 18, 17, 16]
    delta3 = 10
    result3 = group_values_by_delta(values3, delta3)
    print("Example 3:")
    print(f"Values: {values3}")
    print(f"Delta: {delta3}")
    print("Groups:")
    for group_loc, indices in result3.items():
        group_values = [values3[idx] for idx in indices]
        print(f"  Group average: {group_loc:.2f}, Values: {group_values}, Indices: {indices}")

if __name__ == "__main__":
    main()