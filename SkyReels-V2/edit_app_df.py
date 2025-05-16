with open('app_df.py', 'r') as f:
    lines = f.readlines()

if len(lines) >= 15:
    # Comment out line 14 if not already commented
    if not lines[13].lstrip().startswith('#'):
        lines[13] = '#' + lines[13]
    # Uncomment line 15 if commented
    if lines[14].lstrip().startswith('#'):
        # Remove only the first '#' and any following space
        lines[14] = lines[14].lstrip()[1:]
        if not lines[14].startswith('is_shared_ui'):
            lines[14] = 'is_shared_ui = False\n'

with open('app_df.py', 'w') as f:
    f.writelines(lines)
