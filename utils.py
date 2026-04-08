def validate_prompt(prompt):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    return prompt.strip()
