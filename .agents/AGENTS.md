# Project Rules

## Long-Running Commands: Do NOT Poll or Wait

When a task involves a long-running command that does not require agent input or decision-making (e.g., batch processing, model training, data generation, large file downloads), **do NOT**:

- Launch it as a background task and poll/wait for completion
- Set timers to check on it repeatedly
- Burn inference cycles monitoring something that will finish on its own

**Instead**, do the following:

1. Provide the user with the exact command(s) to run manually in their terminal
2. Explain what the command does and roughly how long it will take
3. Tell the user what to check to confirm it finished (e.g., a status file, exit code, output directory)
4. Tell the user to come back and notify you when it's done, so you can proceed with the next step

This applies to any command expected to run longer than ~2 minutes without needing agent interaction.
