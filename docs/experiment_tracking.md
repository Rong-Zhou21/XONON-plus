# Experiment Tracking Workflow

This project keeps raw artifacts and long-running outputs outside Git, while
committing the code, configs, commands, and human-readable result summaries.

## What goes to GitHub

- Source code and configuration files needed to reproduce an experiment.
- Small documentation files under `docs/experiments/`.
- Summaries generated from raw experiment outputs, such as tables, metrics, and
  failure analysis.
- Scripts that launch or analyze experiments.

## What stays local

- Model checkpoints: `checkpoints/`, `checkpoints.bak/`, `*.pt`, `*.weights`.
- Raw logs: `logs/`, `*.log`.
- Videos and images: `videos/`, `imgs/`.
- Raw experiment dumps: `exp_results/`, `results/`, `runs/`.
- Interrupted downloads: `*.part`.

If a raw artifact is important, record its local path, file size, checksum, and
how it was produced in the experiment note instead of committing the file.

## Daily workflow

1. Start from the latest GitHub version.

   ```bash
   git pull --rebase origin main
   ```

2. Create or switch to a branch for one idea.

   ```bash
   git switch -c exp/<short-idea-name>
   ```

3. Record the planned experiment in `docs/experiments/experiment_log.md`.

4. Run the experiment and keep raw outputs under ignored directories such as
   `exp_results/` and `logs/`.

5. Summarize the result in a note copied from
   `docs/experiments/template.md`.

6. Save code progress.

   ```bash
   bash scripts/git_checkpoint.sh "short message"
   ```

7. Push the branch.

   ```bash
   git push -u origin exp/<short-idea-name>
   ```

8. When the idea is stable, merge it into `main`.

## Commit message convention

Use short, searchable prefixes:

- `exp:` experiment code or configuration changes
- `fix:` bug fixes found during experiments
- `docs:` experiment notes and reproduction records
- `analysis:` result aggregation or report changes

Examples:

```text
exp: add vllm planner backend switch
analysis: summarize diamond pickaxe failures
docs: record wooden benchmark baseline
```
