Figure assets for the README. Generate with:

```bash
python training/plot_run_metrics.py --input logs/metrics_full.json
```

This writes `assets/figures/reward_and_pass_by_episode.png` after you run
`python main.py --full --episodes 20 --save-metrics` (or point `--input` to your JSON).

For GRPO training plots, use `trl`’s `training_logs` or the JSON written under `logs/grpo_coding/`.
