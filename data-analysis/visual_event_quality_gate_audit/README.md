# Quality-gate audit

Offline diagnostic of the current Visual Event quality gate on the operational
184-passage cohort. It uses the precomputed single-frame features from the
previous noise audit and never reads or modifies production code/data.

Run:

```bash
.venv/bin/python data-analysis/visual_event_quality_gate_audit/audit_quality_gate.py
```

Outputs are in `output/`:

- `candidate_rule_summary.csv`: confusion metrics for the current rule and
  small alternatives using thresholds already established in the prior audit;
- `current_gate_all_invalid_frames.csv`: every frame currently classified as
  `INVALID`, including human label and alternative-rule decisions;
- `current_gate_invalids_by_passage.csv`: compact manual-review index;
- `human_noise_missed_by_current_gate.csv`: human `ruido` frames not caught by
  the current P99 rule;
- `audit_configuration.json`: exact rules and scope.

`ruido` is the positive class. This is a diagnostic comparison, not a new
threshold search and not a proposal automatically applied to the agent.
