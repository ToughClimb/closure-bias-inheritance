# Reviewer experiment protocols

The revision reuses audited completed experiments and adds fixed-expression PySR observation refits. Manuscript interpretation and theoretical extensions are separate work. Historical inputs remain unchanged; generated records, saved observations, tables and plots belong in a separate local work root.

## Existing supplementary evidence

The completed archive includes 75 five-seed main comparisons; 230 loss ablations; 15 fixed-data restart runs with 150, 500 and 1500-epoch checkpoints; 30 additional mismatch-family runs; 25 stronger 2D experiments; 40 reduced spatial versus time-weak comparisons; 30 quadrature comparisons; 30 PySR searches across A, B and Exp under clean and 3% noise; and short and long validation records. Valid coarse evidence comprises five nx16 and five resolved nx8 runs. Five original unresolved nx8 runs remain diagnostics.

`experiments/audit_reviewer_experiments.py` maps actual completed records to review items. Configured jobs are not counted as completed, and continuation or representative CPU/GPU checks do not create additional independent seeds.

## Additional PySR coefficient fitting

Reviewer 5 asks for observation/equation refitting alongside frozen symbolic models, neural surrogates and polynomial controls. The new comparison includes all 30 archived PySR inputs: A/B/Exp, clean and 3% noise, seeds 0 through 4.

The exact selected expression structure is retained. Distinct floating literals become fitted parameters; repeated equal literals within a branch share a parameter. Integer constants and powers remain fixed. Diffusion and reaction parameters are independent. No new Julia search or expression-family selection is performed. This is a restricted coefficient-refitting experiment.

The fit uses eight observed trajectories on nx64 through T=.1. Eleven uniformly selected saved snapshots, including both endpoints, enter the forward-prediction residual. The initial state is the first observed snapshot, including noise. RK45 uses rtol2e-6, atol1e-8 and maximum step min(5e-4,.35 dx²/maxD). A sampled diffusion-positivity penalty uses 512 support points and floor1e-4; each trial permits at most 20000 RHS evaluations. Truth laws do not enter optimization, model selection or stopping.

The initial budget is 40 least-squares function evaluations, with ftol, xtol and gtol all 1e-8. Fits that exhaust that budget continue from their saved coefficients for at most 80 further evaluations under identical criteria. Already converged fits are retained. Continuation depends on optimizer termination status, with every seed preserved; true closure errors never select continuation or models. Both budgets and actual residual/solver work are recorded.

Unseen validation uses four initial conditions, a full-support regime and a fixed midpoint/amplitude regime, nx64, T=.5 and saved spacing .01. It uses explicit unclipped CFL-controlled RK4 and records stability failures and incomplete horizons. The comparison reports closure and rollout errors separately, all per-seed values, sample SD, Student-t mean intervals and paired error differences. Optimizer termination does not establish a global optimum.

## Observation reconstruction

The portable archive retains configurations, hashes, curves, histories and diagnostics, but not the original observation arrays. Local reconstruction has the same declared configuration and seeds and agrees with recorded observation diagnostics within rtol1e-12 and atol1e-14. Its byte hashes differ from the cloud archive. Using the archived NumPy version alone did not restore byte identity.

By default a byte mismatch fails the new refitter. The explicit `--reconstruct-observations` option enables the separately recorded local-reconstruction protocol after the diagnostic checks pass. Every reconstructed array is saved before fitting. Continuation checks its hash and reuses those exact saved bytes. Bitwise identity with the original cloud observations is not claimed. Archived neural and polynomial results remain historical references; the new frozen versus refitted expression comparison shares the saved local protocol and validation inputs.

Representative repaired-code CPU verification passed all 37 closure and 56 validation numerical comparisons; the strict portability check still failed its observation-hash requirement. This failed check is retained. A full-budget representative GPU 2D run passed the separately reported numerical comparison. These checks do not count as new scientific seeds.

## Commands

Use the retained parent environment in the current workspace, or replace `../.venv/bin/python` with the interpreter for a fresh project installation.

```bash
../.venv/bin/python experiments/run_pysr_observation_refit.py --work-root outputs/pysr-refit-40 --input-root /path/to/archive --device cpu --workers 3 --max-nfev 40 --reconstruct-observations
../.venv/bin/python experiments/continue_pysr_observation_refit.py --work-root outputs/pysr-refit-120 --base-root outputs/pysr-refit-40 --device cpu --workers 3 --additional-nfev 80
../.venv/bin/python experiments/summarize_pysr_observation_refit.py --work-root outputs/pysr-refit-report --input-root /path/to/archive --refit-root outputs/pysr-refit-120
```

The input JSON and portable curves follow the `revision/` runner schema. Each process records source/input hashes, protocol, optimizer status and validation status. In this local campaign, systemd user services retain explicit command lines and logs; the root virtual environment and historical repository stay in place. No remote push or manuscript edit is part of these experiments.
