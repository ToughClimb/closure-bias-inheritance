# Code audit repairs and validation

The repairs preserve the original core implementation and bring the revision experiment tools into this public repository. Historical results remain read-only inputs. New experiments use the repaired core, their own work root and recorded implementation identity.

## Reviewed defects

| Item | Defect | Repair and regression coverage |
| --- | --- | --- |
| 1 | Separate checkpoint, history and curve writes could produce incomplete successful resumes. | Atomic schema 2 checkpoints embed Adam state, RNG, complete history and requested curves. Interrupted exports, corrupt external history and missing final curves are reconstructed. Missing epoch history and empty final snapshots are rejected. CPU and CUDA continuous versus resumed runs are compared. |
| 2 | Four Fourier modes on nx8 normalized a nearly zero Nyquist sine and overflowed Adam. | Unresolved modes and near-zero sampled norms are rejected. Losses, gradients, parameters and Adam tensors are checked for finite values. The new campaign retains the resolved three-mode nx8 arm and retires the five invalid historical configurations. |
| 3 | Failed PySR caches and fresh failures could be counted as completed; cache identity was unchecked. | Fresh, cached and watched failures cause nonzero exits. Explicit retries preserve earlier failure records. Completed caches check configuration, source, frozen target, search budget and metrics. Julia initializes only for a new fit, before Torch in the revision PySR runner. |
| 4 | Portability comparisons silently truncated unequal validation lists. | Method and policy sets, entry counts, array shapes and saved times must match. The representative check requires all 37 closure and 56 validation comparisons. Failures replace stale passed audit records. |
| 5 | Rectangle endpoint weights biased the optional time-weak objective. | Both temporal terms use trapezoidal endpoint weights. Stationary and manufactured time-dependent solutions test zero residual and quadrature convergence. |
| 6 | Rational expressions guessed coefficient layout from total parameter count. | Numerator and denominator degrees are explicit and survive serialization and observation refits. Unequal-degree families are tested against their fitted error. Legacy five-coefficient degree-2/degree-2 exports remain supported. |
| 7 | Nondividing strides produced nonuniform periodic grids treated as uniform. | Both 1D and 2D observation models require divisibility, enough spatial cells and at least three saved times. Valid uniform strides remain supported. |
| 8 | PySR summaries merged cases, operator sets and training protocols. | Groups include case, observation setting, operators, training protocol and search identity; duplicate seeds are rejected. Modern restricted references require the identical frozen target. |
| 9 | Strict rollout assumed nx64 and stopped before a nonmultiple final time. | Resolution comes from the initial state; explicit mismatches are rejected. Exact final times include partial save intervals and sub-save horizons. Discrete heat modes verify 1D and 2D accuracy. |
| 10 | Default PySR outputs overwrote different observation and search protocols. | CSV and intermediate search paths include protocol and operator identities. A simulated backend exercises the public command across changed training budgets and operators. |
| 11 | Polynomial baseline ignored disabled unseen validation. | The disabled flag prevents unseen simulation and omits its metrics. |
| 12 | 2D Markdown reports dropped cases outside a fixed default list. | Reports enumerate actual cases; a nondefault Case B row is covered. |
| 13 | Empty CSV summaries retained obsolete earlier results. | Empty outputs remove the old CSV. |
| 14 | KAN output broadcast across singleton axes. | Spline and affine terms preserve the input shape, including scalar and singleton-axis tensors; independent input derivatives are tested. |

## Validation on 12 October 2026

- CPU regression suite: 83 tests passed, with two opt-in CUDA checks skipped in that run.
- Local WSL/Linux, RTX 5070 Ti, Torch 2.12.0+cu130: both CUDA checks passed. They cover short 1D and 2D Adam resume, with parameters and histories identical to continuous training.
- Combined final run with CUDA enabled: all 85 tests passed.
- Historical nominal nx64 Case Exp, seed 2, five CPU epochs: all 22 exported parameter and curve arrays are bitwise identical between the original core and repaired core.
- Report reconstruction from the existing complete campaign: all eight selected numerical CSV files are byte-for-byte identical. The bundle audit checked 1308 inputs, found no missing files and reconstructed curve metrics within 3.56e-15.
- Five historical unresolved nx8 configurations are retained and identified as diagnostics. The new declaration has 445 valid 1D configurations; the audit checks exact required configurations and related validation records.
- The existing root virtual environment stays in place. Existing dependency versions were retained; SciPy 1.17.0 and pandas 2.2.3 were added from the already validated migration stack. The editable core package now points at this repository.

The PySR regression backend is simulated; these checks do not establish new Julia search results. Full GPU campaigns, native Windows execution and scientific equivalence of every repaired protocol require their own experiments. Completed historical campaigns were not repeated for this repair.

## Experiment compatibility

Use a distinct work root for repaired experiments and keep historical inputs under a separate input root. Source and configuration mismatches cannot reuse a completed task as though it belonged to the new implementation. Compatible legacy 1D optimizer checkpoints require explicit opt-in, complete external history and finite optimizer state; old model-only or insufficiently described 2D checkpoints cannot provide a verified optimizer resume.

The main spatial-weak objective and nominal nx64 Fourier construction are preserved. Time-weak quadrature, invalid coarse-grid tests, arbitrary rational layouts and final save-time handling intentionally change behavior. Existing report reconstruction uses stored records and does not silently recompute those historical experiments.

The scientific interpretation remains conditional: faithful compression of a frozen surrogate can retain its bias. Observation-only symbolic refitting is a separate operation and can reduce that bias; the historical clean Case Exp forward refit reduced mean reaction error from approximately 0.441 to 0.0092.

See [README.md](README.md) for installation, small runs, resume, validation and report commands.
