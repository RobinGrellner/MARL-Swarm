# Code Review — MARL-Swarm

**Date:** 2026-07-06
**Scope:** all first-party Python code (`analysis/`, `training/`, `environments/`, `policies/`, root runner scripts, tests) — ~7,500 lines.
**Method:** full manual read of every module, empirical verification of suspected bugs (small repro scripts), git-history checks for when defects were introduced, a baseline `pytest` run, and a `ruff` lint pass (PEP 8 rules E/W/F/I/N/B/C4/SIM, line length 120 per `pyproject.toml`).

**Ground rule for this document:** nothing here has been acted on yet, with one disclosed exception — `ruff check --fix` was run before this review-first approach was agreed, applying **87 purely mechanical, behavior-preserving fixes** (import sorting, removal of provably unused imports, `f`-prefix removal on placeholder-less strings, `dict.fromkeys` rewrites, `dict()` → literal). No logic was touched. The remaining ~20 ruff findings that require judgment are listed in §6 and were **not** applied.

---

## Executive summary

The codebase is in better shape than typical research code: the `analysis/` package is genuinely well-designed (clear docstrings, dataclasses, caching, a fast model-free test suite that passes completely), and the training pipeline is coherent. The main problems are concentrated in three places:

1. **A confirmed observation bug in `RendezvousEnv`** — all neighbour bearings are rotated by 180° (details in B1). It is *self-consistent* (training and evaluation both use it, and every saved checkpoint was trained on it), so existing results remain internally valid — but the code does not do what its comments and any thesis text describing it would claim.
2. **The environment test suite is stale**: 48 of 143 tests fail on the current code, none of them because of a new defect — they were never updated after the distance-caching and cos/sin-bearing refactors (B7).
3. **Latent bugs in code paths not used by the thesis experiments** (all experiments use `obs_model="global_basic"`): the local-model masking is broken in two different ways (B3, B4), and `obs_model="classic"` cannot be trained at all (B2).

Severity legend: 🔴 incorrect behavior · 🟠 crash/blocked path or misleading result · 🟡 latent/edge-case · ⚪ smell/style.

---

## 1. Bugs

### B1 🔴 Rendezvous neighbour bearings are flipped by 180° (confirmed empirically)

- **Where:** `environments/rendezvous/rendezvous_env.py`, `_compute_and_cache_distance_matrix()` vs. `environments/rendezvous/observations_vectorized.py`, bearing computation.
- **What:** The env caches displacements as `diff[i, j] = p_i − p_j`, but `compute_observations_vectorized` interprets `diff` as `p_j − p_i` when deriving bearings (`arctan2(diff[…,1], diff[…,0]) − orientation`). The cache is *always* populated before observations are built, so the live path always uses the wrong sign: every neighbour bearing is rotated by π (cos and sin both negated). Verified with a two-agent repro: true bearing 0 → observed `cos = −1.0`.
- **The two paths disagree:** calling `compute_observations_vectorized` *without* the cache (as the standalone tests do) produces the correct bearing. Same inputs, opposite answers, depending on an invisible cache argument.
- **Since when / blast radius:** the sign flip entered with the caching refactor (commit `825e44d` / `ef1d9d1`, 2026-02-02/05). Every model in `model/` is from May 2026 → **all saved rendezvous policies were trained on flipped bearings.** `PursuitEvasionEnv` computes `p_j − p_i` correctly, so the two environments use opposite conventions.
- **Why results are still internally valid:** a constant π-rotation of the bearing feature is a fixed, information-preserving input transformation; the φ-network simply learned it. Training curves, embed_dim comparisons, and the generalization eval (which rebuilds the env with the same code) are all self-consistent.
- **Fix options (a decision, not a patch):**
  - **(a) Compatibility-preserving (recommended while trained checkpoints matter):** make the uncached path in `observations_vectorized.py` use the same `p_i − p_j` convention, and document the convention honestly in both files. All code paths then agree with every existing checkpoint. One-line change + docs.
  - **(b) Correctness-restoring:** flip the cached `diff` to `p_j − p_i` in `rendezvous_env.py`. Semantically right and consistent with the pursuit env, **but silently invalidates every saved rendezvous model** (they would be evaluated on inputs they never saw — no error, just degraded behavior). Requires retraining or explicit acceptance.
  - Either way, the thesis text describing the observation model should state the convention actually used.

### B2 🟠 `obs_model="classic"` crashes training (`obs_layout` never set)

- **Where:** `rendezvous_env.py`, `_get_observation_space()` — the `classic` branch returns early (~line 116) without setting `self.obs_layout`; only the non-classic branch sets it (~line 222).
- **Effect:** `run_training()` reads `env.obs_layout` unconditionally (`common_train_utils.py` ~line 709) → `AttributeError`. The `classic` choice is offered by both training CLIs, so the advertised option is dead on arrival.
- **Fix:** set `obs_layout = {"local_dim": base_dim, "neigh_dim": 0, "max_neighbours": 0, "total_dim": base_dim}` in the classic branch (the extractor already supports `neigh_dim == 0`).

### B3 🔴(latent) Pursuit local models: comm-radius mask is always true

- **Where:** `pursuit_evasion_env.py`, `_get_observations()` (~lines 266–308).
- **What:** distances are first clamped: `neighbor_dists_normalized = min(d / comm_radius, 1.0)`, then masked with `valid_mask = neighbor_dists_normalized <= 1.0`. A clamped value can never exceed 1.0, so **every neighbour is always "valid"** — `local_basic`/`local_extended` silently behave like global models (with distance saturation). The comment above the line documents the intended behavior; the code cannot implement it.
- **Additionally**, the padded slots (when `max_pursuers > num_pursuers`) index agent 0 (see B4) and, with the always-true mask, are included as phantom neighbours.
- **Impact:** none on existing results (all configs use `global_basic`), but any future local-model pursuit experiment would produce wrong science with no warning.
- **Fix:** mask on the *unclamped* distance (`d <= comm_radius`) AND on slot validity (`slot < actual_neighbors`).

### B4 🔴(latent) Rendezvous local models: padded neighbour slots alias agent 0

- **Where:** `observations_vectorized.py`, steps 6–8 (~lines 196–276).
- **What:** when `max_neighbours > num_agents − 1` (the scale-invariance setup: `max_agents=100`, fewer actual agents), surplus slots are padded with index `0` into `neighbor_indices`. For **global** models an explicit `valid_mask[:, :actual_neighbors]` fixes this. For **local** models the mask is purely distance-based (`neighbor_dists <= comm_radius/world_size`), and a padded slot's "distance" is `distances[i, 0]` — the distance to agent 0 (and for agent 0 itself, distance 0 → *always* valid). Result: phantom duplicate/self neighbours enter the mean embedding with mask = 1.
- **Impact:** latent (thesis experiments are `global_basic`), but it breaks exactly the combination the generalization pipeline is built around (max_agents padding × local obs), and the existing test `test_padded_slots_have_zero_mask` asserts the correct behavior — it currently fails.
- **Fix:** same as B3 — combine the distance mask with a slot-validity mask.

### B5 🟠 `evaluate_rendezvous.py`: `--deterministic` can never be disabled

- **Where:** `parse_args()`: `parser.add_argument("--deterministic", action="store_true", default=True, ...)`.
- **Effect:** `store_true` with `default=True` is always `True`; stochastic evaluation is impossible from the CLI. Also `choices=["human", "rgb_array", None]` on `--render-mode` is dead — an argparse string can never equal `None` (only the default sneaks past validation).
- **Fix:** invert to a `--stochastic` flag (keeping `args.deterministic` as the derived attribute so `run_demo_experiments.py`'s `Namespace(deterministic=True)` keeps working), and drop `None` from `choices`.

### B6 🟠(portability) `MemoryDiagnosticCallback` is Windows-only

- **Where:** `common_train_utils.py`, `_on_rollout_end()` — `self._process.num_handles()`.
- **Effect:** `psutil.Process.num_handles()` exists only on Windows; on Linux/macOS the callback raises `AttributeError` (not caught by the `except (NoSuchProcess, AccessDenied)`), killing training runs on any non-Windows machine (e.g. a cluster). Fix: `num_handles` / `num_fds` fallback via `getattr`.

### B7 🟠 Test suite is stale: 48 of 143 environment tests fail (pre-existing)

Baseline: `pytest analysis/tests environments/tests --ignore=environments/tests/benchmarks` → **48 failed, 95 passed**. All `analysis/` tests pass. Failure clusters, each traced to a stale assumption rather than a live defect:

| Cluster | Count | Root cause |
|---|---|---|
| `test_rendezvous/test_rewards.py` | 14 | Tests set `agent_handler.positions` directly, then call `_calculate_rewards()` — but rewards now read `self._cached_distances` (populated in `_intermediate_steps()`), so they assert against distances of the *pre-injection* random positions. Never updated after the caching refactor. Fix: call `env._intermediate_steps()` after injecting positions. |
| `test_rendezvous/test_termination.py` | 8 | Same caching root cause (`_check_terminations` reads the cache). |
| `test_pursuit/test_rewards.py` | 10 | Same pattern with `_cached_evader_distances`. |
| `test_rendezvous/test_observations.py` | 8 | Fixtures call `RendezvousEnv(...)` without the now-required `world_size` kwarg; parametrized dims expect the pre-cos/sin feature sizes (`local_basic=2`, `local_extended=3`, `local_comm=4` vs. current 3/5/6). One test (`test_padded_slots_have_zero_mask`) additionally fails because of real bug B4. |
| `test_base/test_agent_handler.py` | 5 | Stale API expectations (needs individual inspection). |
| `test_pursuit/test_environment.py` | 2 | Stale obs-dim expectations. |
| `test_observations_standalone.py` | 1 | Expects 3 local features for `local_comm`; current code has 4 (own neighbourhood count). |

Also: `environments/tests/conftest.py` is broken independently — `rendezvous_env` fixture calls `RendezvousEnv(**basic_config)` without `world_size`, and it imports `PursuitEnv` from the empty stub `pursuit_env.py` (a `class PursuitEnv: pass`).

**Consequence:** the suite currently cannot catch regressions in exactly the reward/termination/observation code the thesis depends on. Repairing it is mostly mechanical (insert `_intermediate_steps()`, update dims/kwargs) but should be done deliberately so tests assert *intended* behavior — e.g. the padded-slot test should stay strict and B4 be fixed, not the test loosened.

### B8 🟡 Wrong type annotations

- `base_environment.py` — `reset()` annotated `-> Dict[str, np.ndarray]` but returns `(observations, infos)`; `step()` annotated as a 4-tuple but returns 5 (obs, rewards, terminations, truncations, infos).
- `run_architecture_scalability_continuation.py` — `build_continuation_command()` annotated `-> List[str]` but returns `(cmd, Path)`; `run_continuation()` annotated `-> bool` but returns `None` for "skipped" (the tri-state is then compared with `is True` / `is None` at the call site — works, but the signature lies).

### B9 🟡 Edge-case divisions/`None`s

- `rendezvous_env.py`: `alpha = −1/((n(n−1)/2)·dc)` → `ZeroDivisionError` for `num_agents=1`; `observations_vectorized.py` `local_comm` divides by `num_agents − 1`.
- `rendezvous_env.py` accepts `max_agents < num_agents` silently (pursuit env validates this; rendezvous doesn't) → observation slots < actual neighbours, silently truncated swarm view.
- `run_experiments.py` `compute_total_timesteps()`: `num_agents` is `None` if the config lacks the key → `TypeError` on multiply, far from the cause.

### B10 ⚪ `generalization_eval.py` hardcodes `PPO.load`

Fine while all sweeps are PPO (they are), but `ConfigSpec` carries no algorithm and a TRPO zip would fail obscurely. Worth a guard or comment. Similarly, `run_or_load_raw`'s CSV cache does not record `eval_sizes`/`n_episodes`/`eval_seeds` — rerunning with different flags silently returns the old cache (the `--force` flag exists, but nothing warns).

---

## 2. Dead code

| Location | Item | Notes |
|---|---|---|
| `evasion_agent.py` (~lines 507–600) | `_compute_ridge_escape_direction_correct()` | Never called anywhere; ~95 lines. Superseded by the torus/non-torus pair. |
| `rendezvous_env.py` (~line 268) | `delta()` method | No callers in code or tests (superseded by the vectorized cache). |
| `environments/pursuit/pursuit_env.py` | `class PursuitEnv: pass` + `# TODO` | Stub; only "user" is the broken conftest fixture. |
| `environments/pursuit/observations.py`, `environments/pursuit/rendering.py`, `environments/rendezvous/rendering.py`, `environments/base/utils.py` | empty 0-byte files | Placeholders that never materialized. Delete, or (for `base/utils.py`) use as home for the shared helpers of §3. |
| `common_train_utils.py` / `rendezvous_train_utils.py` | `normalize=` parameter threaded through `wrap_env_for_sb3()` → `run_training()` → `run_training_rendezvous()` | Documented "deprecated, kept for API compatibility"; does nothing (`VecNormalize` was removed). The `__main__` block of `rendezvous_train_utils.py` still prints "VecNormalize stats saved to: …_vecnormalize.pkl" — **that file is never written**; misleading output. |
| `rendezvous_train_utils.py` `__main__` | duplicate ad-hoc training CLI | Near-copy of `train_rendezvous.py` with *divergent* flags (`--n-envs` vs `--num-vec-envs`, fixed PPO, own defaults). Two entry points that drift apart; recommend deleting the `__main__` block. |
| `run_training()` | `resume_algorithm = algorithm` alias + comment "Determine algorithm from file" | Nothing is determined from any file; the alias adds a second name for the same value. |

---

## 3. Duplication (refactor candidates)

1. **Torus wrapping** — the 3-line `np.where(diff > half, …)` idiom appears 5×: `rendezvous_env.py` (cache), `observations_vectorized.py` (fallback), `pursuit_evasion_env.py` (×3: pursuer pairs, evader diff, evader step). One `torus_wrap(diff, world_size)` helper in `environments/base/utils.py` covers all.
2. **Wall distance + wall bearing block** (~35 lines: stack 4 wall distances, argmin, build `wall_targets` via 4 boolean masks, `arctan2`, wrap, cos/sin) — duplicated between `observations_vectorized.py` (steps 5) and `pursuit_evasion_env.py` (~lines 314–347). Same helper module.
3. **Evasion agent ridge logic** — `_compute_ridge_escape_direction_nontorus` and `_torus` are ~80 lines each and differ in exactly one thing: which point is used as the evader reference (`evader_pos` vs `nodes[sub_list[evader_sub]]`). Mergeable into one helper with an `evader_ref` argument; with the dead method of §2 removed, `evasion_agent.py` shrinks by ~180 lines.
4. **Relative-orientation block** in `pursuit_evasion_env.py` — the identical 7 lines computed in both the `global_extended` and `local_extended` branches; hoist above the branch.
5. **`_variant_dim()`** defined twice (`run_generalization.py`, `generalization_loading.py`), identical; belongs in `generalization_resolver.py` next to the variant regex.
6. **Rendezvous observation-space bounds** (`_get_observation_space`, ~lines 160–213) — five nearly identical per-model blocks setting `low/high` element-wise. Every feature is either a distance/count (`[0,1]`) or a cos/sin/velocity (`[−1,1]`); a small per-model bounds table collapses ~55 lines to ~15.
7. **Wrap-to-[−π,π]** `(x + π) % 2π − π` appears 6× across `agent_handler.py`, `observations_vectorized.py`, `pursuit_evasion_env.py` → `wrap_angle()` helper.

---

## 4. AI-typical code smells

- **Comments that narrate the diff, not the code:** "CRITICAL: Apply torus wrapping…" (×3), "OPTIMIZATION: Use cached distance…", "Optimized: longer rollouts for better value estimates", "Reduced: larger batches compensate…", "matching paper specification", "This is the EXACT approach from the original code". These describe the author's/assistant's reasoning at edit time; several are now wrong (see next point).
- **Comments contradicting the code:** `run_training()`'s "Determine algorithm from file" (nothing is); the pursuit local-mask comment claiming a radius mask that is provably a no-op (B3); `rendezvous_train_utils` advertising VecNormalize output that doesn't exist; `# Initialize to avoid UnboundLocalError when resuming` on a variable both branches assign.
- **Numbered step comments with gaps:** `run_training()` has steps `# 1., # 2., # 3., # 6., # 7., # 8.` — steps 4–5 were deleted, numbering never updated. Classic refactor residue.
- **Obfuscated test constant:** `assert mat.shape == (2, 4 // 4 + 1)` in `test_generalization.py` — that's `(2, 2)`.
- **Naming:** `MALRMetricsCallback` — transposed acronym (MARL). `HüttenrauchEvasionAgent` — non-ASCII identifier (PEP 8 recommends ASCII identifiers; also awkward to type/grep; docstrings can keep the umlaut).
- **Inconsistent defaults:** `RendezvousEnv(obs_model=…)` defaults to `"classic"` but its `None`-fallback is `"global_basic"`; `render_mode` defaults to `""` in `RendezvousEnv` but `None` in `PursuitEvasionEnv`/`BaseEnv` (and `build_eval_env` has to know which env wants which sentinel).
- **CLI choices that do nothing:** `--evader-strategy` offers 5 strategies (`simple`, `max_min_distance`, `weighted_escape`, `voronoi_center`, `huttenrauch`) but `create_evasion_agent()` ignores the argument and always returns the Hüttenrauch agent. An experiment sweeping this flag would produce 5 identical conditions. Either implement, or reduce choices to the one real option.
- **Global RNG leakage:** `_weighted_escape_direction()` uses `np.random.uniform` (module-level RNG) while the envs otherwise use a seeded `default_rng`; the evader fallback path is thus unseedable.
- **`functools.lru_cache` on methods** (`observation_space`/`action_space` in `BaseEnv`) — ruff B019; the cache holds `self` forever, and the body is a dict lookup anyway.
- **Path building by string concat:** `CheckpointCallback` does `f"{self.save_path}{name}"` (requires the caller to remember the trailing slash) and `save_path.replace(".zip", "_checkpoints/")` silently misbehaves for non-`.zip` paths.
- **`sys.argv` monkey-patching** in `run_demo_experiments.evaluate_pursuit()` — save/restore is not in `finally` (restored in `except`, but a `KeyboardInterrupt` leaks the patched argv); calling `evaluate_pursuit_evasion.main()` via argv simulation instead of a function argument is fragile.
- **Broad `except Exception` swallowing** in `evasion_agent._voronoi_ridge_strategy` (any bug in the geometry silently degrades to the fallback heuristic — fine as a robustness net, but worth at least a debug log) and in `load_model()` of `evaluate_pursuit_evasion.py` (TRPO→PPO fallback hides real load errors).
- **`env_keys` set literal rebuilt on every loop iteration** in `config_utils.expand_matrix_parameters()` — belongs at module level as a constant.

---

## 5. Design observations (including positives)

**Good:**
- `analysis/` is thesis-grade: `rliable_eval.py` has complete, accurate docstrings; the resolver/loader/eval split is clean; `EpisodeResult` as a dataclass with a shared CSV schema across tasks is a nice touch; the model-free test suite (`test_generalization.py`, `test_rliable_eval.py`) is fast and green.
- The rollout engine's handling of SuperSuit's post-reset-info quirk (recording the last pre-done step's metrics) is correct and — importantly — *documented with the reasoning*.
- Distance-matrix caching per step (`_intermediate_steps`) is the right design; the bug (B1) is in a sign convention, not the architecture.
- `wrap_env_for_sb3`'s num_cpus=0 design decision is documented with a real rationale.

**Questionable but defensible (flag, don't change):**
- `IterationCounterCallback` writes `by_iter/*` scalars one iteration behind via `name_to_value` snooping — documented, works, but is tightly coupled to SB3 logger internals.
- `resolve_run_dirs`'s `architecture_schaling` typo-mapping table — ugly but honest (the typo is baked into directories on disk); keep as long as the dirs exist.
- Duplicated `if __name__ == "__main__"` demo in `rendezvous_env.py` — fine as an executable smoke demo.

---

## 6. PEP 8 / lint status

`ruff` (E, W, F, I, N, B, C4, SIM; line length 120) found **106 issues**. The 87 mechanical ones (unsorted imports ×29, placeholder-less f-strings ×26, unused imports ×14, `dict.fromkeys` ×9, misc SIM/C4) are already applied as disclosed above. The **remaining 20** need judgment:

| Rule | Where | Recommendation |
|---|---|---|
| F841 unused local ×7 | `result = subprocess.run(...)` in the 3 runner scripts; `num_agents`, `n` in `rendezvous_env.py`; `initial_min_dist` in a pursuit test; `e` in `evasion_agent.py` | Drop the assignments (keep the calls). |
| B019 `lru_cache` on methods ×2 | `base_environment.py` | Remove decorators (body is a dict lookup). |
| B904 raise-without-from | `parse_policy_layers()` | `raise … from e`. |
| B028 no stacklevel | `build_algo_params()` warning | `stacklevel=2`. |
| N806 `L_i` ×3 (2 after dead-code removal) | `evasion_agent.py` | Rename to `ridge_len` (or keep the paper notation with `# noqa: N806`). |
| N812 `functional as F` | `mean_embedding_extractor.py` | Idiomatic PyTorch — keep with `# noqa: N812`. |
| C408 `dict(pi=…, vf=…)` etc. ×4 | `common_train_utils.py`, `run_generalization.py`, tests | Literal `{}` form. |
| SIM102/SIM105/SIM114 ×3 | `run_architecture_scalability_continuation.py`, `generalization_eval.py`, `pursuit_evasion_env.py` | Apply (collapse nested if; `contextlib.suppress`; merge identical branches). |

---

## 7. Recommended action plan (in order, with risk)

| Phase | Content | Risk to results |
|---|---|---|
| 1 | **Decide B1** (convention alignment a) vs. correctness b)) — everything else in the obs code should follow that decision. Document the chosen convention in code + thesis. | (a) none · (b) invalidates saved rendezvous checkpoints |
| 2 | Fix latent obs bugs **B3 + B4** (slot-validity ∧ unclamped-radius masks), **B2** (classic `obs_layout`), rendezvous `max_agents ≥ num_agents` validation. | none for existing `global_basic` results |
| 3 | Repair the test suite (**B7**, **B9-conftest**): inject `_intermediate_steps()` after manual state writes, fix fixture kwargs, update expected feature dims to the cos/sin layout. Keep `test_padded_slots_have_zero_mask` strict (it guards B4). | none |
| 4 | Small correctness/portability: **B5** (deterministic flag), **B6** (num_handles), **B8** (annotations), `CheckpointCallback` path join, `sys.argv` try/finally. | none |
| 5 | Dead code removal (§2) incl. `normalize` plumbing and the misleading VecNormalize print; stub/empty file deletion (requires the conftest fix from phase 3). | none |
| 6 | Deduplication (§3): `environments/base/utils.py` helpers (`torus_wrap`, `wall_features`, `wrap_angle`), evasion-agent merge, `_variant_dim` single home, bounds-table refactor. Existing tests (post phase 3) validate behavior is unchanged. | low (covered by tests) |
| 7 | Remaining lint (§6) + naming (`MARLMetricsCallback`, ASCII evasion-agent class) + comment hygiene (§4). | none |

A full test run (`pytest analysis/tests environments/tests`) should gate phases 2–7; phase 3 is what makes that gate meaningful.
