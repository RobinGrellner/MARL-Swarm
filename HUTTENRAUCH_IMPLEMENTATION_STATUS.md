# Hüttenrauch Implementation Status

Comprehensive comparison of Hüttenrauch et al.'s original implementation vs. current MARL-Swarm implementation.

---

## 1. Fundamental Constants

### obs_radius Calculation

**Hüttenrauch (Both PE and Rendezvous):**
```
pursuer_agent.py, line 12:
self.obs_radius = experiment.comm_radius / 2

rendezvous_agent.py, line 12:
self.obs_radius = experiment.comm_radius / 2
```
- **Always**: `obs_radius = comm_radius / 2`
- **Never** varies by observation model
- `comm_radius` is always required (never None)

**Current MARL-Swarm (PE only):**
```python
# pursuit_evasion_env.py, lines 100-103
if self.obs_model.startswith("global"):
    self.obs_radius = world_size / 2.0
else:
    self.obs_radius = comm_radius / 2.0 if comm_radius is not None else world_size / 2.0
```
- **Varies by observation model** (global vs local)
- Falls back to `world_size/2` if `comm_radius` is None
- **MISMATCH**: Global models use `world_size/2` instead of `comm_radius/2`

---

## 2. Bearing Representation

### Key Principle
Hüttenrauch **ALWAYS uses (cos, sin) pairs** instead of raw angle values. This eliminates the discontinuity at ±π.

### Pursuit-Evasion: Evader Bearings

**Hüttenrauch (pursuer_agent.py, lines 300-305):**
```python
if evader_dists < self.obs_radius:
    dist_to_evader = evader_dists / self.obs_radius
    angle_to_evader = [np.cos(evader_bearings), np.sin(evader_bearings)]
else:
    dist_to_evader = 1.
    angle_to_evader = [0, 0]
```
- **Representation**: (cos, sin) pair in observation
- **Normalization**: `evader_dists / obs_radius`

**Current MARL-Swarm (pursuit_evasion_env.py, line 326):**
```python
evader_bearings = evader_bearings_raw - orientations  # (N,)
evader_bearings = (evader_bearings + np.pi) % (2 * np.pi) - np.pi
```
- **Representation**: Raw angle in [-π, π]
- **MISMATCH**: Not using (cos, sin)

---

### Pursuit-Evasion: Pursuer Bearings

**Hüttenrauch (pursuer_agent.py, lines 329-332):**
```python
sum_obs[0:nr_neighbors, 0] = pursuer_dists[pursuers_in_range] / self.comm_radius
sum_obs[0:nr_neighbors, 1] = np.cos(pursuer_bearings[pursuers_in_range])
sum_obs[0:nr_neighbors, 2] = np.sin(pursuer_bearings[pursuers_in_range])
```
- **Representation**: (cos, sin) pair in observation
- **Distance normalization**: `pursuer_dists / comm_radius`

**Current MARL-Swarm (pursuit_evasion_env.py, lines 243-250):**
```python
neighbor_features = np.stack([neighbor_dists, neighbor_bears], axis=2)
```
- **Representation**: Raw angle
- **Distance normalization**: `neighbor_dists / world_size` (line 276)
- **MISMATCH**: Both bearing representation AND distance normalization

---

### Rendezvous: Neighbor Bearings

**Hüttenrauch (rendezvous_agent.py, lines 354-356, sum_obs mode):**
```python
sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
sum_obs[0:nr_neighbors, 1] = np.cos(my_orientation[in_range])
sum_obs[0:nr_neighbors, 2] = np.sin(my_orientation[in_range])
```
- **Representation**: (cos, sin) for orientation/bearing
- **Distance normalization**: `dm / world_size`

**Current MARL-Swarm (observations_vectorized.py, line 191):**
```python
neighbor_features = np.stack([neighbor_dists, neighbor_bears], axis=2)  # (N, max_neighbours, 2)
```
- **Representation**: Raw angle
- **Distance normalization**: `neighbor_dists / world_size` (line 186)
- **MISMATCH**: Bearing representation

---

### Rendezvous: Wall Bearings

**Hüttenrauch (rendezvous_agent.py, lines 428-429, get_local_obs):**
```python
local_obs[1] = np.cos(wall_angles[closest_wall])
local_obs[2] = np.sin(wall_angles[closest_wall])
```
- **Representation**: (cos, sin) pair
- **Line 425**: `wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation`

**Current MARL-Swarm (observations_vectorized.py, lines 166-168):**
```python
wall_bearings = np.arctan2(delta_w[:, 1], delta_w[:, 0]) - orientations
wall_bearings = (wall_bearings + np.pi) % (2 * np.pi) - np.pi
wall_bearings = wall_bearings.astype(np.float32)
```
- **Representation**: Raw angle
- **MISMATCH**: Not using (cos, sin)

---

## 3. Distance Normalization

### Pursuit-Evasion

**Hüttenrauch:**
- **Evader distances**: Normalized by `obs_radius` (line 301)
  ```python
  dist_to_evader = evader_dists / self.obs_radius
  ```
- **Pursuer distances**: Normalized by `comm_radius` (line 329)
  ```python
  sum_obs[0:nr_neighbors, 0] = pursuer_dists[pursuers_in_range] / self.comm_radius
  ```
- **Ratio**: 2:1 (because `obs_radius = comm_radius / 2`)

**Current MARL-Swarm:**
- **Evader distances**: Normalized by `obs_radius` (line 329) ✓
- **Pursuer distances**: Normalized by `world_size` (line 276) ✗
- **MISMATCH**: Wrong normalization scale for pursuers

---

### Rendezvous

**Hüttenrauch:**
- **All neighbor distances**: Normalized by `world_size` (line 274, 294, 314, 333, 354, 373)
  ```python
  sum_obs[0:nr_neighbors, 0] = dm[in_range] / self.world_size
  ```
- **Filtering**: Uses `comm_radius` for communication range filtering (line 346)
  ```python
  in_range = (dm < self.comm_radius) & (0 < dm)
  ```

**Current MARL-Swarm:**
- **All neighbor distances**: Normalized by `comm_radius` (line 186) ✗
  ```python
  neighbor_dists = neighbor_dists / comm_radius
  ```
- **MISMATCH**: Should be `world_size`, not `comm_radius`

---

## 4. Wall Observations

### Pursuit-Evasion

**Hüttenrauch (pursuer_agent.py, lines 309-314, sum_obs_limited):**
```python
if self.torus is False:
    if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= 99):
        wall = 1
    else:
        wall = 0
local_obs[0] = wall
```
- **Representation**: Binary (in wall region or not)
- **No wall distance/bearing** in mean-embedding observations

**Current MARL-Swarm (pursuit_evasion_env.py, lines 330-335):**
```python
wall_dists_normalized = np.minimum(wall_dists / self.world_size, 1.0)
local_features = np.stack([wall_dists_normalized, wall_bearings, evader_dists_normalized, evader_bearings], axis=1)
```
- **Representation**: Continuous distance + raw bearing angle
- **MISMATCH**: Includes wall distance/bearing, not just binary

---

### Rendezvous

**Hüttenrauch (rendezvous_agent.py, lines 420-432, get_local_obs):**
```python
if np.any(self.state.p_pos <= 1) or np.any(self.state.p_pos >= self.world_size - 1):
    wall_dists = np.array([self.world_size - self.state.p_pos[0],
                           self.world_size - self.state.p_pos[1],
                           self.state.p_pos[0],
                           self.state.p_pos[1]])
    wall_angles = np.array([0, np.pi / 2, np.pi, 3 / 2 * np.pi]) - self.state.p_orientation
    closest_wall = np.argmin(wall_dists)
    local_obs[0] = wall_dists[closest_wall]
    local_obs[1] = np.cos(wall_angles[closest_wall])
    local_obs[2] = np.sin(wall_angles[closest_wall])
else:
    local_obs[0] = 1
    local_obs[1:3] = 0
```
- **Only when near wall**: Includes raw distance + (cos, sin) bearing
- **When far from wall**: Sets distance=1, bearing=(0, 0)

**Current MARL-Swarm (observations_vectorized.py, lines 128-168):**
- **Always computed**: Wall distance and bearing for all agents
- **Representation**: Normalized distance + raw bearing angle
- **MISMATCH**: Always included, and bearing is raw angle not (cos, sin)

---

## 5. Observation Space Dimensions

### Pursuit-Evasion

**Hüttenrauch (sum_obs_limited, line 113-119):**
```python
self.dim_evader_o = (self.n_evaders, 3)  # [dist, cos(bearing), sin(bearing)]
self.dim_rec_o = (100, 8)  # [dist/comm_radius, cos, sin, cos(ori), sin(ori), ?, ?, ?]
```

**Current MARL-Swarm (global_basic):**
```python
# Local (4): [wall_dist, wall_bearing, evader_dist, evader_bearing]
# Neighbors: [distance, bearing] per neighbor (2 dims each)
# Mask: binary per neighbor
# Total: 4 + (N-1)*2 + (N-1)
```
- **MISMATCH**: Different structure and dimensions

---

### Rendezvous

**Hüttenrauch (sum_obs_acc, line 80-85):**
```python
self.dim_rec_o = (self.n_agents - 1, 7)  # [dist, cos, sin, cos(ori), sin(ori), vel_x, vel_y]
self.dim_mean_embs = self.dim_rec_o
self.dim_local_o = 2 + 3 * int(not self.torus)
self.dim_flat_o = self.dim_local_o
```

**Current MARL-Swarm (local_basic):**
```python
# Local (2): [wall_dist, wall_bearing]
# Neighbors: [distance, bearing] per neighbor (2 dims each)
# Mask: binary per neighbor
```
- **MISMATCH**: Missing velocity information in neighbors

---

## 6. Termination Logic

### Pursuit-Evasion

**Hüttenrauch (pursuit_evasion.py, lines 199-201):**
```python
done = self.is_terminal  # max timesteps check
if rewards[0] > -1 / self.obs_radius:  # distance < 1 world unit
    done = True
```
- **Termination condition**: `min_distance < 1` (in world units)
- **obs_radius = comm_radius / 2**, so this means `min_distance < obs_radius`
- Triggers **EARLY termination** on capture

**Current MARL-Swarm (pursuit_evasion_env.py, line 360-367):**
```python
def _check_terminations(self) -> Dict[str, bool]:
    distances = self._cached_evader_distances
    captured = np.any(distances < self.capture_radius)
    return {agent: captured for agent in self.agents}
```
- **Termination condition**: `min_distance < capture_radius` (0.5)
- **MISMATCH**: Different threshold, but same principle

---

## 7. Evader Behavior

### Hüttenrauch Implementation

**Algorithm**: Line-of-Control (LoC) strategy with Voronoi geometry (evader_agent.py)

**Key components:**
1. **Boundary reflections** (lines 27-162)
2. **Voronoi ridge computation** with Shapely library
3. **Directional components** (α_h_i, α_v_i) from line of control

```python
# Line 155-158
alpha_h_i = - L_i / 2
alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2 * np.linalg.norm(xi))
d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)
```

**Current MARL-Swarm Implementation:**

**Algorithm**: Simplified Voronoi approach (evasion_agent.py)

**Differences:**
- Uses generic Voronoi ridge finding (not precise line-of-control)
- Missing the proper directional component calculation
- Falls back to weighted escape direction

**MISMATCH**: Fundamentally different evasion strategy

---

## 8. Summary Table

| Component | Hüttenrauch | Current | Match? |
|-----------|------------|---------|--------|
| **obs_radius** | `comm_radius/2` always | `world_size/2` for global | ❌ |
| **Evader dist norm** | `obs_radius` | `obs_radius` | ✓ |
| **Pursuer dist norm (PE)** | `comm_radius` | `world_size` | ❌ |
| **Neighbor dist norm (Rdv)** | `world_size` | `comm_radius` | ❌ |
| **Evader bearing** | (cos, sin) | raw angle | ❌ |
| **Pursuer bearing (PE)** | (cos, sin) | raw angle | ❌ |
| **Neighbor bearing (Rdv)** | (cos, sin) | raw angle | ❌ |
| **Wall bearing (Rdv)** | (cos, sin) | raw angle | ❌ |
| **Evader strategy** | Line-of-Control | Simplified Voronoi | ❌ |
| **Early termination** | Yes, on capture | Yes, on capture | ✓ |

---

## 9. Critical Issues Affecting Training Stability

### High Priority (Primary Cause of Oscillation)

1. **Bearing Representation (ALL)** - Raw angles [-π, π] have discontinuity
   - Affects PE evader bearings, PE pursuer bearings, Rendezvous neighbor bearings, Rendezvous wall bearings
   - Value function confusion at angle boundaries

2. **Distance Normalization Mismatch**
   - PE: Pursuer distances normalized by `world_size` instead of `comm_radius`
   - Rendezvous: Neighbor distances normalized by `comm_radius` instead of `world_size`
   - Creates scale inconsistency in observations

3. **obs_radius Mismatch** - Global models use `world_size/2` instead of `comm_radius/2`
   - Affects reward scaling and observation consistency

### Medium Priority

4. **Evader Strategy** - Simplified Voronoi instead of precise line-of-control
   - Changes game dynamics

5. **Wall Observations** - Continuous instead of conditional
   - Different information provided to agents

### Lower Priority

6. **Missing velocity information** in Rendezvous mean-embedding observations

---

## 10. Recommended Fix Order

1. **Fix bearing representation** (cos, sin) in all environments
2. **Fix distance normalization** per environment type
3. **Fix obs_radius calculation** to use comm_radius
4. **Implement proper line-of-control** evasion strategy
5. **Align wall observation format** with Hüttenrauch

---

## File References

### Hüttenrauch
- Pursuer agent (PE): `../deep_rl_for_swarms/deep_rl_for_swarms/ma_envs/agents/point_agents/pursuer_agent.py`
- Evader agent: `../deep_rl_for_swarms/deep_rl_for_swarms/ma_envs/agents/point_agents/evader_agent.py`
- Rendezvous agent: `../deep_rl_for_swarms/deep_rl_for_swarms/ma_envs/agents/point_agents/rendezvous_agent.py`
- PE environment: `../deep_rl_for_swarms/deep_rl_for_swarms/ma_envs/envs/point_envs/pursuit_evasion.py`
- Rendezvous environment: `../deep_rl_for_swarms/deep_rl_for_swarms/ma_envs/envs/point_envs/rendezvous.py`

### Current Implementation
- PE environment: `environments/pursuit/pursuit_evasion_env.py`
- PE evasion agent: `environments/pursuit/evasion_agent.py`
- Rendezvous environment: `environments/rendezvous/rendezvous_env.py`
- Rendezvous observations: `environments/rendezvous/observations_vectorized.py`
