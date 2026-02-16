# Output Locations Audit

## Executive Summary

**Status**: ✅ **FIXED** - All scripts now use consistent output locations

### Designated Locations
- **Pickle files**: `experimental_results/` (repo root)
- **Plot images**: `experimental_results/plots/` with subfolders:
  - `plots/values/` - Raw metric value plots
  - `plots/pairwise/` - Pairwise difference plots
  - `plots/invalid_presummarization/` - Archived invalid plots

### Issue Found and Fixed
`scripts/run_pca_kllmeans_sweep.py` was saving pickle files to **current working directory** instead of `experimental_results/`. This has been corrected.

---

## Detailed Audit Results

### ✅ Scripts - Correctly Using `experimental_results/`

#### 1. `scripts/run_experimental_sweep.py`
- **Line 37-38**: 
  ```python
  OUTPUT_DIR = Path(__file__).parent.parent / "experimental_results"
  OUTPUT_DIR.mkdir(exist_ok=True)
  ```
- **Output**: `experimental_sweep_entry{X}_{dataset}_labelmax{Y}_{model}_{timestamp}.pkl`
- **Status**: ✅ Correct

#### 2. `scripts/run_custom_full_categories_sweep.py`
- **Line 32**: Imports `OUTPUT_DIR` from `run_experimental_sweep`
- **Lines 216, 253**: Uses `OUTPUT_DIR` for save paths
- **Output**: Same format as `run_experimental_sweep.py`
- **Status**: ✅ Correct

---

### ⚠️ Scripts - ~~Saving to Wrong Location~~ **FIXED**

#### 3. `scripts/run_pca_kllmeans_sweep.py`
- **Line 186-236**: `save_results()` function
- **Line 191-196** (FIXED): 
  ```python
  output_dir = Path(__file__).parent.parent / "experimental_results"
  output_dir.mkdir(exist_ok=True)
  output_file = str(output_dir / f"pca_kllmeans_sweep_results_{timestamp}.pkl")
  ```
- **Previous Problem**: No path prefix, saved to current working directory
- **Fix Applied**: Now saves to `experimental_results/` like other sweep scripts
- **Status**: ✅ **FIXED**

---

### ✅ Notebooks - Correctly Configured

#### 4. `notebooks/pca_kllmeans_sweep.ipynb`
- **Cell with line 3451** (WRITE):
  ```python
  output_dir = Path("..") / "experimental_results"
  output_dir.mkdir(exist_ok=True)
  ```
- **Cells with lines 3062, 3294** (READ):
  ```python
  pickle_files = sorted(Path("../experimental_results").glob("pca_kllmeans_sweep_results_*.pkl"), reverse=True)
  ```
- **Output**: `pca_kllmeans_sweep_results_{timestamp}.pkl`
- **Status**: ✅ Correct (both read and write)

#### 5. `notebooks/colab_pca_kllmeans_sweep.ipynb`
- **Cell with line 1076** (WRITE):
  ```python
  output_dir = Path("experimental_results")
  output_dir.mkdir(exist_ok=True)
  ```
- **Note**: Relative to Colab working directory (typically repo root when mounted)
- **Status**: ✅ Correct (for Colab context)

#### 6. `notebooks/pca_kllmeans_analysis.ipynb`
- **Cell with line 58** (READ): 
  ```python
  pickle_files = sorted(Path("../experimental_results").glob("pca_kllmeans_sweep_results_*.pkl"), reverse=True)
  ```
- **Purpose**: Loads pickles (doesn't create)
- **Status**: ✅ Correct

#### 7. `notebooks/sweep_explorer.ipynb`
- **Loading (line 433)** (READ): 
  ```python
  def load_sweep_dataframe(data_dir: str = "../experimental_results") -> pd.DataFrame:
  ```
- **Plotting - Pairwise (line 1583)** (WRITE):
  ```python
  output_dir = Path("..") / "experimental_results" / "plots"
  ```
- **Plotting - Values (line 1837)** (WRITE):
  ```python
  output_dir = Path("..") / "experimental_results" / "plots" / "values"
  ```
- **Status**: ✅ Correct (both read and write)

---

### ✅ Scripts - Read Operations

#### 8. `scripts/verify_embedding_identity.py`
- **Line 41** (READ):
  ```python
  for pkl_file in results_dir.glob("experimental_sweep_*.pkl"):
  ```
- **Purpose**: Reads experimental sweep pickles from `experimental_results/`
- **Status**: ✅ Correct

---

## Recommended Fix

### ~~Update `scripts/run_pca_kllmeans_sweep.py`~~ **APPLIED**

**Status**: ✅ Fix has been applied to the script.

The change updates lines 191-196 to save pickle files to `experimental_results/` directory:

```python
if output_file is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save to experimental_results directory (consistent with other sweep scripts)
    output_dir = Path(__file__).parent.parent / "experimental_results"
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / f"pca_kllmeans_sweep_results_{timestamp}.pkl")
```

---

## Current `.gitignore` Status

```gitignore
experimental_results/
```

✅ Correctly ignoring the designated output directory.

---

## Summary Table

### Write Operations (Creating Files)

| File | Output Type | Location | Status |
|------|-------------|----------|--------|
| `run_experimental_sweep.py` | Pickles | `experimental_results/` | ✅ |
| `run_custom_full_categories_sweep.py` | Pickles | `experimental_results/` | ✅ |
| `run_pca_kllmeans_sweep.py` | Pickles | `experimental_results/` | ✅ **FIXED** |
| `pca_kllmeans_sweep.ipynb` | Pickles | `experimental_results/` | ✅ |
| `colab_pca_kllmeans_sweep.ipynb` | Pickles | `experimental_results/` | ✅ |
| `sweep_explorer.ipynb` | Plots | `experimental_results/plots/` | ✅ |
| `sweep_explorer.ipynb` | Plots | `experimental_results/plots/values/` | ✅ |

### Read Operations (Loading Files)

| File | Input Type | Read Location | Status |
|------|-----------|---------------|--------|
| `sweep_explorer.ipynb` | Pickles | `../experimental_results/` (default) | ✅ |
| `pca_kllmeans_analysis.ipynb` | Pickles | `../experimental_results/` | ✅ |
| `pca_kllmeans_sweep.ipynb` | Pickles | `../experimental_results/` | ✅ |
| `verify_embedding_identity.py` | Pickles | `experimental_results/` (via glob) | ✅ |
| `run_pca_kllmeans_sweep.py` | Estela dict | User-specified (via arg/env) | N/A |
| `analyze_dataset_lengths.py` | Estela dict | User-specified (via arg/env) | N/A |

**All read and write operations are consistent with `experimental_results/` as the designated location.**

---

**Generated**: 2026-02-13  
**Last Updated**: 2026-02-13  
**All Issues Resolved**: ✅
