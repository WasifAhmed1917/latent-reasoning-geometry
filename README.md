# Latent Reasoning: Geometry of LLM Representations

I am studying whether the geometry of a language model's hidden representations differs between correct and incorrect reasoning, and whether those differences generalize beyond arithmetic. This repository contains my working Gemma notebooks from that ongoing project at LUMS under Dr. Muhammad Tahir.

I started with GSM8K rollouts from Gemma 3 1B IT. I compare representations across layers and reasoning phases using PCA/SVD, effective rank, spectral entropy, anisotropy, and principal angles. I also use random-group controls and prompt-disjoint comparisons to check how much of the observed separation is specific to correctness. The notebooks include a phase-based correctness probe and an extension to paired safe/unsafe responses from PKU-SafeRLHF.

My initial analyses show separation between correct and incorrect representations. I treat this as preliminary observational evidence: it does not by itself establish a causal mechanism, a reliable detector for unseen tasks, or a universal signal of correctness.

## Notebooks

All notebooks are in [`notebooks/`](notebooks/). I have kept the original filenames so the different experiment versions remain identifiable.

| Notebook | Purpose |
|---|---|
| `Gemma_dgen_v3_lean_runready.ipynb` | GSM8K rollout generation, answer evaluation, and raw layer-state extraction |
| `gemma3_analysis_final.ipynb` | Centered and uncentered spectral analysis across layers and phases |
| `gemma_angle_analysis_final.ipynb` | Principal-angle analysis with random-group controls |
| `gemma_angle_analysis_1000_iterations_final.ipynb` | Principal-angle controls over 1,000 random splits |
| `Gemma_Principal_Angle_Extended_Experiments.ipynb` | Rollout-disjoint and prompt-disjoint comparisons |
| `Gemma_Principal_Angle_Extended_Experiments_tqdm (1).ipynb` | Progress-reporting version of the extended experiments |
| `Gemma_Principal_Angle_Follow_Up_Experiments.ipynb` | Repeated analysis at different sample sizes |
| `gemma_final_layer_nll_entropy_logging_final_script.ipynb` | Final-layer token NLL and predictive entropy |
| `gemma_probe.ipynb` | Reasoning-phase classifier with a prompt-level train/test split |
| `Gemma_PKU_SafeRLHF_runready_final.ipynb` | Response-only extraction from existing paired safety responses |
| `Gemma_PKU_SafeRLHF_runready_final_5k_with_analysis.ipynb` | Larger safety-data extraction and analysis variant |

## Running the notebooks

I developed these in Google Colab with GPU access and Google Drive storage. Install the dependencies in each notebook, authenticate with your own Hugging Face account where requested, and update the Drive paths to match your files. Access to `google/gemma-3-1b-it` requires accepting its Hugging Face access terms.

For GSM8K, start with the generation notebook, then point the analysis notebooks at the saved records and activation files. The safety notebooks use existing dataset responses rather than generating new ones. Check the run mode and sample count before starting; some saved configurations enable full collection.

I have included the notebooks and retained research outputs where available, but not the large activation arrays or model weights. These are working research notebooks with external file dependencies, rather than a self-contained benchmark or a finished paper. The safety extension and causal interpretation are still in progress.
