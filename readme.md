> **DEPRECATED — this branch is stale.** `main` predates the IS_Validate dataset-level /
> sample-level schema refactor: `binary_subject_prep` here still reads
> `request['dataset_info']['task_channels']`, a request shape the validation framework no
> longer sends (current requests carry `dataset_level_schema` / `sample_level_schema`
> instead). Discovered 2026-08-28 while cherry-picking an unrelated fix onto this branch —
> the fix couldn't apply because this branch never received the schema-refactor work that
> `gpu_heavy` has. Use **`gpu_heavy`** for anything current; this branch is kept only for
> history until someone decides whether to port it forward or retire it for real.

# SAMMed3D_App 

An adaptation of the SAM-Med3D implementation which was submitted to the SegFM challenge, as part of the evaluation stack for the IS_Evaluation_Framework by Esmaeili et al. 

This implementation was used as the guide for the most part, since it provided the logic for performing inference on images with different voxel counts and image spacings.

For our evaluation, we used SAM-Med3D turbo as the checkpoint. As such, the image normalisation preprocessing was left as-is with reference to the main branch of the SAM-Med3D repository. 