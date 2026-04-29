# Artifact Vocabulary

Preferred field names in data-prep artifacts:

- `benchmark_id`
- `question_id`
- `question_hash`
- `program_key` (DSL/canonical compositional)
- `program`
- `primitive_indices`
- `deviation_key` (legacy fine-routing deviations)
- `edit_sequence`
- `delta_matrix`
- `anchor_utilities`
- `row_metadata`
- `col_metadata`
- `source_path`
- `row_alignment`

## Dense Alignment Notes

Dense matrix artifacts should record:

- Source artifact path
- Row alignment assumption (`question_id` or explicit mapping)
- Column row-space source (`legal_programs/...` or joint catalogue)
- Number of questions and candidates/programs
- Fill value used for missing entries (if any)

When remapping to joint catalogues, write benchmark-specific keep masks so
unmeasured joint rows are excluded during supervision.

