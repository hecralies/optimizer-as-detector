# Source-code package

This ZIP contains the maintained Python scripts, workflow runner, dependency
specification, and input-data checksum manifest for the JASA manuscript. It
does not contain datasets, precomputed numerical results, manuscript source,
or the Author Contributions Checklist (ACC).

To execute the workflow, extract this archive at the root of the complete
reproducibility repository, alongside the `data/` directory described by
`data_manifest.json`. Then follow `README.md`.

For an invited JASA revision, this source-only ZIP should be incorporated into
the journal's requested `reproducibility_materials.zip`, together with the
permitted data, output files needed to verify manuscript claims, and workflow
documentation. Submit the completed ACC PDF separately.

The historical California Housing full-design KS result must be regenerated
with the corrected four-predictor implementation before the final materials
are deposited or submitted; see `README.md`.
