# Public-release checklist

This directory is safe as an internal staging package, not yet as a paper-linked public GitHub release. Complete and document each item before making it public or adding its URL/DOI to the manuscript.

## Scope and code

- [x] Scientific core separated from the production web service.
- [x] No editable frontend source is claimed where none is available.
- [x] Demo configuration, non-web runner, tests and release preflight check included.
- [ ] Confirm that the exact source revision matches the code used for every paper result.
- [ ] Add a deterministic preprocessing and figure/table-generation script for every reported study.
- [ ] Add expected metric/figure-source outputs and checksums for independent verification.
- [ ] Validate the dependency environment on a clean machine and archive a platform/Python-specific lockfile or container digest.

## Data and provenance

- [ ] Obtain written permission or a documented public licence for every dataset and derived table.
- [ ] Add the four paper-study data inputs, or a stable approved access route if data cannot be redistributed.
- [ ] Record source DOI/URL, checksum, row-ID column, preprocessing, retained/excluded rows and split manifest in `data/metadata/data_manifest.csv`.
- [ ] Add the exact per-study configuration: requested and effective generator, selected columns/roles/types, sample counts, seed, target and prediction settings.
- [ ] Confirm that no user, partner, credential-bearing or unpublished data appears in Git history or release artifacts.

## Publication metadata

- [ ] Choose an institutionally approved code licence and add it as `LICENSE`.
- [ ] Complete and rename `CITATION.cff.template` to `CITATION.cff` with the final authors, version, DOI and repository URL.
- [ ] Add a release tag and archive the exact tagged source in a DOI-granting repository if the paper's code-availability statement cites one.
- [ ] Update the manuscript's Code availability and Data availability statements with only verified links, versions and access conditions.

## Final verification

- [ ] Run `python scripts/preflight_release.py --strict-public-release` successfully.
- [ ] Clone the repository into a clean directory; install dependencies; run the demo, tests and each paper-study reproduction command.
- [ ] Have a co-author independently compare generated tables/figures with the submitted manuscript.
