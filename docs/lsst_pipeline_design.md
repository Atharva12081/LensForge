# LSST Pipeline Workflow

This note covers the project-description side of the DeepLense "Data Processing Pipeline for the LSST" task and maps directly to the runnable code in `src/lsst_pipeline/` and `run_lsst_mock_pipeline.py`.

## Goal

Build a practical workflow that connects Rubin/LSST-style data access tools to DeepLense downstream tasks such as:

- lens finding
- classification
- super-resolution

This workflow is now framed around three source-backed ideas from the project links:

- Butler-centered data access from the LSST Science Pipelines documentation, where datasets are identified by dataset type, collection, and data IDs rather than raw file paths alone.
- supervised morphology learning from "Deep Learning the Morphology of Dark Matter Substructure" ([arXiv:1909.07346](https://arxiv.org/abs/1909.07346)), which motivates keeping downstream classification-ready cutouts and labels explicit.
- unsupervised or theory-agnostic representation discovery from "Decoding Dark Matter Substructure without Supervision" ([arXiv:2008.12731](https://arxiv.org/abs/2008.12731)), which motivates preserving provenance-rich packaged artifacts for later embedding, clustering, or anomaly-detection workflows.

## Implemented workflow

1. Discover data products.
Use a query stage to identify candidate objects and stable identifiers.

2. Retrieve science-ready inputs.
Fetch the calibrated image planes and associated metadata needed for a target task.

3. Build task-specific cutouts.
Convert larger survey images into per-object tensors or cutouts centered on candidate systems.

4. Standardize representation.
Normalize filters, align channels, record provenance, and convert data into the tensor layout expected by DeepLense models.

5. Export reproducible artifacts.
Save model-ready arrays plus metadata manifests so downstream experiments can be rerun and audited.

6. Support multiple downstream analysis styles.
Keep the packaged output usable not just for supervised classifiers, but also for unsupervised morphology discovery, retrieval, and future neural-operator experiments.

## Implemented components

### 1. `query`

Responsibility:
- query mock-survey tables or local survey-like folders
- filter to a bounded reproducible subset for testing

Inputs:
- data root
- optional per-folder cap

Outputs:
- a table of candidate objects with stable identifiers

Current implementation:
- `src/lsst_pipeline/query.py`

### 2. `fetch`

Responsibility:
- retrieve calibrated images and ancillary metadata for selected objects
- cache raw results locally to avoid repeated network access

Inputs:
- object identifiers from the query stage
- desired bands / filters

Outputs:
- raw downloaded image files
- metadata sidecars

Current implementation:
- `src/lsst_pipeline/fetch.py`

### 3. `cutout`

Responsibility:
- create fixed-size image cutouts around each object
- align multi-band inputs into a consistent channel stack

Inputs:
- calibrated survey images
- object coordinates and cutout sizes

Outputs:
- per-object image tensors
- stable channel ordering metadata so multi-band downstream models can trust the input semantics

Current implementation:
- `src/lsst_pipeline/cutout.py`

### 4. `preprocess`

Responsibility:
- normalize flux ranges
- clip invalid pixels
- resize or pad to model input sizes
- convert arrays into the target DeepLense tensor schema

Outputs:
- model-ready arrays such as `(C, H, W)`
- preprocessing metadata for reproducibility

Current implementation:
- `src/lsst_pipeline/preprocess.py`

### 5. `package`

Responsibility:
- split data into train / validation / test sets
- write manifests and labels
- expose a stable folder format for DeepLense experiments

Outputs:
- dataset directory
- manifest JSON / CSV
- provenance report

Additional repository-aligned packaging fields:
- `dataset_type`
- `collection`
- `bands`
- `source_kind`

These fields mirror the Butler-style habit of treating dataset identity and provenance as first-class metadata rather than hidden assumptions.

Current implementation:
- `src/lsst_pipeline/package.py`
- `src/lsst_pipeline/pipeline.py`
- `run_lsst_mock_pipeline.py`

## Mock-survey validation plan

The project description explicitly mentions testing on Rubin mock surveys. LensForge now includes a local mock-survey adapter over the Test V dataset so the same pipeline stages can be executed and verified end to end without depending on external services.

Validation path used in this repository:

1. run the pipeline on a small mock-survey subset
2. verify that object ids, band ordering, and cutout geometry are consistent
3. export a packaged dataset in DeepLense format
4. feed that packaged output into a downstream lens-finding or classification baseline
5. confirm that the entire path from query to model input is reproducible

## How this connects to LensForge

LensForge currently implements the downstream model side for:

- Common Test I multi-class classification
- Test V lens finding

The LSST pipeline layer would sit upstream of those models and produce the standardized arrays they consume.

## Current repository artifact

The repository now contains:

- runnable pipeline code in `src/lsst_pipeline/`
- a CLI runner in `run_lsst_mock_pipeline.py`
- a notebook in `output/jupyter-notebook/lsst-mock-pipeline.ipynb`
- a compact run summary in `reports/lsst_mock_pipeline_summary.md`

If Rubin/LSST APIs are available later, the `query` and `fetch` stages are the places that need real service adapters plus the appropriate external Rubin environment and credentials. The downstream cutout, preprocess, package, and model handoff logic can stay the same once those upstream dependencies are satisfied.

## Mock-to-real adapter path

LensForge now separates the two upstream Rubin-facing responsibilities clearly:

- `query` should map to Rubin TAP discovery against `ivoa.ObsCore`, using the
  same `get_tap_service("tap")` access pattern documented in the DP0.2
  tutorials.
- `fetch` should map to Butler-backed dataset discovery and retrieval, using
  `Butler("dp02", collections="2.2i/runs/DP0.2")` plus
  `query_datasets("calexp", ...)` in a Rubin notebook environment where the
  repository index aliases are defined.

The reviewer-facing notebook
`output/jupyter-notebook/rubin-dp02-access.ipynb` exists to make that boundary
concrete. It does not attempt a full end-to-end replacement of the mock
pipeline, and it should not be interpreted as a self-contained local proof that
Rubin services are runnable everywhere. Instead, it demonstrates the exact
external service entry points that the future real `query` and `fetch` adapters
should wrap once the proper Rubin environment is available.

## Paper-grounded task adapters

The linked papers suggest that the LSST-facing pipeline should not stop at one single classifier handoff. LensForge now treats the packaged survey output as a common substrate for multiple DeepLense task families:

- `lens_finding`
  Uses calibrated multi-band cutouts and explicit labels for binary discovery tasks like Test V.
- `classification`
  Supports supervised morphology studies in the style of [arXiv:1909.07346](https://arxiv.org/abs/1909.07346), where distinct substructure classes are compared directly.
- `super_resolution`
  Preserves stable cutout geometry and band ordering so the same upstream packaging can feed image-restoration tasks later.
- `representation_learning`
  Keeps enough provenance for unsupervised workflows in the spirit of [arXiv:2008.12731](https://arxiv.org/abs/2008.12731), where embeddings, clustering, or anomaly scores may be the primary output instead of class labels.

## Why this improves the submission

This iteration makes LensForge look more aligned with the actual project brief and source material because it now:

- uses LSST-native concepts like dataset type, collection, and provenance explicitly
- connects the upstream data pipeline to both supervised and unsupervised DeepLense use cases
- presents the mock pipeline as a credible adapter boundary for future Rubin access, not just a folder reshuffling script
