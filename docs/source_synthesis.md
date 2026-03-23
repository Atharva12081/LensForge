# Source Synthesis for LensForge

This note summarizes how the external references from the DeepLense LSST pipeline brief shaped the LensForge design.

## LSST Pipeline Docs

Primary source:
- [LSST Science Pipelines / Butler docs](https://pipelines.lsst.io/modules/lsst.daf.butler/index.html)

Key concept used:
- the Butler is an abstracted data access interface that reads and writes data without exposing file-format or storage-location details directly
- datasets are organized around dataset types, collections, and data IDs
- provenance matters as much as pixels when building reproducible survey workflows

How LensForge uses it:
- the mock Rubin pipeline is organized into `query -> fetch -> cutout -> preprocess -> package`
- packaged rows now retain `dataset_type`, `collection`, `bands`, and `source_kind`
- the Rubin-facing notebook and design note explicitly separate TAP-style discovery from Butler-backed retrieval

## Paper 1

Primary source:
- [Decoding Dark Matter Substructure without Supervision](https://arxiv.org/abs/2008.12731)

Relevant idea:
- theory-agnostic or unsupervised learning can reveal meaningful lens morphology structure without requiring every downstream analysis to be framed only as supervised classification

How LensForge uses it:
- the LSST packaging layer is described as a shared substrate for future embedding, clustering, anomaly detection, and morphology-discovery workflows
- manifests and provenance are preserved so the same packaged data can support unsupervised experiments later

## Paper 2

Primary source:
- [Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346)

Relevant idea:
- supervised morphology learning on strong-lensing images is feasible and scientifically meaningful when image structure is preserved and class semantics are explicit

How LensForge uses it:
- the pipeline keeps class labels, split metadata, and multi-band channel semantics explicit
- the downstream Common Test I and Test V tasks are framed as consumers of standardized, morphology-aware packaged cutouts

## Repository consequence

Together, these references support a submission architecture with:

- LSST-style upstream access and provenance handling
- task adapters for lens finding, classification, and super-resolution
- room for both supervised and unsupervised downstream DeepLense workflows

This is the key reason LensForge is structured as a pipeline-plus-model repository instead of only a standalone classifier notebook.
