# Read but not Cited? Discerning the Academic Impacts of Papers with Delayed Recognition in Science

This analysis code is the official code for "Read but not Cited? Discerning the Academic Impacts of Papers with Delayed Recognition in Science."

## Details

- 1-1_clustering.py: Identifies the field of each paper through clustering of citation networks.

- 1-2_paper_detail.py: Adds citation information, including citation delay.

- 1-3_field_demographics.ipynb: Retrieves basic information about each field.

- 2_create_data.ipynb: Creates master data.

- 3_read_vs_cite.ipynb: Analyzes citation and readership using pairwise matching.

- 4_regression.ipynb: Estimates variables that affect citation and readership using random effect models and mixed effect models.

- 5_read_per_cite_for_predict.ipynb: Validates prediction accuracy for growing areas.

- 6_read_per_cite_mapping: Mapping.

- check_valid_readers: Supplementary, validation based on data where readership is greater than or equal to one.

## Dataset

The data used is the Scopus Custom Dataset extracted in July 2022. 
This dataset, obtained through a contract with Elsevier, includes metadata and citation information for 70 million documents. 
From this, papers for which readership data could be obtained via the Mendeley API using DOI as the key was targeted.
