# Real-acquisition selection

A deterministic selection of existing M4Raw data, not a newly collected cohort. The original creators' paper, ethics/consent description, preprocessing and CC BY 4.0 license remain authoritative. Attribution/source links: `SOURCE_NOTICE.md`.

180 unchanged source H5 payloads; 100 distinct healthy volunteers: twenty training, thirty calibration, twenty-five expanded-test and twenty-five motion-risk participants. Roles are disjoint across every repetition and contrast. There are 20 training T2 acquisitions, 60 calibration T2/FLAIR acquisitions, 75 test acquisitions (T2 repeats 1/2, FLAIR repeat 1), and 25 T2 motion-risk acquisitions. The motion-risk group contains eight inter-scan and seventeen intra-scan participants. This participant-level label does not certify an artifact in each individual file.

One 0.3-T scanner and one four-channel coil. Stored k-space: `[18 slices,4 coils,256 FE,256 PE]`, complex64. The experiment uses fixed slices 7/8/9. T2 has 195 acquired PE columns; FLAIR 198; remaining stored columns are unavailable zeros. The source applies off-center phase correction and supplies an all-available RSS reference. No noise-free anatomical truth, disease labels or clinician-reviewed unsupported-feature annotations are supplied. Actual repeats remain noisy references.

`manifest.json` freezes creators' mirror revision `46cb84455c0d6653b3b4afe27d56b10fbd4cd424`, source paths, participant/sequence/repetition metadata, sizes and SHA256. Every selected file was verified in the completed cloud run. The 2,221,536,429-byte selection is downloaded directly using `scripts/download_m4raw.py`; raw H5 payloads are not bundled into this repository or relabeled as synthetic data.

CC BY 4.0 applies to M4Raw, derived selection metadata and source-based figures. Code is MIT; original synthetic data remain CC0 under their own card. Preserve source attribution and distinguish actual acquisitions from retrospective sampling modifications. No new demographics or pathology labels are inferred from image appearance.
