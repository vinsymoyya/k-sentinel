# Publication and author-record guide

Author: Vinsy Moyya. ORCID supplied by the author: https://orcid.org/0009-0003-0042-3556. No affiliation is asserted. Date: 3 November 2026. Version 0.1.0 is an unreviewed synthetic methods preprint and software release; it is not journal acceptance.

## Ready artifacts

- `output/pdf/k-sentinel-preprint.pdf`: methods, measured results, figures, references and limitations.
- `output/pdf/medical-imaging-ai-market-analysis.pdf`: source-backed opportunity analysis.
- `paper/citation.bib`: importable work citation without an invented DOI.
- `CITATION.cff`: GitHub software citation metadata.
- `.zenodo.json`: software-deposit metadata, not proof of a Zenodo deposit.
- Source, generated data, checkpoints and complete result record.

## ORCID

ORCID is an identity and research-output record, not a peer-reviewed publication venue. Once signed in, use Works -> Add -> Import BibTeX and select `paper/citation.bib`, or add the preprint manually using its public URL. Choose an accurate type such as preprint. Do not label the work a published journal article. If a DOI is later minted, update the existing record rather than create unnecessary duplicates. Official guidance: https://support.orcid.org/hc/en-us/articles/360006973133-Add-works-to-your-ORCID-record.

## Durable DOI, if desired

A GitHub release does not automatically create a DOI. Zenodo's GitHub integration requires connecting and enabling the repository; a release can then be archived through that integration. An independent manuscript deposit is another option. Complete the authenticated deposit and verify the resulting DOI before adding it to the paper, citation files or ORCID. No DOI has been fabricated in this package. Official integration guide: https://help.zenodo.org/docs/github/enable-repository/.

## Before journal submission

The present study is suitable for public sharing as an unreviewed computational report. Journal publishability and novelty are not guaranteed. The author must complete funding and competing-interest declarations and confirm the target venue's current authorship, disclosure and preprint policies. Real-data experiments, stronger reconstruction baselines and external assessment are the most consequential scientific additions. No institutional ethics exemption, clinician validation, funding statement or acceptance is invented here.

Live publication state is recorded separately in `release_status.json`; this guide is not a claim that profile updates, a DOI deposit or peer review have been completed.
