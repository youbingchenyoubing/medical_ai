# Sample Dataset

This is a simulated dataset for testing radiomics pipelines.

## Dataset Information
- Generated: 2026-05-11 00:07:16
- Number of patients: 10
- Type: Simulated data (for testing purposes only)

## Directory Structure
```
sample/
├── clinical_data.csv      # Clinical and outcome data
├── radiomics_features.csv # Pre-computed radiomics features
├── images/                # Simulated CT images (numpy format)
└── masks/                 # Simulated tumor segmentation masks
```

## Important Note
This is **NOT real medical data** and should only be used for:
- Testing your radiomics analysis pipeline
- Code development and debugging
- Educational purposes

For real research, please download actual public datasets following the
instructions in `docs/DATA_DOWNLOAD_GUIDE.md`.
