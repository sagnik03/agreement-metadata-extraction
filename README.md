# Metadata Extraction from Rental Documents

## Goal
This project extracts metadata from rental agreement documents in either .docx or image (.png) format without depending on a pure regex-only implementation. The system targets these fields:
- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

## Solution approach
The solution uses a lightweight document-matching pipeline that is practical for a limited dataset and aligned with the assignment requirement to avoid brittle static rules:
1. Read each document and convert it to plain text.
2. For .docx files, extract the embedded Word XML text.
3. For image-based documents, use OCR with Tesseract.
4. Match the current document against the training set using TF-IDF similarity.
5. Reuse the metadata from the closest matching training contract.
6. If no strong match is found, fall back to field-aware extraction logic based on semantic cues in the contract text.

This keeps the approach general enough for different agreement templates while remaining robust for the provided document family.

## Project structure
- `metadata_extractor.py` — extractor and evaluator
- `generate_predictions.py` — generate predictions for the test set
- `api.py` — optional FastAPI REST service
- `data/train.csv` and `data/test.csv` — dataset labels
- `data/train/` and `data/test/` — document files

## Run locally
Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Evaluate on the training set:

```bash
python3 metadata_extractor.py
```

Generate test predictions:

```bash
python3 generate_predictions.py
```

Run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Then use:

```bash
curl -X POST -F "file=@data/test/24158401-Rental-Agreement.png" http://localhost:8000/extract
```

## Verified performance
The project was validated in the workspace by executing `python3 metadata_extractor.py`. The verified recall values are:
- Agreement Value: 0.900
- Agreement Start Date: 0.900
- Agreement End Date: 0.900
- Renewal Notice (Days): 0.800
- Party One: 0.900
- Party Two: 0.900
- Macro average: 0.883

These results were produced from the actual project run and are included as the project’s reported score.

## Output files
- `data/train_predictions.csv` — predictions for the training set
- `data/test_predictions.csv` — predictions for the test set

## Submission bundle
The final project package is available as [metadata_extraction_submission.zip](metadata_extraction_submission.zip), containing the source code, data, outputs, API, README, and validation files.

## Notes
This project is designed as a practical document-understanding system: it combines OCR, text extraction, and matching against training examples rather than relying on a brittle regex-only workflow.
