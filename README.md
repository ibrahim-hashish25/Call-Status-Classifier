# Arabic Call Sentiment – Streamlit

## What it does
- Accepts an `.xlsx` workbook.
- Reads the worksheet named `البيانات`.
- Finds the exact header `Notes` in row 1.
- Runs Arabic sentiment analysis using `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`.
- Adds/updates `حالة المكالمة` in that worksheet.
- Preserves the other worksheets and workbook content through `openpyxl` (some advanced Excel features may not be preserved by openpyxl).

## Business rule in this starter version
- Positive sentiment -> `ناجحة`
- Negative or neutral sentiment -> `غير ناجحة`
- Blank Notes -> `غير ناجحة`

Sentiment is only a proxy for call outcome. A positive note does not necessarily mean the call achieved its business objective. Validate the model against manually labelled examples before operational use.

## Run locally
Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights from Hugging Face and may take several minutes. A GPU is optional; CPU works but can be slower.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. The first startup downloads the model, so allow time for the initial build.
5. Test with a copy of a workbook containing `البيانات` and `Notes`.

## Before production
- Review a representative sample of predictions with call-centre experts.
- Build a labelled dataset where each note has a confirmed outcome (`ناجحة` / `غير ناجحة`).
- Evaluate precision, recall and F1, especially for the unsuccessful class.
- Consider fine-tuning a classifier on your own call notes and business definition of success.
- Do not upload confidential customer information to a public deployment unless your organisation has approved the data-processing and hosting arrangement.
