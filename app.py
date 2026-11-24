import io
import re
import streamlit as st
import torch
from openpyxl import load_workbook
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
SHEET_NAME = "البيانات"
NOTES_COLUMN = "Notes"
OUTPUT_COLUMN = "حالة المكالمة"

st.set_page_config(page_title="تحليل حالة المكالمات", page_icon="📊", layout="wide")
st.title("تحليل حالة المكالمات باستخدام الذكاء الاصطناعي")
st.write("ارفع ملف Excel لتحليل محتوى عمود Notes وإضافة حالة المكالمة تلقائياً.")

@st.cache_resource(show_spinner="جاري تحميل نموذج تحليل المشاعر...")
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )
    return classifier, model.config.id2label

def normalize_label(raw_label, id2label):
    """Convert model labels to positive/negative/neutral where possible."""
    label = str(raw_label).strip().lower()
    if label.startswith("label_"):
        try:
            label = str(id2label[int(label.split("_")[-1])]).strip().lower()
        except (ValueError, KeyError, IndexError):
            pass

    # Support common English and Arabic label spellings.
    if any(x in label for x in ["positive", "pos", "إيجابي", "ايجابي"]):
        return "positive"
    if any(x in label for x in ["negative", "neg", "سلبي"]):
        return "negative"
    if any(x in label for x in ["neutral", "neu", "محايد"]):
        return "neutral"
    raise ValueError(
        f"تعذّر التعرّف على تصنيف النموذج: {raw_label}. "
        "تحقق من تسميات النموذج قبل استخدامه."
    )

def classify_notes(notes, classifier, id2label, batch_size=16):
    results = []
    valid_positions = [i for i, value in enumerate(notes) if value and str(value).strip()]
    texts = [str(notes[i]).strip() for i in valid_positions]

    # Initialize blank/empty notes as unsuccessful by the stated business rule.
    results = ["غير ناجحة"] * len(notes)

    if not texts:
        return results

    predictions = classifier(texts, batch_size=batch_size)
    for position, prediction in zip(valid_positions, predictions):
        sentiment = normalize_label(prediction["label"], id2label)
        results[position] = "ناجحة" if sentiment == "positive" else "غير ناجحة"
    return results

uploaded_file = st.file_uploader("اختر ملف Excel بصيغة .xlsx", type=["xlsx"])

if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        workbook = load_workbook(io.BytesIO(file_bytes))
        if SHEET_NAME not in workbook.sheetnames:
            st.error(f'لم يتم العثور على ورقة باسم "{SHEET_NAME}".')
            st.stop()

        worksheet = workbook[SHEET_NAME]
        headers = [cell.value for cell in worksheet[1]]
        if NOTES_COLUMN not in headers:
            st.error(f'لم يتم العثور على عمود "{NOTES_COLUMN}" في الصف الأول من ورقة "{SHEET_NAME}".')
            st.stop()

        notes_col = headers.index(NOTES_COLUMN) + 1
        output_col = headers.index(OUTPUT_COLUMN) + 1 if OUTPUT_COLUMN in headers else worksheet.max_column + 1
        worksheet.cell(row=1, column=output_col, value=OUTPUT_COLUMN)

        notes = [
            worksheet.cell(row=row, column=notes_col).value
            for row in range(2, worksheet.max_row + 1)
        ]

        st.info(f"تم العثور على {len(notes):,} صفاً في ورقة البيانات.")
        if st.button("ابدأ تحليل المكالمات", type="primary"):
            with st.spinner("جاري تحليل الملاحظات... قد يستغرق ذلك بعض الوقت في أول تشغيل."):
                classifier, id2label = load_sentiment_model()
                statuses = classify_notes(notes, classifier, id2label)
                for row, status in enumerate(statuses, start=2):
                    worksheet.cell(row=row, column=output_col, value=status)

                output = io.BytesIO()
                workbook.save(output)
                output.seek(0)

            successful = statuses.count("ناجحة")
            unsuccessful = statuses.count("غير ناجحة")
            c1, c2, c3 = st.columns(3)
            c1.metric("إجمالي المكالمات", f"{len(statuses):,}")
            c2.metric("ناجحة", f"{successful:,}")
            c3.metric("غير ناجحة", f"{unsuccessful:,}")

            st.success("اكتمل تحليل الملف بنجاح.")
            st.download_button(
                "تحميل ملف Excel بعد التحليل",
                data=output.getvalue(),
                file_name=f"analyzed_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
    except Exception as exc:
        st.error(f"حدث خطأ أثناء معالجة الملف: {exc}")
        st.exception(exc)

st.caption(
    "ملاحظة: هذا الإصدار يستخدم المشاعر كمؤشر مبدئي لنجاح المكالمة. "
    "ينبغي مراجعة النتائج على عينة من بياناتك قبل الاعتماد التشغيلي."
)
