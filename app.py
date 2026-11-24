import io, os, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from openpyxl import load_workbook

SHEET="Data"; NOTES="Notes"; STATUS="حالة المكالمة"
BASE="CAMeL-Lab/bert-base-arabic-camelbert-da"
SENTIMENT="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
DATA=Path(os.getenv("DATA_DIR","app_data"))
CSV=DATA/"classified_examples.csv"; MODEL=DATA/"trained_model"
REQUIRED=["Team","Created by","Surrogate","Notes","Created on","Created On Date",
"Wasted Time","Wasted Time 2","Actual Wasted Time","Portfolio Type","Debitor Name",
"Debtor","Account Number","Claim","Amount","Main State","Sub State"]
LABELS={"غير ناجحة":0,"ناجحة":1}; ID_LABELS={0:"غير ناجحة",1:"ناجحة"}

st.set_page_config(page_title="تحليل نشاط المكالمات",page_icon="📊",layout="wide")
st.title("تحليل نشاط المكالمات")
st.caption("تحليل الملاحظات وتدريب النموذج على التصنيفات الصحيحة.")

def cairo_date():
    try: return datetime.now(ZoneInfo("Africa/Cairo")).date()
    except Exception: return datetime.now().date()

def validate(upload, training=False):
    wb=load_workbook(io.BytesIO(upload.getvalue()))
    if wb.sheetnames != [SHEET]:
        raise ValueError('يجب أن يحتوي الملف على ورقة واحدة فقط باسم "Data".')
    ws=wb[SHEET]; headers=[c.value for c in ws[1]]
    missing=[x for x in REQUIRED if x not in headers]
    if missing: raise ValueError("أعمدة مفقودة: "+"، ".join(missing))
    if NOTES not in headers: raise ValueError('عمود "Notes" غير موجود.')
    if training and STATUS not in headers:
        raise ValueError('ملف التدريب يجب أن يحتوي على عمود "حالة المكالمة".')
    return wb,ws,headers

def ensure_status(ws, headers):
    desired=headers.index(NOTES)+2
    if STATUS in headers:
        old=headers.index(STATUS)+1
        if old != desired:
            vals=[ws.cell(r,old).value for r in range(1,ws.max_row+1)]
            ws.delete_cols(old,1)
            if old < desired: desired-=1
            ws.insert_cols(desired,1)
            for r,v in enumerate(vals,1): ws.cell(r,desired,v)
    else:
        ws.insert_cols(desired,1)
    ws.cell(1,desired,STATUS)
    return desired

@st.cache_resource
def tokenizer(): return AutoTokenizer.from_pretrained(BASE)

@st.cache_resource
def baseline():
    tok=AutoTokenizer.from_pretrained(SENTIMENT)
    mdl=AutoModelForSequenceClassification.from_pretrained(SENTIMENT)
    return pipeline("text-classification",model=mdl,tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,truncation=True,max_length=512)

def predict(texts):
    if (MODEL/"config.json").exists():
        tok=tokenizer(); mdl=AutoModelForSequenceClassification.from_pretrained(str(MODEL))
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(dev).eval(); result=[]
        with torch.no_grad():
            for i in range(0,len(texts),16):
                enc=tok(texts[i:i+16],return_tensors="pt",padding=True,truncation=True,max_length=256)
                logits=mdl(**{k:v.to(dev) for k,v in enc.items()}).logits
                probs=torch.softmax(logits,1)
                for idx,conf in zip(probs.argmax(1).tolist(),probs.max(1).values.tolist()):
                    result.append((ID_LABELS[idx],float(conf)))
        del mdl
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result
    outputs=baseline()(texts,batch_size=16); id2=baseline().model.config.id2label
    result=[]
    for out in outputs:
        raw=out["label"].lower()
        if raw.startswith("label_"):
            try: raw=str(id2[int(raw.split("_")[-1])]).lower()
            except Exception: pass
        positive=("positive" in raw or "pos" in raw or "إيجابي" in raw or "ايجابي" in raw)
        result.append(("ناجحة" if positive else "غير ناجحة",float(out["score"])))
    return result

def classify(notes):
    positions=[i for i,n in enumerate(notes) if n is not None and str(n).strip()]
    texts=[str(notes[i]).strip() for i in positions]
    output=[("",0.0) for _ in notes]
    if texts:
        for pos,pred in zip(positions,predict(texts)): output[pos]=pred
    return output

class NoteDataset(Dataset):
    def __init__(self, texts, labels, tok): self.texts=texts; self.labels=labels; self.tok=tok
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        x=self.tok(self.texts[i],truncation=True,padding="max_length",max_length=256,return_tensors="pt")
        item={k:v.squeeze(0) for k,v in x.items()}
        item["labels"]=torch.tensor(self.labels[i],dtype=torch.long)
        return item

def import_labels(upload):
    _,ws,headers=validate(upload,True)
    ni=headers.index(NOTES)+1; li=headers.index(STATUS)+1; rows=[]
    for r in range(2,ws.max_row+1):
        note=ws.cell(r,ni).value; label=ws.cell(r,li).value
        if note is None or not str(note).strip(): continue
        label=str(label).strip()
        if label not in LABELS: raise ValueError(f'تصنيف غير صالح في الصف {r}: استخدم "ناجحة" أو "غير ناجحة".')
        rows.append({"Notes":str(note).strip(),"label":label})
    if not rows: raise ValueError("لم يتم العثور على أمثلة مصنفة.")
    DATA.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(CSV,dtype=str).fillna("") if CSV.exists() else pd.DataFrame(columns=["Notes","label"])
    all_data=pd.concat([old,pd.DataFrame(rows)],ignore_index=True)
    all_data=all_data.drop_duplicates("Notes",keep="last")
    all_data=all_data[all_data.label.isin(LABELS) & all_data.Notes.str.strip().ne("")]
    all_data.to_csv(CSV,index=False,encoding="utf-8-sig")
    return all_data

def train(epochs):
    df=pd.read_csv(CSV,dtype=str).fillna("")
    counts=df.label.value_counts().to_dict()
    if len(df)<20 or any(counts.get(x,0)<2 for x in LABELS):
        raise ValueError("يلزم 20 مثالاً على الأقل ومثالان على الأقل لكل تصنيف. الأفضل توفير مئات الأمثلة المتوازنة.")
    tok=tokenizer()
    ds=NoteDataset(df.Notes.tolist(),[LABELS[x] for x in df.label],tok)
    loader=DataLoader(ds,batch_size=8,shuffle=True)
    mdl=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=2,
        id2label=ID_LABELS,label2id=LABELS)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); mdl.to(dev)
    opt=torch.optim.AdamW(mdl.parameters(),lr=2e-5)
    bar=st.progress(0); total=epochs*len(loader); step=0
    mdl.train()
    for ep in range(epochs):
        for batch in loader:
            batch={k:v.to(dev) for k,v in batch.items()}
            opt.zero_grad(); loss=mdl(**batch).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0); opt.step()
            step+=1; bar.progress(step/total,text=f"Epoch {ep+1}/{epochs}")
    DATA.mkdir(parents=True,exist_ok=True); temp=DATA/"model_tmp"
    if temp.exists(): shutil.rmtree(temp)
    mdl.save_pretrained(temp); tok.save_pretrained(temp)
    if MODEL.exists(): shutil.rmtree(MODEL)
    temp.rename(MODEL); del mdl
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    bar.empty()
    return len(df),counts

t1,t2=st.tabs(["📊 تحليل ملف اليوم","🧠 إضافة بيانات وتدريب النموذج"])
with t1:
    st.subheader("تحليل ملف النشاط اليومي")
    st.write('ارفع ملف Excel يحتوي على ورقة واحدة باسم **Data** والأعمدة المحددة، بما فيها **Notes**.')
    up=st.file_uploader("ملف النشاط (.xlsx)",type=["xlsx"],key="daily")
    if up:
        try:
            wb,ws,headers=validate(up)
            st.success(f"تم التحقق من الملف: {ws.max_row-1:,} سجل.")
            if st.button("تحليل المكالمات وتصدير الملف",type="primary"):
                note_col=headers.index(NOTES)+1; status_col=ensure_status(ws,headers)
                notes=[ws.cell(r,note_col).value for r in range(2,ws.max_row+1)]
                with st.spinner("جاري تحليل الملاحظات..."):
                    preds=classify(notes)
                    for r,(label,score) in enumerate(preds,2): ws.cell(r,status_col,label)
                    out=io.BytesIO(); wb.save(out); out.seek(0)
                labels=[x[0] for x in preds if x[0]]
                a,b,c,d=st.columns(4)
                a.metric("إجمالي الصفوف",len(preds)); b.metric("ناجحة",labels.count("ناجحة"))
                c.metric("غير ناجحة",labels.count("غير ناجحة")); d.metric("ملاحظات فارغة",len(preds)-len(labels))
                filename=f"النشاط {cairo_date():%d-%m-%Y}.xlsx"
                st.success("اكتمل التحليل.")
                st.download_button("تحميل ملف النشاط",out.getvalue(),filename,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary")
        except Exception as e: st.error(str(e))

with t2:
    st.subheader("إضافة التصنيفات الصحيحة وتدريب النموذج")
    st.info('ارفع ملف Data بعد مراجعة التصنيفات يدوياً. يجب أن يحتوي على Notes و"حالة المكالمة"، والقيم المقبولة هي "ناجحة" و"غير ناجحة".')
    labelled=st.file_uploader("ملف Excel المصنف يدوياً (.xlsx)",type=["xlsx"],key="labels")
    if labelled and st.button("إضافة البيانات إلى مجموعة التدريب",type="primary"):
        try:
            df=import_labels(labelled)
            st.success(f"تم حفظ الأمثلة. إجمالي الأمثلة الفريدة المتراكمة: {len(df):,}.")
        except Exception as e: st.error(str(e))
    if CSV.exists():
        df=pd.read_csv(CSV,dtype=str).fillna("")
        st.write(f"**أمثلة التدريب المحفوظة:** {len(df):,}")
        st.dataframe(df.label.value_counts().rename_axis("التصنيف").reset_index(name="عدد الأمثلة"),
                     use_container_width=True,hide_index=True)
        st.caption("كل تدريب يستخدم كامل البيانات المتراكمة، وليس ملف اليوم وحده.")
        epochs=st.select_slider("عدد دورات التدريب",options=[1,2,3,4,5],value=3)
        if st.button("تدريب / تحديث النموذج الآن"):
            try:
                with st.spinner("جاري تدريب النموذج..."):
                    n,counts=train(epochs)
                st.success(f"اكتمل التدريب على {n:,} مثال: {counts.get('ناجحة',0)} ناجحة و{counts.get('غير ناجحة',0)} غير ناجحة.")
                st.cache_resource.clear()
            except Exception as e: st.error(str(e))
    else: st.info("لم تتم إضافة بيانات تدريب بعد.")
st.divider()
st.caption("الملاحظات الفارغة تظل بدون تصنيف. راجع عينة من النتائج بعد كل تدريب. احتفظ بنسخة احتياطية من بيانات التدريب.")
