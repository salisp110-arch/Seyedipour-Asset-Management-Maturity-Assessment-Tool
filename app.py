# app.py
# -*- coding: utf-8 -*-
import re, json, base64
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")

# ---------------- Optional deps ----------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------------- Safe dirs ----------------
BASE = Path(".")
def _safe_dir(p: Path) -> Path:
    if p.exists():
        if p.is_dir():
            return p
        alt = p.with_name(f"_{p.name}_dir")
        alt.mkdir(parents=True, exist_ok=True)
        return alt
    p.mkdir(parents=True, exist_ok=True)
    return p

DATA_DIR   = _safe_dir(BASE / "data")
ASSETS_DIR = _safe_dir(BASE / "assets")

# ---------------- Topics (40) ----------------
TOPICS_PATH = BASE / "topics.json"
EMBEDDED_TOPICS = [
    {"id":1, "name":"هدف و زمینه (Purpose & Context)", "desc":"Purpose و Context نقطه شروع سیستم مدیریت دارایی هستند..."},
    {"id":2, "name":"مدیریت ذی‌نفعان", "desc":"مدیریت ذی‌نفعان به معنای داشتن یک رویکرد ساختاریافته..."},
    {"id":3, "name":"هزینه‌یابی و ارزش‌گذاری دارایی", "desc":"هزینه‌یابی دارایی شامل شناسایی و ثبت کل هزینه‌های..."},
    {"id":4, "name":"خط مشی مدیریت دارایی", "desc":"خط مشی مدیریت دارایی سندی رسمی است که تعهد سازمان..."},
    {"id":5, "name":"سیستم مدیریت دارایی (AMS)", "desc":"سیستم مدیریت دارایی مجموعه‌ای از عناصر مرتبط..."},
    {"id":6, "name":"اطمینان و ممیزی", "desc":"اطمینان و ممیزی فرآیندهای ساختاریافته‌ای برای ارزیابی..."},
    {"id":7, "name":"استانداردهای فنی و قوانین", "desc":"باید اطمینان حاصل شود که تمامی فعالیت‌ها با قوانین..."},
    {"id":8, "name":"آرایش سازمانی", "desc":"آرایش سازمانی نحوه سازمان‌دهی افراد از نظر ساختار..."},
    {"id":9, "name":"فرهنگ سازمانی", "desc":"فرهنگ سازمانی نحوه فکر کردن و رفتار افراد..."},
    {"id":10,"name":"مدیریت شایستگی","desc":"شایستگی یعنی توانایی به‌کارگیری دانش و مهارت..."},
    {"id":11,"name":"مدیریت تغییر سازمانی","desc":"رویکردی ساختاریافته برای هدایت افراد در برابر تغییرات..."},
    {"id":12,"name":"تحلیل تقاضا","desc":"ابزاری برای درک نیازهای آینده ذی‌نفعان..."},
    {"id":13,"name":"توسعه پایدار","desc":"پاسخگویی به نیازهای امروز بدون به خطر انداختن توان..."},
    {"id":14,"name":"استراتژی و اهداف مدیریت دارایی","desc":"در SAMP تعریف می‌شوند و اصول سیاست مدیریت دارایی..."},
    {"id":15,"name":"برنامه‌ریزی مدیریت دارایی","desc":"تهیه برنامه‌های عملیاتی برای تحقق SAMP..."},
    {"id":16,"name":"استراتژی و برنامه‌ریزی توقف‌ها و تعمیرات اساسی","desc":"STO شامل برنامه‌ریزی، زمان‌بندی و اجرای کارهایی..."},
    {"id":17,"name":"برنامه‌ریزی اضطراری و تحلیل تاب‌آوری","desc":"توانایی مقاومت در برابر اختلالات و بازگشت سریع..."},
    {"id":18,"name":"استراتژی و مدیریت منابع","desc":"تعیین نحوه تأمین و مدیریت منابع..."},
    {"id":19,"name":"مدیریت زنجیره تأمین","desc":"تضمین تأمین به‌موقع و باکیفیت تجهیزات/مواد/خدمات..."},
    {"id":20,"name":"تحقق ارزش چرخه عمر","desc":"اطمینان از بیشترین ارزش کل در کل چرخه عمر..."},
    {"id":21,"name":"هزینه‌یابی و ارزش‌گذاری دارایی (تمرکز مالی)","desc":"ثبت دقیق Capex/Opex و ارزش‌گذاری..."},
    {"id":22,"name":"تصمیم‌گیری","desc":"در قلب AM؛ روش متناسب با ریسک/پیچیدگی..."},
    {"id":23,"name":"ایجاد و تملک دارایی","desc":"برنامه‌ریزی تا تحویل به بهره‌برداری..."},
    {"id":24,"name":"مهندسی سیستم‌ها","desc":"رویکرد میان‌رشته‌ای با تمرکز بر RAMS..."},
    {"id":25,"name":"قابلیت اطمینان یکپارچه","desc":"به‌کارگیری اصول/تکنیک‌های قابلیت اطمینان..."},
    {"id":26,"name":"عملیات دارایی","desc":"سیاست‌ها/فرآیندهای بهره‌برداری برای سطح خدمت..."},
    {"id":27,"name":"اجرای نگهداری","desc":"مدیریت برنامه‌ریزی، زمان‌بندی، اجرا و تحلیل..."},
    {"id":28,"name":"مدیریت و پاسخ به رخدادها","desc":"تشخیص، تحلیل، اقدام اصلاحی و بازیابی..."},
    {"id":29,"name":"بازتخصیص و کنارگذاری دارایی","desc":"گزینه‌های بازاستفاده/نوسازی/فروش/بازیافت/کنارگذاری..."},
    {"id":30,"name":"استراتژی داده و اطلاعات","desc":"مشخص می‌کند داده‌های دارایی چگونه جمع‌آوری..."},
    {"id":31,"name":"مدیریت دانش","desc":"شناسایی، ثبت، سازمان‌دهی، اشتراک‌گذاری و نگهداری..."},
    {"id":32,"name":"استانداردهای داده و اطلاعات","desc":"استانداردهای طبقه‌بندی، ویژگی‌ها، مقیاس وضعیت..."},
    {"id":33,"name":"مدیریت داده و اطلاعات","desc":"تضمین دقت، به‌روز بودن، امنیت و دسترس‌پذیری..."},
    {"id":34,"name":"سیستم‌های داده و اطلاعات","desc":"سیستم‌های پشتیبان جمع‌آوری/یکپارچه‌سازی/تحلیل..."},
    {"id":35,"name":"مدیریت پیکربندی","desc":"فرآیند شناسایی، ثبت و کنترل ویژگی‌های..."},
    {"id":36,"name":"مدیریت ریسک","desc":"طبق ISO 31000: اثر عدم قطعیت بر اهداف..."},
    {"id":37,"name":"پایش","desc":"سنجش ارزش تحقق‌یافته با شاخص‌های مالی/غیرمالی..."},
    {"id":38,"name":"بهبود مستمر","desc":"تحلیل عملکرد برای شناسایی فرصت‌ها..."},
    {"id":39,"name":"مدیریت تغییر","desc":"سیستمی برای شناسایی، ارزیابی، اجرا و اطلاع‌رسانی..."},
    {"id":40,"name":"نتایج و پیامدها","desc":"ترکیبی از خروجی‌ها و اثرات کوتاه/بلندمدت..."},
]
if not TOPICS_PATH.exists():
    TOPICS_PATH.write_text(json.dumps(EMBEDDED_TOPICS, ensure_ascii=False, indent=2), encoding="utf-8")
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ---------------- Roles/colors/weights ----------------
ROLES = ["مدیران ارشد","مدیران اجرایی","سرپرستان / خبرگان","متخصصان فنی","متخصصان غیر فنی"]
LEVEL_OPTIONS = [
    ("اطلاعی در این مورد ندارم.",0),
    ("سازمان نیاز به این موضوع را شناسایی کرده ولی جزئیات آن را نمی‌دانم.",1),
    ("سازمان در حال تدوین دستورالعمل‌های مرتبط است و فعالیت‌هایی به‌صورت موردی انجام می‌شود.",2),
    ("بله، این موضوع در سازمان به‌صورت کامل و استاندارد پیاده‌سازی و اجرایی شده است.",3),
    ("بله، چند سال است که نتایج اجرای آن ارزیابی می‌شود و از بهترین تجربه‌ها برای بهبود مستمر استفاده می‌گردد.",4),
]
REL_OPTIONS = [("هیچ ارتباطی ندارد.",1),("ارتباط کم دارد.",3),("تا حدی مرتبط است.",5),("ارتباط زیادی دارد.",7),("کاملاً مرتبط است.",10)]
ROLE_MAP_EN2FA={"Senior Managers":"مدیران ارشد","Executives":"مدیران اجرایی","Supervisors/Sr Experts":"سرپرستان / خبرگان","Technical Experts":"متخصصان فنی","Non-Technical Experts":"متخصصان غیر فنی"}
NORM_WEIGHTS = {  # — همان وزن‌های قبلی —
    1:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    2:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    3:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    4:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    5:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    6:{"Senior Managers":0.1923,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    7:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    8:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    9:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
    10:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    11:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    12:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    13:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    14:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    15:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    16:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    17:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    18:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    19:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    20:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    21:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    22:{"Senior Managers":0.2692,"Executives":0.3846,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    23:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    24:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    25:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.3846,"Non-Technical Experts":0.1154},
    26:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    27:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    28:{"Senior Managers":0.1154,"Executives":0.1923,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.2692,"Non-Technical Experts":0.0385},
    29:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.1154,"Non-Technical Experts":0.2692},
    30:{"Senior Managers":0.1154,"Executives":0.3846,"Supervisors/Sr Experts":0.0385,"Technical Experts":0.2692,"Non-Technical Experts":0.1923},
    31:{"Senior Managers":0.1154,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.0385,"Non-Technical Experts":0.3846},
    32:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    33:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    34:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.1923},
    35:{"Senior Managers":0.0385,"Executives":0.1923,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.3846,"Non-Technical Experts":0.2692},
    36:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1923,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    37:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    38:{"Senior Managers":0.0385,"Executives":0.2692,"Supervisors/Sr Experts":0.3846,"Technical Experts":0.1923,"Non-Technical Experts":0.1154},
    39:{"Senior Managers":0.1923,"Executives":0.3846,"Supervisors/Sr Experts":0.2692,"Technical Experts":0.1154,"Non-Technical Experts":0.0385},
    40:{"Senior Managers":0.3846,"Executives":0.2692,"Supervisors/Sr Experts":0.1154,"Technical Experts":0.0385,"Non-Technical Experts":0.1923},
}

# ---------------- Data helpers ----------------
def _sanitize_company_name(name: str) -> str:
    s = (name or "").strip()
    s = s.replace("/", "／").replace("\\", "＼")
    s = re.sub(r"\s+", " ", s).strip(".")
    return s

def ensure_company(company: str):
    (DATA_DIR / _sanitize_company_name(company)).mkdir(parents=True, exist_ok=True)

def load_company_df(company: str) -> pd.DataFrame:
    company = _sanitize_company_name(company)
    ensure_company(company)
    p = DATA_DIR/company/"responses.csv"
    if p.exists():
        return pd.read_csv(p)
    cols = ["timestamp","company","respondent","role"]
    for t in TOPICS:
        cols += [f"t{t['id']}_maturity",f"t{t['id']}_rel",f"t{t['id']}_adj"]
    return pd.DataFrame(columns=cols)

def save_response(company: str, rec: dict):
    company = _sanitize_company_name(company)
    df_old = load_company_df(company)
    df_new = pd.concat([df_old, pd.DataFrame([rec])], ignore_index=True)
    (DATA_DIR/company/"responses.csv").write_text(df_new.to_csv(index=False), encoding="utf-8")

def get_company_logo_path(company: str) -> Optional[Path]:
    folder = DATA_DIR / _sanitize_company_name(company)
    for ext in ("png","jpg","jpeg"):
        p = folder / f"logo.{ext}"
        if p.exists(): return p
    p2 = folder / "logo.png"
    return p2 if p2.exists() else None

# ---------------- Plot helpers ----------------
PLOTLY_TEMPLATE = "plotly_white"
TARGET = 45

def _angles_deg_40():
    base = np.arange(0,360,360/40.0)
    return (base+90) % 360

def plot_radar(series_dict, tick_numbers, tick_mapping_df, target=45, annotate=False, height=900, point_size=7):
    angles = _angles_deg_40()
    N = len(tick_numbers)
    fig = go.Figure()
    for label, vals in series_dict.items():
        arr = list(vals)
        if len(arr) != N: arr = (arr + [None]*N)[:N]
        fig.add_trace(go.Scatterpolar(
            r=arr+[arr[0]], theta=angles.tolist()+[angles[0]], thetaunit="degrees",
            mode="lines+markers"+("+text" if annotate else ""), name=label,
            text=[f"{v:.0f}" if v is not None else "" for v in arr+[arr[0]]] if annotate else None,
            marker=dict(size=point_size, line=dict(width=1))
        ))
    fig.add_trace(go.Scatterpolar(
        r=[target]*(N+1), theta=angles.tolist()+[angles[0]], thetaunit="degrees",
        mode="lines", name=f"هدف {TARGET}", line=dict(dash="dash", width=3, color="#444"), hoverinfo="skip"
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height,
        polar=dict(radialaxis=dict(visible=True, range=[0,100], dtick=10),
                   angularaxis=dict(thetaunit="degrees", direction="clockwise", rotation=0,
                                    tickmode="array", tickvals=angles.tolist(), ticktext=tick_numbers)))
    c1, c2 = st.columns([3,2])
    with c1: st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### نگاشت شماره ↔ نام موضوع")
        st.dataframe(tick_mapping_df, use_container_width=True, height=min(700, 22*(len(tick_numbers)+2)))

def plot_bars_multirole(per_role, labels, title, target=45, height=600):
    fig = go.Figure()
    for lab, vals in per_role.items():
        fig.add_trace(go.Bar(x=labels, y=vals, name=lab))
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)",
                      barmode="group", height=height)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {target}")
    st.plotly_chart(fig, use_container_width=True)

def plot_bars_top_bottom(series, topic_names, top=10):
    s = pd.Series(series, index=[f"{i+1:02d} — {n}" for i,n in enumerate(topic_names)])
    top_s = s.sort_values(ascending=False).head(top)
    bot_s = s.sort_values(ascending=True).head(top)
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(top_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Top {top} (میانگین سازمان)")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(bot_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Bottom {top} (میانگین سازمان)")
        st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa, topic_id: int):
    w = NORM_WEIGHTS.get(topic_id, {}); num = 0.; den = 0.
    for en_key, weight in w.items():
        fa = ROLE_MAP_EN2FA[en_key]; lst = per_role_norm_fa.get(fa, []); idx = topic_id-1
        if idx < len(lst) and pd.notna(lst[idx]): num += weight * lst[idx]; den += weight
    return np.nan if den == 0 else num/den

# ---------------- Tabs ----------------
tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ======================= پرسشنامه =======================
with tabs[0]:
    # Header (no CSS, just basic layout)
    c1, c2 = st.columns([1,6])
    with c1:
        if (ASSETS_DIR/"holding_logo.png").exists():
            st.image(str(ASSETS_DIR/"holding_logo.png"), width=90, caption="هلدینگ")
    with c2:
        st.markdown("### پرسشنامه تعیین سطح بلوغ مدیریت دارایی فیزیکی (هلدینگ انرژی گستر سینا)")

    with st.expander("⚙️ برندینگ هلدینگ (اختیاری)"):
        holding_logo_file = st.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"], key="upl_holding_logo")
        if holding_logo_file:
            (ASSETS_DIR/"holding_logo.png").write_bytes(holding_logo_file.getbuffer())
            st.success("لوگوی هلدینگ ذخیره شد.")
            st.rerun()

    st.info("برای هر موضوع ابتدا توضیح آن را بخوانید، سپس به دو پرسش زیر پاسخ دهید.")

    company = st.text_input("نام شرکت")
    respondent = st.text_input("نام و نام خانوادگی (اختیاری)")
    role = st.selectbox("نقش / رده سازمانی", ROLES)

    answers = {}
    for t in TOPICS:
        with st.container(border=True):
            st.markdown(f"**{t['id']:02d} — {t['name']}**")
            st.caption(t["desc"])
            st.write(f"۱) سطح بلوغ «{t['name']}» در سازمان شما؟")
            m_choice = st.radio("", options=[opt for (opt,_) in LEVEL_OPTIONS], key=f"mat_{t['id']}", label_visibility="collapsed")
            st.write(f"۲) میزان ارتباط «{t['name']}» با حیطه کاری شما؟")
            r_choice = st.radio("", options=[opt for (opt,_) in REL_OPTIONS], key=f"rel_{t['id']}", label_visibility="collapsed")
            answers[t['id']] = (m_choice, r_choice)

    if st.button("ثبت پاسخ"):
        if not company:
            st.error("نام شرکت را وارد کنید.")
        elif not role:
            st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers) != len(TOPICS):
            st.error("لطفاً همهٔ ۴۰ موضوع را پاسخ دهید.")
        else:
            ensure_company(company)
            rec = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "company": _sanitize_company_name(company),
                   "respondent": respondent, "role": role}
            m_map = dict(LEVEL_OPTIONS); r_map = dict(REL_OPTIONS)
            for t in TOPICS:
                m_label, r_label = answers[t['id']]
                m = m_map.get(m_label, 0); r = r_map.get(r_label, 1)
                rec[f"t{t['id']}_maturity"] = m
                rec[f"t{t['id']}_rel"] = r
                rec[f"t{t['id']}_adj"] = m * r
            save_response(company, rec)
            st.success("✅ پاسخ شما با موفقیت ذخیره شد.")

# ======================= داشبورد =======================
with tabs[1]:
    st.subheader("📊 داشبورد نتایج")

    if not PLOTLY_OK:
        st.error("برای نمایش نمودارها باید Plotly نصب باشد:  pip install plotly")
        st.stop()

    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.warning("رمز درست را وارد کنید.")
        st.stop()

    companies = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and (DATA_DIR/d.name/"responses.csv").exists()])
    if not companies:
        st.info("هنوز هیچ پاسخی ثبت نشده است.")
        st.stop()

    company = st.selectbox("انتخاب شرکت", companies)
    df = load_company_df(company)
    if df.empty:
        st.info("برای این شرکت پاسخی وجود ندارد.")
        st.stop()

    # خلاصه مشارکت
    with st.container(border=True):
        st.markdown("### خلاصه مشارکت شرکت")
        total_n = len(df)
        st.markdown(f"**{_sanitize_company_name(company)}** — تعداد کل پاسخ‌ها: **{total_n}**")
        role_counts = df["role"].value_counts().reindex(ROLES).fillna(0).astype(int)
        rc_df = pd.DataFrame({"نقش/رده": role_counts.index, "تعداد پاسخ‌ها": role_counts.values})
        st.dataframe(rc_df, use_container_width=True, hide_index=True)
        fig_cnt = px.bar(rc_df, x="نقش/رده", y="تعداد پاسخ‌ها", template="plotly_white", title="تعداد پاسخ‌دهندگان به تفکیک رده سازمانی")
        st.plotly_chart(fig_cnt, use_container_width=True)

    # لوگوها
    c1, c2 = st.columns([1,6])
    with c1:
        if (ASSETS_DIR/"holding_logo.png").exists():
            st.image(str(ASSETS_DIR/"holding_logo.png"), width=90, caption="هلدینگ")
        comp_logo_file = st.file_uploader("آپلود/به‌روزرسانی لوگوی شرکت", key="uplogo", type=["png","jpg","jpeg"])
        if comp_logo_file:
            (DATA_DIR/_sanitize_company_name(company)/"logo.png").write_bytes(comp_logo_file.getbuffer())
            st.success("لوگوی شرکت ذخیره شد.")
            st.rerun()
        p = get_company_logo_path(company)
        if p: st.image(str(p), width=90, caption=company)

    # نرمال‌سازی 0..100
    for t in TOPICS:
        c = f"t{t['id']}_adj"
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].apply(lambda x: (x/40)*100 if pd.notna(x) else np.nan)

    # میانگین نقش‌ها
    role_means = {}
    for r in ROLES:
        sub = df[df["role"]==r]
        role_means[r] = [sub[f"t{t['id']}_adj"].mean() if not sub.empty else np.nan for t in TOPICS]

    # میانگین سازمان (فازی)
    per_role_norm_fa = {r: role_means[r] for r in ROLES}
    org_series = [org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]

    # KPI ها
    with st.container(border=True):
        st.markdown("### شاخص‌های کلی")
        nanmean_org = np.nanmean(org_series)
        org_avg = float(nanmean_org) if np.isfinite(nanmean_org) else 0.0
        pass_rate = (np.mean([1 if (v >= TARGET) else 0 for v in org_series if pd.notna(v)]) * 100
                     if any(pd.notna(v) for v in org_series) else 0)
        simple_means = [np.nanmean([role_means[r][i] for r in ROLES if pd.notna(role_means[r][i])]) for i in range(40)]
        has_any = any(np.isfinite(x) for x in simple_means)
        if has_any:
            best_idx = int(np.nanargmax(simple_means)); worst_idx = int(np.nanargmin(simple_means))
            best_label = f"{best_idx+1:02d} — {TOPICS[best_idx]['name']}"
            worst_label = f"{worst_idx+1:02d} — {TOPICS[worst_idx]['name']}"
        else:
            best_label = "-"; worst_label = "-"

        cA, cB, cC, cD = st.columns(4)
        cA.metric("میانگین سازمان (فازی)", f"{org_avg:.1f}", "از 100")
        cB.metric("نرخ عبور از هدف", f"{pass_rate:.0f}%", f"≥ {TARGET}")
        cC.metric("بهترین موضوع", best_label)
        cD.metric("ضعیف‌ترین موضوع", worst_label)

    # فیلترها
    with st.container(border=True):
        st.markdown("### فیلترهای نمایش")
        annotate_radar = st.checkbox("نمایش اعداد روی نقاط رادار", value=False)
        col_sz1, col_sz2 = st.columns(2)
        with col_sz1:
            radar_point_size = st.slider("اندازه نقاط رادار", 4, 12, 7, key="rad_pt")
        with col_sz2:
            radar_height = st.slider("ارتفاع رادار (px)", 600, 1100, 900, 50, key="rad_h")
        bar_height = st.slider("ارتفاع نمودار میله‌ای (px)", 400, 900, 600, 50, key="bar_h")

        roles_selected = st.multiselect("نقش‌های قابل نمایش", ROLES, default=ROLES)
        topic_range = st.slider("بازهٔ موضوع‌ها", 1, 40, (1,40))
        label_mode = st.radio("حالت برچسب محور X / زاویه", ["شماره (01..40)","نام کوتاه","نام کامل"], horizontal=True)

    idx0, idx1 = topic_range[0]-1, topic_range[1]
    topics_slice = TOPICS[idx0:idx1]
    names_full = [t['name'] for t in topics_slice]
    names_short = [n if len(n)<=14 else n[:13]+"…" for n in names_full]
    labels_bar = [f"{i+idx0+1:02d}" for i,_ in enumerate(topics_slice)] if label_mode=="شماره (01..40)" else (names_short if label_mode=="نام کوتاه" else names_full)
    tick_numbers = [f"{i+idx0+1:02d}" for i,_ in enumerate(topics_slice)]
    tick_mapping_df = pd.DataFrame({"شماره":tick_numbers, "نام موضوع":names_full})
    role_means_filtered = {r: role_means[r][idx0:idx1] for r in roles_selected}
    org_series_slice = org_series[idx0:idx1]

    with st.container(border=True):
        st.markdown("### رادار ۴۰‌بخشی (نقش‌ها)")
        if role_means_filtered:
            plot_radar(role_means_filtered, tick_numbers, tick_mapping_df, target=TARGET,
                       annotate=annotate_radar, height=radar_height, point_size=radar_point_size)
        else:
            st.info("نقشی برای نمایش انتخاب نشده است.")

    with st.container(border=True):
        st.markdown("### رادار میانگین سازمان (وزن‌دهی فازی)")
        plot_radar({"میانگین سازمان": org_series_slice}, tick_numbers, tick_mapping_df,
                   target=TARGET, annotate=annotate_radar, height=radar_height, point_size=radar_point_size)

    with st.container(border=True):
        st.markdown("### نمودار میله‌ای گروهی (نقش‌ها)")
        plot_bars_multirole({r: role_means[r][idx0:idx1] for r in roles_selected},
                            labels_bar, "مقایسه رده‌ها (0..100)", target=TARGET, height=bar_height)

    with st.container(border=True):
        st.markdown("### Top/Bottom — میانگین سازمان")
        plot_bars_top_bottom(org_series_slice, names_full, top=10)

    with st.container(border=True):
        st.markdown("### Heatmap و Boxplot")
        heat_df = pd.DataFrame({"موضوع":labels_bar})
        for r in roles_selected: heat_df[r] = role_means[r][idx0:idx1]
        hm = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
        fig_heat = px.density_heatmap(hm, x="نقش", y="موضوع", z="امتیاز", color_continuous_scale="RdYlGn", height=560, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_heat, use_container_width=True)
        fig_box = px.box(hm.dropna(), x="نقش", y="امتیاز", points="all", color="نقش", template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_box, use_container_width=True)

    with st.container(border=True):
        st.markdown("### ماتریس همبستگی و خوشه‌بندی")
        corr_base = heat_df.set_index("موضوع")[roles_selected]
        if not corr_base.empty:
            corr = corr_base.T.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", aspect="auto", height=620, template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_corr, use_container_width=True)
        if SKLEARN_OK and not corr_base.empty:
            try:
                X_raw = corr_base.values
                imp_med = SimpleImputer(strategy="median"); X_med = imp_med.fit_transform(X_raw)
                X = X_med if not np.isnan(X_med).any() else SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(X_raw)
                if np.allclose(X, 0) or np.nanstd(X) == 0:
                    st.info("دادهٔ کافی/متغیر برای خوشه‌بندی وجود ندارد.")
                else:
                    k = st.slider("تعداد خوشه‌ها (K)", 2, 6, 3)
                    K = min(k, X.shape[0]) if X.shape[0] >= 2 else 2
                    if X.shape[0] >= 2:
                        km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
                        clusters = km.labels_
                        cl_df = pd.DataFrame({"موضوع":corr_base.index,"خوشه":clusters}).sort_values("خوشه")
                        st.dataframe(cl_df, use_container_width=True)
                    else:
                        st.info("برای خوشه‌بندی حداقل به ۲ موضوع نیاز است.")
            except Exception as e:
                st.warning(f"خوشه‌بندی انجام نشد: {e}")
        else:
            st.caption("برای فعال‌شدن خوشه‌بندی، scikit-learn را نصب کنید (اختیاری).")

    with st.container(border=True):
        st.markdown("### دانلود")
        st.download_button("⬇️ دانلود CSV پاسخ‌های شرکت",
                           data=load_company_df(company).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{_sanitize_company_name(company)}_responses.csv", mime="text/csv")
