# app.py
# -*- coding: utf-8 -*-
import base64, json, re
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")

# ---------------- Optional libs ----------------
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

# ---------------- Paths ----------------
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

# ---------------- Safe CSS + Vazir font (no raw text) ----------------
def inject_css():
    css = """
:root{--brand:#16325c;--accent:#0f3b8f;--border:#e8eef7;--font:Vazir,Tahoma,Arial,sans-serif}
html,body,*{font-family:var(--font)!important;direction:rtl}
.block-container{padding-top:.6rem;padding-bottom:3rem}
h1,h2,h3,h4{color:var(--brand)}
.header-sticky{position:sticky;top:0;z-index:999;background:#ffffffcc;backdrop-filter:blur(6px);
  border-bottom:1px solid #eef2f7;padding:8px 12px;margin:-10px -1rem 10px -1rem}
.header-sticky .wrap{display:flex;align-items:center;gap:12px}
.header-sticky .title{font-weight:800;color:var(--brand);font-size:18px;margin:0}
.question-card{background:#fff;border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin:10px 0 16px;
  box-shadow:0 6px 16px rgba(36,74,143,.06),inset 0 1px 0 rgba(255,255,255,.6)}
.q-head{font-weight:800;color:var(--brand);font-size:15px;margin-bottom:8px}
.q-desc{color:#222;font-size:14px;line-height:1.9;margin-bottom:10px;text-align:justify}
.q-num{display:inline-block;background:#e8f0fe;color:var(--brand);font-weight:700;border-radius:8px;padding:2px 8px;margin-left:6px;font-size:12px}
.q-question{color:var(--accent);font-weight:700;margin:.2rem 0 .4rem}
.kpi{border-radius:14px;padding:16px 18px;border:1px solid #e6ecf5;background:linear-gradient(180deg,#fff 0%,#f6f9ff 100%);
  box-shadow:0 8px 20px rgba(0,0,0,.05);min-height:96px}
.kpi .title{color:#456;font-size:13px;margin-bottom:6px}
.kpi .value{color:var(--accent);font-size:22px;font-weight:800}
.kpi .sub{color:#6b7c93;font-size:12px}
.panel{background:linear-gradient(180deg,#f2f7ff 0%,#eaf3ff 100%);border:1px solid #d7e6ff;border-radius:16px;
  padding:16px 18px;margin:12px 0 18px 0;box-shadow:0 10px 24px rgba(31,79,176,.1),inset 0 1px 0 rgba(255,255,255,.8)}
.panel h3,.panel h4{margin-top:0;color:#17407a}
.mapping table{font-size:12px}.mapping .row_heading,.mapping .blank{display:none}
"""
    b64 = base64.b64encode(css.encode("utf-8")).decode()
    st.markdown(
        f"""
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css">
<link rel="stylesheet" href="data:text/css;base64,{b64}">
""",
        unsafe_allow_html=True
    )

inject_css()

PLOTLY_TEMPLATE = "plotly_white"
TARGET = 45  # 🎯

# ---------------- Topics ----------------
TOPICS_PATH = BASE/"topics.json"
EMBEDDED_TOPICS = [
    {"id":1,"name":"هدف و زمینه (Purpose & Context)","desc":"Purpose و Context نقطه شروع سیستم مدیریت دارایی هستند. Purpose همان مأموریت و ارزش‌هایی است که سازمان برای ذی‌نفعان خلق می‌کند. Context محیطی است که سازمان در آن فعالیت دارد: شامل شرایط اجتماعی، سیاسی، اقتصادی، فناورانه و داخلی. این دو باید در SAMP و اهداف مدیریت دارایی منعکس شوند تا اقدامات سازمان همسو با مأموریت اصلی باشد. ابزارهایی مانند SWOT و PESTLE برای تحلیل محیط و شناسایی ریسک‌ها و فرصت‌ها استفاده می‌شوند. سازمان‌هایی که Purpose و Context را به‌طور منظم بازنگری می‌کنند، بهتر می‌توانند منابع خود را بهینه کنند، ریسک‌ها را کاهش دهند و فرصت‌ها را شناسایی نمایند."},
    {"id":2,"name":"مدیریت ذی‌نفعان","desc":"مدیریت ذی‌نفعان به معنای داشتن یک رویکرد ساختاریافته و مستند برای شناسایی، درگیر کردن و مدیریت نیازها و انتظارات افرادی است که می‌توانند بر سازمان اثر بگذارند یا از آن اثر بپذیرند."},
    {"id":3,"name":"هزینه‌یابی و ارزش‌گذاری دارایی","desc":"هزینه‌یابی دارایی شامل شناسایی و ثبت کل هزینه‌های سرمایه‌ای و عملیاتی در طول چرخه عمر است."},
    {"id":4,"name":"خط مشی مدیریت دارایی","desc":"بیان تعهد سازمان به مدیریت دارایی و هم‌سویی با چشم‌انداز و اهداف کلان."},
    {"id":5,"name":"سیستم مدیریت دارایی (AMS)","desc":"مجموعه عناصر مرتبط سیاست‌ها، اهداف و فرآیندهای مدیریت دارایی؛ همسو با سایر سیستم‌ها."},
    {"id":6,"name":"اطمینان و ممیزی","desc":"ممیزی‌های داخلی/خارجی و الگوی سه خط دفاع برای بهبود AMS."},
    {"id":7,"name":"استانداردهای فنی و قوانین","desc":"انطباق با قوانین و استانداردهای مرتبط و پایش مستمر الزامات."},
    {"id":8,"name":"آرایش سازمانی","desc":"ساختار، مسئولیت‌ها و جایگاه مدیریت دارایی در چارت سازمانی."},
    {"id":9,"name":"فرهنگ سازمانی","desc":"رفتار و نگرش‌ها در جهت اهداف AM؛ حمایت مدیریت ارشد."},
    {"id":10,"name":"مدیریت شایستگی","desc":"ارزیابی و توسعه مهارت‌ها در سطوح مختلف سازمان."},
    {"id":11,"name":"مدیریت تغییر سازمانی","desc":"الگوهای تغییر (ADKAR/کاتر) و عوامل کلیدی موفقیت."},
    {"id":12,"name":"تحلیل تقاضا","desc":"پیش‌بینی نیازهای آینده و ورودی برای ریسک و برنامه‌ریزی."},
    {"id":13,"name":"توسعه پایدار","desc":"LCA، کاهش کربن، هم‌سویی با SDGs/BS8900-1."},
    {"id":14,"name":"استراتژی و اهداف مدیریت دارایی","desc":"تعریف در SAMP و تبدیل سیاست‌ها به اقدامات SMART."},
    {"id":15,"name":"برنامه‌ریزی مدیریت دارایی","desc":"برنامه‌های عملیاتی، منابع، زمان‌بندی و بازنگری."},
    {"id":16,"name":"استراتژی و برنامه‌ریزی توقف‌ها و تعمیرات اساسی","desc":"برنامه‌ریزی STO برای کارهای غیرقابل انجام در بهره‌برداری."},
    {"id":17,"name":"برنامه‌ریزی اضطراری و تحلیل تاب‌آوری","desc":"ISO 22301، تحلیل سناریو و بازگشت سریع."},
    {"id":18,"name":"استراتژی و مدیریت منابع","desc":"تأمین و مدیریت منابع انسانی/تجهیزات/خدمات."},
    {"id":19,"name":"مدیریت زنجیره تأمین","desc":"انتخاب و ارزیابی پیمانکاران، مدیریت ریسک تأمین."},
    {"id":20,"name":"تحقق ارزش چرخه عمر","desc":"بهینه‌سازی ارزش کل چرخه عمر (LCC/TCO/CBA)."},
    {"id":21,"name":"هزینه‌یابی و ارزش‌گذاری دارایی (تمرکز مالی)","desc":"ثبت دقیق Capex/Opex و ارزش‌گذاری برای تصمیم‌گیری."},
    {"id":22,"name":"تصمیم‌گیری","desc":"روش متناسب با ریسک/پیچیدگی؛ ابزارهای کمی و ماتریس ریسک."},
    {"id":23,"name":"ایجاد و تملک دارایی","desc":"از برنامه‌ریزی تا تحویل با ملاحظه RAMS و هزینه کل."},
    {"id":24,"name":"مهندسی سیستم‌ها","desc":"V-Model، مدیریت واسط‌ها، ISO 15288."},
    {"id":25,"name":"قابلیت اطمینان یکپارچه","desc":"RCM، FMECA، افزونگی و تحلیل خرابی."},
    {"id":26,"name":"عملیات دارایی","desc":"سطح خدمت، HSE، اتوماسیون و پایش."},
    {"id":27,"name":"اجرای نگهداری","desc":"PM/CM، پایش وضعیت، EAMS و پیش‌بینانه."},
    {"id":28,"name":"مدیریت و پاسخ به رخدادها","desc":"FRACAS، RCA، 5Why، ایشیکاوا."},
    {"id":29,"name":"بازتخصیص و کنارگذاری دارایی","desc":"بازاستفاده/نوسازی/فروش/بازیافت/کنارگذاری."},
    {"id":30,"name":"استراتژی داده و اطلاعات","desc":"جمع‌آوری، ذخیره، تحلیل، امنیت و حذف داده‌ها."},
    {"id":31,"name":"مدیریت دانش","desc":"ثبت/اشتراک‌گذاری دانش، درس‌آموخته‌ها، دوقلوی دیجیتال."},
    {"id":32,"name":"استانداردهای داده و اطلاعات","desc":"طبقه‌بندی، مقیاس وضعیت، KPI و کیفیت داده."},
    {"id":33,"name":"مدیریت داده و اطلاعات","desc":"دقت، به‌روزرسانی، امنیت و مسئولیت‌ها."},
    {"id":34,"name":"سیستم‌های داده و اطلاعات","desc":"یکپارچگی سیستم‌ها و هزینه-فایدهٔ داده‌ها."},
    {"id":35,"name":"مدیریت پیکربندی","desc":"کنترل تغییر، گزارش وضعیت و ممیزی."},
    {"id":36,"name":"مدیریت ریسک","desc":"ISO 31000، ماتریس ریسک، Bow-tie، ۴T."},
    {"id":37,"name":"پایش","desc":"KPIهای مالی/غیرمالی، سطح خدمت و وضعیت دارایی."},
    {"id":38,"name":"بهبود مستمر","desc":"چرخه PDCA و تغییرات تدریجی."},
    {"id":39,"name":"مدیریت تغییر","desc":"شناسایی/ارزیابی/اجرای تغییرات داخلی و بیرونی."},
    {"id":40,"name":"نتایج و پیامدها","desc":"خروجی‌ها و اثرات؛ Value Framework و 6 Capitals."}
]
if not TOPICS_PATH.exists():
    TOPICS_PATH.write_text(json.dumps(EMBEDDED_TOPICS, ensure_ascii=False, indent=2), encoding="utf-8")
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ---------------- Roles / colors / weights ----------------
ROLES = ["مدیران ارشد","مدیران اجرایی","سرپرستان / خبرگان","متخصصان فنی","متخصصان غیر فنی"]
ROLE_COLORS = {
    "مدیران ارشد":"#d62728","مدیران اجرایی":"#1f77b4","سرپرستان / خبرگان":"#2ca02c",
    "متخصصان فنی":"#ff7f0e","متخصصان غیر فنی":"#9467bd","میانگین سازمان":"#111"
}
LEVEL_OPTIONS = [
    ("اطلاعی در این مورد ندارم.",0),
    ("سازمان نیاز به این موضوع را شناسایی کرده ولی جزئیات آن را نمی‌دانم.",1),
    ("سازمان در حال تدوین دستورالعمل‌های مرتبط است و فعالیت‌هایی به‌صورت موردی انجام می‌شود.",2),
    ("بله، این موضوع در سازمان به‌صورت کامل و استاندارد پیاده‌سازی و اجرایی شده است.",3),
    ("بله، چند سال است که نتایج اجرای آن بر اساس شاخص‌های استاندارد ارزیابی می‌شود و از بهترین تجربه‌ها برای بهبود مستمر استفاده می‌گردد.",4),
]
REL_OPTIONS = [("هیچ ارتباطی ندارد.",1),("ارتباط کم دارد.",3),("تا حدی مرتبط است.",5),("ارتباط زیادی دارد.",7),("کاملاً مرتبط است.",10)]
ROLE_MAP_EN2FA={"Senior Managers":"مدیران ارشد","Executives":"مدیران اجرایی","Supervisors/Sr Experts":"سرپرستان / خبرگان","Technical Experts":"متخصصان فنی","Non-Technical Experts":"متخصصان غیر فنی"}
NORM_WEIGHTS = {
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
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
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
        cols += [f"t{t['id']}_maturity", f"t{t['id']}_rel", f"t{t['id']}_adj"]
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
        if p.exists():
            return p
    return None

# ---------------- Charts ----------------
def _angles_deg_40():
    base = np.arange(0,360,360/40.0)
    return (base+90) % 360

def plot_radar(series_dict, tick_numbers, tick_mapping_df, target=45, annotate=False, height=900, point_size=7):
    angles = _angles_deg_40()
    N = len(tick_numbers)
    fig = go.Figure()
    for label, vals in series_dict.items():
        arr = list(vals)
        if len(arr) != N:
            arr = (arr + [None]*N)[:N]
        fig.add_trace(go.Scatterpolar(
            r=arr+[arr[0]], theta=angles.tolist()+[angles[0]], thetaunit="degrees",
            mode="lines+markers"+("+text" if annotate else ""), name=label,
            text=[f"{v:.0f}" if v is not None else "" for v in arr+[arr[0]]] if annotate else None,
            marker=dict(size=point_size, line=dict(width=1), color=ROLE_COLORS.get(label))
        ))
    fig.add_trace(go.Scatterpolar(
        r=[target]*(N+1), theta=angles.tolist()+[angles[0]], thetaunit="degrees",
        mode="lines", name=f"هدف {target}", line=dict(dash="dash", width=3, color="#444"), hoverinfo="skip"
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"), height=height,
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], dtick=10, gridcolor="#e6ecf5"),
            angularaxis=dict(thetaunit="degrees", direction="clockwise", rotation=0,
                             tickmode="array", tickvals=angles.tolist(),
                             ticktext=tick_numbers, gridcolor="#edf2fb"),
            bgcolor="white"
        ),
        paper_bgcolor="#ffffff",
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(t=40,b=120,l=10,r=10)
    )
    c1, c2 = st.columns([3,2])
    with c1: st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### نگاشت شماره ↔ نام موضوع")
        st.dataframe(tick_mapping_df, use_container_width=True, height=min(700, 22*(len(tick_numbers)+2)))

def plot_bars_multirole(per_role, labels, title, target=45, height=600):
    fig = go.Figure()
    for lab, vals in per_role.items():
        fig.add_trace(go.Bar(x=labels, y=vals, name=lab, marker_color=ROLE_COLORS.get(lab)))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)",
        xaxis=dict(tickfont=dict(size=10)), barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(t=40,b=120,l=10,r=10), paper_bgcolor="#ffffff", height=height)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {TARGET}")
    st.plotly_chart(fig, use_container_width=True)

def plot_bars_top_bottom(series, topic_names, top=10):
    s = pd.Series(series, index=[f"{i+1:02d} — {n}" for i,n in enumerate(topic_names)])
    top_s = s.sort_values(ascending=False).head(top)
    bot_s = s.sort_values(ascending=True).head(top)
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(top_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Top {top} (میانگین سازمان)")
        fig.update_layout(font=dict(family="Vazir, Tahoma")); st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(bot_s[::-1], orientation="h", template=PLOTLY_TEMPLATE, title=f"Bottom {top} (میانگین سازمان)")
        fig.update_layout(font=dict(family="Vazir, Tahoma")); st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa, topic_id: int):
    w = NORM_WEIGHTS.get(topic_id, {}); num = 0.; den = 0.
    en2fa = ROLE_MAP_EN2FA
    for en_key, weight in w.items():
        fa = en2fa[en_key]; lst = per_role_norm_fa.get(fa, []); idx = topic_id-1
        if idx < len(lst) and pd.notna(lst[idx]): num += weight * lst[idx]; den += weight
    return np.nan if den == 0 else num/den

def _logo_html(assets_dir: Path, fname: str = "holding_logo.png", height: int = 44) -> str:
    p = assets_dir / fname
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{height}" alt="logo">'
    return ""

def reset_survey_state():
    for t in TOPICS:
        st.session_state.pop(f"mat_{t['id']}", None)
        st.session_state.pop(f"rel_{t['id']}", None)
    for k in ["company_input", "respondent_input", "role_select"]:
        st.session_state.pop(k, None)

# ---------------- Sidebar navigation (robust) ----------------
page = st.sidebar.radio("بخش را انتخاب کنید:", ["📝 پرسشنامه","📊 داشبورد"], index=0)

# ================= Survey Page =================
def render_survey():
    st.markdown(
        f'''
        <div class="header-sticky">
          <div class="wrap">
            {_logo_html(ASSETS_DIR, "holding_logo.png", 44)}
            <div class="title">پرسشنامه تعیین سطح بلوغ هلدینگ انرژی گستر سینا و شرکت‌های تابعه در مدیریت دارایی فیزیکی</div>
          </div>
        </div>
        ''', unsafe_allow_html=True
    )

    if st.session_state.pop("submitted_ok", False):
        st.success("✅ پاسخ شما با موفقیت ذخیره شد و فرم ریست شد.")

    with st.expander("⚙️ برندینگ هلدینگ (اختیاری)"):
        holding_logo_file = st.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"], key="upl_holding_logo")
        if holding_logo_file:
            (ASSETS_DIR/"holding_logo.png").write_bytes(holding_logo_file.getbuffer())
            st.success("لوگوی هلدینگ ذخیره شد.")
            st.rerun()

    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس با توجه به دو پرسش ذیل هر موضوع، یکی از گزینه‌های زیر هر پرسش را انتخاب بفرمایید.")

    with st.form("survey_form", clear_on_submit=False):
        company = st.text_input("نام شرکت", key="company_input")
        respondent = st.text_input("نام و نام خانوادگی (اختیاری)", key="respondent_input")
        role = st.selectbox("نقش / رده سازمانی", ROLES, key="role_select")

        answers = {}
        for t in TOPICS:
            st.markdown(
                f'''
                <div class="question-card">
                  <div class="q-head"><span class="q-num">{t["id"]:02d}</span>{t["name"]}</div>
                  <div class="q-desc">{t["desc"].replace("\n","<br>")}</div>
                </div>
                ''', unsafe_allow_html=True
            )
            st.markdown(f'<div class="q-question">۱) «{t["name"]}» در سازمان شما در چه سطحی است؟</div>', unsafe_allow_html=True)
            m_choice = st.radio("", options=[opt for (opt, _) in LEVEL_OPTIONS], key=f"mat_{t['id']}", label_visibility="collapsed")
            st.markdown(f'<div class="q-question">۲) «{t["name"]}» چقدر به کار شما مرتبط است؟</div>', unsafe_allow_html=True)
            r_choice = st.radio("", options=[opt for (opt, _) in REL_OPTIONS], key=f"rel_{t['id']}", label_visibility="collapsed")
            answers[t['id']] = (m_choice, r_choice)

        submitted = st.form_submit_button("ثبت پاسخ")

    if submitted:
        if not company:
            st.error("نام شرکت را وارد کنید.")
        elif not role:
            st.error("نقش/رده سازمانی را انتخاب کنید.")
        elif len(answers) != len(TOPICS):
            st.error("لطفاً همهٔ ۴۰ موضوع را پاسخ دهید.")
        else:
            ensure_company(company)
            m_map = dict(LEVEL_OPTIONS); r_map = dict(REL_OPTIONS)
            rec = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "company": company, "respondent": respondent, "role": role}
            for t in TOPICS:
                m_label, r_label = answers[t['id']]
                m = m_map.get(m_label, 0); r = r_map.get(r_label, 1)
                rec[f"t{t['id']}_maturity"] = m
                rec[f"t{t['id']}_rel"] = r
                rec[f"t{t['id']}_adj"] = m * r
            save_response(company, rec)
            reset_survey_state()
            st.session_state["submitted_ok"] = True
            st.rerun()

# ================= Dashboard Page =================
def render_dashboard():
    st.subheader("📊 داشبورد نتایج")

    # رمز عبور؛ تب حذف نمی‌شود اگر نادرست باشد
    password = st.text_input("🔑 رمز عبور داشبورد را وارد کنید", type="password")
    if password != "Emacraven110":
        st.warning("رمز درست را وارد کنید تا نمودارها نمایش داده شوند.")
        return

    # شرکت‌های دارای پاسخ
    companies = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and (DATA_DIR/d.name/"responses.csv").exists()])
    if not companies:
        st.info("هنوز هیچ پاسخی ثبت نشده است.")
        return

    company = st.selectbox("انتخاب شرکت", companies)
    df = load_company_df(company)
    if df.empty:
        st.info("برای این شرکت پاسخی وجود ندارد.")
        return

    # خلاصه مشارکت
    st.markdown('<div class="panel"><h4>خلاصه مشارکت شرکت</h4>', unsafe_allow_html=True)
    total_n = len(df)
    st.markdown(f"**{_sanitize_company_name(company)}** — تعداد کل پاسخ‌ها: **{total_n}**")
    role_counts = df["role"].value_counts().reindex(ROLES).fillna(0).astype(int)
    rc_df = pd.DataFrame({"نقش/رده": role_counts.index, "تعداد پاسخ‌ها": role_counts.values})
    st.dataframe(rc_df, use_container_width=True, hide_index=True)
    if PLOTLY_OK:
        fig_cnt = px.bar(rc_df, x="نقش/رده", y="تعداد پاسخ‌ها", template=PLOTLY_TEMPLATE, title="تعداد پاسخ‌دهندگان به تفکیک رده سازمانی")
        fig_cnt.update_layout(font=dict(family="Vazir, Tahoma"))
        st.plotly_chart(fig_cnt, use_container_width=True)
    else:
        st.error("Plotly نصب نیست. برای نمودارها: `pip install plotly`")
    st.markdown('</div>', unsafe_allow_html=True)

    # لوگوها
    colL, colH, colC = st.columns([1,1,6])
    with colH:
        if (ASSETS_DIR/"holding_logo.png").exists():
            st.image(str(ASSETS_DIR/"holding_logo.png"), width=90, caption="هلدینگ")
    with colL:
        st.caption("لوگوی شرکت:")
        comp_logo_file = st.file_uploader("آپلود/به‌روزرسانی لوگو", key="uplogo", type=["png","jpg","jpeg"])
        if comp_logo_file:
            (DATA_DIR/_sanitize_company_name(company)/"logo.png").write_bytes(comp_logo_file.getbuffer())
            st.success("لوگوی شرکت ذخیره شد.")
            st.rerun()
        comp_logo_path = get_company_logo_path(company)
        if comp_logo_path:
            st.image(str(comp_logo_path), width=90, caption=company)

    if not PLOTLY_OK:
        st.info("برای بقیه نمودارها Plotly لازم است.")
        return

    # نرمال‌سازی و میانگین‌گیری
    for t in TOPICS:
        c = f"t{t['id']}_adj"
        df[c] = pd.to_numeric(df[c], errors="coerce").apply(lambda x: (x/40)*100 if pd.notna(x) else np.nan)

    role_means = {}
    for r in ROLES:
        sub = df[df["role"]==r]
        role_means[r] = [sub[f"t{t['id']}_adj"].mean() if not sub.empty else np.nan for t in TOPICS]

    per_role_norm_fa = {r: role_means[r] for r in ROLES}
    org_series = [org_weighted_topic(per_role_norm_fa, t["id"]) for t in TOPICS]

    # KPI
    st.markdown('<div class="panel">', unsafe_allow_html=True)
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

    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"""<div class="kpi"><div class="title">میانگین سازمان (فازی)</div>
    <div class="value">{org_avg:.1f}</div><div class="sub">از 100</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi"><div class="title">نرخ عبور از هدف</div>
    <div class="value">{pass_rate:.0f}%</div><div class="sub">نقاط ≥ {TARGET}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi"><div class="title">بهترین موضوع</div>
    <div class="value">{best_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi"><div class="title">ضعیف‌ترین موضوع</div>
    <div class="value">{worst_label}</div><div class="sub">میانگین ساده نقش‌ها</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # فیلترها
    st.markdown('<div class="panel"><h4>فیلترها و تنظیمات نمایش</h4>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    # رادار نقش‌ها
    st.markdown('<div class="panel"><h4>رادار ۴۰‌بخشی (خوانا)</h4>', unsafe_allow_html=True)
    if role_means_filtered:
        plot_radar(role_means_filtered, tick_numbers, tick_mapping_df, target=TARGET,
                   annotate=annotate_radar, height=radar_height, point_size=radar_point_size)
    else:
        st.info("نقشی برای نمایش انتخاب نشده است.")
    st.markdown('</div>', unsafe_allow_html=True)

    # رادار میانگین سازمان
    st.markdown('<div class="panel"><h4>رادار میانگین سازمان (وزن‌دهی فازی)</h4>', unsafe_allow_html=True)
    plot_radar({"میانگین سازمان": org_series_slice}, tick_numbers, tick_mapping_df,
               target=TARGET, annotate=annotate_radar, height=radar_height, point_size=radar_point_size)
    st.markdown('</div>', unsafe_allow_html=True)

    # میله‌ای گروهی
    st.markdown('<div class="panel"><h4>نمودار میله‌ای گروهی (نقش‌ها)</h4>', unsafe_allow_html=True)
    plot_bars_multirole({r: role_means[r][idx0:idx1] for r in roles_selected},
                        labels_bar, "مقایسه رده‌ها (0..100)", target=TARGET, height=bar_height)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top/Bottom
    st.markdown('<div class="panel"><h4>Top/Bottom — میانگین سازمان</h4>', unsafe_allow_html=True)
    plot_bars_top_bottom(org_series_slice, names_full, top=10)
    st.markdown('</div>', unsafe_allow_html=True)

    # Heatmap و Boxplot
    st.markdown('<div class="panel"><h4>Heatmap و Boxplot</h4>', unsafe_allow_html=True)
    heat_df = pd.DataFrame({"موضوع":labels_bar})
    for r in roles_selected: heat_df[r] = role_means[r][idx0:idx1]
    hm = heat_df.melt(id_vars="موضوع", var_name="نقش", value_name="امتیاز")
    fig_heat = px.density_heatmap(hm, x="نقش", y="موضوع", z="امتیاز",
                                  color_continuous_scale="RdYlGn", height=560, template=PLOTLY_TEMPLATE)
    fig_heat.update_layout(font=dict(family="Vazir, Tahoma"))
    st.plotly_chart(fig_heat, use_container_width=True)
    fig_box = px.box(hm.dropna(), x="نقش", y="امتیاز", points="all", color="نقش",
                     color_discrete_map=ROLE_COLORS, template=PLOTLY_TEMPLATE)
    fig_box.update_layout(font=dict(family="Vazir, Tahoma"))
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # خوشه‌بندی (اختیاری)
    st.markdown('<div class="panel"><h4>ماتریس همبستگی و خوشه‌بندی</h4>', unsafe_allow_html=True)
    corr_base = heat_df.set_index("موضوع")[roles_selected]
    if not corr_base.empty:
        corr = corr_base.T.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                             aspect="auto", height=620, template=PLOTLY_TEMPLATE)
        fig_corr.update_layout(font=dict(family="Vazir, Tahoma"))
        st.plotly_chart(fig_corr, use_container_width=True)
    if SKLEARN_OK and not corr_base.empty:
        try:
            X_raw = corr_base.values
            imp_med = SimpleImputer(strategy="median"); X_med = imp_med.fit_transform(X_raw)
            if np.isnan(X_med).any():
                imp_zero = SimpleImputer(strategy="constant", fill_value=0.0); X = imp_zero.fit_transform(X_raw)
            else:
                X = X_med
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
    st.markdown('</div>', unsafe_allow_html=True)

    # دانلود
    st.markdown('<div class="panel"><h4>دانلود</h4>', unsafe_allow_html=True)
    st.download_button("⬇️ دانلود CSV پاسخ‌های شرکت",
                       data=load_company_df(company).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{_sanitize_company_name(company)}_responses.csv", mime="text/csv")
    st.caption("برای دانلود تصویر نمودارها، می‌توانید بستهٔ اختیاری `kaleido` را نصب کنید.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Router ----------------
if page == "📝 پرسشنامه":
    render_survey()
else:
    render_dashboard()
