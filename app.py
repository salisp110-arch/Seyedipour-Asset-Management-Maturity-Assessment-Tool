# app.py
# -*- coding: utf-8 -*-
import os, json, base64, re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional

# ---------------- Page config ----------------
st.set_page_config(page_title="پرسشنامه و داشبورد مدیریت دارایی", layout="wide")

# ---------------- Plotly (برای داشبورد لازم) ----------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------------- scikit-learn اختیاری ----------------
try:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# ---------------- مسیرها (ایمن) ----------------
BASE = Path(".")

def _safe_dir(p: Path) -> Path:
    """اگر مسیر وجود داشت ولی دایرکتوری نبود، به مسیر جایگزین برود تا FileExistsError نگیریم."""
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

# ---------------- CSS/Font: تزریق امن در هر اجرا + مخفی‌سازی pre/code ----------------
def inject_css():
    css = """
:root{--brand:#16325c;--accent:#0f3b8f;--border:#e8eef7;--font:Vazir,Tahoma,Arial,sans-serif}
html,body,*{font-family:var(--font)!important;direction:rtl}
.block-container{padding-top:.6rem;padding-bottom:3rem}
h1,h2,h3,h4{color:var(--brand)}
/* جلوگیری از نمایش هر خروجی code/pre (مثلاً اگر موتور موقتاً CSS خام را به صورت کد رندر کند) */
.stMarkdown pre, .stMarkdown code{display:none!important}

/* هدر چسبنده */
.header-sticky{position:sticky;top:0;z-index:999;background:#ffffffcc;backdrop-filter:blur(6px);border-bottom:1px solid #eef2f7;padding:8px 12px;margin:-10px -1rem 10px -1rem}
.header-sticky .wrap{display:flex;align-items:center;gap:12px}
.header-sticky .title{font-weight:800;color:var(--brand);font-size:18px;margin:0}

/* کارت سوال */
.question-card{background:#fff;border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin:10px 0 16px;box-shadow:0 6px 16px rgba(36,74,143,.06),inset 0 1px 0 rgba(255,255,255,.6)}
.q-head{font-weight:800;color:var(--brand);font-size:15px;margin-bottom:8px}
.q-desc{color:#222;font-size:14px;line-height:1.9;margin-bottom:10px;text-align:justify}
.q-num{display:inline-block;background:#e8f0fe;color:var(--brand);font-weight:700;border-radius:8px;padding:2px 8px;margin-left:6px;font-size:12px}
.q-question{color:var(--accent);font-weight:700;margin:.2rem 0 .4rem}

/* KPI */
.kpi{border-radius:14px;padding:16px 18px;border:1px solid #e6ecf5;background:linear-gradient(180deg,#fff 0%,#f6f9ff 100%);box-shadow:0 8px 20px rgba(0,0,0,.05);min-height:96px}
.kpi .title{color:#456;font-size:13px;margin-bottom:6px}
.kpi .value{color:var(--accent);font-size:22px;font-weight:800}
.kpi .sub{color:#6b7c93;font-size:12px}

/* پنل */
.panel{background:linear-gradient(180deg,#f2f7ff 0%,#eaf3ff 100%);border:1px solid #d7e6ff;border-radius:16px;padding:16px 18px;margin:12px 0 18px 0;box-shadow:0 10px 24px rgba(31,79,176,.1),inset 0 1px 0 rgba(255,255,255,.8)}
.panel h3,.panel h4{margin-top:0;color:#17407a}

/* جدول نگاشت کنار رادار */
.mapping table{font-size:12px}
.mapping .row_heading,.mapping .blank{display:none}

/* تب‌ها راست‌به‌چپ */
.stTabs [role=tab]{direction:rtl}
"""
    b64css = base64.b64encode(css.encode("utf-8")).decode()
    st.markdown(
        f"""
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css">
<link rel="stylesheet" href="data:text/css;base64,{b64css}">
""",
        unsafe_allow_html=True,
    )

# تزریق در هر اجرا
inject_css()

PLOTLY_TEMPLATE = "plotly_white"
TARGET = 45  # 🎯

# ---------------- موضوعات (اگر topics.json نبود، بساز) ----------------
TOPICS_PATH = BASE/"topics.json"
EMBEDDED_TOPICS = [
    {"id":1,"name":"هدف و زمینه (Purpose & Context)","desc":"Purpose و Context نقطه شروع سیستم مدیریت دارایی هستند. Purpose همان مأموریت و ارزش‌هایی است که سازمان برای ذی‌نفعان خلق می‌کند. Context محیطی است که سازمان در آن فعالیت دارد: شامل شرایط اجتماعی، سیاسی، اقتصادی، فناورانه و داخلی. این دو باید در SAMP و اهداف مدیریت دارایی منعکس شوند تا اقدامات سازمان همسو با مأموریت اصلی باشد. ابزارهایی مانند SWOT و PESTLE برای تحلیل محیط و شناسایی ریسک‌ها و فرصت‌ها استفاده می‌شوند. سازمان‌هایی که Purpose و Context را به‌طور منظم بازنگری می‌کنند، بهتر می‌توانند منابع خود را بهینه کنند، ریسک‌ها را کاهش دهند و فرصت‌ها را شناسایی نمایند."},
    {"id":2,"name":"مدیریت ذی‌نفعان","desc":"مدیریت ذی‌نفعان به معنای داشتن یک رویکرد ساختاریافته و مستند برای شناسایی، درگیر کردن و مدیریت نیازها و انتظارات افرادی است که می‌توانند بر سازمان اثر بگذارند یا از آن اثر بپذیرند. این ذی‌نفعان می‌توانند داخلی یا خارجی باشند. هدف، ایجاد شفافیت و اطمینان از این است که ارزش‌های مورد انتظار ذی‌نفعان در فعالیت‌های مدیریت دارایی منعکس شود. ابزارهایی مانند Stakeholder Mapping و ماتریس نفوذ-علاقه به سنجش اهمیت و تعریف راهکار ارتباط مؤثر کمک می‌کنند. پایش مستمر و سازوکارهای رسمی مشارکت، مدیریت ریسک و مشروعیت اجتماعی را تقویت می‌کند."},
    {"id":3,"name":"هزینه‌یابی و ارزش‌گذاری دارایی","desc":"هزینه‌یابی دارایی شامل شناسایی و ثبت کل هزینه‌های سرمایه‌ای (Capex) و عملیاتی (Opex) در طول چرخه عمر است. ارزش‌گذاری دارایی فرآیند سنجش ارزش مالی دارایی‌ها طبق استانداردهای حسابداری است. این دو حوزه برای تصمیم‌گیری سرمایه‌گذاری و گزارش‌دهی مالی حیاتی‌اند. ابزارهایی مانند NPV، IRR، Payback و LCC به‌کار می‌روند."},
    {"id":4,"name":"خط مشی مدیریت دارایی","desc":"خط مشی مدیریت دارایی سندی رسمی است که تعهد سازمان به مدیریت دارایی را بیان می‌کند و با چشم‌انداز، مأموریت و اهداف کلان همسو می‌شود. این سیاست چارچوبی جهت‌دار برای هم‌سویی برنامه‌های استراتژیک و اهداف دارایی فراهم می‌کند و معمولاً بخشی از SAMP است و با سایر خط‌مشی‌های کلان یکپارچه می‌شود. سازمان‌های پیشرو این سیاست را به‌طور منظم بازبینی و به کارکنان ابلاغ می‌کنند."},
    {"id":5,"name":"سیستم مدیریت دارایی (AMS)","desc":"سیستم مدیریت دارایی مجموعه‌ای از عناصر مرتبط برای ایجاد، به‌روزرسانی و پایدارسازی سیاست‌ها، اهداف و فرآیندهای مدیریت دارایی است و باید با سایر سیستم‌های مدیریتی مانند ISO 9001/14001/45001 همسو باشد. این سیستم شامل فرآیندهایی برای ارزیابی اثربخشی، شناسایی عدم انطباق‌ها و اجرای بهبود مستمر است. ISO 55001 چارچوب طراحی و ممیزی ارائه می‌دهد."},
    {"id":6,"name":"اطمینان و ممیزی","desc":"اطمینان و ممیزی فرآیندهای ساختاریافته‌ای برای ارزیابی اثربخشی دارایی‌ها، فعالیت‌های مدیریت دارایی و خود AMS هستند. الگوی «سه خط دفاع» معمولاً برای تفکیک مسئولیت‌های عملیاتی، کنترل ریسک و ممیزی مستقل استفاده می‌شود. ممیزی‌های داخلی و خارجی، ورودی‌های کلیدی برای بازنگری مدیریت و بهبود AMS محسوب می‌شوند."},
    {"id":7,"name":"استانداردهای فنی و قوانین","desc":"باید اطمینان حاصل شود که تمامی فعالیت‌ها با قوانین، مقررات و استانداردهای فنی مرتبط (ملی، بین‌المللی یا صنعتی) سازگارند. علاوه بر قوانین الزام‌آور، «کدهای عملی» و استانداردهای صنعتی معیار قضاوت خوب محسوب می‌شوند. فرآیندهای شناسایی، پایش و اعمال الزامات در SAMP و برنامه‌های چرخه عمر ضروری است. ممیزی مستقل ابزار کلیدی اطمینان از انطباق است."},
    {"id":8,"name":"آرایش سازمانی","desc":"آرایش سازمانی نحوه سازمان‌دهی افراد از نظر ساختار، مسئولیت‌ها و خطوط ارتباطی است. جایگاه مدیریت دارایی در چارت سازمانی نشانه مهمی از جدیت سازمان در این حوزه است. تعریف نقش‌ها و مسئولیت‌های مدیریت دارایی در سطح ارشد برای همکاری بین‌رشته‌ای ضروری است."},
    {"id":9,"name":"فرهنگ سازمانی","desc":"فرهنگ سازمانی نحوه فکر کردن و رفتار افراد در جهت اهداف مدیریت دارایی است. فرهنگ باید فعالانه مدیریت شود تا همکاری، شفافیت، مسئولیت‌پذیری و یادگیری مستمر تقویت شود. حمایت مشهود مدیریت ارشد و سازگاری رفتارها پایه‌های فرهنگ مطلوب‌اند."},
    {"id":10,"name":"مدیریت شایستگی","desc":"شایستگی یعنی توانایی به‌کارگیری دانش و مهارت برای دستیابی به نتایج مورد انتظار. مدیریت شایستگی شامل ارزیابی، ثبت و توسعه مهارت‌های افراد از سطح هیئت‌مدیره تا کارگاه است. چارچوب‌هایی مانند IAM Competence Framework و ISO 55012 برای تعریف و پایش شایستگی‌ها به کار می‌آیند."},
    {"id":11,"name":"مدیریت تغییر سازمانی","desc":"رویکردی ساختاریافته برای هدایت افراد در برابر تغییرات فرآیندها، فناوری، ساختار یا فرهنگ. مدل‌هایی مانند ADKAR یا ۸گام کاتر کمک می‌کنند. عوامل کلیدی موفقیت: رهبری متعهد، مشارکت ذی‌نفعان، ارتباطات شفاف و برنامه آموزشی."},
    {"id":12,"name":"تحلیل تقاضا","desc":"ابزاری برای درک نیازهای آینده ذی‌نفعان و تغییرات احتمالی آنها. خروجی تحلیل تقاضا ورودی مهمی برای مدیریت ریسک، برنامه‌ریزی سرمایه‌ای و عملیاتی است. شامل پیش‌بینی سناریو، تحلیل روند و مدل‌های کمی."},
    {"id":13,"name":"توسعه پایدار","desc":"پاسخگویی به نیازهای امروز بدون به خطر انداختن توان نسل‌های آینده. تعیین معیارهای پایداری، LCA، کاهش کربن و هم‌سویی با SDGs/BS8900-1 توصیه می‌شود."},
    {"id":14,"name":"استراتژی و اهداف مدیریت دارایی","desc":"در SAMP تعریف می‌شوند و اصول سیاست مدیریت دارایی را به اقدامات عملی تبدیل می‌کنند. اهداف باید SMART باشند و نیاز ذی‌نفعان، ریسک، چرخه عمر و قابلیت‌های سازمان لحاظ شوند."},
    {"id":15,"name":"برنامه‌ریزی مدیریت دارایی","desc":"تهیه برنامه‌های عملیاتی برای تحقق SAMP شامل فعالیت‌ها، منابع، هزینه‌ها، زمان‌بندی‌ها و مسئولیت‌ها. ادغام با سایر برنامه‌های سازمانی و بازنگری منظم اهمیت دارد."},
    {"id":16,"name":"استراتژی و برنامه‌ریزی توقف‌ها و تعمیرات اساسی","desc":"STO شامل برنامه‌ریزی، زمان‌بندی و اجرای کارهایی است که در زمان بهره‌برداری قابل انجام نیست. این فعالیت‌ها پرهزینه و پرریسک‌اند و نیازمند هماهنگی واحدها هستند."},
    {"id":17,"name":"برنامه‌ریزی اضطراری و تحلیل تاب‌آوری","desc":"توانایی مقاومت در برابر اختلالات و بازگشت سریع. ابزارها: چرخه تاب‌آوری، ISO 22301، تحلیل سناریو."},
    {"id":18,"name":"استراتژی و مدیریت منابع","desc":"تعیین نحوه تأمین و مدیریت منابع انسانی، تجهیزاتی، خدمات و مواد لازم؛ شامل استخدام، برون‌سپاری، شراکت، مدیریت پیمانکاران و هم‌راستایی با SAMP."},
    {"id":19,"name":"مدیریت زنجیره تأمین","desc":"تضمین تأمین به‌موقع و باکیفیت تجهیزات/مواد/خدمات؛ انتخاب و ارزیابی پیمانکاران، مدیریت قراردادها و ریسک تأمین‌کنندگان."},
    {"id":20,"name":"تحقق ارزش چرخه عمر","desc":"اطمینان از بیشترین ارزش کل در کل چرخه عمر (ایجاد، بهره‌برداری، نگهداری، بهبود، نوسازی و کنارگذاری). ابزارها: تحلیل ارزش، LCC، TCO، CBA."},
    {"id":21,"name":"هزینه‌یابی و ارزش‌گذاری دارایی (تمرکز مالی)","desc":"ثبت دقیق Capex/Opex و ارزش‌گذاری برای تصمیم‌گیری سرمایه‌ای و گزارش‌دهی مالی با استفاده از ابزارهای کمی."},
    {"id":22,"name":"تصمیم‌گیری","desc":"در قلب AM؛ روش متناسب با ریسک/پیچیدگی؛ چارچوب تصمیم‌گیری، مشارکت بین‌رشته‌ای و ابزارهای کمی و ماتریس ریسک."},
    {"id":23,"name":"ایجاد و تملک دارایی","desc":"برنامه‌ریزی تا تحویل به بهره‌برداری با درنظرگرفتن RAMS و هزینه‌های کل؛ روش‌های قراردادی مانند PPP/BOT/اجاره نیز رایج است."},
    {"id":24,"name":"مهندسی سیستم‌ها","desc":"رویکرد میان‌رشته‌ای با تمرکز بر RAMS؛ V-Model از نیازمندی تا آزمون/اعتبارسنجی و مدیریت واسط‌ها؛ ISO 15288 راهنماست."},
    {"id":25,"name":"قابلیت اطمینان یکپارچه","desc":"به‌کارگیری اصول/تکنیک‌های قابلیت اطمینان در سراسر چرخه عمر (RCM, FMECA, تحلیل خرابی، افزونگی) برای کاهش ریسک خرابی."},
    {"id":26,"name":"عملیات دارایی","desc":"سیاست‌ها/فرآیندهای بهره‌برداری برای سطح خدمت با رعایت HSE، قابلیت اطمینان و عملکرد مالی؛ توجه به خطای انسانی، اتوماسیون و پایش."},
    {"id":27,"name":"اجرای نگهداری","desc":"مدیریت برنامه‌ریزی، زمان‌بندی، اجرا و تحلیل نگهداری؛ بازرسی/پایش وضعیت، PM، CM و استفاده از EAMS و روش‌های پیش‌بینانه."},
    {"id":28,"name":"مدیریت و پاسخ به رخدادها","desc":"تشخیص، تحلیل، اقدام اصلاحی و بازیابی پس از خرابی‌ها/حوادث؛ FRACAS، RCA، 5Why، ایشیکاوا؛ سازوکار واکنش سریع متناسب با ریسک."},
    {"id":29,"name":"بازتخصیص و کنارگذاری دارایی","desc":"گزینه‌های بازاستفاده/نوسازی/فروش/بازیافت/کنارگذاری با توجه به اثرات اقتصادی، زیست‌محیطی و اجتماعی؛ اقتصاد دایره‌ای."},
    {"id":30,"name":"استراتژی داده و اطلاعات","desc":"مشخص می‌کند داده‌های دارایی چگونه جمع‌آوری، ذخیره، تحلیل، نگهداری و حذف می‌شوند؛ هم‌سویی با SAMP، کیفیت داده، امنیت و یکپارچگی."},
    {"id":31,"name":"مدیریت دانش","desc":"شناسایی، ثبت، سازمان‌دهی، اشتراک‌گذاری و نگهداری دانش ضمنی/صریح؛ درس‌آموخته‌ها، جانشین‌پروری، BIM و دوقلوی دیجیتال."},
    {"id":32,"name":"استانداردهای داده و اطلاعات","desc":"استانداردهای طبقه‌بندی، ویژگی‌ها، مقیاس وضعیت، دسته‌بندی خرابی، KPIها و کیفیت داده؛ استفاده از BIM/DT/ISO 8000."},
    {"id":33,"name":"مدیریت داده و اطلاعات","desc":"تضمین دقت، به‌روز بودن، امنیت و دسترس‌پذیری؛ تعیین مسئولیت‌ها، فرکانس به‌روزرسانی و کیفیت؛ سطح اعتماد به داده مشخص شود."},
    {"id":34,"name":"سیستم‌های داده و اطلاعات","desc":"سیستم‌های پشتیبان جمع‌آوری/یکپارچه‌سازی/تحلیل؛ یکپارچگی سیستم‌ها و هزینه-فایدهٔ داده‌ها برای تصمیم‌گیری بهتر."},
    {"id":35,"name":"مدیریت پیکربندی","desc":"فرآیند شناسایی، ثبت و کنترل ویژگی‌های عملکردی/فیزیکی دارایی‌ها، نرم‌افزارها و اسناد؛ کنترل تغییر، گزارش وضعیت و ممیزی."},
    {"id":36,"name":"مدیریت ریسک","desc":"طبق ISO 31000: اثر عدم قطعیت بر اهداف؛ تهدید/فرصت؛ Criticality، ماتریس ریسک، رجیستر، Bow-tie، FTA، ETA؛ ۴T، اشتهای ریسک و تحمل ریسک."},
    {"id":37,"name":"پایش","desc":"سنجش ارزش تحقق‌یافته با شاخص‌های مالی/غیرف مالی، سطح خدمت و وضعیت دارایی‌ها؛ بازخورد برای بهینه‌سازی سرمایه‌گذاری/عملیات/نگهداری."},
    {"id":38,"name":"بهبود مستمر","desc":"تحلیل عملکرد برای شناسایی فرصت‌ها و ایجاد تغییرات تدریجی؛ چرخه PDCA پراستفاده‌ترین ابزار است."},
    {"id":39,"name":"مدیریت تغییر","desc":"سیستمی برای شناسایی، ارزیابی، اجرا و اطلاع‌رسانی تغییرات ناشی از قوانین جدید، فناوری نو، تغییرات کارکنان یا شرایط بحرانی."},
    {"id":40,"name":"نتایج و پیامدها","desc":"ترکیبی از خروجی‌ها و اثرات کوتاه/بلندمدت مالی/غیرف مالی؛ چارچوب‌های Value Framework و 6 Capitals برای سنجش ارزش به‌کار می‌روند."}
]
if not TOPICS_PATH.exists():
    TOPICS_PATH.write_text(json.dumps(EMBEDDED_TOPICS, ensure_ascii=False, indent=2), encoding="utf-8")
TOPICS = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
if len(TOPICS) != 40:
    st.warning("⚠️ تعداد موضوعات باید دقیقاً ۴۰ باشد.")

# ---------------- نقش‌ها و رنگ‌ها و وزن‌ها ----------------
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

# ---------------- کمک‌توابع داده ----------------
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
        cols += [f"t{t['id']}_maturity",f"t{t['id']}_rel",f"t{t['id']}_adj"]
    return pd.DataFrame(columns=cols)

def save_response(company: str, rec: dict):
    company = _sanitize_company_name(company)
    df_old = load_company_df(company)
    df_new = pd.concat([df_old, pd.DataFrame([rec])], ignore_index=True)
    out = DATA_DIR/company/"responses.csv"
    df_new.to_csv(out, index=False)

def get_company_logo_path(company: str) -> Optional[Path]:
    folder = DATA_DIR / _sanitize_company_name(company)
    for ext in ("png","jpg","jpeg"):
        p = folder / f"logo.{ext}"
        if p.exists():
            return p
    return None

# ---------------- توابع رسم ----------------
def _angles_deg_40():
    base = np.arange(0,360,360/40.0); return (base+90) % 360

def plot_radar(series_dict, tick_numbers, tick_mapping_df, target=45, annotate=False, height=900, point_size=7):
    N = len(tick_numbers); angles = _angles_deg_40()
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
        template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        height=height,
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
    with c1:
        st.plotly_chart(fig, use_container_width=True)
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

def plot_lines_multirole(per_role, title, target=45):
    x = [f"{i+1:02d}" for i in range(len(list(per_role.values())[0]))]; fig = go.Figure()
    for lab, vals in per_role.items():
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=lab, line=dict(width=2, color=ROLE_COLORS.get(lab))))
    fig.update_layout(template=PLOTLY_TEMPLATE, font=dict(family="Vazir, Tahoma"),
        title=title, xaxis_title="موضوع", yaxis_title="نمره (0..100)", paper_bgcolor="#ffffff", hovermode="x unified")
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=target-5, y1=target+5,
                  fillcolor="rgba(255,0,0,0.06)", line_width=0)
    fig.add_hline(y=target, line_dash="dash", line_color="red", annotation_text=f"هدف {target}")
    st.plotly_chart(fig, use_container_width=True)

def org_weighted_topic(per_role_norm_fa, topic_id: int):
    w = NORM_WEIGHTS.get(topic_id, {}); num = 0.; den = 0.
    en2fa = ROLE_MAP_EN2FA
    for en_key, weight in w.items():
        fa = en2fa[en_key]; lst = per_role_norm_fa.get(fa, []); idx = topic_id-1
        if idx < len(lst) and pd.notna(lst[idx]):
            num += weight * lst[idx]; den += weight
    return np.nan if den == 0 else num/den

# ---------------- هدر/لوگو ----------------
def _logo_html(assets_dir: Path, fname: str = "holding_logo.png", height: int = 44) -> str:
    p = assets_dir / fname
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" height="{height}" alt="logo">'
    return ""

# ---------------- ریست فرم پس از ثبت ----------------
def reset_survey_state():
    for t in TOPICS:
        st.session_state.pop(f"mat_{t['id']}", None)
        st.session_state.pop(f"rel_{t['id']}", None)
    for k in ["company_input", "respondent_input", "role_select"]:
        st.session_state.pop(k, None)

# ---------------- تب‌ها ----------------
tabs = st.tabs(["📝 پرسشنامه","📊 داشبورد"])

# ======================= پرسشنامه =======================
with tabs[0]:
    # هدر چسبنده با لوگو
    st.markdown(
        f'''
        <div class="header-sticky">
          <div class="wrap">
            {_logo_html(ASSETS_DIR, "holding_logo.png", 44)}
            <div class="title">پرسشنامه تعیین سطح بلوغ هلدینگ انرژی گستر سینا و شرکت‌های تابعه در مدیریت دارایی فیزیکی</div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # پیام موفقیت پس از رفرش
    if st.session_state.pop("submitted_ok", False):
        st.success("✅ پاسخ شما با موفقیت ذخیره شد و فرم ریست شد.")

    # برندینگ هلدینگ (آپلود لوگو)
    with st.expander("⚙️ برندینگ هلدینگ (اختیاری)"):
        holding_logo_file = st.file_uploader("لوگوی هلدینگ انرژی گستر سینا", type=["png","jpg","jpeg"], key="upl_holding_logo")
        if holding_logo_file:
            (ASSETS_DIR/"holding_logo.png").write_bytes(holding_logo_file.getbuffer())
            st.success("لوگوی هلدینگ ذخیره شد.")
            st.rerun()

    st.info("برای هر موضوع ابتدا توضیح فارسی آن را بخوانید، سپس با توجه به دو پرسش ذیل هر موضوع، یکی از گزینه‌های زیر هر پرسش را انتخاب بفرمایید.")

    # فرم پرسشنامه (برای ریست تمیز)
    with st.form("survey_form", clear_on_submit=False):
        company = st.text_input("نام شرکت", key="company_input")
        respondent = st.text_input("نام و نام خانوادگی (اختیاری)", key="respondent_input")
        role = st.selectbox("نقش / رده سازمانی", ROLES, key="role_select")

        answers = {}
        for t in TOPICS:
            st.markdown(f'''
            <div class="question-card">
              <div class="q-head"><span class="q-num">{t["id"]:02d}</span>{t["name"]}</div>
              <div class="q-desc">{t["desc"].replace("\n","<br>")}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown(f'<div class="q-question">۱) به نظر شما، موضوع «{t["name"]}» در سازمان شما در چه سطحی قرار دارد؟</div>', unsafe_allow_html=True)
            m_choice = st.radio("", options=[opt for (opt,_) in LEVEL_OPTIONS], key=f"mat_{t['id']}", label_visibility="collapsed")
            st.markdown(f'<div class="q-question">۲) موضوع «{t["name"]}» چقدر به حیطه کاری شما ارتباط مستقیم دارد؟</div>', unsafe_allow_html=True)
            r_choice = st.radio("", options=[opt for (opt,_) in REL_OPTIONS], key=f"rel_{t['id']}", label_visibility="collapsed")
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
            rec = {"timestamp": datetime
