# Q6 — AE Decision Engine for Industrial Edge Computing

**درس:** محاسبات لبه | **استاد:** دکتر انصاری | **دانشجو:** کیهان مسعودی ۴۰۱۱۰۶۵۰۹

---

## معرفی کلی

این پروژه یک موتور تصمیم‌گیری سبک برای پردازش و تحلیل سیگنال‌های Acoustic Emission (AE) تجهیزات صنعتی در معماری رایانش لبه است. سناریو: نصب سنسور AE روی پمپ/کمپرسور در صنایع پتروشیمی ایران.

---

## ساختار فایل‌ها

| فایل | شرح |
|------|-----|
| `data_generator.py` | **بخش الف** — مولد داده مصنوعی (۸۰ پنجره AE) |
| `feature_extractor.py` | **بخش ب** — استخراج ۱۲ ویژگی زمانی/فرکانسی |
| `model.py` | **بخش ج** — مدل سبک (Random Forest) |
| `passport.py` | **بخش د** — Data Passport و قیدهای انتقال داده |
| `decision_engine.py` | **بخش هـ** — موتور تصمیم‌گیری لبه‌ای |
| `zero_trust.py` | **بخش و** — ماژول Zero Trust |
| `simulation.py` | **بخش ز** — آزمایش سناریوهای چالشی |
| `baseline.py` | **بخش ح** — مقایسه با baseline‌ها |
| `main.py` | اجرای کامل تمام بخش‌ها |

---

## نیازمندی‌ها

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

---

## اجرا

```bash
cd q6
python main.py
```

### اجرای تکی هر بخش:

```bash
python data_generator.py       # Part A
python feature_extractor.py    # Part B
python model.py                # Part C
python passport.py             # Part D
python decision_engine.py      # Part E
python zero_trust.py           # Part F
python simulation.py            # Part G
python baseline.py  # Part H
```

---

## فایل‌های خروجی

| فایل | شرح |
|------|-----|
| `ae_samples_meta.csv` | متادیتای نمونه‌های مصنوعی |
| `feature_table.csv` | ویژگی‌های استخراج‌شده از همه پنجره‌ها |
| `decision_output.csv` | خروجی موتور تصمیم‌گیری برای همه نمونه‌ها |
| `audit_log.csv` | لاگ Zero Trust برای همه درخواست‌های دسترسی |
| `passport_table.csv` | Data Passport هر نمونه |
| `scenario_results.csv` | نتایج سه سناریوی چالشی |
| `comparison_normal/degraded/offline.csv` | مقایسه با baseline‌ها |
| `comparison_*.png` | نمودار مقایسه |

---

## توضیح بخش‌ها

### تولید داده
- ۸۰ پنجره AE مصنوعی با ۷ برچسب خرابی
- حداقل ۲۰٪ non_transferable، ۲۰٪ high/critical، ۱۰ نمونه degraded/offline، ۵ نمونه unknown_anomaly

### استخراج ویژگی
۱۲ ویژگی شامل:
- **حوزه زمان (۸):** RMS, peak_amplitude, crest_factor, energy, ZCR, kurtosis, skewness, signal_entropy
- **حوزه فرکانس (۳):** dominant_frequency, spectral_centroid, band_power_ratio
- **رخدادهای گذرا (۱):** burst_count


### Zero Trust
سه سناریوی مشخص:
1. **Gateway با trust پایین** درخواست داده raw می‌دهد → رد
2. **Cloud مدل بهتری دارد** اما داده ممنوع‌الخروج است → رد
3. **اپراتور درخواست اضطراری** دارد → break-glass با ثبت لاگ

### سناریوهای چالشی
۳ سناریو (normal/degraded/offline) × ۸۰ نمونه = ۲۴۰ تصمیم

### مقایسه با Baseline
| متریک | our_engine | first_cloud | first_latency |
|--------|-----------|-------------|---------------|
| avg_latency | متوسط | بالا | کم (ولی ناامن) |
| privacy_violations | صفر | بالا | صفر |
| non_transferable_violations | صفر | بالا | صفر |

---