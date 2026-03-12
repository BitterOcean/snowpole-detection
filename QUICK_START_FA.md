# 🚀 راهنمای سریع - فردا در دانشگاه

## قبل از رفتن به دانشگاه

1. **این پوشه را کپی کنید** روی یک فلش مموری یا آپلود کنید به Google Drive/OneDrive
2. **مطمئن شوید همه فایل‌ها موجود هستند:**
   - ✅ data_analysis.py
   - ✅ train.py
   - ✅ evaluate.py
   - ✅ requirements.txt
   - ✅ run_complete_project.sh
   - ✅ run_idun.sh (برای IDUN cluster)
   - ✅ README.md
   - ✅ PRESENTATION_GUIDE.md

---

## در دانشگاه - راهنمای گام به گام

### گزینه 1: استفاده از Cybele Lab (توصیه می‌شود - سریع‌تر)

#### مرحله 1: Setup (5 دقیقه)
```bash
# کپی فایل‌ها به سیستم
cd ~
mkdir -p projects
cd projects

# اگر از فلش کپی می‌کنید:
cp -r /path/to/usb/snow_pole_detection .

# یا اگر از Git pull می‌کنید:
git clone [YOUR_REPO_URL] snow_pole_detection

# وارد پوشه شوید
cd snow_pole_detection

# ساخت virtual environment
python3 -m venv venv
source venv/bin/activate

# نصب کتابخانه‌ها
pip install --upgrade pip
pip install -r requirements.txt
```

#### مرحله 2: اجرای کامل پروژه (2-4 ساعت)
```bash
# اجرای تمام مراحل یکجا
bash run_complete_project.sh datasets/TDT17/ad/Poles2025

# یا اگر خواستید مرحله به مرحله:

# 1. تحلیل داده (5-10 دقیقه)
python data_analysis.py --data_path datasets/TDT17/ad/Poles2025

# 2. آموزش مدل (1-2 ساعت)
python train.py --data_path datasets/TDT17/ad/Poles2025 \
                --model_size n \
                --epochs 100 \
                --batch_size 16 \
                --device cuda

# 3. ارزیابی مدل (10-15 دقیقه)
python evaluate.py --model_path runs/yolov8n_*/weights/best.pt \
                   --data_path datasets/TDT17/ad/Poles2025
```

---

### گزینه 2: استفاده از IDUN Cluster

#### مرحله 1: اتصال به IDUN
```bash
# از ترمینال خودتان
ssh [username]@idun-login1.hpc.ntnu.no

# رفتن به پوشه پروژه
cd $HOME
mkdir -p projects
cd projects
```

#### مرحله 2: آپلود فایل‌ها
```bash
# از کامپیوتر محلی (ترمینال جدید):
scp -r snow_pole_detection [username]@idun-login1.hpc.ntnu.no:~/projects/
```

#### مرحله 3: Submit Job
```bash
# در IDUN:
cd ~/projects/snow_pole_detection

# Submit job
sbatch run_idun.sh

# چک کردن وضعیت
squeue -u $USER

# دیدن لاگ (پس از شروع job)
tail -f slurm_output_*.log
```

#### مرحله 4: دانلود نتایج
```bash
# از کامپیوتر محلی:
scp -r [username]@idun-login1.hpc.ntnu.no:~/projects/snow_pole_detection/runs .
scp -r [username]@idun-login1.hpc.ntnu.no:~/projects/snow_pole_detection/analysis_results .
```

---

## 📊 چک کردن نتایج

### پس از اتمام هر مرحله:

**1. تحلیل داده:**
```bash
ls analysis_results/
# باید ببینید:
# - data_analysis.png
# - sample_annotations.png
# - analysis_report.json
```

**2. آموزش مدل:**
```bash
ls runs/yolov8n_*/
# باید ببینید:
# - weights/best.pt
# - weights/last.pt
# - results.csv
# - training_info.json
# - sustainability_metrics.json
```

**3. ارزیابی مدل:**
```bash
ls runs/yolov8n_*/evaluation/
# باید ببینید:
# - validation_metrics.json
# - prediction_samples.png
# - confidence_distribution.png
# - test_predictions/
```

---

## 🎯 چک لیست نهایی قبل از ترک دانشگاه

- [ ] تمام فایل‌های output کپی شده (به فلش یا کلود)
- [ ] `analysis_results/` کامل است
- [ ] `runs/yolov8n_*/` کامل است
- [ ] مدل آموزش دیده (`best.pt`) موجود است
- [ ] نتایج ارزیابی موجود است
- [ ] فایل‌های JSON متریک‌ها موجود هستند

---

## 📈 مشاهده نتایج سریع

```bash
# نمایش متریک‌های validation
cat runs/yolov8n_*/evaluation/validation_metrics.json

# نمایش sustainability metrics
cat runs/yolov8n_*/sustainability_metrics.json

# نمایش آمار داده
cat analysis_results/analysis_report.json
```

---

## ⚠️ مشکلات احتمالی و راه حل

### مشکل: CUDA out of memory
```bash
# کاهش batch size
python train.py --data_path ... --batch_size 8
```

### مشکل: Training خیلی طول می‌کشد
```bash
# کاهش epochs برای تست
python train.py --data_path ... --epochs 50

# یا استفاده از مدل کوچکتر
python train.py --data_path ... --model_size n
```

### مشکل: Dataset path پیدا نمی‌شود
```bash
# چک کردن وجود dataset
ls /cluster/projects/vc/courses/TDT17/ad/Poles2025  # IDUN
ls datasets/TDT17/ad/Poles2025  # Cybele

# اگر وجود ندارد، با استاد تماس بگیرید
```

### مشکل: Dependency install نمی‌شود
```bash
# استفاده از conda به جای venv
conda create -n snow_pole python=3.10
conda activate snow_pole
pip install -r requirements.txt
```

---

## 🔥 نکات مهم

1. **زمان را مدیریت کنید:**
   - تحلیل داده: 10 دقیقه
   - Training: 1-3 ساعت (بسته به GPU)
   - Evaluation: 15 دقیقه
   - جمع: حدود 2-4 ساعت

2. **همیشه backup بگیرید:**
   - هر ساعت نتایج را کپی کنید
   - از Google Drive یا OneDrive استفاده کنید

3. **Cybele Lab ساعت کاری دارد:**
   - اجتناب کنید از:
     - دوشنبه 14:15 - 19:00
     - سه‌شنبه 14:15 - 16:00
     - چهارشنبه 14:15 - 16:00

4. **IDUN queue زمان‌بر است:**
   - زود Submit کنید
   - در ساعات غیر اوج (شب/آخر هفته) بهتر است

5. **کد را تست کنید:**
   - قبل از training کامل، با epochs=5 تست کنید
   - مطمئن شوید همه چیز کار می‌کند

---

## 📞 در صورت مشکل

1. خطاها را بخوانید و Google کنید
2. از ChatGPT/Claude برای دیباگ استفاده کنید
3. از همکلاسی‌ها کمک بگیرید
4. با استاد تماس بگیرید: frankl@ntnu.no

---

## 🎬 پس از تمام شدن پروژه

1. همه فایل‌ها را جمع کنید
2. PRESENTATION_GUIDE.md را بخوانید
3. ویدیوی ارائه را بسازید (12/14 دقیقه)
4. قبل از 29 نوامبر upload کنید

---

**موفق باشید! 💪🚀**
