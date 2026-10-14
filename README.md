# Garbage Classification — จำแนกประเภทขยะด้วย CNN

เว็บแอปที่รับรูปขยะ แล้วใช้โมเดล CNN ที่ฝึกไว้ทำนายว่าเป็นขยะประเภทใด (12 ประเภท) พร้อมค่า confidence

## 🔗 URL เว็บแอป

| แอป | URL |
|---|---|
| Image Classification App (โปรเจกต์นี้) | https://garbage-classification-m9ti.onrender.com/ |
| Regression App | https://moid-regression.onrender.com/ |

> Render แพ็กเกจฟรีจะหลับเมื่อไม่มีผู้ใช้งาน เปิดลิงก์ครั้งแรกอาจรอ 30–60 วินาที

## 1. แนวคิด ปัญหาที่ต้องการแก้ และประโยชน์

**ปัญหา:** ขยะหลายประเภทถูกทิ้งปะปนกัน ทำให้รีไซเคิลยากและผู้ใช้ทั่วไปมักไม่แน่ใจว่าชิ้นนั้นควรทิ้งที่ใด

**แอป Image Classification:** ผู้ใช้อัปโหลดรูปขยะ แอปจะแสดงประเภทที่โมเดลทำนาย Top-5 และ confidence ช่วยตัดสินใจคัดแยกขยะก่อนทิ้ง

**แอป Regression:** `<อธิบายสั้น ๆ ว่าทำนายค่าอะไร หน่วยอะไร ใช้ประโยชน์อย่างไร>`

## 2. แหล่งที่มา จำนวนข้อมูล และการแบ่งชุดฝึก/ชุดทดสอบ

- **แหล่งที่มา:** [Garbage Classification – Kaggle (mostafaabla)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) ค้นหาเมื่อ 30 ก.ย. 2026
- **สิทธิ์การใช้งาน (License):** `Database: Open Database, Contents: © Original Authors`
- **Label:** battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
- **จำนวนข้อมูล:** 15,515 รูป, 12 ประเภท

| Class | รูป | Class | รูป |
|---|---:|---|---:|
| battery | 945 | paper | 1,050 |
| biological | 985 | plastic | 865 |
| brown-glass | 607 | shoes | 1,977 |
| cardboard | 891 | trash | 697 |
| clothes | 5,325 | white-glass | 775 |
| green-glass | 629 | metal | 769 |

ข้อมูลไม่สมดุล: `clothes` มีประมาณ 34% ของทั้งหมด ขณะที่ `brown-glass` มีเพียง 607 รูป

**การเตรียมภาพ:** อ่านเป็น RGB → ย่อเป็น 64×64 → หาร 255 ให้ค่าอยู่ในช่วง 0–1 (`app.py` ทำขั้นตอนเดียวกันกับรูปที่ผู้ใช้อัปโหลด)

**การแบ่งข้อมูล** (ใน notebook `garbage_classification_cnn.ipynb` หัวข้อ 4):

| ชุด | สัดส่วนของข้อมูลทั้งหมด | จำนวนรูป | ใช้ทำอะไร |
|---|---|---:|---|
| Training | 64% | 9,929 | ฝึกโมเดล |
| Validation | 16% | 2,483 | ติดตามการฝึกและเลือก epoch ที่ดีที่สุด (early stopping) |
| Testing | 20% | 3,103 | ประเมินผลขั้นสุดท้ายเท่านั้น |

- แบ่งสองขั้น: กัน Test 20% ออกก่อน แล้วแบ่ง 20% ของที่เหลือเป็น Validation (= 16% ของทั้งหมด)
- ใช้ stratified split (คงสัดส่วนแต่ละ class) ด้วย `random_state=42` จึงแบ่งซ้ำได้เหมือนเดิม
- Test Set ไม่ถูกใช้ฝึก เลือก epoch หรือปรับพารามิเตอร์
- จำนวนรูปแต่ละชุดคำนวณจากรูปทั้งหมด 15,515 รูป ตามวิธีของ `train_test_split` ควรตรงกับที่ notebook พิมพ์ในหัวข้อ 4 (`Training / Validation / Test`) — ถ้าไม่ตรงให้แก้ตามที่ notebook พิมพ์

## 3. แบบจำลองที่ใช้ ผลประเมินหลัก และข้อจำกัด

**แบบจำลอง:** CNN (สร้างและฝึกใน `garbage_classification_cnn.ipynb`)

```
Input 64×64×3
→ Conv2D(32) + MaxPool → Conv2D(64) + MaxPool → Conv2D(128) + MaxPool
→ Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(12, Softmax)
```
Adam, sparse categorical cross-entropy, batch 32, สูงสุด 20 epochs พร้อม EarlyStopping (`val_loss`, patience 4, คืนน้ำหนักจาก epoch ที่ดีที่สุด) และ ModelCheckpoint

**เมตริกหลัก: Macro F1-score** เนื่องจากข้อมูลไม่สมดุล Accuracy จึงถูกดึงขึ้นโดย class ใหญ่อย่าง `clothes` ได้ง่าย Macro average เฉลี่ย F1 ของทุก class ให้น้ำหนักเท่ากัน จึงสะท้อน class ที่มีรูปน้อยด้วย (รายงาน Weighted average และ Accuracy ควบคู่)

**ผลบน Test Set** (3,103 รูป ที่ไม่ได้ใช้ฝึกหรือเลือก epoch) — ฝึกหยุดที่ epoch 13 ด้วย early stopping และใช้น้ำหนักจาก epoch 9 (Test Loss 0.8740)

| Accuracy | Macro F1 (หลัก) | Weighted F1 |
|---:|---:|---:|
| 74.41% | **0.6647** | 0.7368 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| battery | 0.6635 | 0.7407 | 0.7000 | 189 |
| biological | 0.8198 | 0.4619 | 0.5909 | 197 |
| brown-glass | 0.6118 | 0.7623 | 0.6788 | 122 |
| cardboard | 0.7386 | 0.7303 | 0.7345 | 178 |
| clothes | 0.8821 | 0.9343 | 0.9074 | 1065 |
| green-glass | 0.8699 | 0.8492 | 0.8594 | 126 |
| metal | 0.5038 | 0.4286 | 0.4632 | 154 |
| paper | 0.6916 | 0.7048 | 0.6981 | 210 |
| plastic | 0.5371 | 0.5434 | 0.5402 | 173 |
| shoes | 0.6622 | 0.7494 | 0.7031 | 395 |
| trash | 0.5951 | 0.6978 | 0.6424 | 139 |
| white-glass | 0.7222 | 0.3355 | 0.4581 | 155 |
| **macro avg** | 0.6915 | 0.6615 | 0.6647 | 3103 |
| **weighted avg** | 0.7457 | 0.7441 | 0.7368 | 3103 |

**แปลผล:**
- **จำแนกได้ดี:** `clothes` (F1 0.91) และ `green-glass` (0.86) ซึ่งหนึ่งในเหตุผลคือ `clothes` มีรูปมากที่สุด ส่วน `cardboard` (0.73), `shoes` (0.70), `battery` (0.70) และ `paper` (0.70) อยู่ในระดับกลาง
- **ยังสับสน:** `white-glass` (F1 0.46, recall ต่ำเพียง 0.34), `metal` (0.46) และ `plastic` (0.54) และ `biological` มี precision สูง (0.82) แต่ recall ต่ำ (0.46) คือโมเดลทำนายว่าเป็น `biological` น้อยเกินไป
- **คู่ที่สับสนมากสุด** (จาก Confusion Matrix): `white-glass` ถูกทำนายเป็น `trash` 31 รูปและ `plastic` 26 รูป, `clothes` ↔ `shoes` (31 และ 29 รูป), `biological` ถูกทำนายเป็น `brown-glass` 28 รูปและ `clothes` 26 รูป, `plastic` เป็น `trash` 21 รูป และ `metal` เป็น `battery`/`clothes` อย่างละ 18 รูป ซึ่งเป็นวัสดุที่มีสี พื้นผิวหรือความโปร่งใสใกล้กัน
- **Accuracy (74.41%) สูงกว่า Macro F1 (0.66)** เพราะ `clothes` ที่มีรูปมากและทำนายได้ดีดึง Accuracy ขึ้น จึงใช้ Macro F1 เป็นเมตริกหลัก ภาพที่ทำนายผิดมี 794 จาก 3,103 รูป

**ข้อจำกัด**
- ย่อภาพเหลือ 64×64 ทำให้เสียรายละเอียดพื้นผิว จึงอาจสับสนระหว่างวัสดุที่คล้ายกัน (เช่น plastic / metal / glass)
- โครงสร้าง CNN อย่างง่าย ไม่ได้ใช้ data augmentation หรือ transfer learning
- ข้อมูลไม่สมดุล และภาพส่วนใหญ่เป็นวัตถุเดี่ยวฉากหลังเรียบ ภาพจริงที่มีหลายวัตถุ แสงน้อย หรือฉากรกอาจทำนายผิด
- โมเดลทำนายได้เฉพาะ 12 class ที่ฝึก ภาพนอกกลุ่มนี้ยังถูกทำนายเป็นหนึ่งใน 12 class
- Confidence ไม่ใช่การรับประกันความถูกต้อง

## 4. ติดตั้งแพ็กเกจและรันแอปบนเครื่อง

```bash
git clone <repo-url>
cd garbage_classification_project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**ฝึกโมเดลใหม่** (ไม่บังคับ ถ้ามีไฟล์ `model/trash_class_cnn_mvp.keras` และ `model/metadata.json` อยู่แล้ว):

```bash
pip install jupyter
jupyter notebook garbage_classification_cnn.ipynb   # รันทุกเซลล์ตามลำดับ
```
Notebook ดาวน์โหลดชุดข้อมูลจาก Kaggle ผ่าน `kagglehub` (ต้องต่ออินเทอร์เน็ตในครั้งแรก) ฝึกโมเดล ประเมินผล แล้วบันทึก `model/trash_class_cnn_mvp.keras` และ `model/metadata.json` ที่ `app.py` ใช้งาน

**รันเว็บแอป:**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
เปิด http://127.0.0.1:8000

## 5. วิธีใช้งาน พร้อมตัวอย่างข้อมูลนำเข้า

**ผ่านหน้าเว็บ**
1. เปิด URL ของแอป
2. เลือก/อัปโหลดรูปขยะ (ไฟล์ภาพ .jpg .jpeg .png .bmp .webp)
3. กดทำนาย → แสดงประเภทที่ทำนาย, Top-5 และ confidence

**ตัวอย่างข้อมูลนำเข้า:** รูปขยะ 1 รูป เช่น รูปขวดพลาสติก (`plastic_01.jpg`) → คาดว่าได้ `predicted_class = plastic` (ใช้รูปขยะจริงที่ถ่ายเองหรือรูปจากชุดข้อมูล Kaggle ที่ตรงกับ class ใด class หนึ่งก็ได้)

**ผ่าน API**

```bash
curl -X POST -F "file=@plastic_01.jpg" http://127.0.0.1:8000/predict
```
ผลลัพธ์ (รูปแบบ):
```json
{
  "predicted_class": "plastic",
  "confidence": 0.82,
  "predictions": [
    {"class": "plastic", "confidence": 0.82},
    {"class": "metal", "confidence": 0.09}
  ]
}
```
(`predictions` จะมี 5 อันดับแรก ตัวเลขข้างต้นเป็นเพียงตัวอย่างรูปแบบ)

Endpoint อื่น: `GET /health` ตรวจสถานะแอปและโมเดล, `GET /info` ดูรายชื่อ class และข้อมูลโมเดล

## 6. คำอธิบายไฟล์และโฟลเดอร์สำคัญ

```text
garbage_classification_project/
├── garbage_classification_cnn.ipynb  # notebook: สำรวจข้อมูล → แบ่งข้อมูล → ฝึก → ประเมิน → บันทึกโมเดล
├── app.py              # FastAPI: เสิร์ฟหน้าเว็บ + endpoint /predict /health /info
├── requirements.txt    # แพ็กเกจที่ต้องใช้
├── render.yaml         # ตั้งค่า deploy บน Render
├── static/             # หน้าเว็บ (index.html, script.js, style.css)
├── data/               # ที่ใส่ชุดข้อมูล (ดูรายละเอียดใน data/README.txt)
└── model/              # trash_class_cnn_mvp.keras และ metadata.json (สร้างโดย notebook)
```

## 7. Deploy บน Render

1. ฝึกโมเดลใน notebook ก่อน แล้ว push โค้ดพร้อมโฟลเดอร์ `model/` (ไฟล์ `.keras` และ `metadata.json`) ขึ้น GitHub
2. Render → New → Web Service → เลือก repository
3. Build: `pip install -r requirements.txt` / Start: `uvicorn app:app --host 0.0.0.0 --port $PORT` (ตั้งไว้ใน `render.yaml` แล้ว)
4. ตรวจสอบ `/health` ว่า `model_loaded` เป็น `true` (ตรวจแล้ว: https://garbage-classification-m9ti.onrender.com/health ตอบ `status: ok`, `model_loaded: true`, 12 classes)

## 8. ผู้จัดทำ (เรียงตามรหัสนิสิตน้อยไปมาก)

| ลำดับ | รหัสนิสิต | ชื่อ-นามสกุล |
|---|---|---|
| 1 | 6730300035 | นางสาวกานต์ธิดา เปรื่องนนท์ |
| 2 | 6730300531 | นางสาววิรชา เชื้อสายสิทธิ์ |
| 3 | 6730300639 | นางสาวสุวนันท์ พิงสันเทียะ |
| 4 | 6730300728 | นางสาวณัฏฐ์อาภา เลิศศุภจิรโชติ |
| 5 | 6730300990 | นางสาวฐนิชา สอนสุข |