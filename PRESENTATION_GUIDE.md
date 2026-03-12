# Video Presentation Guide
## Snow Pole Detection - TDT17 Mini-Project

**Duration**: 12 minutes (solo) or 14 minutes (group of 2)

---

## 📋 Presentation Structure & Time Allocation

### 1. Background / Motivation (2 minutes)

**Key Points to Cover:**
- Challenge: Autonomous driving struggles in winter conditions
- Snow obscures road markings and lane lines
- Snow poles provide reliable reference points along roads
- Project goal: Real-time detection for winter AD systems

**Visual Aids:**
- Sample image showing snowy road conditions
- Image highlighting snow poles
- Diagram showing where snow poles are positioned

**Script Template:**
```
"Autonomous driving has made significant progress, but winter conditions 
remain challenging. When snow covers the road, traditional lane detection 
fails. However, snow poles—erected along Norwegian roads—provide a reliable 
way to identify road boundaries. Our project aims to develop a real-time 
detection system for these poles."
```

---

### 2. Data Analysis (2 minutes)

**Key Points to Cover:**
- Dataset size (train/validation/test splits)
- Number of images and annotations
- Distribution of poles per image
- Box size statistics
- Position distribution (where poles appear in images)
- Challenges identified (small objects, varying lighting, occlusion)

**Visual Aids:**
- Show: `analysis_results/data_analysis.png`
- Show: `analysis_results/sample_annotations.png`
- Highlight interesting patterns from analysis

**Script Template:**
```
"Our dataset contains [X] training images and [Y] validation images from 
the Trøndelag region. On average, each image contains [Z] snow poles. 
The analysis revealed that poles vary in size, with an average box area 
of [W] pixels. We observed that poles appear predominantly at [positions], 
which makes sense given road geometry."
```

---

### 3. Approach / Strategy (1 minute)

**Key Points to Cover:**
- Why YOLO? (real-time detection requirement)
- Edge deployment consideration (model size matters)
- Choice of YOLOv8-Nano/Small
- Single-class detection task

**Script Template:**
```
"For this task, we selected YOLOv8 due to its excellent balance of speed 
and accuracy. Since the final system must run on edge devices in vehicles, 
we chose the Nano/Small variant to minimize computational requirements 
while maintaining good detection performance. YOLOv8 is particularly 
well-suited for single-class object detection tasks like ours."
```

---

### 4. Methods / Models (2 minutes)

**Key Points to Cover:**
- YOLOv8 architecture overview
  - Backbone: CSPDarknet
  - Neck: PANet
  - Head: Detection head
- Training configuration:
  - Image size: 640x640
  - Batch size: 16
  - Epochs: 100
  - Optimizer: AdamW
  - Augmentation: Built-in YOLO augmentations
- Loss functions: Box loss, classification loss, DFL loss

**Visual Aids:**
- YOLOv8 architecture diagram (from internet/papers)
- Training configuration table

**Script Template:**
```
"YOLOv8 uses a CSPDarknet backbone for feature extraction, followed by 
a PANet neck for multi-scale feature fusion. We trained for 100 epochs 
with 640x640 input resolution and batch size 16. The model optimizes 
three loss components: bounding box regression, classification, and 
distribution focal loss for better localization."
```

---

### 5. Results (3 minutes)

**Key Points to Cover:**
- Training curves (loss, mAP over epochs)
- Final metrics:
  - Precision: [value]
  - Recall: [value]
  - mAP@0.5: [value]
  - mAP@0.5:0.95: [value]
- Visual results (show predictions on test images)
- Confidence distribution
- Success cases and failure cases

**Visual Aids:**
- Show: `runs/yolov8X_*/training_curves.png`
- Show: `evaluation/prediction_samples.png`
- Show: `evaluation/confidence_distribution.png`
- Confusion matrix if available

**For Groups of 2:**
- Compare two models side-by-side
- Table showing metric comparison
- Discuss trade-offs (speed vs accuracy, size vs performance)

**Script Template:**
```
"Our model achieved excellent results: Precision of [X], Recall of [Y], 
mAP@0.5 of [Z], and mAP@0.5:0.95 of [W]. As we can see in the training 
curves, the model converged smoothly after ~50 epochs. The validation 
loss plateaued, indicating good generalization. Here are some visual 
examples of the model's predictions..."

[For groups]: "Comparing our two models, YOLOv8-Small achieved higher 
accuracy ([metrics]) but with increased inference time. The Nano model 
offers better real-time performance, making it more suitable for edge 
deployment."
```

---

### 6. Discussion (1 minute)

**Key Points to Cover:**
- What worked well
  - Model generalizes across different lighting
  - Good detection of distant poles
- Limitations
  - Occasional false positives (signs, posts)
  - Struggles with heavily occluded poles
  - Performance in heavy snow/fog could improve
- Potential improvements
  - More training data in challenging conditions
  - Data augmentation for adverse weather
  - Temporal consistency (video tracking)

**Script Template:**
```
"The model performs well in most conditions, successfully detecting poles 
at various distances and lighting. However, we observed challenges with 
heavily occluded poles and occasional confusion with similar roadside 
objects. Future work could incorporate temporal information from video 
sequences and additional training data in extreme weather conditions."
```

---

### 7. Sustainability (1 minute)

**Key Points to Cover:**
- Training time: [X] hours
- Energy consumption: [Y] kWh
- Tesla Model Y comparison: [Z] km
- Percentage of Trondheim-Oslo trip

**Visual Aids:**
- Simple infographic showing energy comparison
- Tesla icon with distance

**Script Template:**
```
"Regarding sustainability, our model training consumed [X] hours of GPU 
time, equivalent to [Y] kWh of energy. To put this in perspective, the 
same energy could power a Tesla Model Y for [Z] kilometers—that's [P]% 
of the distance from Trondheim to Oslo. This highlights the importance 
of efficient model selection for both environmental and economic reasons."
```

---

### 8. Key Learning Points (1 minute)

**Key Points to Cover:**
- Technical insights
  - Importance of exploratory data analysis
  - Hyperparameter tuning impact
  - Trade-offs in model selection
- Process insights
  - GPU resource management
  - Dealing with queue times
  - Iterative experimentation

**Script Template:**
```
"This project reinforced several important concepts: thorough data analysis 
guides better model decisions, hyperparameter choices significantly impact 
performance, and practical constraints like inference speed matter as much 
as raw accuracy. Working with GPU clusters taught us valuable lessons in 
resource planning and efficient experimentation."
```

---

## 🎬 Recording Tips

### Technical Setup
1. **Screen Recording Software:**
   - OBS Studio (free, powerful)
   - Zoom (simple, reliable)
   - PowerPoint built-in recording

2. **Audio:**
   - Use a decent microphone (headset is fine)
   - Quiet environment
   - Test audio levels first

3. **Video Quality:**
   - 1080p recommended
   - 30 fps minimum
   - MP4 format for compatibility

### Presentation Tips
1. **Practice run-through** (at least twice)
2. **Stay within time limit** (use timer)
3. **Clear, confident speaking**
4. **Point to specific elements** when showing visuals
5. **Smooth transitions** between sections
6. **Professional but natural** tone

### Structure Your Slides/Screen
1. **Title slide** with project name
2. **Section headers** for each part
3. **Visual-heavy** (less text, more images/plots)
4. **Code snippets** only if discussing specific techniques
5. **Results first** approach (show the cool stuff early)

---

## 📊 Essential Visuals to Include

### Must-Have Figures:
1. ✅ Sample images with snow poles highlighted
2. ✅ Data distribution plots
3. ✅ Training curves (loss and mAP)
4. ✅ Prediction examples (good and bad)
5. ✅ Metrics table/comparison
6. ✅ Sustainability infographic

### Optional But Good:
- YOLOv8 architecture diagram
- Confusion matrix
- Confidence distribution
- Processing speed comparison
- Real-world application mockup

---

## 🎯 Common Mistakes to Avoid

❌ **Reading slides verbatim** → Explain naturally  
❌ **Too much technical jargon** → Make it accessible  
❌ **Ignoring time limits** → Practice with timer  
❌ **No visual examples** → Show actual results  
❌ **Rushed ending** → Plan good conclusion  
❌ **Forgetting to mention sustainability** → It's required!  
❌ **Not explaining metrics** → Define P, R, mAP briefly  

---

## ✅ Checklist Before Recording

- [ ] All results generated and saved
- [ ] Figures exported and ready
- [ ] Slides/screen layout prepared
- [ ] Script notes ready (not reading verbatim)
- [ ] Practiced at least twice
- [ ] Timing checked (within 12/14 min limit)
- [ ] Audio/video tested
- [ ] Quiet recording environment
- [ ] References prepared
- [ ] "Who did what" section ready (if group)

---

## 📤 Submission Checklist

- [ ] Video recorded (MP4 format)
- [ ] Duration: 12 min (solo) or 14 min (group)
- [ ] All required sections covered
- [ ] Code available (GitHub link or local)
- [ ] Uploaded to Teams before deadline
- [ ] File named appropriately: `TDT17_SnowPole_[YourName].mp4`

---

## 🎓 Grading Criteria Reminder

1. **Understanding of dataset** ← Show good EDA
2. **Data-driven model development** ← Explain choices based on analysis
3. **Understanding of method** ← Explain YOLO clearly
4. **Complexity of method** ← Appropriate for task
5. **Thoroughness** ← Complete pipeline with evaluation
6. **Clear presentation** ← Easy to follow, good visuals

---

**Good luck! You've got this! 🚀**
