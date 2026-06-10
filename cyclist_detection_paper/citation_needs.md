# Citation Needs Document

This document maps every empty `\cite{}` call in `main.tex` to the topic/sentence that needs a reference.

## Empty Citations Identified

### 1. YOLO family `\cite{}`
**Location:** Related Work, Pedestrian and Cyclist Detection section (~line 64)  
**Context:** "General object detection backbones including the YOLO family~\cite{}, RT-DETR~\cite{}, and RTMDet~\cite{}..."  
**Needed:** A recent survey or foundational YOLO paper. The YOLOv8 paper (Ultralytics, 2023) or the original YOLOv1-v8 survey is appropriate.

### 2. RT-DETR `\cite{}`
**Location:** Related Work (~line 64)  
**Context:** Same sentence as YOLO.  
**Needed:** The RT-DETR paper (Real-Time Detection Transformer). Typically Zhao et al., 2023.

### 3. RTMDet `\cite{}`
**Location:** Related Work (~line 64)  
**Context:** Same sentence as YOLO/RT-DETR.  
**Needed:** The RTMDet paper (OpenMMLab). Typically Li et al., 2022.

### 4. SORT `\cite{}`
**Location:** Related Work, Multi-Object Tracking section (~line 66)  
**Context:** "SORT~\cite{} introduced Kalman-filter-based tracking..."  
**Needed:** The original SORT paper. Bewley et al., 2016.

### 5. DeepSORT `\cite{}`
**Location:** Related Work (~line 66)  
**Context:** "...while DeepSORT~\cite{} added appearance embeddings for re-identification."  
**Needed:** The original DeepSORT paper. Wojke et al., 2017.

### 6. ByteTrack `\cite{}`
**Location:** Related Work (~line 66)  
**Context:** "ByteTrack~\cite{} improved association through secondary low-confidence detections."  
**Needed:** The ByteTrack paper. Zhang et al., 2022.

### 7. SAHI `\cite{}`
**Location:** Related Work, Small-Object Detection section (~line 68)  
**Context:** "SAHI~\cite{} demonstrated that tiled inference on high-resolution inputs significantly improves small-object recall."  
**Needed:** The SAHI paper (Slicing Aided Hyper Inference). Akyon et al., 2022.

### 8. Homography / IPM `\cite{}`
**Location:** Related Work (~line 68)  
**Context:** "Homography and Inverse Perspective Mapping are widely used in autonomous driving and surveillance to rectify angled views~\cite{}."  
**Needed:** A recent survey or foundational paper on IPM/homography for traffic surveillance. Could be a recent autonomous driving survey or a specific paper on perspective correction for traffic cameras.

### 9. PET (Post-Encroachment Time) `\cite{}`
**Location:** Related Work, Safety Metrics section (~line 70)  
**Context:** "Post-Encroachment Time (PET)~\cite{} measures the time gap between one road user leaving a conflict zone and another entering it."  
**Needed:** The original PET definition paper in traffic safety. PET was introduced by Allen et al. (1978) or a more recent traffic safety paper that uses PET.

### 10. RT-DETR `\cite{}` (Background section)
**Location:** Background, RT-DETR subsection (~line 81)  
**Context:** "RT-DETR~\cite{} is a real-time detection transformer that retains the end-to-end advantages of DETR..."  
**Needed:** Same as #2 (RT-DETR paper). Could also cite original DETR paper (Carion et al., 2020).

### 11. DeepSORT `\cite{}` (Background section)
**Location:** Background, DeepSORT subsection (~line 85)  
**Context:** "DeepSORT~\cite{} combines a Kalman filter for motion prediction with a MobileNet appearance embedding network..."  
**Needed:** Same as #5 (DeepSORT paper).

### 12. EuroCity Persons dataset `\cite{}`
**Location:** Method, Dataset Curation section (~line 99)  
**Context:** "...the EuroCity Persons dataset~\cite{}."  
**Needed:** The EuroCity Persons dataset paper. Braun et al., 2019.

---

## Additional Citations Worth Adding (not currently marked with `\cite{}`)

- **COCO / Cityscapes:** Mentioned in Intro (~line 51) as benchmarks. Could cite Lin et al. 2014 (COCO) and Cordts et al. 2016 (Cityscapes).
- **DETR:** Mentioned in Background (~line 81) as the original transformer detector. Carion et al. 2020.
- **Ultralytics:** Mentioned in Method (~line 121) for the framework. Jocher et al. 2023.
