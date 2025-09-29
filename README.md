# Action Recognition on UCF101, Deep Learning vs Machine Learning

This project compares **deep learning** and **traditional machine learning** approaches for **video action recognition** on **UCF101**. We build a **CNN plus LSTM** pipeline and two **ML** pipelines based on **hand crafted features** and **SVM**, then evaluate accuracy, F1, efficiency, and error patterns.

## Dataset
- **Source**: **UCF101**, 13,320 videos across **101 action classes**, grouped into **25 groups** with diverse viewpoints and backgrounds.
- **Task**: classify a video into one of the **101 actions**. Standard train and test splits are provided by the dataset.

## Project Overview

### Deep Learning pipeline
- **Model**: **ResNet18** backbone for per frame **spatial features**, followed by an **LSTM** for **temporal modeling**, then a **linear classifier** with softmax.
- **Setup**: 16 frames per video sequence, **Adam** optimizer, **learning rate 1e-4**, **batch size 4**, **10 epochs**.
- **Goal**: learn end to end spatiotemporal representations from frames without manual feature engineering.

### Machine Learning pipelines
- **Pipeline 1, STIP plus BoVW plus SVM**  
  Extract **STIP** keypoints, compute **HOG** and **HOF** descriptors, quantize with **K Means** to build **Bag of Visual Words histograms** around 500 dimensions, train **SVM** with **RBF kernel** using **GridSearchCV**.
- **Pipeline 2, Global HOG plus HOF plus PCA plus SVM**  
  Aggregate **HOG** and **HOF** per video, **concatenate**, reduce with **PCA** to around 256 dimensions, train **SVM** with **GridSearchCV**.

## Results
- **Top accuracy**: **Deep Learning 80.15 percent** accuracy, **F1 about 0.80**.  
- **ML accuracy**: **STIP plus BoVW plus SVM about 53 percent**, **Global HOG plus HOF about 37.8 percent**.  
- **Confusion patterns**: DL struggles with **Lunges**, **BrushingTeeth**, **BoxingPunchingBag**. ML struggles with **PlayingCello**, **BrushingTeeth**, **HandstandWalking**.

## Efficiency
- **Total pipeline time**: **DL about 27.8 minutes** for 10 epochs including data handling, **ML about 36.2 minutes** due to heavy **BoVW histogram generation** in STIP parsing and quantization.
- **Training only**: **DL training about 27.8 minutes** for 10 epochs, **ML SVM training about 1.5 minutes** including **GridSearchCV**.

## Findings
- **Feature learning wins**: **CNN plus LSTM** outperforms **hand crafted features** by a large margin on this subset, especially for actions with richer temporal dynamics.
- **Class specific errors**: low F1 for **fine motor or subtle actions** such as **BrushingTeeth** and **PlayingCello** indicates limits of both global pooling and local STIPs for nuanced motion.
- **Cost profile**: ML appears simpler, yet **feature extraction time** can dominate runtime. DL needs **GPU**, but avoids large offline feature steps.
