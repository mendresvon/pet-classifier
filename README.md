---
title: AI Pet Species Classifier
emoji: 🐾
colorFrom: zinc
colorTo: slate
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🐾 AI Pet Species Classifier | 寵物物種辨識系統

### 👤 Developer Information
* **Name / 姓名:** 馬盛中 (Ma Sheng-Zhong)
* **Student ID / 學號:** 4B1YZ001
* **Institution / 學校:** Southern Taiwan University of Science and Technology (STUST)
* **Department / 系所:** Computer Science and Information Engineering (CSIE)

---

## 📖 Project Overview
This project is a deep learning-based image classifier designed to recognize 7 common household pet species with high confidence. The model was trained using the **fastai** framework and is deployed via **Hugging Face Spaces** for real-time inference.

### 🎯 Supported Species
The model is optimized to identify the following classes:
1. **Cat** (貓)
2. **Dog** (狗)
3. **Goldfish** (金魚)
4. **Hamster** (倉鼠)
5. **Turtle** (烏龜)
6. **Parrot** (鸚鵡)
7. **Snake** (蛇)

---

## 🛠️ Technical Stack
* **Architecture:** ResNet34 (Transfer Learning)
* **Framework:** fastai v2 / PyTorch
* **Deployment:** Gradio & Hugging Face Spaces
* **Language:** Python 3.x
* **Dataset:** Animal Image Dataset (90 Different Animals)



---

## 🚀 How to Use
1. **Upload:** Drag and drop an image of a pet into the input box.
2. **Analyze:** Click the "Analyze / 開始辨識" button.
3. **Results:** View the top 3 most likely species and their corresponding confidence scores.

---

## 🎓 Academic Context
This project was developed as part of a deep learning coursework at **STUST CSIE**. It demonstrates the complete machine learning pipeline:
* Data collection and cleaning via symbolic links.
* Model selection and fine-tuning.
* Performance evaluation using confusion matrices.
* Full-stack deployment of a trained model.