<div align="center">

# 🪐 TransitNet
### Explainable 1D-CNN Vetting of Kepler Exoplanet Transits

*NASA Space Apps Challenge 2026 · Theme: The Next Frontier*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NASA Data](https://img.shields.io/badge/Data-NASA%20Kepler%20%2F%20MAST-0B3D91)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview
When a planet crosses in front of its star, the star's light dips slightly. NASA's Kepler mission recorded thousands of these dips, but many are **false positives**: eclipsing binary stars, stellar variability, or instrument artifacts. Vetting them by hand is slow.

**TransitNet** is a two-branch 1D convolutional neural network that reads phase-folded Kepler light curves and predicts whether a signal is a real planet transit. It also explains *why*, using Grad-CAM saliency over the light curve.

The approach follows Shallue & Vanderburg (2018), the work that led to the discovery of Kepler-90i.

## ✨ Features
| Feature | Description |
|---|---|
| Two-view input | Global view (2001 bins, full orbit) + local view (201 bins, zoom on the dip) |
| Real NASA data | Light curves from MAST via `lightkurve`, labels from the NASA Exoplanet Archive KOI table |
| Synthetic mode | Physics-based generator (U-shaped planet vs V-shaped binary with secondary eclipse) for offline demos |
| Imbalance + augmentation | Weighted loss, time-reversal and noise augmentation |
| Evaluation suite | ROC, precision-recall, confusion matrix, validation loss, `metrics.json` |
| Explainability | Grad-CAM highlights which part of the orbit drove each decision |
| Vet any star | `--predict "Kepler-90"` runs a BLS period search then the CNN |

## 🧠 Architecture
```
Kepler light curve
      │  flatten → BLS/KOI period → phase-fold
      ├──► Global view (2001) ─► [Conv×2 + Pool] ×4 ─┐
      │                                               ├─► concat ─► Dense(128) ─► P(planet)
      └──► Local view  (201)  ─► [Conv×2 + Pool] ×2 ─┘
                                           └─► Grad-CAM saliency
```

## 🚀 Quick Start
```bash
git clone https://github.com/BhavishyasriMatangi-rgb/kepler-transit-cnn.git
cd kepler-transit-cnn
pip install -r requirements.txt

python exoplanet_cnn.py --mode synthetic --n 600 --epochs 10     # fast demo
python exoplanet_cnn.py --mode real --n 100 --epochs 15          # real Kepler data
python exoplanet_cnn.py --predict "Kepler-90"                    # vet one star
```
Runs on Google Colab too. Outputs are saved to `./outputs/`.

## 📊 Results
> Fill in from your own run. Report only numbers from `outputs/metrics.json`.

| Dataset | Test samples | ROC-AUC | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Real Kepler subset | _n_ | _x.xx_ | _x.xx_ | _x.xx_ | _x.xx_ |

<p align="center">
  <img src="outputs/evaluation.png" width="90%"><br>
  <img src="outputs/gradcam.png" width="90%">
</p>

**Note:** synthetic mode uses simulated light curves and scores very high because the task is easy. It exists to demo the pipeline, not to measure real performance.

## 🗂 Project Structure
```
├── exoplanet_cnn.py   # data, model, training, evaluation, Grad-CAM, prediction
├── requirements.txt
├── outputs/           # model weights, metrics.json, plots
└── README.md
```

## ⚠️ Limitations & Future Work
- Trained on a small subset of Kepler objects of interest; scaling to the full catalog is the next step.
- No centroid, odd/even depth, or secondary-eclipse side inputs yet (used by AstroNet/ExoMiner).
- Extend to TESS light curves.
- Not a replacement for expert vetting.

## 📚 References & Data
- Shallue, C. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning.* The Astronomical Journal, 155, 94.
- NASA Kepler Mission, MAST archive · NASA Exoplanet Archive (KOI cumulative table) · `lightkurve`

## 🤖 AI Disclosure
AI assistance (Claude) was used to help write parts of the code and documentation.

## 👩‍💻 Author
**Bhavishya Sri Matangi** · AI & ML undergraduate · [GitHub](https://github.com/BhavishyasriMatangi-rgb) · [LinkedIn](https://linkedin.com/in/bhavishya-sri-matangi-35618b35a)

## 📄 License
MIT
