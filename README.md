# CS-TMSS


## Table of Contents
- [Introduction](#introduction)
- [File Descriptions](#file-descriptions)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [File Structure](#file-structure)
  - [Installation](#installation)
  - [Usage](#Usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction
This project presents a deep learning-based solution for Prognosis and Segmentation of Head and Neck Cancer on HECTOR 2022 Dataset. Cancer is a leading global cause of death, accounted for almost 10 million deaths in 2023, according to the World Health Organization. Utilizing a BioLinkBert embeddings, Clip embeddings and Swintransformers , our model excels in Survival prediction and Segmentation. It is designed to work with the 'Hector 2022 Dataset', enabling the segmentation and Prognosis of Patients from 9 different centres. This model represents a significant improvement in CI index and Dice Score . It not only demonstrates the potential of deep learning in survival prediction and segmentation but also sets a new benchmark for future research endeavors. The model's ability to prognosied and segment patients from 9 differerent centre is currently unmatched, showcasing our commitment in easing the job of oncologists . Our approach beat the state-of-the-art performance, particularly in terms of CI index and Dice score, further contributing to the field of research.

## Technical Highlights

- **Advanced Text Embeddings from EHR**: We employed CLIP and BioLinkBERT
 to convert EHR textual data into semantic embeddings.
- **Formulating Specialized Text Prompts**:Our tailored prompts efficiently extract vital data
from patient records.
- **Implementation of the Swin Transformer Encoderr**: We replaced the ViT encoder with
the Swin Transformer.The Swin Transformer’s hierarchical and shifted window self-attention
mechanisms allow for more effective feature extraction at various scales.


Our Model Weights can be found at [Model Weights](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/sarim_hashmi_mbzuai_ac_ae/EtHUz5Tl53RBsVz9ihqxIBMBtRgzqOf8f8gqIu99aeeJRg?e=FDK88w)

<p align="center">
  <img src="https://github.com/Sarim-MBZUAI/TMSS-V2/blob/main/tmss.png" alt="TMSS old Architecture" width="75%"/>
  <br>
  <strong>Figure 1:</strong> TMSS Old Architecture
</p>
<p align="center">
  <img src="https://github.com/Sarim-MBZUAI/HC701_project/blob/main/architecture.png" alt="Our Architecture" width="75%"/>
  <br>
  <strong>Figure 2:</strong> Our Architecture
</p>




## File Descriptions

- `my_model.py`: Trains the deep learning model on HECTOR 2022 Dataset.
- `my_data.py`: code for medical imaging dataset processing and model training.
- `utils.py`: utilities for medical image processing and model training optimization. ​​
- `net.py`: implementation of UNETR for 3D medical image segmentation.
- `DEEP_MTLR.py`: module for configurable deep learning models in medical imaging.
- `swin.py`: implementation for Swin Transformer-based neural networks
  

## Getting Started

### Prerequisites
Download the dataset

| Data File | Download |
|-----------|----------|
| `HECTOR2022 Dataset` |  (https://hecktor.grand-challenge.org/)|

¹ Hosted at [MICCAI 2022](https://hecktor.grand-challenge.org/data-download-2/).


### Installation
Clone the repository and navigate to the project directory:

```bash
git clone [https://github.com/Sarim-MBZUAI/AI702_project.git]
cd AI702_project


### Environment Setup

Create a virtual environment using Conda to manage dependencies:

```bash
conda create -n tmssv2 python=3.8
conda activate tmssv2
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install -r torchmtlr/requirements.txt
cd torchmtlr
pip install -e .
```

### File Structure

project structure:

```
tmssv2/
│
├── weight/
├── log/
├── samples/
│     ├── gt/
│     ├── predictions/
├── src/
│   ├── Datamodules/
│   │   ├── components/
│   │   │   └── hector_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── components/
│   │   │   ├── net.py
│   │   │   └── swin.py
│   │   └── DEEP_MTLR.py
│   ├── vendor/
│   ├── my_data.py
│   ├── my_model.py
│   └── utils.py
│
├── torchmtlr/
│   ├── notebooks/
│   │   ├── MTLR for survival prediction.ipynb
│   │   └── time_bins.png
│   ├── tests/
│   │   ├── test_mtlr.py
│   │   └── test_utils.py
│   ├── torchmtlr/
│   │   └── utils.py
│   ├── torchmtlr.egg-info/
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt
│   └── setup.py
│
├── Prompt.txt
├── requirements.txt
├── train_loader.pkl
└── val_loader.pkl
```

<!--  ## Usage -->

### Usage

To train our model, use this command line:

```bash
python src/my_model.py
```

<!--  ## Usage -->

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact
This was done as a course project for HC701 at Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
For any inquiries or further information, please reach out to [sarim.hashmi@mbzuai.ac.ae].
## Reference
This work was built upon:
- Hatamizadeh, Ali, et al. "Swin unetr: Swin transformers for semantic segmentation of brain tumors in mri images." International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021.

- Saeed, Numan, et al. "TMSS: an end-to-end transformer-based multimodal network for segmentation and survival prediction." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022
