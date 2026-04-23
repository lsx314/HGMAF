
# Hierarchical Generation of Multi-Evidence Alignment and Fusion Model for Multimodal Entity and Relation Extraction

This repository contains the code and resources for the paper titled **"The More Quality Information the Better: Hierarchical Generation of Multi-Evidence Alignment and Fusion Model for Multimodal Entity and Relation Extraction"** by He X, Li S, Zhang Y, et al. The paper proposes a novel hierarchical approach to extract entities and relations from multimodal data (text and images) by aligning and fusing evidence from multiple sources.

---

## Citation
If you use this code or the associated paper in your research, please cite the following paper:

```bibtex
@article{he2025more,
  title={The more quality information the better: Hierarchical generation of multi-evidence alignment and fusion model for multimodal entity and relation extraction},
  author={He, X and Li, S and Zhang, Y and others},
  journal={Information Processing \& Management},
  volume={62},
  number={1},
  pages={103875},
  year={2025},
  publisher={Elsevier}
}
```

---

## Abstract
The paper introduces a hierarchical model that aligns and fuses multi-evidence from text and images to improve the accuracy of entity and relation extraction. The model leverages deep learning techniques to process and integrate information from different modalities, enabling more robust and context-aware extraction. The proposed approach is evaluated on a multimodal dataset, demonstrating superior performance compared to existing methods.

---

## Repository Structure
The repository is organized as follows:

```
HGMAF/
├── HGMAF_Stage1/                 # Stage 1: hierarchical multi-evidence generation
├── HGMAF_Stage2_AdaSeq/          # Stage 2: AdaSeq-based HGMAF alignment and fusion implementation
├── LICENSE                       # MIT License file
├── README.md                     # Project overview and usage instructions
└── requirements.txt              # Python dependencies

```

---

## Usage

### Stage 1: Multi-Evidence Generation

The first stage is located in:

```bash
cd HGMAF_Stage1
```

This stage is used to generate hierarchical evidence, including textual evidence and visual evidence. The generated evidence can include emotion-related text, entity-related text, entity interpretation text, sentence-level generated images, and entity-level generated images.

Before running the scripts, please modify the input and output paths according to your local dataset location.

Example usage:

```bash
python emotion_prompt.py
python entity_prompt.py
python entity_interpretation_prompt.py
python generated_sentences_image.py
python generated_entity_image.py
```

The outputs of this stage can be used as auxiliary evidence for the second-stage HGMAF alignment and fusion model.

### Stage 2: HGMAF Alignment and Fusion

The second stage is located in:

```bash
cd HGMAF_Stage2_AdaSeq
```

This stage implements the HGMAF Stage2 model based on the AdaSeq framework. It is used for sequence labeling tasks such as multimodal named entity recognition.

Example training commands:

```bash
python -W ignore -m scripts.train -c examples/HGMAF/twitter-15-HGMAF.yaml
python -W ignore -m scripts.train -c examples/HGMAF/twitter-17-HGMAF.yaml
```

## Recommended Workflow

```text
Raw multimodal data
        ↓
HGMAF_Stage1
        ↓
Generate textual and visual evidence
        ↓
HGMAF_Stage2_AdaSeq
        ↓
Multi-evidence alignment and fusion
        ↓
Multimodal entity extraction results
```

## Requirements

The code is written in Python and requires the following dependencies, which can be installed using the `requirements.txt` file:

```txt
datasets>=2.0.0
huggingface-hub>=0.10.0
modelscope>=1.4.0,<2.0.0
numpy>=1.21.0,<2.0.0
packaging>=21.0
PyYAML>=6.0
requests>=2.28.0
sentencepiece>=0.1.99
seqeval>=1.2.2
torch>=1.11.0
tqdm>=4.64.0
transformers>=4.21.0
urllib3>=1.26.0
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement

The implementation of the second stage is based on the open-source sequence understanding framework [AdaSeq](https://github.com/modelscope/AdaSeq). We sincerely thank the AdaSeq team for their public work on data processing, training pipelines, and model components, which provided an important engineering foundation for the reproduction and extension of this project. AdaSeq is open-sourced under the Apache License 2.0, and its well-designed implementation has been of great help to this work.
