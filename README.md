
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
sdxl-turbo/
├── README.md                   # This file, providing an overview of the repository
├── emotion_prompt.py           # Script for generating emotion-based prompts
├── entity_interpretation_prompt.py  # Script for generating entity interpretation prompts
├── entity_prompt.py            # Script for generating entity-related prompts
├── generated_entity_image.py   # Script for generating images associated with extracted entities
├── generated_sentences_image.py # Script for generating sentences and corresponding images
└── requirements.txt            # List of Python dependencies
└── ....           # More thing
```

---

## Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name/sdxl-turbo
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
The repository contains several scripts that are used in the hierarchical generation and fusion process. Below is a brief description of each script and how to use it:

### 1. Emotion Prompt Generation (`emotion_prompt.py`)
This script generates emotion-based prompts that are used to guide the model in understanding the emotional context of the text and images.

**Usage**:
```bash
python emotion_prompt.py --input_text "Your input text here" --output_file "emotion_prompts.json"
```

### 2. Entity Interpretation Prompt Generation (`entity_interpretation_prompt.py`)
This script generates prompts that help the model interpret entities in the context of multimodal data.

**Usage**:
```bash
python entity_interpretation_prompt.py --input_text "Your input text here" --output_file "entity_interpretation_prompts.json"
```

### 3. Entity Prompt Generation (`entity_prompt.py`)
This script generates prompts related to entities, which are used to extract and align entities from text and images.

**Usage**:
```bash
python entity_prompt.py --input_text "Your input text here" --output_file "entity_prompts.json"
```

### 4. Generated Entity Image (`generated_entity_image.py`)
This script generates images associated with the extracted entities, which are then used for multimodal alignment.

**Usage**:
```bash
python generated_entity_image.py --entity "Entity Name" --output_image "entity_image.png"
```

### 5. Generated Sentences and Images (`generated_sentences_image.py`)
This script generates sentences and corresponding images, which are used for aligning text and image data.

**Usage**:
```bash
python generated_sentences_image.py --input_text "Your input text here" --output_image "sentence_image.png"
```

---

## Requirements
The code is written in Python and requires the following dependencies, which can be installed using the `requirements.txt` file:

- `torch`
- `transformers`
- `PIL`
- `numpy`
- `pandas`
- `scikit-learn`

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
