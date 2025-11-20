# TabRefine

Official implementation of the paper **"TabRefine: A Post-Processing Framework for Enhancing Synthetic Tabular Data in Network Intrusion Detection Systems"**, accepted at the **8th Annual Workshop on Cyber Threat Intelligence and Hunting (CyberHunt)**, held in conjunction with the **2025 IEEE International Conference on Big Data (BigData)**.

## 📦 Installation

1. Clone this repository.
2. Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📂 Data Preparation

Please download the necessary datasets from the links provided in `data_link.txt`.

After downloading, extract and place the data so that the `real_data`, `refined_data`, and `synth_data` folders are located in the same directory as `tabrefine.py`.

The directory structure should look like this:

```text
.
├── tabrefine.py
├── utility_evaluation.py
├── privacy_evaluation.py
├── requirements.txt
├── data_link.txt
├── real_data/
│   └── ...
├── refined_data/
│   └── ...
└── synth_data/
    └── ...
```

## 🚀 Usage

### 1. Running TabRefine

To perform the refinement process using TabRefine:

1. Open `tabrefine.py` and modify the `dataset_name` and `baseline_method` variables to select the target data.
2. Run the script:

```bash
python3 tabrefine.py
```

### 2. Evaluation

To evaluate the utility or privacy of the data:

1. Open `utility_evaluation.py` or `privacy_evaluation.py`.
2. Modify the `dataset_name` and `baseline_method` variables as needed.
3. Run the corresponding script:

**Utility Evaluation:**

```bash
python3 utility_evaluation.py
```

**Privacy Evaluation:**

```bash
python3 privacy_evaluation.py
```

## 👏 Acknowledgements

The implementation of privacy metrics (specifically in `syntheval`) is adapted from [syntheval](https://github.com/schneiderkamplab/syntheval) by [schneiderkamplab].
We have modified the code to fit our data format and improved the calculation efficiency.

## 📝 Citation

If you use this code or our method in your research, please cite our paper:

```bibtex
@inproceedings{ishii2025tabrefine,
  title={TabRefine: A Post-Processing Framework for Enhancing Synthetic Tabular Data in Network Intrusion Detection Systems},
  author={Ishii, Haruto and Akashi, Kunio and Sekiya, Yuji},
  booktitle={2025 IEEE International Conference on Big Data (BigData)},
  pages={},
  year={2025},
  organization={IEEE}
}
```