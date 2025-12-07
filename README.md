# TabRefine

Official implementation of the paper **"TabRefine: A Post-Processing Framework for Enhancing Synthetic Tabular Data in Network Intrusion Detection Systems"**, accepted at the **8th Annual Workshop on Cyber Threat Intelligence and Hunting (CyberHunt)**, held in conjunction with the **2025 IEEE International Conference on Big Data (BigData)**.

## 📦 Installation

1. Clone this repository.
2. Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 📂 Data Preparation

### 1. Included Datasets
We currently provide the pre-processed data (real, synthetic, and refined) for the **CIC-IDS2018** dataset.

* **CSE-CIC-IDS2018**:
    * **Source**: [https://registry.opendata.aws/cse-cic-ids2018/](https://registry.opendata.aws/cse-cic-ids2018/)
    * **Citation**:
        > Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018

    The data located in `real_data/CIC-IDS2018/` has been pre-processed (separated into numerical/categorical features and formatted as `.npy`) for immediate use with this framework.

### 2. Other Datasets
Due to licensing restrictions, we do not publicly distribute the **UNSW-NB15** and **CICIoT2023** datasets (including their synthetic versions) in this repository.

If you are interested in reproducing the experiments with these datasets or require the processed data for research purposes, please **contact the author** directly.

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