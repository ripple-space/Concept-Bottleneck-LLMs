# Concept-Bottleneck-LLM

This is the extended and reproduced implementation of the paper: [Concept Bottleneck Large Language Models](https://arxiv.org/abs/2412.07992).  
We faithfully reproduce the CB-LLM classification pipeline and introduce two additional contributions:
- **NEC (Number of Effective Concepts) Analysis**: Study the effect of sparsity on performance.
- **BCE-trained Concept Bottleneck Layer**: Explore alternative concept training objectives.

This repo is adapted to run under limited GPU resources and includes updates to improve efficiency and compatibility.

---

### 🔧 Setup

We recommend using:
- CUDA 12.1  
- Python 3.10  
- PyTorch 2.2  

After cloning the repo, install dependencies:
```bash
cd classification
pip install -r requirements.txt```

Note: We updated torchvision in requirements.txt from `0.17.0` to `0.19.0` for compatibility.

Download the finetuned CB-LLM checkpoints from HuggingFace:
```bash
git lfs install
git clone https://huggingface.co/cesun/cbllm-classification temp_repo
mv temp_repo/mpnet_acs .
rm -rf temp_repo```

---

## 📊 Part I: CB-LLM (Classification)

###  Automatic Concept Scoring (ACS)
To generate concept scores for a dataset, run:
```bash
python get_concept_labels.py

This will generate the concept scores for the SST2 dataset using our predefined concept set, and store the scores under `mpnet_acs/SetFit_sst2/`. Set the argument `--dataset ag_news`, `--dataset yelp_polarity`, or `--dataset dbpedia_14` to switch the dataset.

**Updates:**

- Reduced `batch_size` for large datasets.
- Cast model weights and features to `float16`.
- Used efficient batching for tokenization.

### Train the Concept Bottleneck Layer (CBL)
To train the CBL, run
```bash
python train_CBL.py --automatic_concept_correction

This will train the CBL with Automatic Concept Correction for the SST2 dataset, and store the model under `mpnet_acs/SetFit_sst2/roberta_cbm/`. To disable Automatic Concept Correction, remove the given argument. Set the argument `--backbone gpt2` to switch the backbone from roberta to gpt2. Set the argument `--dataset ag_news`, `--dataset yelp_polarity`, or `--dataset dbpedia_14` to switch the dataset.

**Update:**
Checkpoints are saved automatically to avoid losing progress if your connection breaks.

### Train the Final Predictor
To train the final predictor, run:
```bash
python train_FL.py --cbl_path mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt

This will train the linear predictor of the CBL for the SST2 dataset, and store the linear layer in the same directory.
Please change the argument `--cbl_path` accordingly for other settings.
For example, without Automatic Concept Correction, the model will be saved as `cbl.pt`.

**Update:**
The code now supports more flexible backbone detection (e.g. both `roberta` and `gpt2` in string).




## 🧪 Additional Experiments

