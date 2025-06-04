# Concept-Bottleneck-LLM

This is the extended and reproduced implementation of the paper: [Concept Bottleneck Large Language Models](https://arxiv.org/abs/2402.06634).  
We faithfully reproduce the CB-LLM classification pipeline and introduce two additional contributions:
- **NEC (Number of Effective Concepts) Analysis**: Study the effect of sparsity on performance.
- **BCE-trained Concept Bottleneck Layer**: Explore alternative concept training objectives.

This repo is adapted to run under limited GPU resources and includes updates to improve efficiency and compatibility.

---

## 🔧 Setup

We recommend using:
- CUDA 12.1  
- Python 3.10  
- PyTorch 2.2  

After cloning the repo, install dependencies:
```bash
cd classification
pip install -r requirements.txt
