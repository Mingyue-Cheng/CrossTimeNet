<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b>  CrossTimeNet: Cross-Domain Pre-training with Language Models for Transferable Time Series Representations (WSDM2025, Accepted) </b></h2>
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{cheng2024learning,
  title={Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model},
  author={Cheng, Mingyue and Tao, Xiaoyu and Liu, Qi and Zhang, Hao and Chen, Yiheng and Lei, Chenyi},
  journal={arXiv preprint arXiv:2403.12372},
  year={2024}
}
```
### Updates/News:
🚩 **News** (): 


### Introduction
Pre-training universal models across multiple domains to enhance downstream tasks is a prevalent learning paradigm. However, there has been minimal progress in pre-training transferable models across domains for time series representation. This dilemma is incurred by two key factors: the limited availability of training set within each domain and the substantial differences in data characteristics between domains. To address these challenges, we present a novel framework, namely CrossTimeNet, designed to perform cross-domain self-supervised pre-training to benefit target tasks. Specifically, to address the issue of data scarcity, we utilize a pretrained language model as the backbone network to effectively capture the sequence dependencies of the input sequence. Meanwhile, we adopt the recovery of corrupted inputs as a self-supervised optimization objective, taking into account the locality of time series. To address discrepancies in data characteristics, we introduce a novel tokenization module that converts continuous time series inputs into discrete token sequences using vector quantization techniques. This approach facilitates the learning of transferable time series models across different domains. Extensive experimental results on diverse time series tasks, including classification and forecasting, demonstrate the effectiveness of our approach. 

<img width="1101" alt="截屏2024-09-13 11 34 11" src="https://github.com/user-attachments/assets/6403fff1-2215-40e6-a4e4-c371317dda5a">
<img width="1403" alt="截屏2024-09-13 11 34 44" src="https://github.com/user-attachments/assets/ba2e5a01-d553-4b25-b1c4-0980f069fc53">


### Further Reading

1, [**Diffusion Auto-regressive Transformer for Effective Self-supervised Time Series Forecasting**](https://arxiv.org/pdf/2410.05711).

**Authors**: Daoyu Wang, Mingyue Cheng, Zhiding Liu, Qi Liu, Enhong Chen

```bibtex
@article{wang2024diffusion,
  title={Diffusion Auto-regressive Transformer for Effective Self-supervised Time Series Forecasting},
  author={Wang, Daoyu and Cheng, Mingyue and Liu, Zhiding and Liu, Qi and Chen, Enhong},
  journal={arXiv preprint arXiv:2410.05711},
  year={2024}
}
```
2, [**Generative pretrained hierarchical transformer for time series forecasting**](https://arxiv.org/pdf/2402.16516).

**Authors**: Liu, Zhiding, Jiqian Yang, Mingyue Cheng, Yucong Luo, and Zhi Li.

```bibtex
@inproceedings{liu2024generative,
  title={Generative pretrained hierarchical transformer for time series forecasting},
  author={Liu, Zhiding and Yang, Jiqian and Cheng, Mingyue and Luo, Yucong and Li, Zhi},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={2003--2013},
  year={2024}
}
```
3, [**Timemae: Self-supervised representations of time series with decoupled masked autoencoders**](https://arxiv.org/pdf/2303.00320).

**Authors**: Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong.

```bibtex
@article{cheng2023timemae,
  title={Timemae: Self-supervised representations of time series with decoupled masked autoencoders},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong},
  journal={arXiv preprint arXiv:2303.00320},
  year={2023}
}
```

### Contact
If you have any questions, we encourage you to either create Github issues or get in touch with us at <mycheng@ustc.edu.cn>.


