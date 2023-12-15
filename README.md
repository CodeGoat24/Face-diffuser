# High-fidelity Person-centric Subject-to-Image Synthesis

Official implementation of [High-fidelity Person-centric Subject-to-Image Synthesis](https://arxiv.org/pdf/2311.10329.pdf).

> **PrimeComposer: Faster Progressively Combined Diffusion for Image Composition with Attention Steering**<br>
> Yibin Wang, Weizhong Zhang, Jianwei Zheng, and Cheng Jin <br>
> 
## Abstract
Current subject-driven image generation methods encounter significant challenges in person-centric image generation. The reason is that they learn the semantic scene and person generation by fine-tuning a common pre-trained diffusion, which involves an irreconcilable training imbalance. Precisely,  to generate realistic persons, they need to sufficiently tune the pre-trained model, which inevitably causes the model to forget the rich semantic scene prior and makes scene generation over-fit to the training data. 
Moreover, even with sufficient fine-tuning, these methods can still not generate high-fidelity persons since joint learning of the scene and person generation also lead to quality compromise. In this paper, we propose  Face-diffuser, an effective collaborative generation pipeline to eliminate the above training imbalance and quality compromise. Specifically, we first develop two specialized pre-trained diffusion models, i.e., Text-driven Diffusion Model (TDM) and Subject-augmented Diffusion Model (SDM), for scene and person generation, respectively. The sampling process is divided into three sequential stages, i.e., semantic scene construction, subject-scene fusion, and subject enhancement. The first and last stages are performed by TDM and SDM respectively. The subject-scene fusion stage, that is the collaboration achieved through a novel and highly effective mechanism, Saliency-adaptive Noise Fusion (SNF). Specifically, it is based on our key observation that there exists a robust link between classifier-free guidance responses and the saliency of generated images. In each time step, SNF leverages the unique strengths of each model and allows for the spatial blending of predicted noises from both models automatically in a saliency-aware manner, all of which can be seamlessly integrated into the DDIM sampling process. Extensive experiments confirm the impressive effectiveness and robustness of the Face-diffuser in generating high-fidelity person images depicting multiple unseen persons with varying contexts.

![multi-subject](figures/display.png)

![framework](figures/framework.png)

## TODOs

- [ ] Release inference code
- [ ] Release pre-trained models
- [ ] Release demo
- [ ] Release training code and data
- [ ] Release evaluation code and data

## Citation

If you find Face-Diffuser relevant to your research, please kindly cite our paper:

```bibtex
@article{wang2023high,
  title={High-fidelity Person-centric Subject-to-Image Synthesis},
  author={Wang, Yibin and Zhang, Weizhong and Zheng, Jianwei and Jin, Cheng},
  journal={arXiv preprint arXiv:2311.10329},
  year={2023}
}
```