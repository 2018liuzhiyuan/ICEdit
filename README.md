This is an **Ascend (Huawei NPU)-powered** version of ICEdit. 

It has been rigorously tested on the 910B platform with CANN-8.0.RC2. 

Moreover, this version supports multi-NPU training, enabling efficient and scalable training processes.

<div align="center">

<h1>In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer</h1>

<div>
    <a href="https://river-zhang.github.io/zechuanzhang//" target="_blank">Zechuan Zhang</a>&emsp;
    <a href="https://horizonwind2004.github.io/" target="_blank">Ji Xie</a>&emsp;
    <a href="https://yulu.net.cn/" target="_blank">Yu Lu</a>&emsp;
    <a href="https://z-x-yang.github.io/" target="_blank">Zongxin Yang</a>&emsp;
    <a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=zh-CN&oi=ao" target="_blank">Yi Yang✉</a>&emsp;
</div>
<div>
    ReLER, CCAI, Zhejiang University; Harvard University
</div>
<div>
     <sup>✉</sup>Corresponding Author
</div>
<div>
    <a href="https://arxiv.org/abs/2504.20690" target="_blank">Arxiv</a>&emsp;
    <a href="https://huggingface.co/spaces/RiverZ/ICEdit" target="_blank">Huggingface Demo 🤗</a>&emsp;
    <a href="https://huggingface.co/RiverZ/normal-lora/tree/main" target="_blank">Model 🤗</a>&emsp;
    <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">Project Page</a>
</div>


<div style="width: 80%; margin:auto;">
    <img style="width:100%; display: block; margin: auto;" src="docs/images/teaser.png">
    <p style="text-align: left;"><strong>Image Editing is worth a single LoRA!</strong> We present In-Context Edit, a novel approach that achieves state-of-the-art instruction-based editing <b>using just 0.5% of the training data and 1% of the parameters required by prior SOTA methods</b>. The first row illustrates a series of multi-turn edits, executed with high precision, while the second and third rows highlight diverse, visually impressive single-turn editing results from our method.</p>
</div>

:open_book: For more visual results, go checkout our <a href="https://river-zhang.github.io/ICEdit-gh-pages/" target="_blank">project page</a>


<div align="left">

# ⚠️ Tips

### If you encounter such a failure case, please **try again with a different seed**!

- Our base model, FLUX, does not inherently support a wide range of styles, so a large portion of our dataset involves style transfer. As a result, the model **may sometimes inexplicably change your artistic style**.

- Our training dataset is **mostly targeted at realistic images**. For non-realistic images, such as **anime** or **blurry pictures**, the success rate of the editing **drop and could potentially affect the final image quality**.

- While the success rates for adding objects, modifying color attributes, applying style transfer, and changing backgrounds are high, the success rate for object removal is relatively lower due to the low quality of the removal dataset we use.

The current model is the one used in the experiments in the paper, trained with only 4 A800 GPUs (total `batch_size` = 2 x 2 x 4 = 16). In the future, we will enhance the dataset, and do scale-up, finally release a more powerful model.

### ⚠️ Clarification

We've noticed numerous web pages related to ICEdit, including [https://icedit.net/](https://icedit.net/), [https://icedit.org/](https://icedit.org/). Kudos to those who built these pages!

However, we'd like to emphasize two important points:
- **No Commercial Use**: Our project **cannot** be used for commercial purposes. Please check the [LICENSE](https://github.com/River-Zhang/ICEdit/blob/main/LICENSE) for details.
- **Official Page**: The official project page is [https://river-zhang.github.io/ICEdit-gh-pages/](https://river-zhang.github.io/ICEdit-gh-pages/).



# 💼 Installation

## Conda environment setup

```bash
conda create -n icedit python=3.10
conda activate icedit
pip install -r requirements.txt
pip install -U huggingface_hub
```

## Download pretrained weights

If you can connect to Huggingface, you don't need to download the weights. Otherwise, you need to download the weights to local.

- [Flux.1-fill-dev](https://huggingface.co/black-forest-labs/flux.1-fill-dev).
- [ICEdit-normal-LoRA](https://huggingface.co/RiverZ/normal-lora/tree/main).

Note: Due to some cooperation permission issues, we have to withdraw the weights and codes of moe-lora temporarily. What is released currently is just the ordinary lora, but it still has powerful performance. If you urgently need the moe lora weights of the original text, please email the author.

## Inference in bash (w/o VLM Inference-time Scaling)

Now you can have a try!

> Our model can **only edit images with a width of 512 pixels** (there is no restriction on the height). If you pass in an image with a width other than 512 pixels, the model will automatically resize it to 512 pixels.

> If you found the model failed to generate the expected results, please try to change the `--seed` parameter. Inference-time Scaling with VLM can help much to improve the results.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --seed 304897401

```

Editing a 512×768 image requires 35 GB of NPU memory. If you need to run on a system with 24 GB of NPU memory, you can add the `--enable-model-cpu-offload` parameter.

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --enable-model-cpu-offload
```

If you have downloaded the pretrained weights locally, please pass the parameters during inference, as in: 

```bash
python scripts/inference.py --image assets/girl.png \
                            --instruction "Make her hair dark green and her clothes checked." \
                            --flux-path /path/to/flux.1-fill-dev \
                            --lora-path /path/to/ICEdit-normal-LoRA
```


# 🔧 Training

Found more details in here: [Training Code](./train/)

Challenges:
1. Different from GPU, there are many `Ascend-toolkit` related **system variables** to add.
2. Some python libraries are not compatible with NPU. For convenience, I provide available [combination](./train/requirements.txt) with exact lib versions. 

When facing underlying problems of Ascend NPU, it's recommended to refer to github issues of related repositories or ask `Kimi` which is more professional on Ascend NPU than other AI.

Some demos after training 12000 global steps:

Change it to look like it's in the style of an impasto painting.
![flux-fill-test-12000-3-edit](https://github.com/user-attachments/assets/8b61e84c-5f6b-40da-8926-9dbd6594b061)
Make it pop art.
![flux-fill-test-12000-9-edit](https://github.com/user-attachments/assets/49798605-d0cf-4148-95a2-2a49cf1cc9ef)
Make the image look like it's from an ancient Egyptian mural.
![flux-fill-test-12000-0-edit](https://github.com/user-attachments/assets/2c7f9116-1242-47a1-bcfa-a37fa18b5e50)


# 💪 Comparison with Commercial Models

<div align="center">
<div style="width: 80%; text-align: left; margin:auto;">
    <img style="width:100%" src="docs/images/gpt4o_comparison.png">
    <p style="text-align: left;">Compared with commercial models such as Gemini and GPT-4o, our methods are comparable to and even superior to these commercial models in terms of character ID preservation and instruction following. <b>We are more open-source than them, with lower costs, faster speed (it takes about 9 seconds to process one image), and powerful performance</b>.</p>
</div>


<div align="left">


# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{zhang2025ICEdit,
      title={In-Context Edit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer}, 
      author={Zechuan Zhang and Ji Xie and Yu Lu and Zongxin Yang and Yi Yang},
      year={2025},
      eprint={2504.20690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.20690}, 
}
```
