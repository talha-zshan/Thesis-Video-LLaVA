## Create A virtual Environment
use either python or conda to create a virtual environment

For Python, use the following command
```python
python3 -m venv <environment_name>
source /path/to/env/bin/activate
```
or for Conda, use

```bash
conda create -n <environment_name> python=3.10 -y
conda activate <environment_name>
```

## 🛠️ Autogen And Kafka Requirements and Installation
In the virtual environment, use the following pip command to install the necessary packages
```bash
pip install pyautogen
pip install confluent-kafka
```
## 🛠️ Video-LLava Requirements and Installation
* Python >= 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/Video-LLaVA
cd Video-LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```
## Initialization
First, run the docker-compose file to get the kafka broker up and running. In the root directory, run the following command:
```bash
docker-compose up -d
```
Wait a minute or so to ensure enough time for the broker to set up

Run the following command to initalize the necessary Kafka topics for this demo.
```bash
python Kafka.py
```

Once the set up is done, run the following file to run autogen + video_llava pipeline

```bash
python app.py
```

## ✏️ Citation

```BibTeX
@article{lin2023video,
  title={Video-LLaVA: Learning United Visual Representation by Alignment Before Projection},
  author={Lin, Bin and Zhu, Bin and Ye, Yang and Ning, Munan and Jin, Peng and Yuan, Li},
  journal={arXiv preprint arXiv:2311.10122},
  year={2023}
}
```

```BibTeX
@article{zhu2023languagebind,
  title={LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment},
  author={Zhu, Bin and Lin, Bin and Ning, Munan and Yan, Yang and Cui, Jiaxi and Wang, HongFa and Pang, Yatian and Jiang, Wenhao and Zhang, Junwu and Li, Zongwei and others},
  journal={arXiv preprint arXiv:2310.01852},
  year={2023}
}
```

<!---->
## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/Video-LLaVA&type=Date)](https://star-history.com/#PKU-YuanGroup/Video-LLaVA&Date)

## 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/Video-LLaVA/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/Video-LLaVA" />
</a>


