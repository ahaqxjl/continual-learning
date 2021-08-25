# Continual Learning
This is a PyTorch implementation of the continual learning experiments described in the following papers:
* Three scenarios for continual learning ([link](https://arxiv.org/abs/1904.07734))
* Generative replay with feedback connections as a general strategy 
for continual learning ([link](https://arxiv.org/abs/1809.10635))


## Requirements
The current version of the code has been tested with:
* `pytorch 1.1.0`
* `torchvision 0.2.2`


## Running the experiments
Individual experiments can be run with `main.py`. Main options are:
- `--experiment`: which task protocol? (`splitMNIST`|`permMNIST`)
- `--scenario`: according to which scenario? (`task`|`domain`|`class`)
- `--tasks`: how many tasks?

To run specific methods, you can use the following:
- Context-dependent-Gating (XdG): `./main.py --xdg=0.8`
- Elastic Weight Consolidation (EWC): `./main.py --ewc --lambda=5000`
- Online EWC:  `./main.py --ewc --online --lambda=5000 --gamma=1`
- Synaptic Intelligence (SI): `./main.py --si --c=0.1`
- Learning without Forgetting (LwF): `./main.py --replay=current --distill`
- Generative Replay (GR): `./main.py --replay=generative`
- GR with distillation: `./main.py --replay=generative --distill`
- Replay-trough-Feedback (RtF): `./main.py --replay=generative --distill --feedback`
- Experience Replay (ER): `./main.py --replay=exemplars --budget=2000`
- Averaged Gradient Episodic Memory (A-GEM): `./main.py --replay=exemplars --agem --budget=2000`
- iCaRL: `./main.py --icarl --budget=2000`

To run the two baselines (see the papers for details):
- None: `./main.py`
- Offline: `./main.py --replay=offline`

For information on further options: `./main.py -h`.

The code in this repository only supports MNIST-based experiments. An extension to more challenging problems (e.g., with
natural images as inputs) can be found here: <https://github.com/GMvandeVen/brain-inspired-replay>.

Another extension, with several additional class-incremental learing methods
(BI-R, CWR, AR1, SLDA & Generative Classifier), can be found here:
<https://github.com/GMvandeVen/class-incremental-learning>.

## Running comparisons from the papers
#### "Three CL scenarios"-paper
[This paper](https://arxiv.org/abs/1904.07734) describes three scenarios for continual learning (Task-IL, Domain-IL &
Class-IL) and provides an extensive comparion of recently proposed continual learning methods. It uses the permuted and
split MNIST task protocols, with both performed according to all three scenarios.

A comparison of all methods included in this paper can be run with `compare_all.py` (this script includes extra
methods and reports additional metrics compared to the paper). The comparison in Appendix B can be run with
`compare_taskID.py`, and Figure C.1 can be recreated with `compare_replay.py`.

#### "Replay-through-Feedback"-paper
The three continual learning scenarios were actually first identified in [this paper](https://arxiv.org/abs/1809.10635),
after which this paper introduces the Replay-through-Feedback framework as a more efficent implementation of generative
replay. 

A comparison of all methods included in this paper can be run with
`compare_time.py`. This includes a comparison of the time these methods take to train (Figures 4 and 5).

Note that the results reported in this paper were obtained with
[this earlier version](https://github.com/GMvandeVen/continual-learning/tree/9c0ca78f43c29594b376ca59516031fcdaa5d7ba)
of the code. 


## On-the-fly plots during training
With this code it is possible to track progress during training with on-the-fly plots. This feature requires `visdom`, 
which can be installed as follows:
```bash
pip install visdom
```
Before running the experiments, the visdom server should be started from the command line:
```bash
python -m visdom.server
```
The visdom server is now alive and can be accessed at `http://localhost:8097` in your browser (the plots will appear
there). The flag `--visdom` should then be added when calling `./main.py` to run the experiments with on-the-fly plots.

For more information on `visdom` see <https://github.com/facebookresearch/visdom>.


### Citation
Please consider citing our papers if you use this code in your research:
```
@article{vandeven2019three,
  title={Three scenarios for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1904.07734},
  year={2019}
}

@article{vandeven2018generative,
  title={Generative replay with feedback connections as a general strategy for continual learning},
  author={van de Ven, Gido M and Tolias, Andreas S},
  journal={arXiv preprint arXiv:1809.10635},
  year={2018}
}
```


### Acknowledgments
The research projects from which this code originated have been supported by an IBRO-ISN Research Fellowship, by the 
Lifelong Learning Machines (L2M) program of the Defence Advanced Research Projects Agency (DARPA) via contract number 
HR0011-18-2-0025 and by the Intelligence Advanced Research Projects Activity (IARPA) via Department of 
Interior/Interior Business Center (DoI/IBC) contract number D16PC00003. Disclaimer: views and conclusions 
contained herein are those of the authors and should not be interpreted as necessarily representing the official
policies or endorsements, either expressed or implied, of DARPA, IARPA, DoI/IBC, or the U.S. Government.


## 运行代码注意事项

conda环境

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

如果下载太慢，就用下面的命令

conda create -n testIL --clone other_env_with_pytorch_1.7

conda install pandas

conda install -c conda-forge visdom



坑1. torchvision安装不上

需要用conda install pytorch=1.1.0 torchvision==0.2.2 -c pytorch安装



坑2. visdom用conda装不上

需要用conda install -c conda-forge visdom



坑3. RuntimeError: cublas runtime error : the GPU program failed to execute at /opt/conda/conda-bld/pytorch_1556653114079/work/aten/src/THC/THCBlas.cu:259

可能是cuda版本不对，安装cuda11对应的包，于是换成了conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch



坑4. 坑3解决后遇到symbol free_gemm_select version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference

需要用pip安装，不能用conda安装



坑5. pip安装pytorch太慢

看缘分，有玄学。可以用其他已经有pytorch的环境clone
