# Backdoor Attacks
> A Pytroch Implementation of Some Backdoor Attack Algorithms, Including BadNets, SIG, FIBA, FTrojan ...

## Usage

run on `MNIST` dataset
```
python main.py --dataset mnist --epochs 10
```

output:
```
Epoch: 0         loss=0.13787
Epoch: 1         loss=0.03434
Epoch: 2         loss=0.01981
Epoch: 3         loss=0.01124
Epoch: 4         loss=0.00698
Epoch: 5         loss=0.00316
Epoch: 6         loss=0.00167
Epoch: 7         loss=0.00091
Epoch: 8         loss=0.00081
Epoch: 9         loss=0.00064
Original Acc@1: 99.32000
Poisoned Acc@1: 99.99000
```

---

run on `CIFAR-10` dataset
```
python main.py --dataset cifar-10
```

output:
```
Epoch: 0         loss=1.54291
Epoch: 1         loss=1.03342
Epoch: 2         loss=0.83719
Epoch: 3         loss=0.69116
Epoch: 4         loss=0.58158
Epoch: 5         loss=0.47433
Epoch: 6         loss=0.38273
Epoch: 7         loss=0.31041
Epoch: 8         loss=0.25565
...
Epoch: 98        loss=0.00003
Epoch: 99        loss=0.00005
Original Acc@1: 73.88000
Poisoned Acc@1: 95.78000
```

## Paper And Code
**BadNets (2017)**
- `paper`: [Badnets: Identifying vulnerabilities in the machine learning model supply chain](https://arxiv.org/abs/1708.06733)
- `code`: [BadNets](https://github.com/Kooscii/BadNets)

**SIG (2019, ICIP)**
- `paper`: [A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning](https://arxiv.org/abs/1902.11237)

**FIBA (2022, CVPR)**
- `paper`: [FIBA: Frequency-Injection Based Backdoor Attack in Medical Image Analysis](https://arxiv.org/abs/2112.01148)
- `code`: [FIBA](https://github.com/HazardFY/FIBA)

**FTrojan (2022, ECCV)**
- `paper`: [An Invisible Black-Box Backdoor Attack Through Frequency Domain](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_23)
- `code`: [FTrojan](https://github.com/SoftWiser-group/FTrojan)

**ISSBA (ICCV 2021)**
> In this repository, we did not implement ISSBA's trigger generator due to its complexity. Please refer to its official implementation.

- `paper`: [Invisible Backdoor Attack with Sample-Specific Triggers
](http://openaccess.thecvf.com/content/ICCV2021/html/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.html)
- `code`: [ISSBA](https://github.com/yuezunli/ISSBA)

**LCBA (2019)**
- `paper`: [Label-Consistent Backdoor Attacks](https://arxiv.org/abs/1912.02771)
- `code`: [LCBA](https://github.com/MadryLab/label-consistent-backdoor-code)

## LICENSE
This project is under the MIT license. See [LICENSE](LICENSE) for details.