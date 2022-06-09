# Backdoor Attacks
> A Pytroch Implementation of Some Backdoor Attack Algorithms, Including BadNets ...

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
BadNets (2017)
- `paper`: [Badnets: Identifying vulnerabilities in the machine learning model supply chain](https://arxiv.org/abs/1708.06733)
- `code`: [BadNets](https://github.com/Kooscii/BadNets)

ISSBA (ICCV 2021)
- `paper`: [Invisible Backdoor Attack with Sample-Specific Triggers
](http://openaccess.thecvf.com/content/ICCV2021/html/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.html)
- `code`: [ISSBA](https://github.com/yuezunli/ISSBA)

## LICENSE
This project is under the MIT license. See [LICENSE](LICENSE) for details.