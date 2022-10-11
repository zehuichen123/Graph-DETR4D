<div align="center">
<h1> Graph-DETR3D </h1>
<h3>Rethinking Overlapping Regions for Multi-View 3D Object Detection</h3>
<br>Zehui Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinhong Jiang, Feng Zhao. 
<br>
<a href="https://arxiv.org/abs/2204.11582">[ACM MM 2022 Paper]</a>
<center>
<img src='figs/framework.png'>
</center>
</div>

## Usage

### Train

```
./tools/dist_train.sh projects/configs/graphdetr3d/graphdetr3d_res50_gridmask_cbgs_fullset_1x.py 8
```

### Infer

```
./tools/dist_test.sh projects/configs/graphdetr3d/graphdetr3d_res50_gridmask_cbgs_fullset_1x.py work_dirs/graphdetr3d_res50_gridmask_cbgs_fullset_1x/epoch_12.pth 8 --eval bbox
```


## Performance

### Nuscenes Validation Set
| Model | mAP | NDS |
| -|-|-|
| DETR3D | 33.6 |  41.4  |
| Graph-DETR3D | 35.1 | 43.3 |


### Nuscenes Test Leaderboard
| Model | mAP | NDS |
| -|-|-|
| DETR3D | 41.2 |  47.9  |
| Graph-DETR3D | 42.5 | 49.5 |


## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{chen2022graph,
  title={Graph-DETR3D: Rethinking Overlapping Regions for Multi-View 3D Object Detection},
  author={Chen, Zehui and Li, Zhenyu and Zhang, Shiquan and Fang, Liangji and Jiang, Qinhong and Zhao, Feng},
  journal={ACM MM},
  year={2022}
}
```