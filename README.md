<div align="center">
<h1> Graph-DETR4D </h1>
<h3>Graph-DETR4D: Spatio-Temporal Graph Modeling for Multi-View 3D Object Detection</h3>
<br>Zehui Chen, Zheng Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinhong Jiang, Feng Zhao. 
<br>
<center>
<img src='figs/framework.png'>
</center>
</div>

## Usage

### Train

```
./tools/dist_train.sh projects/configs/graphdetr4d/graphdetr3d_res50_gridmask_cbgs_fullset_1x.py 8
```

### Infer

```
./tools/dist_test.sh projects/configs/graphdetr3d/graphdetr3d_res50_gridmask_cbgs_fullset_1x.py work_dirs/graphdetr3d_res50_gridmask_cbgs_fullset_1x/epoch_12.pth 8 --eval bbox
```


## Performance

### Nuscenes Validation Set
| Model | mAP | NDS |
| -|-|-|
| DETR3D | 28.9 |  34.2  |
| Graph-DETR3D | 32.2 | 38.1 |
| Graph-DETR4D | 34.2 | 44.5 |


### Nuscenes Test Leaderboard
| Model | mAP | NDS |
| -|-|-|
| DETR3D | 41.2 |  47.9  |
| Graph-DETR3D | 42.5 | 49.5 |
| Graph-DETR4D | 54.1 | 62.0 |

## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{chen2023graph,
  title={Graph-DETR4D: Spatio-Temporal Graph Modeling for Multi-View 3D Object Detection},
  author={Chen, Zehui and Chen, Zheng and Li, Zhenyu and Zhang, Shiquan and Fang, Liangji and Jiang, Qinhong and Zhao, Feng},
  journal={},
  year={2023}
}
```