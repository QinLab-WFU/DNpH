# Deep Neighborhood-preserving Hashing with Quadratic Spherical Mutual Information for Cross-modal Retrieval [Paper](https://ieeexplore.ieee.org/document/10379137)
This paper is accepted for publication with TMM.


## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128


## Citation
@ARTICLE{10379137,   
  author={Qin, Qibing and Huo, Yadong and Huang, Lei and Dai, Jiangyan and Zhang, Huihui and Zhang, Wenfeng},  
  journal={IEEE Transactions on Multimedia},  
  title={Deep Neighborhood-preserving Hashing with Quadratic Spherical Mutual Information for Cross-modal Retrieval},  
  year={2023},  
  volume={},  
  number={},  
  pages={1-14},  
  doi={[10.1109/TMM.2023.3349075](https://ieeexplore.ieee.org/document/10379137)}}  


## Acknowledgements
[QSMI](https://github.com/passalis/qsmi)  
[DCHMT](https://github.com/kalenforn/DCHMT)
