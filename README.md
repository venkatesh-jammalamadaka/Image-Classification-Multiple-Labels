General Multi-label Image Classification with Transformers

**Original Authors:** Jack Lanchantin, Tianlu Wang, Vicente Ordóñez Román, Yanjun Qi  
**Conference:** Computer Vision and Pattern Recognition (CVPR) 2021  
**Paper:** [[arXiv]](https://arxiv.org/abs/2011.14027) [[Poster]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_poster.pdf) [[Slides]](https://github.com/QData/C-Tran/blob/main/supplemental/ctran_slides.pdf)

This repository contains the implementation of C-Tran with additional features and bug fixes.

---

## Requirements

Python 3.7+ is required. Major packages and versions are listed in `requirements.txt`.
```bash
pip install -r requirements.txt

Dataset Used:

wget https://www.cs.virginia.edu/~yq2h/jack/vision/voc.tar.gz
mkdir -p data/
tar -xvf voc.tar.gz -C data/

Train Model:

python main.py --batch_size 16 --lr 0.00001 --optim 'adam' --layers 3 \
               --dataset 'voc' --use_lmt --grad_ac_step 2 --dataroot data/


Training Results:

1. Trained model on VOC2007 dataset:

mAP: 99.4%
CF1 Score: 97.0%
OF1 Score: 97.3%
Training: 10 epochs on Tesla T4 GPU (4-5 hours)

Additional Features / bug fixes:

Added a Visualization Module.
Added visualize_predictions.py for visualizing model predictions with confidence scores.

Usage: 
python visualize_predictions.py --image path/to/image.jpg \
                                --model path/to/model.pt \
                                --threshold 0.5 \
                                --save output.png


2. Bug Fixes and Improvements:

PyTorch 2.x Compatibility:
  1. Fixed deprecated ByteTensor usage → replaced with .bool()
  2. Updated checkpoint loading to handle weights_only parameter
  3. Fixed tensor type conversions in metrics calculations



Dataset Loading:

  1. Fixed VOC dataset loader to properly handle 6-digit zero-padded image IDs
  2. Improved error handling for missing annotation files


Citation:

@article{lanchantin2020general,
  title={General Multi-label Image Classification with Transformers},
  author={Lanchantin, Jack and Wang, Tianlu and Ordonez, Vicente and Qi, Yanjun},
  journal={arXiv preprint arXiv:2011.14027},
  year={2020}
}

License:

MIT License (see LICENSE file)

Acknowledgments:
This implementation is based on the original C-Tran paper by Lanchantin et al. (CVPR 2021).
Original repository: https://github.com/QData/C-Tran