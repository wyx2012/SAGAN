SAGAN, A domain adaptive interpretable substructure-aware graph attention network for DDI prediction. Based on attention mechanism and unsupervised clustering algorithm.

## Requirements  

numpy==1.18.1 \
tqdm==4.42.1 \
pandas==1.0.1 \
rdkit==2009.Q1-1 \
scikit_learn==1.0.2 \
torch==1.11.0 \
torch_geometric==2.0.4 \
torch_scatter==2.0.9

## Step-by-step running:  
### 1. SAGAN
  In SAGAN folder, run the "train.py" file.

  Explanation of parameters
  - --n_iter: number of iterations
  - --fold: {0, 1, 2}
  - --epochs: number of epochs
  - --weight_decay: weight decay
  - --batch_size: batch size
  - --save_model: whether save the model or not, for example, 'python train.py' will not save the model and 'python train.py --save_model' will save the model.
  - --lr: learning rate
### 2. SAGAN-CDAN
- Second, run train.py using 
  In SAGAN-CDAN folder, run the "train.py" file.
  The training record can be found in save/ folder.
# MMD-DTI
