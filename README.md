# Action Knowledge for Video Captioning with Graph Neural Networks
## Description
The official implementation of paper [**Action Knowledge for Video Captioning with Graph Neural Networks**](https://www.sciencedirect.com/science/article/pii/S1319157823000666)

Our approach for video captioning introduces a new technique that leverages action as edge features within a graph neural network (GNN), with objects represented as nodes. By integrating object-action relationships into the GNN, our method enhances the visual representation and generates more precise captions. Furthermore, we enhance the performance by combining the proposed edge representation with a node representation based on grids. By overlapping the grids, the model captures more comprehensive information about the objects, leading to further improvements in performance.

It is demonstrated  in the experiments of MSVD and MSR-VTT that our method significantly outperformed the existing methods.

The illustration of our proposed action-graph model with overlapping grid is shown below:
![alt text](/assets/action-graph.png)



## Prepare the Environment 
Install and create conda environment with the provided `environment.yml` file.
This conda environment was tested with the NVIDIA A6000 and NVIDIA RTX 3090.

The details of each dependency can be found in the environment.yml file.
```
conda env create -f environment.yml
conda activate action_graph_env
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap

Install torch following this page: https://pytorch.org/get-started/locally 
pip install opencv-python
pip install seaborn

```
## Prepare the Dataset

### Dataset Folder Structure
```bash
├── dataset
│   ├── MSVD
│   │   ├── raw # put the 1970 raw videos in here
│   │   ├── captions 
│   │   ├── raw-captions_mapped.pkl # mapping between video id with captions
│   │   ├── train_list_mapping.txt
│   │   ├── val_list_mapping.txt
│   │   ├── test_list_mapping.txt
│   ├── MSRVTT
│   │   ├── raw # put the 10000 raw videos in here
│   │   ├── msrvtt.csv # list of video id in msrvtt dataset
│   │   ├── MSRVTT_data.json # metadata of msrvtt dataset, which includes video url, video id, and caption
```
### MSR-VTT
Raw videos can be downloaded from this [link](https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6).
### MSVD

Raw videos can be downloaded from this [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/).

## Extract the Features
### Feature Extractor Folder Structure
```bash
├── model
│   ├── i3d
├── modules # Copy the files from CLIP4Clip repository: https://github.com/ArrowLuo/CLIP4Clip/tree/master/modules
├── pretrained 
│   ├── [trained CLIP4Clip model].bin # Train your own PyTorch CLIP4Clip model
│   ├── rgb_imagenet.pt # Download I3D model: https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_imagenet.pt
├── utility # Some helper functions to generate the features
```


**Note** 
```
- Please make sure you have copy modules from CLIP4Clip https://github.com/ArrowLuo/CLIP4Clip/tree/master/modules into feature_extractor/modules
- Please make sure you have downloaded rgb_imagenet.pt into feature_extractor/pretrained
- Please change args in each notebook based on requirement e.g,. args.msvd = False for MSR-VTT and args.msvd = True for MSVD
```

## CLIP-based features 
Steps:
1. Train CLIP4Clip based on https://github.com/ArrowLuo/CLIP4Clip and put the best model in the **pretrained** folder
   - For MSR-VTT, we use 6513 clips for training, 497 clips for validation and 2990 clips for testing when training the CLIP4Clip
2. Extract the CLIP-based features by using **clip4clip_theta_2_feature_extraction.ipynb**

## Features of Grid-based Action Graph and Object-based Action Graph
### Grid Based Action Graph
Steps: 
1. Extract grid node by using **grid_node_theta_1_feature_extractor.ipynb**
2. Extract spatial action graph by using **grid_based_spatial_action_graph.ipynb**
3. Extract temporal action graph by using **temporal_similarity_graph.ipynb**
4. Create the grid based action graph: 

   - run **action_spatio_temporal_graph_feature_extractor.ipynb** then ,
   - run **transform-graph-to-geometric.ipynb**

### Object Based Action Graph
Steps: 
1. Extract object node by using **object_node_theta_1_feature_extractor.ipynb**
2. Extract spatial action graph by using **object_based_spatial_action_graph.ipynb**
3. Extract temporal action graph by using **temporal_similarity_graph.ipynb**
4. Create the object based action graph: 
   - run **action_spatio_temporal_graph_feature_extractor.ipynb** then ,
   - run **transform-graph-to-geometric.ipynb**

## Training

1. Download a pretrained BERT model
This is used as word embedding and tokenizer for the captions.
```
mkdir modules/bert-model
cd modules/bert-model/
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../../
```
2. Download a pretrained weight of UniVL
This is used to initialize our caption generator.
```
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```
3. Prepare the CLIP-based features as mentioned in section **[Extract the Features] -> [CLIP-based features]**.
4. Prepare the graph features, i.e., Grid-based Action Graph or Object-based Action Graph, as mentioned in section **[Extract the Features] -> [Features of Grid-based Action Graph and Object-based Action Graph]**.
5. Open a train script (.sh) in folder `scripts`, and change following parameters based on the specs of your machine:
    - **N_GPU** = [Total GPU to use]
    - **N_THREAD** = [Total thread to use]
6. If needed, change also the following parameters according to the location of the data in your machine:
    - **DATA_PATH** = [MSR-VTT JSON file location] or [MSVD dataset location]
    - **CKPT_ROOT** = [Your desired folder for saving the models and results]
    - **INIT_MODEL_PATH** = [UniVL pretrained model location]
    - **FEATURES_PATH** = [CLIP-based features]
    - **DATA_GEOMETRIC_PATH** = [Generated Action Graph feature path (Grid-based action graph or Object-based action graph)]
7. Based on object detection model, change **node_feat_dim** according to the object feature dimension, e.g for Yolo the node_feat_dim is 1024
8. Execute the following scripts to start the training process:

#### Train our proposed method
#### MSVD
```
cd scripts/
./msvd_train_GNN.sh 
```
#### MSRVTT
```
cd scripts/
./msrvtt_train_GNN.sh  
```

## Evaluation
After the training is done, an evaluation process on the test set will be automatically executed using the best checkpoint among all epochs. However, if you want to evaluate a checkpoint at a specific epoch, you can use the provided training shell script by modifying the value of `INIT_MODEL_PATH` to the location of the desired checkpoint, and replacing the `--do_train` to `--do_eval`.

## Our Results

The comparison with the existing methods and also the ablation study of our method can be found in our paper.

### MSVD

| Method  | CLIP Model | BLEU@4 | METEOR | ROUGE-L | CIDEr |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Ours (Action + Object) | ViT-B/32 | 62.56 | 41.53 | 78.62 | 120.64 |
| Ours (Action + Grid) | ViT-B/32 | 62.90 | 41.81 | 78.80 | 119.07 |
| Ours (Action + Grid) | ViT-B/16 | 64.07 | 42.41 | 79.72 | 124.18 |

#### MSR-VTT

| Method  | CLIP Model | BLEU@4 | METEOR | ROUGE-L | CIDEr |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Ours (Action + Object) | ViT-B/32 | 48.31 | 31.35 | 65.34 | 60.00 |
| Ours (Action + Grid) | ViT-B/32 | 49.10 | 31.57 | 65.52 | 61.27 |
| Ours (Action + Grid) | ViT-B/16 | 51.02 | 32.19 | 66.55 | 63.02 |

## Citation
Please cite our paper in your publications if it helps your research:
```
@article{Hendria2023,
  author   = {W. F. Hendria and V. Velda and B. H. H. Putra and F. Adzaka and C. Jeong},
  title        = {Action Knowledge for Video Captioning with Graph Neural Networks},
  journal   = {J. King Saud Univ.-Comput. Inf. Sci.},
  volume   = {35},
  number   = {4},
  pages     = {50-62},
  month     = apr,
  year       = {2023}"
}
```

## Acknowledgements
Our code is developed based on https://github.com/microsoft/UniVL, which is also developed based on https://github.com/huggingface/transformers/tree/v0.4.0 and https://github.com/antoine77340/howto100m .
