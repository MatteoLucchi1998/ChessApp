# ChessApp
Deep Learning applicated to a computer vision project.<br>
ChessApp is an Android application for chess board detection and pieces recognition, using a neural network for classification trained on home-made dataset and tested with different techniques and approaches.<br>
The application also creates a virtual representation of the board allowing the user to visualize the classification results.<br>
Created in collaboration with [Simone Mattioli](https://github.com/SimoneMattioli98) 
<br>
<img src="/img/Demo.gif" width="40%" height="40%"/>

## Datasets
We created our own dataset by collenting chess board images, crop each box and label them.<br>
We have a small dataset made of the 12.500 original images and a larger one made by 110.600 images.
<br>
All the datasets can be found <a href="https://drive.google.com/drive/folders/1DSgQq6am82dlhBs5cbA85SECThUErqKb?usp=sharing">here</a>.
  
## Models
We generated up to 13 models using different approaches and training techniques.<br>
### 13 classes approach (one model)
<img src="/img/13classes.PNG" width="70%" height="70%"/>
### 2 + 7 classes approach (two models)
2 classes:
<img src="/img/2classes.PNG" width="70%" height="70%"/>
7 classes:
<img src="/img/7classes.PNG" width="70%" height="70%"/>

All the checkpoints can be found <a href="https://drive.google.com/drive/folders/1ra-H5OGlfVveFzrJjEBJtBPslbJNzMaF?usp=sharing">here</a>.
