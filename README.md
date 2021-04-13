# ChessApp
Deep Learning for computer vision project. Android application for chess board detection and pieces prediction using a neural network for classification trained on home-made dataset and tested on different techniques. The application also creates a virtual representation of the board allowing the user to visualize the classification results. <br>
<img src="/img/Demo.gif" width="30%" height="30%"/>

<h2>Datasets</h2>
  For the dataset we created our own by collenting chess board images, crop each box and label them.
  We have a small dataset made of 12.500 images and an augmented one made by 110.600 images.<br>
  All the datasets can be found <a href="https://drive.google.com/drive/folders/1DSgQq6am82dlhBs5cbA85SECThUErqKb?usp=sharing">here</a>.
  
<h2>Models</h2>
  We generated up to 13 models using different approaches and training techniques.<br>
  <h4>13 classes approach</h4>
  <img src="/img/13classes.PNG" width="70%" height="70%"/>
  <h4>2 + 7 classes approach</h4>
  2 classes:
  <img src="/img/2classes.PNG" width="70%" height="70%"/>
  7 classes:
  <img src="/img/7classes.PNG" width="70%" height="70%"/>
  
  All the checkpoints can be found <a href="https://drive.google.com/drive/folders/1ra-H5OGlfVveFzrJjEBJtBPslbJNzMaF?usp=sharing">here</a>.
 




