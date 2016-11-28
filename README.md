# harpreif
![jigsaw](/images/jigsaw.png)
Visual Representation Learning by solving Jigsaw puzzles using Deep Reinforcement Learning
![jigsaw]

## Contents
 - [Dataset](#dataset)
 - [Input Construction](#input-construction)
 - [Deep Network](#deep-q-network)
 - [Experimental Results](#experimental-results)
 
## Dataset
![dataset](/images/caltech-256.png)

We take 240 objects and randomly choose 80 images from each of them. Then divide it into (50/10/20) for Training/Validation/Testing respectively. Then for testing for transfer learning, we take 30 images from the rest 16 object categories, and use that for transfer testing.

## Input Construction
For input construction, a windowed HOG gradient (across 8 directions) is calculated for the image and then subsequently discretized, which gives us a state representation, as shown below:

![input](/images/input.png)

## Deep Q Network 
The Deep Q network is used for evaluation function for Reinforcement Learning. The network is shown below:

![dqn](/images/dqn.png)

## Experimental Results

### Test Images
The T-Sne plot for the image features (penultimate layer activation - FC3 layer) for the test images are plot across iterations. The results shows that RL agent learns to generate cluster to improve Learning. 

![tsne-test](/images/test_im.png)

#### 20 neighbors
![plot-test-20](/images/plot_20nb_test.png)
#### 100 neighbors
![plot-test-100](/images/plot_100nb_test.png)

### Transfer Learning Test Images
The T-Sne plot for the image features (penultimate layer activation - FC3 layer) for the transfer test images are plot across iterations. The results shows that RL agent learns to generate cluster to improve Learning. The images were not used for training, and thus this shows transfer learning.

![tsne-tftest](/images/transfer_im.png)

#### 20 neighbors
![plot-tftest-20](/images/plot_20nb_transfer.png)
 
