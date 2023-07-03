## PyTorch-GAN
<strong>Lab Objective:</strong><br>
In this assignment, you will need to implement 3 kinds of GAN with EMNIST dataset.<br>

<strong>Requirements:</strong><br>
1. Make your own dataset to refer to the EMIST dataset. If you can't do it, please refer to the example or use the program in the example to complete the data loader. If you still can't, you can use the MNIST dataset for the assignment, but the score will be very low (60-70).
2. Plot the generator and discriminator training loss during training. And in each GAN you should output a result image with 8x8 generate images or more.
3. Try reading only uppercase "ABDEFGHNQRT" into the data loader as training data (only requires to implement in one of the GANs to demonstrate your ability to customize the data loader)
4. Compare performance changes due to different parameters and model structures and write them into reports.
5. Set “torch.manual_seed(42)” and “torch.backends.cudnn.deterministic = True” in your code for model’s training reproducibility.

## Conditional GAN
#### Example
<p align="left">
    <img src="cgan/cgan_pred_opti.gif" width="360"\>
</p>

#### Training Loss
<p align="left">
    <img src="cgan/cgan_training_loss.jpg" width="360"\>
</p>

## Deep Convolutional GAN
#### Example
<p align="left">
    <img src="dcgan/dcgan_pred_opti.gif" width="360"\>
</p>

#### Training Loss
<p align="left">
    <img src="dcgan/dcgan_training_loss.jpg" width="360"\>
</p>

## Wasserstein GAN GP
#### Example
<p align="left">
    <img src="wgan_gp/wgan_gp_pred_opti.gif" width="360"\>
</p>

#### Training Loss
<p align="left">
    <img src="wgan_gp/wgan_gp_training_loss.jpg" width="360"\>
</p>