# cnn-toys

Convolutional neural networks are fairly simple, and it's easy to apply them. However, it's also easy to lose track of which architectures work well for different applications.

I want to use this repo to play with different applications of CNNs. That way, I gain some intuition for when to use transposed convolutions, upsampling, residual connections, pixel CNNs, leaky ReLUs, etc.

# Contents

 * [colorize](cnn_toys/colorize) - Grayscale -> color predictor. *Current status:* the model sometimes colors skies in correctly, but it's generally pretty terrible.
 * [cyclegan](cnn_toys/cyclegan) - A re-implementation of [CycleGAN](https://github.com/junyanz/CycleGAN). *Current status:* works fairly well.
 * [real_nvp](cnn_toys/real_nvp) - A re-implementation of [real NVP](https://arxiv.org/abs/1605.08803). *Current status:* works on problems that I've tested it on. Generally requires messing with the architecture (usually making it way deeper).
