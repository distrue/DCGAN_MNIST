# DCGAN_MNIST

Deep Convolutional Generative Adverserial Network implementation with Pytorch

### How to run

``` python
import DCGAN_MNIST

try:
    # train
    DCGAN_MNIST.train.run()
except KeyboardInterrupt:
    try:
        # if in colab, download checkpoint in colab
        from google.colab import files
        files.download('checkpoint_generator.pt')
        files.download('checkpoint_discriminator.pt')
    except ImportError:
        pass

# print image on trained neuralnet
DCGAN_MNIST.showimg.printImg()

```
