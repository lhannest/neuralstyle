import matplotlib.pyplot as plt

def plot_all(images, rows, cols):
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(image)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
