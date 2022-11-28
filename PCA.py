import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import math

test_img = "att_faces/s1/1.pgm"

from numpy.random.mtrand import noncentral_chisquare
folder_num = range(1, 41)
img_num = range(1, 11)

data_dir = "att_faces"

def read_images():
    images = []
    for f in folder_num:
        for i in img_num:
            img_path = data_dir + f"/s{f}/{i}.pgm"
            # img = cv2.imread(img_path, 0)
            img = mpimg.imread(img_path, 0)
            if type(img) != type(None):
                flat_img = img.flatten()
                images.append(flat_img)

    # to np array
    images = np.asarray(images)
    print("Total images: ", images.shape[0], "dim: ", images.shape[1])
    return images

def get_eigen(images):
    X = images
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    return sorted_eigenvalue, sorted_eigenvectors

def plot_many(img_list, reshape=True, file_name='result/test.png', vars=None):
    plt.figure()
    for i in range(1,17):
        ax = plt.subplot(4,4,i)
        img = img_list[i].reshape(112, 92) if reshape else img_list[i]
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        ax.set_aspect('auto')
        ax.set_title(f"{round(vars[i]*100, 2)}%")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(file_name, bbox_inches='tight')



if __name__ == "__main__":
    # read images:
    images = read_images()
    # get eigen values:
    eigen_val, eigen_vectors = get_eigen(images)
    vars = eigen_val/sum(eigen_val)
    # plot eigen faces:
    plot_many(eigen_vectors.T, file_name="result/eigen_faces.png", vars=vars)
    # get error vs components
