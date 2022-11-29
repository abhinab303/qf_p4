import numpy as np
import matplotlib.pyplot as plt
# import cv2
import matplotlib.image as mpimg
# import math
import pandas as pd
from tqdm import tqdm
import pdb

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

def get_eigen(x_norm):
    cov_mat = np.cov(x_norm, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    return sorted_eigenvalue, sorted_eigenvectors

def plot_many(img_list, reshape=True, file_name='result/test.png', vars=None):
    plt.figure()
    # for i in range(1,17):
    for i in range(0,8):
        # ax = plt.subplot(4,4,i)
        ax = plt.subplot(2,4,i+1)
        img = img_list[i].reshape(112, 92) if reshape else img_list[i]
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        ax.set_aspect('auto')
        ax.set_title(f"{round(vars[i]*100, 2)}%")
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig(file_name, bbox_inches='tight')


def reconstruct(x_norm, eig_vectors, num_components):
    eigenvector_subset = eig_vectors[:,0:num_components]
    x_reduced = np.dot(eigenvector_subset.transpose(), x_norm.transpose()).transpose()
    reconstructed_x = x_reduced @ eigenvector_subset.T
    return reconstructed_x


if __name__ == "__main__":
    # read images:
    images = read_images()
    x_norm = images - np.mean(images , axis = 0)
    # get eigen values:
    print("generating Eigen vectors... ")
    eigen_val, eigen_vectors = get_eigen(images)
    vars = eigen_val/sum(eigen_val)

    # PCA: Part 1:
    # plot eigen faces:
    plot_many(eigen_vectors.T, file_name="result/eigen_faces_0.png", vars=vars)
    # plot less important eigen faces:
    plot_many(eigen_vectors.T[17:34], file_name="result/eigen_faces_17.png", vars=vars[17:34])
    plot_many(eigen_vectors.T[34:51], file_name="result/eigen_faces_34.png", vars=vars[34:51])
    plot_many(eigen_vectors.T[51:68], file_name="result/eigen_faces_51.png", vars=vars[51:68])
    plot_many(eigen_vectors.T[68:85], file_name="result/eigen_faces_68.png", vars=vars[68:85])
    plot_many(eigen_vectors.T[85:102], file_name="result/eigen_faces_85.png", vars=vars[85:102])
    plot_many(eigen_vectors.T[200:217], file_name="result/eigen_faces_200.png", vars=vars[200:217])
    plot_many(eigen_vectors.T[300:317], file_name="result/eigen_faces_300.png", vars=vars[300:317])

    # PCA: Part 2:
    # reconstruction with increasing number of components:

    # Plot some reconstructed images:
    plt.figure()
    for n in range(50, 850, 50):
        reconstructed_images = reconstruct(x_norm, eigen_vectors, n)
        ax = plt.subplot(4, 4, int(n/50))
        img = reconstructed_images[int(n*2/50)].reshape(112, 92)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        ax.set_aspect('auto')
        ax.set_title(f"n: {n}")
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plt.savefig('result/reconstruct.png', bbox_inches='tight')

    # reconstruction error calculation:
    result_dict = {
        "N": [],
        "MSE": []
    }

    # max_iter = images.shape[1]
    max_iter = 1000
    for n in tqdm(range(1, max_iter)):
        reconstructed_images = reconstruct(x_norm, eigen_vectors, n)
        mse = (np.square(x_norm - reconstructed_images)).mean(axis=None)
        result_dict["N"].append(n)
        result_dict["MSE"].append(mse)
        error_df = pd.DataFrame.from_dict(result_dict)
        csv_file_path = "result/mse_vs_n.csv"
        error_df.to_csv(csv_file_path, index=False)
        if mse < 0.001:
            print("MSE is less than 0.001 at Number of components = ", n)
            break

    # pdb.set_trace()