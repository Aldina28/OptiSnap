import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import linalg
from PIL import Image
import skimage
from skimage import io, img_as_float, img_as_uint
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import sys


def svd_compress_gs(img, k):
    U, singular_vals, V = linalg.svd(img)
    rank = len(singular_vals)
    print("Image rank {}".format(rank))
    if k > rank:
        print("k is larger than rank of image {}".format(rank))
        return img
    # Take columns less than k from U
    U_p = U[:, :k]
    # Take rows less than k from V
    V_p = V[:k, :]
    # Build the new S matrix with top k diagonal elements
    S_p = np.zeros((k, k), img.dtype)
    for i in range(k):
        S_p[i][i] = singular_vals[i]
    print("U_p shape {}, S_p shape {}, V_p shape {}".format(U_p.shape, S_p.shape, V_p.shape))
    compressed = np.dot(np.dot(U_p, S_p), V_p)
    ss = ssim(img, compressed, data_range=1.0, dynamic_range=compressed.max() - compressed.min())
    print(f"Structural similarity: {ss}")
    return compressed

def svd_compress_rgb(img, k_r, k_g, k_b):
    # Split into separate channels
    comp_r = svd_compress_gs(img[:, :, 0], k_r)
    comp_g = svd_compress_gs(img[:, :, 1], k_g)
    comp_b = svd_compress_gs(img[:, :, 2], k_b)
    new_img = np.zeros(img.shape, img.dtype)
    nrows = img.shape[0]
    ncols = img.shape[1]
    nchans = img.shape[2]
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchans):
                val = 0
                if c == 0:
                    val = comp_r[i][j]
                elif c == 1:
                    val = comp_g[i][j]
                else:
                    val = comp_b[i][j]
                # Float64 values must be between -1.0 and 1.0
                if val < -1.0:
                    val = -1.0
                elif val > 1.0:
                    val = 1.0
                new_img[i][j][c] = val
    return new_img

def compress_ratio(orig_img, k):
    m = float(orig_img.shape[0])
    n = float(orig_img.shape[1])
    comp_bytes = 0
    if len(orig_img.shape) > 2:
        comp_bytes += k[0] * (m + n + 1)
        comp_bytes += k[1] * (m + n + 1)
        comp_bytes += k[2] * (m + n + 1)
        return comp_bytes / (3 * m * n)
    else:
        comp_bytes = k[0] * (m + n + 1)
        return comp_bytes / (m * n)
    


def main():
    parser = argparse.ArgumentParser(description='Image compression with SVD or SSIM')
    parser.add_argument('-c', dest='compress', nargs='?', help='compress image using SVD')
    parser.add_argument('-k', dest='k', nargs='*', help='compression factor k (default 5)')
    parser.add_argument('-s', dest='ssim', nargs=2, help='calculate SSIM between 2 images')
    parser.add_argument('-r', dest='size', type=int, default=100, help='image resize percentage (default 100)')
    parser.add_argument('-f', dest='fname', nargs='?', help='save compressed image to file')
    args = parser.parse_args()
    if args.k:
        args.k = [int(x) for x in args.k]

    if args.ssim:
        img1 = img_as_float(io.imread(args.ssim[0]))
        img2 = img_as_float(io.imread(args.ssim[1]))
        ss = ssim(img1, img2, win_size=3, data_range=img1.max() - img1.min())
        print("Structural similarity: {}".format(ss))
        
  
    elif args.compress:
        img = img_as_float(io.imread(args.compress))  # Convert to floating point
        print("Original image dimensions {0}".format(img.shape))
        if args.size < 100:
            img = resize(img, (int(img.shape[0] * args.size / 100), int(img.shape[1] * args.size / 100), img.shape[2]), anti_aliasing=True)
        compressed_svd = None
       

        if args.k:
            if len(img.shape) > 2:
                if len(args.k) != img.shape[2]:
                    print("Provide the correct number of k values ({})".format(img.shape[2]))
                    return
                compressed_svd = svd_compress_rgb(img, args.k[0], args.k[1], args.k[2])
                compressed_svd = (compressed_svd * 255).astype(np.uint8)
            else:
                compressed_svd = svd_compress_gs(img, args.k[0])
            print("SVD Compression ratio: {}".format(compress_ratio(img, args.k)))

        if args.fname:
            io.imsave(args.fname, compressed_svd)

        if compressed_svd is not None:
            io.imshow(compressed_svd)
            io.show()
      
    elif args.size < 100:
        print("Resizing image to {0}%".format(args.size))
        img = resize(img, (int(img.shape[0] * args.size / 100), int(img.shape[1] * args.size / 100), img.shape[2]), anti_aliasing=True)
        plt.figure(figsize=(10, 3.6))
        plt.imshow(img)
        plt.show()
    else:
        parser.print_help()

if __name__ == '__main__':
    sys.stdout.flush()
    main()
