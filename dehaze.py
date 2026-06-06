import os
import cv2
import glob
import numpy as np
import argparse

def GetDarkChannel(im, wsz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (wsz, wsz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    imsz = np.prod(im.shape[:2])
    darkvec = dark.flatten()
    indices = darkvec.argsort()[imsz-max(imsz//1000, 1):]
    A = np.mean(im.reshape(-1, 3)[indices], axis=0, keepdims=True)
    return A

def TransmissionEstimate(im, A, wsz, omega):
    normI = im / A
    transmission = 1 - omega*GetDarkChannel(normI, wsz)
    return transmission

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(q, et):
    r = 80
    eps = 0.0004
    t = Guidedfilter(q, et, r, eps)

    return t

def Recover(im, t, A, t0 = 0.1):
    t = cv2.max(t, t0)
    res = (im - A)/t[:,:, np.newaxis] + A
    res = res.clip(0., 1.)
    return res

def Dehaze(input, wsz = 15, t0 = 0.3, omega = 0.95):
    I = input / 255.
    q = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY) / 255.
 
    dark = GetDarkChannel(I, wsz)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, wsz, omega)
    tr = TransmissionRefine(q, te) 
    J = Recover(I, tr, A, t0)
    output = (J * 255).astype(np.uint8)
    return output

def main():
    parser = argparse.ArgumentParser(description='Haze Removal')
    parser.add_argument('--input', default="input", type=str, help='Path to the input image')
    parser.add_argument('--output', default="output", type=str, help='Path to save the output image')
    parser.add_argument("--ext", type=str, default='auto', help="Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto")

    args = parser.parse_args()
    if args.input.endswith('/'):
        args.input = args.input[:-1]
        
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*.[jpJP][pnPN]*[gG]')))

    os.makedirs(args.output, exist_ok=True)

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        output_img = Dehaze(input_img)

        if args.ext == 'auto':
            extension = ext[1:]
        else:
            extension = args.ext
        cv2.imwrite(os.path.join(args.output, f'{basename}_hazeremoval.{extension}'), output_img)

if __name__ == "__main__":
    main()
