
from colour import delta_E
from colour.models import RGB_to_XYZ, XYZ_to_Lab
from colour.colorimetry import CCS_ILLUMINANTS

def sRGB_to_Lab(srgb_image):
    srgb = srgb_image.astype(np.float32) / 255.0
    linear_rgb = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    xyz = RGB_to_XYZ(linear_rgb, 'sRGB')
    illuminant_D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    lab = XYZ_to_Lab(xyz, illuminant_D65)
    return lab

def canvas_delta(c1, c2):
    lab1 = sRGB_to_Lab(c1)
    lab2 = sRGB_to_Lab(c2)
    delta = delta_E(lab1.reshape(-1, 3), lab2.reshape(-1, 3))
    return np.sum(delta)

def main():
    # Create two blank images (all zeros)
    img1 = np.zeros((1, 1, 3), dtype=np.uint8)
    img2 = np.ones((1, 1, 3), dtype=np.uint8) * 255

    start_time = time.time()
    delta = canvas_delta(img1, img2)
    end_time = time.time()

    print("Delta:", delta)
    print("Time taken: {:.4f} seconds".format(end_time - start_time))

if __name__ == "__main__":
    main()