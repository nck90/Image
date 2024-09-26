import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from imagehash import phash
from PIL import Image
    
def load_and_resize_images(img_bytes1, img_bytes2, width=800):
    img_array1 = np.frombuffer(img_bytes1, np.uint8)
    img_array2 = np.frombuffer(img_bytes2, np.uint8)
    img1 = cv2.imdecode(img_array1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img_array2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded properly.")

    img1 = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2 = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    return img1, img2


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_pixel_similarity(imageA, imageB):
    try:
        if imageA.shape != imageB.shape:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

        diff = cv2.absdiff(imageA, imageB)
        mean_diff = np.mean(diff)

        similarity = (1 - mean_diff / 100) * 100
        return float(max(min(similarity, 100), 0)) 
    except Exception as e:
        print(f"Pixel similarity error: {str(e)}")
        return 0.0 

def calculate_ssim_similarity(imageA, imageB):
    try:
        if imageA.shape != imageB.shape:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

        win_size = min(7, min(imageA.shape[:2]) // 2)  
        score, _ = ssim(imageA, imageB, full=True, win_size=win_size, channel_axis=2)
        similarity = score * 100
        similarity = similarity * 0.85  
        return float(similarity)
    except Exception as e:
        print(f"SSIM similarity error: {str(e)}")
        return 0.0  

def calculate_phash_similarity(imageA, imageB):
    try:
        pil_imageA = Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
        pil_imageB = Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
        hashA = phash(pil_imageA)
        hashB = phash(pil_imageB)
        hash_diff = (hashA - hashB) / len(hashA.hash) ** 2
        similarity = (1 - hash_diff * 1.5) * 100  
        return float(max(min(similarity, 100), 0))
    except Exception as e:
        print(f"pHash similarity error: {str(e)}")
        return 0.0  


def calculate_dct_similarity(imageA, imageB):
    try:
        grayA = convert_to_grayscale(imageA)
        grayB = convert_to_grayscale(imageB)
        if grayA.shape != grayB.shape:
            grayB = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]))

        dctA = cv2.dct(np.float32(grayA))
        dctB = cv2.dct(np.float32(grayB))
        diff = np.mean(np.abs(dctA - dctB))
        similarity = (1 - diff / 15) * 100  # 차이점 확대
        return float(max(min(similarity, 100), 0))
    except Exception as e:
        print(f"DCT similarity error: {str(e)}")
        return 0.0  # 오류 발생 시 0으로 반환

# 히스토그램 유사도 계산 (차이점 확대)
def calculate_histogram_similarity(imageA, imageB):
    try:
        histA = cv2.calcHist([imageA], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        histB = cv2.calcHist([imageB], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
        histA = cv2.normalize(histA, histA).flatten()
        histB = cv2.normalize(histB, histB).flatten()
        similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL) * 100
        # 차이점을 더 크게 반영
        similarity = similarity * 0.8
        return float(max(min(similarity, 100), 0))
    except Exception as e:
        print(f"Histogram similarity error: {str(e)}")
        return 0.0  # 오류 발생 시 0으로 반환

# SIFT 유사도 계산 (차이점 확대)
def calculate_sift_similarity(imageA, imageB):
    try:
        sift = cv2.SIFT_create()
        keypointsA, descriptorsA = sift.detectAndCompute(imageA, None)
        keypointsB, descriptorsB = sift.detectAndCompute(imageB, None)

        if descriptorsA is None or descriptorsB is None:
            return 0.0  # 매칭할 키포인트가 없는 경우 0 반환

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(descriptorsA, descriptorsB)
        similarity = len(matches) / max(len(keypointsA), len(keypointsB)) * 100
        # 차이점을 크게 반영
        similarity = similarity * 0.75
        return float(max(min(similarity, 100), 0))
    except Exception as e:
        print(f"SIFT similarity error: {str(e)}")
        return 0.0  # 오류 발생 시 0으로 반환

# 전체 유사도 계산
def calculate_total_similarity(imageA, imageB):
    pixel_similarity = calculate_pixel_similarity(imageA, imageB)
    ssim_similarity = calculate_ssim_similarity(imageA, imageB)
    phash_similarity = calculate_phash_similarity(imageA, imageB)
    dct_similarity = calculate_dct_similarity(imageA, imageB)
    hist_similarity = calculate_histogram_similarity(imageA, imageB)
    sift_similarity = calculate_sift_similarity(imageA, imageB)

    weights = {
        'pixel': 0.05,  # 픽셀 가중치를 더 낮게
        'ssim': 0.2,
        'phash': 0.25,  # pHash 가중치 증가
        'dct': 0.2,
        'hist': 0.15,
        'sift': 0.15
    }

    total_similarity = (pixel_similarity * weights['pixel'] +
                        ssim_similarity * weights['ssim'] +
                        phash_similarity * weights['phash'] +
                        dct_similarity * weights['dct'] +
                        hist_similarity * weights['hist'] +
                        sift_similarity * weights['sift'])

    return {
        'pixel_similarity': pixel_similarity,
        'ssim_similarity': ssim_similarity,
        'phash_similarity': phash_similarity,
        'dct_similarity': dct_similarity,
        'hist_similarity': hist_similarity,
        'sift_similarity': sift_similarity,
        'total_similarity': total_similarity
    }