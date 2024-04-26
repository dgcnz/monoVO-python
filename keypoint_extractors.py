from roma import roma_outdoor, roma_indoor
from PIL import Image
import cv2
import numpy as np


class KeypointExtractor(object):
    def extract(
        self, im_ref: Image.Image, im_cur: Image.Image, kp_ref: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract keypoints from two images"""
        raise NotImplementedError


class SIFTKeypointExtractor(KeypointExtractor):
    def __init__(self, dr: float = 0.75):
        """SIFT keypoint extractor
        param dr: distance ratio for Lowe's ratio test
        """
        self.sift = cv2.SIFT_create()
        self.dr = dr

    def extract(
        self, im_ref: Image.Image, im_cur: Image.Image, kp_ref: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # ignoring kp_ref
        im_ref = cv2.cvtColor(np.array(im_ref), cv2.COLOR_RGB2BGR)
        im_cur = cv2.cvtColor(np.array(im_cur), cv2.COLOR_RGB2BGR)

        kp_ref, des_ref = self.sift.detectAndCompute(im_ref, None)
        kp_cur, des_cur = self.sift.detectAndCompute(im_cur, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ref, des_cur, k=2)
        good = [m for m, n in matches if m.distance < self.dr * n.distance]
        kp_ref = np.array([kp_ref[m.queryIdx].pt for m in good])
        kp_cur = np.array([kp_cur[m.trainIdx].pt for m in good])
        return kp_ref, kp_cur


class RomaKeypointExtractor(KeypointExtractor):
    def __init__(
        self, device: str = "cuda", model: str = "outdoor", sample_size: int = 5000
    ):
        if model == "outdoor":
            self.roma = roma_outdoor(device=device)
        else:
            self.roma = roma_indoor(device=device)
        self.device = device
        self.sample_size = sample_size

    def extract(
        self, im_ref: np.ndarray, im_cur: np.ndarray, kp_ref: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        im_ref = Image.fromarray(cv2.cvtColor(im_ref, cv2.COLOR_BGR2RGB))
        im_cur = Image.fromarray(cv2.cvtColor(im_cur, cv2.COLOR_BGR2RGB))
        dense_matches, dense_certainty = self.roma.match(
            im_ref, im_cur, device=self.device
        )
        sparse_matches, sparse_certainty = self.roma.sample(
            dense_matches, dense_certainty, self.sample_size
        )
        kp_ref, kp_cur = self.roma.to_pixel_coordinates(
            sparse_matches, *im_ref.size[::-1], *im_cur.size[::-1]
        )
        kp_ref, kp_cur = kp_ref.cpu().numpy(), kp_cur.cpu().numpy()
        return kp_ref, kp_cur

class LKKeypointExtractor(KeypointExtractor):
    def __init__(self):
        self.lk_params = dict(winSize  = (21, 21), 
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def extract(
        self, im_ref: Image.Image, im_cur: Image.Image, kp_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert kp_ref is not None, "kp_ref must be provided"
        kp2, st, err = cv2.calcOpticalFlowPyrLK(im_ref, im_cur, kp_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        kp1 = kp_ref[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2