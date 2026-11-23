"""背景模型。采用 MOG2。KNN 只在备用条件成立时替换这一处。"""


def create_background(kind="mog2"):
    import cv2

    if kind == "mog2":
        return cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
    if kind == "knn":
        return cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400.0, detectShadows=False
        )
    raise ValueError(kind)
