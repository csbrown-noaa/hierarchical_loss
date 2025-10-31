import numpy as np
import cv2
from PIL import Image
import torch

def rescale_boxes(
    pred_boxes: np.ndarray | torch.Tensor,
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> np.ndarray | torch.Tensor:
    """Rescales predicted boxes from model input shape to original image shape.

    This function works for both NumPy arrays and PyTorch tensors.

    Parameters
    ----------
    pred_boxes : np.ndrray | torch.Tensor
        An array or tensor of shape (..., 4) containing boxes in
        [x1, y1, x2, y2] format.
    from_shape : tuple[int, int]
        The original (height, width) of the model input, e.g., (640, 640).
    to_shape : Tuple[int, int]
        The target (height, width) of the original image.

    Returns
    -------
    np.ndarray | torch.Tensor
        The rescaled boxes, in the same type as `pred_boxes`.

    Examples
    --------
    >>> boxes_np = np.array([[10, 10, 60, 60]], dtype=np.float32)
    >>> rescale_boxes(boxes_np, from_shape=(100, 100), to_shape=(200, 400))
    array([[ 40.,  20., 240., 120.]], dtype=float32)
    >>> boxes_torch = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
    >>> rescale_boxes(boxes_torch, from_shape=(100, 100), to_shape=(200, 400))
    tensor([[ 40.,  20., 240., 120.]])
    """
    gain_w = to_shape[1] / from_shape[1]
    gain_h = to_shape[0] / from_shape[0]

    if hasattr(pred_boxes, "new_tensor"):
        gain = pred_boxes.new_tensor([gain_w, gain_h, gain_w, gain_h])
    else:
        gain = np.array([gain_w, gain_h, gain_w, gain_h], dtype=pred_boxes.dtype)

    return pred_boxes * gain
    

def draw_boxes_on_image(
    pil_img: Image.Image,
    boxes: np.ndarray | torch.Tensor,
    labels: list[str] | None = None,
    scores: list[float] | None = None,
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    scaled: tuple[int, int] = (640, 640),
) -> Image.Image:
    """Draws bounding boxes with optional labels and scores onto an image.

    This function is vectorized and handles scaling of boxes internally.

    Parameters
    ----------
    pil_img : Image.Image
        The original image in PIL format.
    boxes : np.ndarray | torch.Tensor
        An array or tensor of shape (R, 4) containing R boxes in
        [x1, y1, x2, y2] format.
    labels : list[str], optional
        A list of class names for each box.
    scores : list[float], optional
        A list of confidence scores for each box.
    box_color : tuple[int, int, int], optional
        The (B, G, R) color for the bounding boxes, by default (0, 255, 0).
    text_color : tuple[int, int, int], optional
        The (B, G, R) color for the text labels, by default (255, 255, 255).
    scaled : tuple[int, int], optional
        The (height, width) shape the boxes are scaled *from* (i.e., the
        model input shape), by default (640, 640).

    Returns
    -------
    Image.Image
        A new PIL Image with the annotations drawn on it.
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    img_area = pil_img.size[::-1][0] * pil_img.size[::-1][1]
    expected_area = 1080*1920
    area_scale_factor = 4 * img_area / expected_area 

    rescaled_boxes = rescale_boxes(boxes, scaled, pil_img.size[::-1])

    for i, box in enumerate(rescaled_boxes):

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)

        label_text = ''
        if labels is not None:
            label_text += labels[i]
        if scores is not None:
            label_text += f' {scores[i]:.2f}' if label_text else f'{scores[i]:.2f}'

        if label_text:
            cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, area_scale_factor, text_color, 2)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

