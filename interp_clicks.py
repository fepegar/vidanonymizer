from pathlib import Path

import cv2
import click
import numpy as np
import pandas as pd
from scipy import interpolate


extrapolate = False
show_time = True
time_delay = 0.25


def get_duration_fps(path):
    video = cv2.VideoCapture(str(path))
    _, image = video.read()
    shape = image.shape
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video.release()
    return duration, fps, shape

def get_interpolated_df(csv_path, video_duration, frame_duration, shape, scale=False):
    df = pd.read_csv(csv_path, index_col=0)
    t1 = df.ms.values / 1000  # in seconds
    t2 = np.arange(0, video_duration + 1, step=frame_duration)  # in seconds
    x1 = np.stack((df.x.values, df.y.values, df.diameter.values))
    fill_value = 'extrapolate' if extrapolate else -1
    bounds_error = None if extrapolate else False
    f = interpolate.interp1d(t1, x1, fill_value=fill_value, kind='linear', bounds_error=bounds_error)
    x2 = f(t2)
    if scale:  # for small view
        x2 /= 2.4
    x, y, diameter = x2
    radii = (diameter / 2).astype(np.int32)
    h, w = shape[:2]
    x = (x * w).astype(np.int32)
    y = (y * h).astype(np.int32)
    return t2, x, y, radii


def blur_with_mask(image, face_mask, sigma=None):
    from me import get_bounding_box, get_subimage, replace_subimage
    if sigma is None:
        sigma = image.shape[0] // 8 + 1
    bounding_box = get_bounding_box(face_mask)
    if bounding_box is None:
        return
    face = get_subimage(image, bounding_box)
    # blurred = cv2.GaussianBlur(face, (sigma, sigma), 0)
    blurred = cv2.blur(face, (sigma, sigma), 0)
    replace_subimage(image, blurred, bounding_box, face_mask)


def blur(image, time, interp_time, x, y, radii):
    index = np.argmin(np.abs(time - interp_time))
    point = x[index], y[index]
    radius = radii[index]
    color = 255, 255, 255
    thickness = -1
    face_mask = np.zeros(image.shape[:2])
    cv2.circle(
        face_mask,
        point,
        radius,
        color,
        thickness,
    )
    blur_with_mask(image, face_mask, sigma=radius)


@click.command()
@click.argument('csv-large-path', type=click.Path(exists=True))
@click.argument('csv-small-path', type=click.Path(exists=True))
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--show/--hide', '-s', default=True)
def main(csv_large_path, csv_small_path, input_path, output_path, show):
    input_path = Path(input_path)
    output_path = Path(output_path)

    video_duration, frame_rate, shape = get_duration_fps(input_path)
    frame_duration = 1 / frame_rate

    t2_large, x_large, y_large, radii_large = get_interpolated_df(
        csv_large_path,
        video_duration,
        frame_duration,
        shape,
    )

    t2_small, x_small, y_small, radii_small = get_interpolated_df(
        csv_small_path,
        video_duration,
        frame_duration,
        shape,
        scale=True,
    )

    frame_idx = 0
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    size = shape[1::-1]
    out = cv2.VideoWriter(str(output_path), fourcc, 25, size)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        time = frame_idx * frame_duration
        if show_time:
            print(f'{time:.2f} / {video_duration:.2f}')

        time += time_delay  # this is to avoid the delay of my clicks
        blur(image, time, t2_large, x_large, y_large, radii_large)
        blur(image, time, t2_small, x_small, y_small, radii_small)
        image[:70] = 128  ##################################################
        if show:
            cv2.imshow("image", image)
        out.write(image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_idx += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
