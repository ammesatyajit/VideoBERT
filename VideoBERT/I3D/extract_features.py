import numpy as np
import cv2
import os.path
import tensorflow_hub as hub
import tensorflow as tf
import torch, torchvision
from torchvision.utils import save_image


batch_count = 20
clip_frame_count = 15
im_size = 224
model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-600/1").signatures['default']

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

def process_batch(device_name, batch, batch_id, save_dir):
    print('processing batch...')

    # normalize values to (-1,1)
    normalized_batch = np.interp(batch, (batch.min(), batch.max()), (-1, +1))

    # reshape to N x clip_frame_count x im_size x im_size x 3
    N = np.size(normalized_batch, axis=0) // clip_frame_count
    clips_batch = normalized_batch.reshape((N, clip_frame_count, im_size, im_size, 3))

    with tf.device(device_name):
        features = model(tf.constant(clips_batch, dtype=tf.float32))['default']

    np.save(os.path.join(save_dir, 'features-{id:04}'.format(id=batch_id)), features)


def extract_features(device_name, path, features_save_dir, imgs_save_dir):
    batch_id = 1
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
    interval = fps / 10
    print(fps, "fps", "\ninterval:", interval)

    batch_total_frames = batch_count * clip_frame_count
    batch = np.zeros((batch_total_frames, im_size, im_size, 3))

    i = 0
    counter = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            # reach to the end of the video file
            break

        if int(counter % interval) == 0:
            frame = cv2.resize(frame, (im_size, im_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch[i] = frame

            i += 1
            if (counter / interval) % 15 == 8:
                img_id = (counter // interval - 8) // 15
                frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
                frame = frame / 255
                save_image(frame,
                           os.path.join(imgs_save_dir, 'img-{id:04}-{row:02}.jpg'.format(id=batch_id, row=img_id % 20)))

            if i == batch_total_frames:
                i = 0
                process_batch(device_name, batch, batch_id, features_save_dir)
                batch_id += 1
        counter += 1

    if i > 0:
        # nclips = i // clip_frame_count
        final_batch = batch[:i]

        if i % clip_frame_count != 0:
            # fractional clip is left over
            nframes_clip_extend = clip_frame_count - (i % clip_frame_count)

            last_frame = batch[i - 1]

            for d, frame in enumerate(np.repeat([last_frame], nframes_clip_extend, axis=0)):
                batch[i + d - 1] = frame

            final_batch = batch[:i + nframes_clip_extend]

        process_batch(device_name, final_batch, batch_id, features_save_dir)

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
