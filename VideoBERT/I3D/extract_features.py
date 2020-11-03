import numpy as np
import cv2
import os.path
from VideoBERT.I3D.I3DModelWrapper import I3DModel


batch_count = 20
clip_frame_count = 15
im_size = 224
model = I3DModel('VideoBERT/I3D/i3d-checkpoint/rgb_scratch_kin600/model.ckpt')


def return_model():
    out_model = I3DModel('VideoBERT/I3D/i3d-checkpoint/rgb_scratch_kin600/model.ckpt')
    return out_model


def process_batch(batch, batch_id, save_dir):
    print('processing batch...')

    # normalize values to (-1,1)
    normalized_batch = np.interp(batch, (batch.min(), batch.max()), (-1, +1))

    # reshape to N x clip_frame_count x im_size x im_size x 3
    N = np.size(normalized_batch, axis=0) // clip_frame_count
    clips_batch = normalized_batch.reshape((N, clip_frame_count, im_size, im_size, 3))

    features = model.generate_features(clips_batch)

    np.save(os.path.join(save_dir, 'batch-{id:04}'.format(id=batch_id)), features)


def extract_features(path, save_dir):
    batch_id = 1
    cap = cv2.VideoCapture(path)
    interval = int(cap.get(cv2.CAP_PROP_FPS) / 10 + 0.5)

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

        if counter % interval == 0:
            frame = cv2.resize(frame, (im_size, im_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch[i] = frame  # np.interp(frame, (frame.min(), frame.max()), (-1, +1))

            i += 1
            if i == batch_total_frames:
                i = 0
                process_batch(batch, batch_id, save_dir)
                batch_id += 1

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

        process_batch(final_batch, batch_id, save_dir)

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
