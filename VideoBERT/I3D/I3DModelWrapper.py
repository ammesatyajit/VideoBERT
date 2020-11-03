import tensorflow as tf
import atexit


class I3DModel:
    def __init__(self, frozen_graph_path):
        self.graph = self.load_graph(frozen_graph_path)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        self.input = self.graph.get_tensor_by_name('graph/Placeholder:0')
        self.output = self.graph.get_tensor_by_name('graph/AvgPool3D:0')

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.sess is not None:
            print('closing tf session in I3DModel')
            self.sess.close()

    def load_graph(self, frozen_graph_filename):
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="graph")
        return graph

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def generate_features(self, batch):
        features = self.sess.run(self.output, feed_dict={
            self.input: batch
        })
        return features.squeeze()


if __name__ == '__main__':
    import numpy as np
    with I3DModel('graph_optimized.pb') as model:
        for _ in range(1):
            features = model.generate_features(np.random.rand(20, 15, 224, 224, 3))
            print(features)
            print(features.shape)
