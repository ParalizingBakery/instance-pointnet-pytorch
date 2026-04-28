import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class ScannetDatasetWholeScene:
    # prepare to give prediction on each points
    def __init__(
        self,
        root,
        block_points=4096,
        block_size=1.0,
        struct_coord_min=None,
        struct_coord_max=None,
    ):
        self.block_points = block_points
        self.block_size = block_size
        self.root = root
        self.scene_points_num = []

        self.file_list = sorted([d for d in os.listdir(root)])

        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:  # For every room
            data = np.load(os.path.join(root, file))
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])

            if (struct_coord_min is None) or (struct_coord_max is None):
                struct_coord_min, struct_coord_max = (
                    np.amin(points, axis=0)[:3],
                    np.amax(points, axis=0)[:3],
                )
            self.room_coord_min.append(struct_coord_min), self.room_coord_max.append(
                struct_coord_max
            )
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        # From my look at the add_vote() in test_semseg.py, label weights
        # are not used in testing at all, plus we are not using the same classes
        for seg in self.semantic_labels_list:
            self.scene_points_num.append(seg.shape[0])

    """Return Scene or room in this case"""

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        assert isinstance(point_set_ini, np.ndarray)
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        # Minimum and Maximum of the structure
        coord_min, coord_max = self.room_coord_min[index], self.room_coord_max[index]
        data_room, label_room, sample_weight, index_room = (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

        sample_min = np.amin(points[:, :3], axis=0)
        point_idxs = np.arange(0, points.shape[0])

        if point_idxs.size == 0:
            raise Exception("No points for sample")
        num_batch = int(np.ceil(point_idxs.size / self.block_points))
        point_size = int(num_batch * self.block_points)

        # duplicate points to a multiple of self.block_points
        replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
        point_idxs_repeat = np.random.choice(
            point_idxs, point_size - point_idxs.size, replace=replace
        )
        point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
        np.random.shuffle(point_idxs)
        data_batch = points[point_idxs, :]
        normlized_xyz = np.zeros((point_size, 3))

        # Readded Sample weights because zero weights (filler points for filling up to batch size)
        # Has to be discarded using weights
        sample_weight = np.ones((point_size))

        # For XY, normalize within structure
        normlized_xyz[:, 0] = (data_batch[:, 0] - coord_min[0]) / (
            coord_max[0] - coord_min[0]
        )
        normlized_xyz[:, 1] = (data_batch[:, 1] - coord_min[1]) / (
            coord_max[1] - coord_min[1]
        )

        # For Z axis, normalize within sample
        normlized_xyz[:, 2] = (data_batch[:, 2] - sample_min[2]) / (np.max(points[:, 2] - sample_min[2]))

        # normalize sample xy to [-0.5, 0.5] when bs is 1.0
        # for Z, still use absolute scale but set min to 0
        data_batch[:, 0] = data_batch[:, 0] - (sample_min[0] + (self.block_size / 2.0))
        data_batch[:, 1] = data_batch[:, 1] - (sample_min[1] + (self.block_size / 2.0))
        data_batch[:, 2] = data_batch[:, 2] - sample_min[2]
        data_batch[:, 3:6] /= 255.0
        data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
        label_batch = labels[point_idxs].astype(int)

        # Possible optimization: stacking copies both array to new index
        # Keeping them as a list then stacking once should be faster
        data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
        label_room = (
            np.hstack([label_room, label_batch]) if label_room.size else label_batch
        )
        index_room = (
            np.hstack([index_room, point_idxs]) if index_room.size else point_idxs
        )
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == "__main__":
    loader = ScannetDatasetWholeScene("data/wps_collected")
    data, labels, _, index = loader[0]

    # (N, 4096, 9)
    print(data.shape)
    np.savetxt('test_points_0.txt', data.reshape((-1, 9)))
