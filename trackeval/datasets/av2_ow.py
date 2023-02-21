from genericpath import isfile
import os
from matplotlib.pyplot import axis
import numpy as np
import json
import itertools
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
import glob
from pyarrow import feather
from scipy.spatial.transform import Rotation
# from pytorch3d.ops.iou_box3d import box3d_overlap
import torch
import sklearn.metrics
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType


_class_dict = {
    1: 'REGULAR_VEHICLE',
    2: 'PEDESTRIAN',
    3: 'BOLLARD',
    4: 'CONSTRUCTION_CONE',
    5: 'CONSTRUCTION_BARREL',
    6: 'STOP_SIGN',
    7: 'BICYCLE',
    8: 'LARGE_VEHICLE',
    9: 'WHEELED_DEVICE',
    10: 'BUS',
    11: 'BOX_TRUCK',
    12: 'SIGN',
    13: 'TRUCK',
    14: 'MOTORCYCLE',
    15: 'BICYCLIST',
    16: 'VEHICULAR_TRAILER',
    17: 'TRUCK_CAB',
    18: 'MOTORCYCLIST',
    19: 'DOG',
    20: 'SCHOOL_BUS',
    21: 'WHEELED_RIDER',
    22: 'STROLLER',
    23: 'ARTICULATED_BUS',
    24: 'MESSAGE_BOARD_TRAILER',
    25: 'MOBILE_PEDESTRIAN_SIGN',
    26: 'WHEELCHAIR',
    27: 'RAILED_VEHICLE',
    28: 'OFFICIAL_SIGNALER',
    29: 'TRAFFIC_LIGHT_TRAILER',
    30: 'ANIMAL',
    31: 'MOBILE_PEDESTRIAN_CROSSING_SIGN'}

class_dict = {v: k for k, v in _class_dict.items()}


class AV2_OW(_BaseDataset):
    """Dataset class for TAO tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/argoverse2'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'out'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': None,  # Classes to eval (if None, all classes)
            'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'val'
            'PRINT_CONFIG': False,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'MAX_DETECTIONS': 300,  # Number of maximal allowed detections per image (0 for unlimited)
            'SUBSET': 'all',
            'NON_DRIVE': True,
            'REMOVE_FAR': True,
            'REMOVE_NON_MOVE': True,
            'REMOVE_NON_MOVE_STRATEGY': 'per_frame',
            'REMOVE_NON_MOVE_THRESH': 0.5
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.gt_fol = os.path.join(self.config['GT_FOLDER'], self.config['SPLIT_TO_EVAL'])
        self.loader = AV2SensorDataLoader(data_dir=Path(self.gt_fol), labels_dir=Path(self.gt_fol))
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.should_classes_combine = True
        self.use_super_categories = False

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']
        self._default_known_unknowns()

        # LOAD GT FILES
        gt_dir_files = glob.glob(os.path.join(self.gt_fol, '*', 'annotations.feather'))
        self.gt_data = self.read_feather_files(gt_dir_files)

        # assign each track unique id
        self.gt_track_to_track_id = {track: i for i, track in enumerate(
            self.gt_data['track_uuid'].unique())}
        a = lambda x: self.gt_track_to_track_id[x]
        self.gt_data['track_id'] = self.gt_data['track_uuid'].apply(a)

        self.subset = self.config['SUBSET']
        if self.subset != 'all':
            # Split GT data into `known`, `unknown` or `distractor`
            self._split_known_unknown_distractor()
            self.gt_data = self._filter_gt_data(self.gt_data)

        # Get sequence names and ids
        self.seq_list = self.gt_data['seq'].unique()
        self.seq_name_to_seq_id = {seq: i for i, seq in enumerate(self.seq_list)}

        # get timestamps and compute sequence lengths
        self.timestamps = {seq: set() for seq in self.gt_data['seq'].unique()}
        for seq in self.seq_list:
            times = self.gt_data[self.gt_data['seq']==seq]['timestamp_ns'].unique().tolist()
            self.timestamps[seq] = sorted(times)
        self.timestamps_to_id = {seq: {t: i for i, t in enumerate(timestamps)}\
            for seq, timestamps in self.timestamps.items()}
        self.seq_lengths = {k: len(v) for k, v in self.timestamps.items()}

        # get classes per sequence
        self.seq_to_classes = dict()
        for seq, i in self.seq_name_to_seq_id.items():
            data = self.gt_data[self.gt_data['seq'] == seq]
            class_dict['pos_cat_ids'] = [\
                cat for cat in data['category'].unique() if cat in self.knowns]
            class_dict['neg_cat_ids'] = [\
                cat for cat in data['category'].unique() if cat in self.unknowns]
            class_dict['distractors'] = [\
                cat for cat in data['category'].unique() if cat in self.distractors]
            self.seq_to_classes[i] = class_dict

        # get sequences to gt track ids
        self.sequences_to_gt_tracks = {
            i: self.gt_data[self.gt_data['seq']==seq]['track_id'].unique() \
                for seq, i in self.seq_name_to_seq_id.items()}

        # get positive classes (classes where there are labels)
        '''considered_vid_ids = [self.seq_name_to_seq_id[vid] for vid in self.seq_list]'''
        '''seen_cats = set([cat_id for vid_id in considered_vid_ids for cat_id
                         in self.seq_to_classes[vid_id]['pos_cat_ids']])'''
        
        '''self.valid_classes = [cls for cls in self.gt_data['category'] \
            if cls['id'] in seen_cats]'''

        # however only evaluate on generic class object in OW setting
        if self.config['CLASSES_TO_EVAL']:
            self.class_list = ["object"]  # class-agnostic
            if not all(self.class_list):
                raise TrackEvalException('Attempted to evaluate an invalid class. Only classes ' +
                                         ', '.join(self.valid_classes) +
                                         ' are valid (classes present in ground truth data).')
        else:
            self.class_list = ["object"]  # class-agnostic
        
        # create name to id dict for classes
        self.class_name_to_class_id = {"object": 1}  # class-agnostic

        # GET TRACKERS TO EVAL
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        self.tracker_data = {tracker: dict() for tracker in self.tracker_list}
        self.track_to_track_id = dict()
        for tracker in self.tracker_list:
            # get tracker files
            tracker_dir = os.path.join(self.tracker_fol, tracker, self.config['SPLIT_TO_EVAL'])
            tr_dir_files = glob.glob(os.path.join(tracker_dir, '*', 'annotations.feather'))
            curr_data = self.read_feather_files(tr_dir_files, is_gt=False)
            curr_data = curr_data.astype({
                'timestamp_ns': 'int64',
                'track_uuid': 'str',
                'length_m': 'float',
                'width_m': 'float',
                'height_m': 'float',
                'qw': 'float',
                'qx': 'float',
                'qy': 'float',
                'qz': 'float',
                'tx_m': 'float',
                'ty_m': 'float',
                'tz_m': 'float',
                'num_interior_pts': 'int',
                'seq': 'str'}, copy=False)
            
            # assign each track unicke id
            self.track_to_track_id[tracker] = {track: i for i, track in enumerate(
                curr_data['track_uuid'].unique())}
            a = lambda x: self.track_to_track_id[tracker][x]
            curr_data['track_id'] = curr_data['track_uuid'].apply(a)

            # get tracker sequence information
            self.tracker_data[tracker]['data'] = curr_data
            self.tracker_data[tracker]['vids_to_tracks'] = dict()
            for seq_name, seq_id in self.seq_name_to_seq_id.items():
                self.tracker_data[tracker]['vids_to_tracks'][seq_id] = \
                    curr_data[curr_data['seq'] == seq_name]['track_id'].unique()

    def _calculate_3d_distance(self, bboxes1, bboxes2, dist='euclidean'):
        if dist == '3dIoU':
            similarity_scores = self._calculate_box_ious3D(bboxes1, bboxes2)
        elif dist == 'euclidean':
            similarity_scores = self._calculate_3d_euclidean(bboxes1, bboxes2)
        elif dist == 'rot':
            similarity_scores = self._calculate_3d_rotation(bboxes1, bboxes2)
        
        return similarity_scores

    def _calculate_3d_euclidean(self, bboxes1, bboxes2, clip_val=10):
        '''
        input values of numpy array:
            "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"
        '''
        dist = sklearn.metrics.pairwise_distances(bboxes1[:, :3], bboxes2[:, :3])
        dist = np.clip(dist, a_min=0, a_max=clip_val)
        dist = 1 - dist/clip_val
        return dist
    
    def _calculate_3d_rotation(self, bboxes1, bboxes2):
        '''
        FROM AV2 geometry / detection evaluation
        input values of numpy array:
            "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"
        '''
        # Convert quaternion from scalar first to scalar last.
        rot1 = self.quat_to_mat(bboxes1[:, [6, 7, 8, 9]])
        rot2 = self.quat_to_mat(bboxes2[:, [6, 7, 8, 9]])

        # Convert a 3D rotation matrix to a sequence of _extrinsic_ rotations.
        xyz_rad1 = Rotation.from_matrix(rot1).as_euler("xyz", degrees=False)
        xyz_rad2 = Rotation.from_matrix(rot2).as_euler("xyz", degrees=False)

        # angle distance
        angles = xyz_rad2 - xyz_rad1

        # Map angles to [0, ∞].
        angles = np.abs(angles)

        # Calculate floor division and remainder simultaneously.
        divs, mods = np.divmod(angles, np.pi)

        # Select angles which exceed specified period.
        angle_complement_mask = np.nonzero(divs)

        # Take set complement of `mods` w.r.t. the set [0, π].
        # `mods` must be nonzero, thus the image is the interval [0, π).
        angles[angle_complement_mask] = np.pi - mods[angle_complement_mask]
        return angles
    
    def quat_to_mat(self, quat_wxyz):
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        mat = Rotation.from_quat(quat_xyzw).as_matrix()
        return mat

    def _calculate_box_ious3D(self, bboxes1, bboxes2, box_format='quat'):
        '''
        bboxes:
             input dim: (B, 8, 3) where B does not have to be the same for both
        box_format:
            If box_format pytorch3D already fine:

                (4) +---------+. (5)
                    | ` .     |  ` .
                    | (0) +---+-----+ (1)
                    |     |   |     |
                (7) +-----+---+. (6)|
                    ` .   |     ` . |
                    (3) ` +---------+ (2)

                    box_corner_vertices = [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]

            If box_format argoverse, first transform:
                5------4
                |\\    |\\
                | \\   | \\
                6--\\--7  \\
                \\  \\  \\ \\
            l    \\  1-------0    h
            e    \\ ||   \\ ||   e
            n    \\||    \\||   i
            g    \\2------3    g
                t      width.     h
                h.               t.

            box_corner_vertices = [
                [+1, +1, +1],  # 0
                [+1, -1, +1],  # 1
                [+1, -1, -1],  # 2
                [+1, +1, -1],  # 3
                [-1, +1, +1],  # 4
                [-1, -1, +1],  # 5
                [-1, -1, -1],  # 6
                [-1, +1, -1],  # 7
            ]

            If box_format quat: get argoverse first and then pytorch3D
        '''
        print(bboxes1.shape)
        if box_format == 'quat':
            bboxes1 = self.to_argoverse(bboxes1)
            bboxes2 = self.to_argoverse(bboxes2, pr=False)
            print(bboxes2)
            print(bboxes1)

        if box_format == 'quat' or box_format == 'argoverse':
            # rotate boxes to bring to same coordinate system as torch3d
            rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
            bboxes1 = bboxes1 @ rotation
            bboxes2 = bboxes2 @ rotation
            
            # bring to same order as box 3D
            order = [1, 0, 3, 2, 5, 4, 7, 6]
            order = [6, 2, 3, 7, 5, 1, 0, 4]
            bboxes1 = bboxes1[:, order, :]
            bboxes2 = bboxes2[:, order, :]

        _, iou = box3d_overlap(torch.from_numpy(bboxes2).float(), torch.from_numpy(bboxes1).float())
        
        print(iou.type(torch.int64), iou.shape)
        quit()
        return iou

    def to_argoverse(self, bbox, pr=False):
        '''
        input values of numpy array:
            "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"
        '''
        rotation = self.quat_to_mat(bbox[:, [6, 7, 8, 9]])
        translation_m = bbox[:, [0, 1, 2]]
        if pr:
            rot_eps = np.array(
                [[np.cos(1), -np.sin(1), 0],
                [np.sin(1), np.cos(1), 0],
                [0, 0, 1]])
            rotation = rot_eps @ rotation
            trans_eps = np.array([0.1, 0.1, 0])
            translation_m = translation_m + trans_eps
        dims_lwh_m = bbox[:, [3, 4, 5]]
        vertices_dst_xyz_m = np.stack([self.vertices(dim_lwh_m, rot, trans)\
            for dim_lwh_m, rot, trans in zip(dims_lwh_m, rotation, translation_m)])

        return vertices_dst_xyz_m
    
    def vertices(self, dim_lwh_m, rotation, translation_m):
        unit_vertices_obj_xyz_m = np.array(
            [
                [+1, +1, +1],  # 0
                [+1, -1, +1],  # 1
                [+1, -1, -1],  # 2
                [+1, +1, -1],  # 3
                [-1, +1, +1],  # 4
                [-1, -1, +1],  # 5
                [-1, -1, -1],  # 6
                [-1, +1, -1],  # 7
            ],
        )

        vertices_obj_xyz_m = (dim_lwh_m / 2.0) * unit_vertices_obj_xyz_m
        vertices_dst_xyz_m = vertices_obj_xyz_m @ rotation.T + translation_m
        return vertices_dst_xyz_m

    def read_feather_files(self, paths, is_gt=True):
        df = None
        for i, path in enumerate(paths):
            data = feather.read_feather(path)
            convert2int = lambda x: class_dict[x]
            data['category'] = data['category'].apply(convert2int)
            data['seq'] = [os.path.basename(os.path.dirname(path))] * data.shape[0]
            if df is None:
                df = data
            else:
                df = df.append(data)
        if not is_gt:
            if 'score' not in df.columns:
                df['score'] = [1] * df.shape[0]
        
        # filter out objects in non-drivable area or far away
        if is_gt:
            # get file name
            split_dir = os.path.dirname(os.path.dirname(path))
            split = os.path.basename(split_dir)
            split_dir = os.path.dirname(split_dir)

            file = 'filtered_version.feather'
            file = 'remove_non_drive_' + file if self.config['NON_DRIVE'] else file
            file = 'remove_far_' + file if self.config['REMOVE_FAR'] else file
            file = 'remove_non_move_' + file if self.config['REMOVE_NON_MOVE'] else file
            file = self.config['REMOVE_NON_MOVE_STRATEGY'] + '_' + file if self.config['REMOVE_NON_MOVE'] else file
            file = str(self.config['REMOVE_NON_MOVE_THRESH'])[:5] + '_' + file if self.config['REMOVE_NON_MOVE'] else file
            file = split + '_' + file if self.config['REMOVE_FAR'] else file

            path_filtered = \
                os.path.join(split_dir, file)

            print(path_filtered)

            # check if filtered version already exists
            if os.path.isfile(path_filtered):
                return feather.read_feather(path_filtered)
            
            # generate filtered version
            filtered = None
            map_dict = dict()
            track_ids = list()
            # iterate over sequences
            num_seqs = df['seq'].unique().shape[0]
            for m, seq in enumerate(df['seq'].unique()):
                print(f'Sequence {m}/{num_seqs}...')
                seq_df = df[df['seq'] == seq]
                timestamps = sorted(seq_df['timestamp_ns'].unique().tolist())

                if self.config['REMOVE_NON_MOVE'] and self.config['REMOVE_NON_MOVE_STRATEGY'] == 'per_seq':
                    centroid_dict = defaultdict(list)
                    movers = list()
                    centroid_time = defaultdict(list)
                    for i, t in enumerate(sorted(seq_df['timestamp_ns'].unique().tolist())):
                        labels = self.loader.get_labels_at_lidar_timestamp(
                            log_id=seq, lidar_timestamp_ns=int(t))

                        city_SE3_ego = self.loader.get_city_SE3_ego(seq, int(t))

                        labels = labels.transform(city_SE3_ego)
                        for label in labels:
                            centroid_dict[label.track_id].append(label.dst_SE3_object.translation)
                            centroid_time[label.track_id].append(int(t))
                    
                    for track, centroids in centroid_dict.items():
                        diff_cent = np.linalg.norm(np.asarray(centroids)[:-1, :-1] - np.asarray(centroids)[1:, :-1] , axis=1)
                        diff_time = np.asarray(centroid_time[track])[1:] - np.asarray(centroid_time[track])[:-1]
                        diff_time = diff_time / np.power(10, 9) 
                        if np.mean(diff_cent/diff_time) > self.config['REMOVE_NON_MOVE_THRESH']:
                            movers.append(track)

                # iterate over timesyeps to get argoverse map and labels
                for i, t in enumerate(sorted(seq_df['timestamp_ns'].unique().tolist())):
                    time_df = seq_df[seq_df['timestamp_ns']==t]

                    # get labels
                    labels = self.loader.get_labels_at_lidar_timestamp(
                        log_id=seq, lidar_timestamp_ns=int(t))

                    # remove labels that are non in dirvable area
                    if self.config['NON_DRIVE']:
                        if seq not in map_dict.keys():
                            log_map_dirpath = Path(self.gt_fol) / seq / "map"
                            map_dict[seq] = ArgoverseStaticMap.from_map_dir(
                                log_map_dirpath, build_raster=True)
                        centroids_ego = np.asarray(
                            [label.dst_SE3_object.translation for label in labels])
                        city_SE3_ego = self.loader.get_city_SE3_ego(seq, int(t))
                        centroids_city = city_SE3_ego.transform_point_cloud(
                                centroids_ego)
                        bool_labels = map_dict[seq].get_raster_layer_points_boolean(
                            centroids_city, layer_name=RasterLayerType.DRIVABLE_AREA)
                        labels = [l for i, l in enumerate(labels) if bool_labels[i]]

                    # remove points that are far away (>80,)
                    if self.config['REMOVE_FAR']:
                        all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])
                        dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
                        ind = np.where(dists_to_center <= 80)[0]
                        labels = [labels[i] for i in ind]

                    if self.config['REMOVE_NON_MOVE']:
                        if self.config['REMOVE_NON_MOVE_STRATEGY'] == 'per_seq':
                            labels = [label for label in labels if label.track_id in movers]
                        elif self.config['REMOVE_NON_MOVE_STRATEGY'] == 'per_frame':
                            if i < len(timestamps) - 1:
                                # labels at t+1 
                                labels_t2 = self.loader.get_labels_at_lidar_timestamp(
                                    log_id=seq, lidar_timestamp_ns=int(timestamps[i+1]))
                                city_SE3_t2 = self.loader.get_city_SE3_ego(seq, int(timestamps[i+1]))
                                labels_t2 = labels_t2.transform(city_SE3_t2)
                                ids_t2 = [label.track_id for label in labels_t2]
                            else:
                                labels_t2 = list()
                                ids_t2 = list()

                            if i > 0:
                                labels_t0 = self.loader.get_labels_at_lidar_timestamp(
                                    log_id=seq, lidar_timestamp_ns=int(timestamps[i-1]))
                                city_SE3_t0 = self.loader.get_city_SE3_ego(seq, int(timestamps[i-1]))
                                labels_t0 = labels_t0.transform(city_SE3_t0)
                                ids_t0 = [label.track_id for label in labels_t0]
                            else:
                                labels_t0 = list()
                                ids_t0 = list()

                            city_SE3_t1 = self.loader.get_city_SE3_ego(seq, int(timestamps[i]))
                            labels_city = [l.transform(city_SE3_t1) for l in labels]

                            
                            other_trans = list()
                            bool_labels = list()
                            vels = list()
                            for label in labels_city:
                                if len(labels_t2) and label.track_id in ids_t2:
                                    other_trans = labels_t2[ids_t2.index(label.track_id)].dst_SE3_object.translation
                                    dist = np.linalg.norm(label.dst_SE3_object.translation - other_trans)
                                    diff_time = (timestamps[i+1]-t) / np.power(10, 9)
                                    vel = dist/diff_time
                                    vels.append(vel)
                                    bool_labels.append(vel > self.config['REMOVE_NON_MOVE_THRESH'])
                                elif len(labels_t0) and label.track_id in ids_t0:
                                    other_trans = labels_t0[ids_t0.index(label.track_id)].dst_SE3_object.translation
                                    dist = np.linalg.norm(label.dst_SE3_object.translation - other_trans)
                                    diff_time = (t-timestamps[i-1]) / np.power(10, 9)
                                    vel = dist/diff_time
                                    vels.append(vel)
                                    bool_labels.append(vel > self.config['REMOVE_NON_MOVE_THRESH'])
                                else:
                                    vels.append(None)
                                    bool_labels.append(False)

                            labels = [l for i, l in enumerate(labels) if bool_labels[i]]
                        else:
                            assert self.config['REMOVE_NON_MOVE_STRATEGY'] in ['per_frame', 'per_seq'], 'remove strategy for static objects not defined'

                    # get track id of remaining objects
                    track_ids.extend([l.track_id for l in labels])

                    # filter time df by track ids
                    time_df = time_df[time_df['track_uuid'].isin(track_ids)]
                    if filtered is None and time_df.shape[0]:
                        filtered = time_df
                    elif filtered is not None:
                        filtered = filtered.append(time_df)

            # store filtered df
            df = filtered
            with open(path_filtered, 'wb') as f:
                feather.write_feather(df, f)

        return df

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the TAO format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets]: list (for each timestep) of lists of detections.
        [classes_to_gt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_gt_track_ids, classes_to_gt_track_areas, classes_to_gt_track_lengths]: dictionary with class values
                                as keys and lists (for each track) as values

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        [classes_to_dt_tracks]: dictionary with class values as keys and list of dictionaries (with frame indices as
                                keys and corresponding segmentations as values) for each track
        [classes_to_dt_track_ids, classes_to_dt_track_areas, classes_to_dt_track_lengths]: dictionary with class values
                                                                                           as keys and lists as values
        [classes_to_dt_track_scores]: dictionary with class values as keys and 1D numpy arrays as values
        """
        seq_id = self.seq_name_to_seq_id[seq]
        # File location
        if is_gt:
            data = self.gt_data[self.gt_data['seq'] == seq]
        else:
            data = self.tracker_data[tracker]['data']
            data = data[data['seq'] == seq]

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        timestamps = sorted(self.timestamps[seq])
        data_keys = ['ids', 'classes', 'dets']
        if not is_gt:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}
        for i, t in enumerate(timestamps):
            # some tracker data contains images without any ground truth information, these are ignored
            annotations = data[data['timestamp_ns'] == t]
            if annotations.shape[0] == 0:
                continue

            raw_data['dets'][i] = np.atleast_2d(
                [ann[["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]].to_numpy() \
                    for _, ann in annotations.iterrows()]).astype(float)

            raw_data['ids'][i] = np.atleast_1d(
                [ann['track_id'] for _, ann in annotations.iterrows()]).astype(int)

            raw_data['classes'][i] = np.atleast_1d(
                [1 for _, _ in annotations.iterrows()]).astype(int)   # class-agnostic
            
            if not is_gt:
                raw_data['tracker_confidences'][i] = np.atleast_1d(
                    [ann['score'] for _, ann in annotations.iterrows()]).astype(float)

        for t, d in enumerate(raw_data['dets']):
            if d is None:
                raw_data['dets'][t] = np.empty((0, 6)).astype(float)
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)
                if not is_gt:
                    raw_data['tracker_confidences'][t] = np.empty(0)

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        # all_classes = [self.class_name_to_class_id[cls] for cls in self.class_list]
        all_classes = [1]  # class-agnostic

        if is_gt:
            classes_to_consider = all_classes
            all_tracks = self.sequences_to_gt_tracks[seq_id]
        else:
            classes_to_consider = all_classes  # class-agnostic
            all_tracks = self.tracker_data[tracker]['vids_to_tracks'][seq_id]

        classes_to_tracks = {1: all_tracks}  # class-agnostic

        # mapping from classes to track information
        raw_data['classes_to_tracks'] = dict()
        for cls, tracks in classes_to_tracks.items():
            get_tracks = list()
            for track in tracks:
                for _, det in data[data['track_id']==track].iterrows():
                    get_tracks.append({self.timestamps_to_id[seq][int(det['timestamp_ns'])]: \
                        np.atleast_1d(det[["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]].to_numpy())})

            raw_data['classes_to_tracks'][cls] = get_tracks

        raw_data['classes_to_track_ids'] = {cls: [track for track in tracks]
                                            for cls, tracks in classes_to_tracks.items()}

        raw_data['classes_to_track_lengths'] = {cls: [data[data['track_id']==track].shape[0] for track in tracks]
                                                for cls, tracks in classes_to_tracks.items()}

        if not is_gt:
            raw_data['classes_to_dt_track_scores'] = {cls: np.array([data[data['track_id']==track]['score'].mean()
                                                                     for track in tracks])
                                                      for cls, tracks in classes_to_tracks.items()}

        if is_gt:
            key_map = {'classes_to_tracks': 'classes_to_gt_tracks',
                       'classes_to_track_ids': 'classes_to_gt_track_ids',
                       'classes_to_track_lengths': 'classes_to_gt_track_lengths'}
        else:
            key_map = {'classes_to_tracks': 'classes_to_dt_tracks',
                       'classes_to_track_ids': 'classes_to_dt_track_ids',
                       'classes_to_track_lengths': 'classes_to_dt_track_lengths'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data['num_timesteps'] = num_timesteps
        raw_data['neg_cat_ids'] = self.seq_to_classes[seq_id]['neg_cat_ids']
        raw_data['distractors'] = self.seq_to_classes[seq_id]['distractors']
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.
        TAO:
            In TAO, the 4 preproc steps are as follow:
                1) All classes present in the ground truth data are evaluated separately.
                2) No matched tracker detections are removed.
                3) Unmatched tracker detections are removed if there is not ground truth data and the class does not
                    belong to the categories marked as negative for this sequence. Additionally, unmatched tracker
                    detections for classes which are marked as not exhaustively labeled are removed.
                4) No gt detections are removed.
            Further, for TrackMAP computation track representations for the given class are accessed from a dictionary
            and the tracks from the tracker data are sorted according to the tracker confidence.
        """
        cls_id = self.class_name_to_class_id[cls]
        is_not_exhaustively_labeled = cls_id in raw_data['distractors']
        is_neg_category = cls_id in raw_data['neg_cat_ids']

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):
            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm).
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            if gt_ids.shape[0] == 0 and not is_neg_category:
                to_remove_tracker = unmatched_indices
            elif is_not_exhaustively_labeled:
                to_remove_tracker = unmatched_indices
            else:
                to_remove_tracker = np.array([], dtype=np.int)

            # remove all unwanted unmatched tracker detections
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # get track representations
        data['gt_tracks'] = raw_data['classes_to_gt_tracks'][cls_id]
        data['gt_track_ids'] = raw_data['classes_to_gt_track_ids'][cls_id]
        data['gt_track_lengths'] = raw_data['classes_to_gt_track_lengths'][cls_id]
        data['dt_tracks'] = raw_data['classes_to_dt_tracks'][cls_id]
        data['dt_track_ids'] = raw_data['classes_to_dt_track_ids'][cls_id]
        data['dt_track_lengths'] = raw_data['classes_to_dt_track_lengths'][cls_id]
        data['dt_track_scores'] = raw_data['classes_to_dt_track_scores'][cls_id]
        data['not_exhaustively_labeled'] = is_not_exhaustively_labeled
        data['iou_type'] = 'bbox'

        # sort tracker data tracks by tracker confidence scores
        if data['dt_tracks']:
            idx = np.argsort([-score for score in data['dt_track_scores']], kind="mergesort")
            data['dt_track_scores'] = [data['dt_track_scores'][i] for i in idx]
            data['dt_tracks'] = [data['dt_tracks'][i] for i in idx]
            data['dt_track_ids'] = [data['dt_track_ids'][i] for i in idx]
            data['dt_track_lengths'] = [data['dt_track_lengths'][i] for i in idx]
        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_3d_distance(gt_dets_t, tracker_dets_t)
        return similarity_scores

    def _split_known_unknown_distractor(self):
        all_ids = set([i for i in range(1, 32)])  # 2000 is larger than the max category id in TAO-OW.
        # `knowns` includes 78 TAO_category_ids that corresponds to 78 COCO classes.
        # (The other 2 COCO classes do not have corresponding classes in TAO).
        self.knowns = {1, 2, 7, 10, 14, 15, 18}
        # `distractors` is defined as in the paper "Opening up Open-World Tracking"
        self.unknowns = {8, 9, 11, 13, 16, 17, 19, 20, 21, 22, 23, 26, 27, 30}
        self.distractors = all_ids.difference(self.knowns.union(self.distractors))

    def _default_known_unknowns(self):
        all_ids = set([i for i in range(1, 32)])  # 2000 is larger than the max category id in TAO-OW.
        # `knowns` includes 78 TAO_category_ids that corresponds to 78 COCO classes.
        # (The other 2 COCO classes do not have corresponding classes in TAO).
        self.knowns = {1, 2, 7, 10, 14, 15, 18, 8, 9, 11, 13, 16, 17, 19, 20, 21, 22, 23, 26, 27, 30}
        # `distractors` is defined as in the paper "Opening up Open-World Tracking"
        self.unknowns = {}
        self.distractors = all_ids.difference(self.knowns.union(self.unknowns))

    def _filter_gt_data(self, raw_gt_data):
        """
        Filter out irrelevant data in the raw_gt_data
        Args:
            raw_gt_data: directly loaded from json.

        Returns:
            filtered gt_data
        """
        valid_cat_ids = list()
        if self.subset == "known":
            valid_cat_ids = self.knowns
        elif self.subset == "distractor":
            valid_cat_ids = self.distractors
        elif self.subset == "unknown":
            valid_cat_ids = self.unknowns
        # elif self.subset == "test_only_unknowns":
        #     valid_cat_ids = test_only_unknowns
        else:
            raise Exception("The parameter `SUBSET` is incorrect")

        filtered = raw_gt_data[raw_gt_data['category'].isin(valid_cat_ids)]

        return filtered
