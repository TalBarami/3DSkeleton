class SkeletonLayout:
    def __init__(self):
        self.joints = {
            'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3,
            'L_Knee': 4, 'R_Knee': 5, 'Spine2': 6, 'L_Ankle': 7, 'R_Ankle': 8,
            'Spine3': 9, 'L_Foot': 10, 'R_Foot': 11, 'Neck': 12,
            'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
            'L_Shoulder': 16, 'R_Shoulder': 17, 'L_Elbow': 18, 'R_Elbow': 19,
            'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand': 22, 'R_Hand': 23
        }
        self._parts = {
            'Head': 0, 'Body': 1
        }
        i = 2
        for m in ['L', 'R']:
            for p in ['Arm', 'Forearm', 'Hand', 'Thigh', 'Calf', 'Foot']:
                self._parts[f'{m}_{p}'] = i
                i += 1
        self._parts_segmentation = {
            'Head': [(self.joints['Head'], self.joints['Neck'])],
            'Body': [(self.joints['Neck'], self.joints['Spine3']),
                     (self.joints['Spine3'], self.joints['Spine2']),
                     (self.joints['Spine2'], self.joints['Spine1']),
                     (self.joints['Spine1'], self.joints['Pelvis'])]
        }
        for m in ['L', 'R']:
            self._parts_segmentation['Body'] += [(self.joints['Spine3'], self.joints[f'{m}_Collar']),
                                                 (self.joints[f'{m}_Collar'], self.joints[f'{m}_Shoulder']),
                                                 # (self.joints[f'{m}_Collar'], self.joints[f'{m}_Hip']),
                                                 (self.joints['Pelvis'], self.joints[f'{m}_Hip']), ]
            self._parts_segmentation[f'{m}_Arm'] = [(self.joints[f'{m}_Shoulder'], self.joints[f'{m}_Elbow'])]
            self._parts_segmentation[f'{m}_Forearm'] = [(self.joints[f'{m}_Elbow'], self.joints[f'{m}_Wrist'])]
            self._parts_segmentation[f'{m}_Hand'] = [(self.joints[f'{m}_Wrist'], self.joints[f'{m}_Hand'])]
            self._parts_segmentation[f'{m}_Thigh'] = [(self.joints[f'{m}_Hip'], self.joints[f'{m}_Knee'])]
            self._parts_segmentation[f'{m}_Calf'] = [(self.joints[f'{m}_Knee'], self.joints[f'{m}_Ankle'])]
            self._parts_segmentation[f'{m}_Foot'] = [(self.joints[f'{m}_Ankle'], self.joints[f'{m}_Foot'])]
        self._parts_segmentation = {self._parts[k]: v for k, v in self._parts_segmentation.items()}
        self._pairs = [p for ps in self._parts_segmentation.values() for p in ps]
        self._limbs = {s: k for (k, v) in self._parts_segmentation.items() for s in v}

        # self.pairs = [
        #     (0, 1), (0, 2), (0, 3),
        #     (1, 4), (2, 5), (3, 6),
        #     (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
        #     (9, 13), (9, 14), (12, 15),
        #     (13, 16), (14, 17),
        #     (16, 18), (17, 19),
        #     (18, 20), (19, 21),
        #     (20, 22), (21, 23),  # 23 is actually 37
        # ]

    def edge_joints(self):
        return {k: v for k, v in self.joints.items() if any([x in k for x in ['Hand', 'Foot', 'Head']])}

    def body_joints(self):
        return {k: v for k, v in self.joints.items() if any([x in k for x in ['Pelvis', 'Spine', 'Collar']])}

    def limb_joints(self):
        return {k: v for k, v in self.joints.items() if any([x in k for x in ['Neck', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle']])}

    def pairs(self):
        return list(self._pairs)

    def limbs(self):
        return dict(self._limbs)

    def is_edge_limb(self, part):
        return part in [self._parts[k] for k in ['Head', 'L_Hand', 'R_Hand', 'L_Foot', 'R_Foot']]

    def parts(self, joint):
        return list(set([v for k, v in self.limbs().items() if joint in k]))

    def neighbors(self, joint):
        return list(set([j for tup in self.pairs() for j in tup if joint in tup and j != joint]))