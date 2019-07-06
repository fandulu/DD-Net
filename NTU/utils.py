import numpy as np
import os
import pandas as pd
import random
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt


###################################################################################
    
    
#Rescale to be 64 frames
def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p_new[:,m,n] = medfilt(p_new[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new

from scipy.spatial.distance import cdist

def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)


def get_CG_single(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)
    return M


def get_CG_double(p_0,p_1,C):
    M_0 = []
    M_1 = []
    # index refers to [0,1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20] to get [1, 3, 7, 11, 14, 18]
    inter_ind = np.array([1,3,6,9,12,16])
    
    iu_self = np.triu_indices(C.joint_n,1,C.joint_n)
    iu_inter = np.triu_indices(6,1,6)
    
    for f in range(C.frame_l):
        #correlation graph 
        d_m_0 = cdist(p_0[f],p_0[f],'euclidean')[iu_self]
        d_m_1 = cdist(p_1[f],p_1[f],'euclidean')[iu_self]
        d_m_01 = cdist(p_0[f],p_1[f],'euclidean')[iu_inter]
        d_m_10 = cdist(p_1[f],p_0[f],'euclidean')[iu_inter]
        
        M_0.append(np.concatenate([d_m_0,d_m_01]))
        M_1.append(np.concatenate([d_m_1,d_m_10]))
 
    M_0 = np.stack(M_0)
    M_1 = np.stack(M_1)
    
    M_0 = norm_scale(M_0)
    M_1 = norm_scale(M_1)
        
    return M_0,M_1



def sampling_frame_double(p_0,p_1,C):
    full_l = p_0.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.85,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p_0 = p_0[int(s):int(e),:,:]
        p_1 = p_1[int(s):int(e),:,:]     
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p_0 = p_0[index,:,:]
        p_1 = p_1[index,:,:]
    p_0 = zoom(p_0,C.frame_l,C.joint_n,C.joint_d)
    p_1 = zoom(p_1,C.frame_l,C.joint_n,C.joint_d)
    return p_0,p_1

def sampling_frame_single(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.85,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]    
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
    return p
   
###################################################################################

###################################################################################
# Following functions are orginally from https://github.com/huguyuehuhu/HCN-pytorch/blob/master/feeder/ntu_read_skeleton.py (Thanks to huguyuehuhu). Here, some modifications are done to fit our needs.

def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


# modified read_xyz to (max_body, seq_info['numFrame'], num_joint, 3)
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j,:] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


# modified to save all data in one file by pandas
def gendata(data_path,
            out_path,
            training_subjects=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38],
            training_cameras=[2, 3],
            ignored_sample_path=None,
            benchmark='xview',
            part='val'):

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []

    sample = pd.Series()
    sample['name'] = []
    sample['label'] = []
    sample['poses'] = []

    data_list = os.listdir(data_path)
    for i in tqdm(range(len(data_list))):
        filename = data_list[i]
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            pose = read_xyz(os.path.join(data_path,filename))
            sample['name'].append(filename)
            sample['label'].append(action_class)
            sample['poses'].append(pose)

    sample.to_pickle(out_path+'{}_{}.pkl'.format(benchmark,part),compression='gzip')
###################################################################################    



def rotaion_two(p_0,p_1,R):
    p_0_new = np.zeros_like(p_0)
    p_1_new = np.zeros_like(p_1)
    for i in range(len(p_0)):
        p_0_new[i] = np.dot(R,p_0[i].T).T
    for i in range(len(p_1)):
        p_1_new[i] = np.dot(R,p_1[i].T).T
    return p_0_new,p_1_new

def rotaion_one(p,R):
    p_new = np.zeros_like(p)
    for i in range(len(p)):
        p_new[i] = np.dot(R,p[i].T).T
    return p_new




