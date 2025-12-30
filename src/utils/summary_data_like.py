import numpy as np
import pandas as pd

def make_feature(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', ]
    return df[fe_names].values

def make_feature_with_comments(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', 
                    'is_like', 'is_follow',
                    'comment0_id', 'comment1_id', 'comment2_id', 'comment3_id', 'comment4_id', 'comment5_id',]
    return df[fe_names].values

def cal_field_dims(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['user_id', 'follow_user_num_range','register_days_range', 'fans_user_num_range', 'friend_user_num_range','user_active_degree',
                    'video_id', 'author_id', 
                    'is_like', 'is_follow']
    field_dims = [int(df[fe].max()) + 1 for fe in fe_names]
    print(fe_names)
    print(field_dims)
    print([df[fe].max() for fe in fe_names])
    return field_dims

def cal_comments_dims(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['comment0_id', 'comment1_id', 'comment2_id', 'comment3_id', 'comment4_id', 'comment5_id']
    
    # 计算每一列的最大值
    max_values = [df[fe].max() for fe in fe_names]
    field_dim = max(max_values) + 1  # 加 1 是为了将最大值作为合法索引
    
    print(f"Comments feature names: {fe_names}")
    print(f"Total unique field dimensions: {field_dim}")
    
    return field_dim


def make_comment_weights_feature(df, data_name):
    if data_name == 'KuaiComt':
        fe_names = ['comment0_explicit_weight', 'comment1_explicit_weight', 
                    'comment2_explicit_weight', 'comment3_explicit_weight', 
                    'comment4_explicit_weight', 'comment5_explicit_weight']
    else:
        fe_names = []

    # 如果缺少列，返回全0数组
    for fe in fe_names:
        if fe not in df.columns:
            df[fe] = 0.0

    if len(fe_names) == 0:
        return np.zeros((len(df), 6), dtype=np.float32)

    weights = df[fe_names].fillna(0.0).values
    return weights.astype(np.float32)
