import scipy.io as sio
import pandas as pd


def convert_feature(matfn):
    data = sio.loadmat(matfn)
    df = pd.DataFrame(data['data_all'],
                      columns=['subject_id', 'time', 'shock_index', 'sbp', 'GCS', 'rr', 'hr', 'fio2', 'fio2_flag',
                               'sofa_neur', 'Spo2', 'sofa_respiratory', 't', 'map', 'rsas', 'dbp', 'BUN', 'BUN_CR',
                               'ph',
                               'Pao2', 'sofa_hepatic', 'wbc', 'sofa_renal', 'platelets', 'sodium', 'PaCO2',
                               'sofa_hematologic', 'creatinine', 'potassium', 'hematocrit', 'bicarbonate', 'glucose',
                               'hemoglobin', 'antibiotics_firsttime', 'antibiotics_firsttime_flag', 'uo_6',
                               'chronic_liver_disease', 'cardiac_surgery', 'immunocompromised', 'sirs',
                               'hematological_malignancy', 'chronic_heart_failure', 'chronic_organ_insufficiency',
                               'diabetes', 'metastatic_carcinoma', 'dialysis', 'weight_admission', 'current_careunit',
                               'hypotension', 'weight', 'HIV', 'uo_6_kg', 'age', 'organ_dysfunction_firsttime',
                               'organ_dysfunction_firsttime_flag', 'chronic_renal'])

    df[['subject_id', 'time', 'current_careunit']] = df[['subject_id', 'time', 'current_careunit']].astype(int)
    dummies_df = pd.get_dummies(df['current_careunit'], prefix='ICU')
    dummies_df.rename(columns={'ICU_1': 'CCU', 'ICU_2': 'CSRU', 'ICU_3': 'MICU', 'ICU_4': 'SICU', 'ICU_5': 'TSICU'},
                      inplace=True)
    df = pd.concat([df, dummies_df], axis=1)
    df.drop('current_careunit', axis=1, inplace=True)
    df.drop(['antibiotics_firsttime', 'organ_dysfunction_firsttime'], axis=1, inplace=True)
    df.to_csv('../rawdata/feature.csv', index=False)


# 将标签mat文件转换为dataframe


def convert_label(matln):
    data = sio.loadmat(matln)
    df = pd.DataFrame(data['shock_info'], columns=['subject_id', 'time'])
    df = df.astype(int)
    df.to_csv('../rawdata/label.csv', index=False)
    return df


if __name__ == '__main__':
    matfn = '../rawdata/data_all.mat'
    matln = u'F:/301/数据集/shock_info.mat'
    convert_feature(matfn)
    convert_label(matln)
