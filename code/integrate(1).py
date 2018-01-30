import pandas as pd
import pickle as pk
import numpy as np


def integrate(feature, label):
    sArray = feature['subject_id'].unique()  # 取出所有样本的id
    truelist = label['subject_id'].unique()  # 取出患病患者的id
    xlist = []  # 存放输入特征的列表
    ylist = []  # 存放样本标签的列表

    count = 0;
    count1 = 0
    for sid in sArray:
        shock_index = feature[feature['subject_id'] == sid]['shock_index']
        sbp = feature[feature['subject_id'] == sid]['sbp']
        GCS = feature[feature['subject_id'] == sid]['GCS']
        rr = feature[feature['subject_id'] == sid]['rr']
        hr = feature[feature['subject_id'] == sid]['hr']
        fio2 = feature[feature['subject_id'] == sid]['fio2']
        fio2_flag = feature[feature['subject_id'] == sid]['fio2_flag']
        sofa_neur = feature[feature['subject_id'] == sid]['sofa_neur']
        Spo2 = feature[feature['subject_id'] == sid]['Spo2']
        sofa_respiratory = feature[feature['subject_id'] == sid]['sofa_respiratory']
        t = feature[feature['subject_id'] == sid]['t']
        map = feature[feature['subject_id'] == sid]['map']
        rsas = feature[feature['subject_id'] == sid]['rsas']
        dbp = feature[feature['subject_id'] == sid]['dbp']
        BUN = feature[feature['subject_id'] == sid]['BUN']
        BUN_CR = feature[feature['subject_id'] == sid]['BUN_CR']
        ph = feature[feature['subject_id'] == sid]['ph']
        Pao2 = feature[feature['subject_id'] == sid]['Pao2']
        sofa_hepatic = feature[feature['subject_id'] == sid]['sofa_hepatic']
        wbc = feature[feature['subject_id'] == sid]['wbc']
        sofa_renal = feature[feature['subject_id'] == sid]['sofa_renal']
        platelets = feature[feature['subject_id'] == sid]['platelets']
        sodium = feature[feature['subject_id'] == sid]['sodium']
        PaCO2 = feature[feature['subject_id'] == sid]['PaCO2']
        sofa_hematologic = feature[feature['subject_id'] == sid]['sofa_hematologic']
        creatinine = feature[feature['subject_id'] == sid]['creatinine']
        potassium = feature[feature['subject_id'] == sid]['potassium']
        hematocrit = feature[feature['subject_id'] == sid]['hematocrit']
        bicarbonate = feature[feature['subject_id'] == sid]['bicarbonate']
        glucose = feature[feature['subject_id'] == sid]['glucose']
        hemoglobin = feature[feature['subject_id'] == sid]['hemoglobin']
        # antibiotics_firsttime=feature[feature['subject_id']==sid]['antibiotics_firsttime']
        antibiotics_firsttime_flag = feature[feature['subject_id'] == sid]['antibiotics_firsttime_flag']
        uo_6 = feature[feature['subject_id'] == sid]['uo_6']
        chronic_liver_disease = feature[feature['subject_id'] == sid]['chronic_liver_disease']
        cardiac_surgery = feature[feature['subject_id'] == sid]['cardiac_surgery']
        immunocompromised = feature[feature['subject_id'] == sid]['immunocompromised']
        sirs = feature[feature['subject_id'] == sid]['sirs']
        hematological_malignancy = feature[feature['subject_id'] == sid]['hematological_malignancy']
        chronic_heart_failure = feature[feature['subject_id'] == sid]['chronic_heart_failure']
        chronic_organ_insufficiency = feature[feature['subject_id'] == sid]['chronic_organ_insufficiency']
        diabetes = feature[feature['subject_id'] == sid]['diabetes']
        metastatic_carcinoma = feature[feature['subject_id'] == sid]['metastatic_carcinoma']
        dialysis = feature[feature['subject_id'] == sid]['dialysis']
        weight_admission = feature[feature['subject_id'] == sid]['weight_admission']
        hypotension = feature[feature['subject_id'] == sid]['hypotension']
        weight = feature[feature['subject_id'] == sid]['weight']
        HIV = feature[feature['subject_id'] == sid]['HIV']
        uo_6_kg = feature[feature['subject_id'] == sid]['uo_6_kg']
        age = feature[feature['subject_id'] == sid]['age']
        # organ_dysfunction_firsttime = feature[feature['subject_id'] == sid]['organ_dysfunction_firsttime']
        organ_dysfunction_firsttime_flag = feature[feature['subject_id'] == sid]['organ_dysfunction_firsttime_flag']
        chronic_renal = feature[feature['subject_id'] == sid]['chronic_renal']
        CCU = feature[feature['subject_id'] == sid]['CCU']
        CSRU = feature[feature['subject_id'] == sid]['CSRU']
        MICU = feature[feature['subject_id'] == sid]['MICU']
        SICU = feature[feature['subject_id'] == sid]['SICU']
        TSICU = feature[feature['subject_id'] == sid]['TSICU']
        featuredata = zip(shock_index, sbp, GCS, rr, hr, fio2, fio2_flag, sofa_neur, Spo2, sofa_respiratory, t, map,
                          rsas,
                          dbp, BUN, BUN_CR, ph, Pao2, sofa_hepatic, wbc, sofa_renal, platelets, sodium, PaCO2,
                          sofa_hematologic, creatinine, potassium, hematocrit, bicarbonate, glucose,
                          hemoglobin, antibiotics_firsttime_flag, uo_6,
                          chronic_liver_disease, cardiac_surgery, immunocompromised, sirs,
                          hematological_malignancy, chronic_heart_failure, chronic_organ_insufficiency,
                          diabetes, metastatic_carcinoma, dialysis, weight_admission,
                          hypotension, weight, HIV, uo_6_kg, age,
                          organ_dysfunction_firsttime_flag, chronic_renal)

        featuredata = list(featuredata)
        if sid in truelist:
            Time = int(label[label['subject_id'] == sid]['time'])
            featuredata = featuredata[:Time]  # 截取患病之前的特征数据

        featuredata = featuredata[-24:]
        if len(featuredata) == 24:
            count += 1
            xlist.append(featuredata)
            if sid in truelist:
                ylist.append(1)
                count1 += 1
            else:
                ylist.append(0)
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    # xlist = (xlist - np.mean(xlist, axis=0)[None, :, :]) / np.std(xlist, axis=0)[None, :, :]
    # xlist = np.around(xlist, decimals = 3)
    return xlist, ylist


if __name__ == '__main__':
    feature = pd.read_csv('../rawdata/feature.csv')
    label = pd.read_csv('../rawdata/label.csv')
    x, y = integrate(feature, label)
    out1 = open('../data/data.pkl', 'wb')
    out2 = open('../data/label.pkl', 'wb')
    pk.dump(x, out1)
    pk.dump(y, out2)

