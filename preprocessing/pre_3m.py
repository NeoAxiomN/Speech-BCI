import os, glob, scipy.io, pickle, mne, numpy as np, pandas as pd
from tqdm import tqdm

# --- 路径 (严禁修改) ---
derivatives_base = '../../LinJunyi/DATA/3M/derivatives/preproc'
raw_base = '../../LinJunyi/DATA/3M'
save_path = '../data/3M'
os.makedirs(save_path, exist_ok=True)

# 10-20 系统标准 32 通道
CH_NAMES = ['Fp1','Fp2','AF3','AF4','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO3','PO4','O1','Oz','O2']

def get_labels(sub, ses, num_trials):
    """尝试从 TSV 提取，失败则返回 None"""
    f = glob.glob(os.path.join(raw_base, sub, ses, "eeg", "*_events.tsv"))
    if not f: return None
    try:
        df = pd.read_csv(f[0], sep='\t')
        for col in ['trial_type', 'value']:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').dropna().astype(int).values
                valid = vals[(vals >= 1) & (vals <= 10)]
                if len(valid) >= 40:
                    # 3M 特有逻辑：按比例广播标签
                    labels = np.repeat(valid, np.ceil(num_trials/len(valid)))[:num_trials]
                    return (labels - 1).astype(int)
    except: pass
    return None

def run_preprocess():
    mat_files = glob.glob(os.path.join(derivatives_base, 'sub-*', 'ses-*', '*.mat'))
    stats = []

    for m_path in tqdm(mat_files):
        fname = os.path.basename(m_path).replace('.mat', '')
        sub, ses = fname.split('_')[0], fname.split('_')[1]
        
        try:
            mat = scipy.io.loadmat(m_path)
            data = np.transpose(mat['data'], (2, 0, 1)) # (N, 32, 1000)
            
            # 标签提取
            labels = get_labels(sub, ses, data.shape[0])
            if labels is None:
                stats.append({'file': fname, 'status': 'SKIP: No Labels'})
                continue

            # MNE 流程
            info = mne.create_info(ch_names=CH_NAMES, sfreq=500, ch_types='eeg')
            epochs = mne.EpochsArray(data, info, verbose=False)
            
            # 1. 减去均值基线 (解决 Baseline 报错)
            epochs.apply_baseline(baseline=(None, None), verbose=False)
            # 2. 重参考
            epochs.set_eeg_reference('average', verbose=False)
            # 3. 滤波
            epochs.filter(0.5, 100, method='iir', verbose=False)

            # 保存
            out_dir = os.path.join(save_path, sub)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{fname}.pkl"), 'wb') as f_out:
                pickle.dump({'data': epochs.get_data().astype(np.float32), 'label': labels}, f_out)
            
            stats.append({'file': fname, 'status': 'Success'})
        except Exception as e:
            stats.append({'file': fname, 'status': f'Error: {str(e)}'})

    df_res = pd.DataFrame(stats)
    print("\n结果统计:\n", df_res['status'].value_counts())

if __name__ == "__main__":
    run_preprocess()