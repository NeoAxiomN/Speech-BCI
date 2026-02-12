import pickle
import os
import glob
import csv

BASE_PATH = '/mnt/main/LinJunyi/DATA/chisco/derivatives/preprocessed_pkl'
SAVE_PATH = './chisco_imagine_only_audit.csv'

def scroll_audit_imagine():
    # 核心：只匹配包含 task-imagine 的文件
    all_files = sorted(glob.glob(os.path.join(BASE_PATH, "sub-*/eeg/*task-imagine*.pkl")))
    total_files = len(all_files)
    
    header = ['Subject', 'File', 'Count', 'C', 'H', 'W']
    total_trials = 0
    reference_shape = None

    print(f"🎯 Chisco 审计：仅针对 Imagine 任务 (共 {total_files} 个文件)")
    print(f"{'Index':<6} | {'File Name':<40} | {'Trials':<6} | {'Shape':<15} | {'Status'}")
    print("-" * 85)
    
    with open(SAVE_PATH, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for idx, f_path in enumerate(all_files):
            f_name = os.path.basename(f_path)
            sub_id = f_name.split('_')[0]
            
            try:
                with open(f_path, 'rb') as f:
                    data_list = pickle.load(f)
                
                if data_list:
                    feat = data_list[0]['input_features']
                    c, h, w = feat.shape
                    current_shape = (c, h, w)
                    count = len(data_list)
                    
                    if reference_shape is None:
                        reference_shape = current_shape
                    
                    status = "OK" if current_shape == reference_shape else f"⚠️ MISMATCH {reference_shape}"
                    
                    # 滚动输出
                    print(f"[{idx+1:03d}/{total_files:03d}] | {f_name:<40} | {count:<6} | {str(current_shape):<15} | {status}")

                    writer.writerow({
                        'Subject': sub_id, 'File': f_name, 
                        'Count': count, 'C': c, 'H': h, 'W': w
                    })
                    total_trials += count
                
                del data_list
            except Exception as e:
                print(f"[{idx+1:03d}/{total_files:03d}] | {f_name:<40} | ERROR: {str(e)[:20]}")

    print("-" * 85)
    print(f"✅ 审计完成！Imagine 总 Trials: {total_trials}")

if __name__ == "__main__":
    scroll_audit_imagine()