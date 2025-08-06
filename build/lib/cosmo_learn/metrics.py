import numpy as np


def Ch2_H0(H0, errH0, refH0):
    dev_abs = np.abs(H0 - refH0)
    dev_rel = dev_abs/np.sqrt(errH0**2)
    return dev_rel**2

def D0(Qi, SigmaQi, Qj, SigmaQj):
    num = np.abs(Qi - Qj)
    den = np.sqrt( SigmaQi**2 + SigmaQj**2 )
    N_dat = (len(Qi) + len(Qj))/2 # Qi & Qj have same length so this don't matter
    return np.sum((num/den)**2)/N_dat

def D1(Qi, SigmaQi, Qj, SigmaQj):
    num = np.abs(Qi - Qj)
    den = np.sqrt(np.abs( SigmaQi**2 - SigmaQj**2 ))
    N_dat = (len(Qi) + len(Qj))/2 # Qi & Qj have same length so this don't matter
    return np.sum(num/den)/N_dat

def D2(Qi, SigmaQi, Qj, SigmaQj):
    t1 = np.abs(Qi - Qj)
    t2 = np.sqrt(np.abs( SigmaQi**2 - SigmaQj**2 ))
    N_dat = (len(Qi) + len(Qj))/2 # Qi & Qj have same length so this don't matter
    Sigma_Ave = np.sqrt((np.mean(SigmaQi)**2) + (np.mean(SigmaQj)**2))
    return np.sum(t1 + t2)/(N_dat*Sigma_Ave)

def DWstat(residuals):
    # DW statistic
    n = len(residuals)
    diff_residuals = [residuals[i] - residuals[i - 1] for i in range(1, n)]

    numerator = np.sum([diff_res ** 2 for diff_res in diff_residuals])

    # ebar = np.mean(residuals)
    # denominator = np.sum([(residual - ebar) ** 2 for residual in residuals])
    denominator = np.sum([(residual) ** 2 for residual in residuals])
    durbin_watson = numerator / denominator

    return durbin_watson


# for combining Hz and fs8z data
def get_metrics(bgData, ptData, HzTest, errHzTest, fs8zTest, errfs8zTest):
    D0_val = (D0(HzTest, errHzTest, bgData['test']['y'], bgData['test']['yerr']) + \
              D0(fs8zTest, errfs8zTest, ptData['test']['y'], ptData['test']['yerr']))/2

    D1_val = D1(HzTest, errHzTest, bgData['test']['y'], bgData['test']['yerr']) + \
    D1(fs8zTest, errfs8zTest, ptData['test']['y'], ptData['test']['yerr'])

    D2_val = (D2(HzTest, errHzTest, bgData['test']['y'], bgData['test']['yerr']) + \
              D2(fs8zTest, errfs8zTest, ptData['test']['y'], ptData['test']['yerr']))/2

    DW_val = (DWstat(bgData['test']['y'] - HzTest) + \
              DWstat(ptData['test']['y'] - fs8zTest))/2
    return D0_val, D1_val, D2_val, DW_val