"""Reproducible re-analysis of archived NYU raster outputs.

This script intentionally does NOT claim metric-depth accuracy. It evaluates the
archived scalar depth/disparity renderings against the archived NYU reference
renderings after per-image affine alignment, which is appropriate only as an
image-space/relative-depth diagnostic when the original metric tensors are not
available.
"""
from pathlib import Path
import csv, math, re
import numpy as np
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
TEX = (ROOT / "sn-article.tex").read_text(encoding="utf-8")
OUT = Path(__file__).resolve().parent

NAMES = {
    'tab:mono':'Tosi et al. monoResMatch',
    'tab:mono2':'Godard et al. 2019 / archived UnSupervisedMonocular',
    'tab:ranftl2021vision':'Ranftl et al. 2021 DPT (Jason)',
    'tab:monodepth17':'Godard et al. 2017 city2kitti',
    'tab:monodepth17-1':'Godard et al. 2017 city2eigen',
    'tab:monodepth17-2':'Godard et al. 2017 cityscapes',
    'tab:monodepth17-3':'Godard et al. 2017 eigen',
    'tab:monodepth17-4':'Godard et al. 2017 kitti',
    'tab:monodepth17-5':'Dai et al. fusion / LeRes50',
    'tab:dai2022multi':'Dai et al. fusion / MiDaS',
    'tab:watson2019self':'Watson et al. 2019 Depth Hints',
    'tab:ranftl2021vision-beto':'Ranftl et al. 2021 DPT (Beto)',
    'tab:ranftl2020towards':'Ranftl et al. 2020 MiDaS archived toward_robust',
    'tab:alhashim2018high-nyu':'Alhashim and Wonka NYU',
    'tab:alhashim2018high-kitti':'Alhashim and Wonka KITTI',
}

def scalar_image(path: Path):
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        rgb = a[..., :3].astype(np.float64)
        # Accept RGBA/RGB only when it is actually a grayscale scalar raster.
        if np.max(np.abs(rgb[...,0]-rgb[...,1])) > 1 or np.max(np.abs(rgb[...,0]-rgb[...,2])) > 1:
            raise ValueError(f"Non-grayscale color rendering cannot be treated as a scalar depth proxy: {path}")
        a = rgb[...,0]
    return a.astype(np.float64), im.mode

def resize_float(a, size):
    im = Image.fromarray(a.astype(np.float32), mode='F').resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float64)

def evaluate(pred, gt):
    if pred.shape != gt.shape:
        pred = resize_float(pred, (gt.shape[1], gt.shape[0]))
    g = gt.astype(np.float64) / 255.0
    p = pred.astype(np.float64)
    mask = np.isfinite(p) & np.isfinite(g)
    pv, gv = p[mask], g[mask]
    if pv.size == 0 or np.std(pv) < 1e-12 or np.std(gv) < 1e-12:
        return None

    # Per-image least-squares affine alignment: alpha*p + beta -> normalized reference.
    A = np.stack([pv, np.ones_like(pv)], axis=1)
    alpha, beta = np.linalg.lstsq(A, gv, rcond=None)[0]
    aligned_v = alpha * pv + beta
    residual = aligned_v - gv
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    # Absolute correlations make the diagnostic invariant to near/far rendering polarity.
    pearson_abs = float(abs(pearsonr(pv, gv).statistic))
    spearman_abs = float(abs(spearmanr(pv, gv).statistic))
    kendall_abs = float(abs(kendalltau(pv, gv, variant='b').statistic))

    aligned = alpha * p + beta
    ssim_aligned = float(structural_similarity(
        g, aligned, data_range=1.0, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False
    ))
    psnr_aligned = float(-20.0*np.log10(rmse)) if rmse > 0 else float('inf')
    return dict(alpha=float(alpha), beta=float(beta), polarity='same' if alpha >= 0 else 'reversed',
                aa_rmse=rmse, aa_mae=mae, pearson_abs=pearson_abs,
                spearman_abs=spearman_abs, kendall_abs=kendall_abs,
                aa_ssim=ssim_aligned, aa_psnr=psnr_aligned)

rows=[]
# Locate table blocks from their labels by taking the nearest preceding begin{table}.
for label, method in NAMES.items():
    pos = TEX.find('\\label{' + label + '}')
    if pos < 0:
        continue
    start = TEX.rfind('\\begin{table}', 0, pos)
    end = TEX.find('\\end{table}', pos)
    if start < 0 or end < 0:
        continue
    block = TEX[start:end + len('\\end{table}')]
    has_gray = '\\textbf{grayscale}' in block.lower() or '\\textbf{Grayscale}' in block
    pred_index = 2 if has_gray else 1
    for row in re.split(r'\\\\\s*(?:\\hline)?', block):
        if 'Groundtruth/GroundtruthNYU' not in row:
            continue
        paths = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', row)
        if len(paths) <= pred_index:
            continue
        pred_rel = paths[pred_index].replace('./','')
        gt_rel = [x for x in paths if 'Groundtruth/GroundtruthNYU' in x][-1].replace('./','')
        pred_path, gt_path = ROOT/pred_rel, ROOT/gt_rel
        rec=dict(label=label, method=method, prediction=pred_rel, ground_truth=gt_rel)
        try:
            pred, mode = scalar_image(pred_path)
            gt, _ = scalar_image(gt_path)
            rec.update(pred_mode=mode, pred_width=pred.shape[1], pred_height=pred.shape[0],
                       pred_min=float(np.nanmin(pred)), pred_max=float(np.nanmax(pred)))
            m=evaluate(pred,gt)
            if m is None:
                raise ValueError('Degenerate scalar prediction or reference')
            rec.update(m); rec['status']='ok'
        except Exception as exc:
            rec['status']='excluded'; rec['reason']=str(exc)
            for k in ['alpha','beta','aa_rmse','aa_mae','pearson_abs','spearman_abs','kendall_abs','aa_ssim','aa_psnr']:
                rec[k]=math.nan
            rec['polarity']='NA'
        rows.append(rec)

fields=sorted(set().union(*(r.keys() for r in rows)))
with open(OUT/'archived_relative_metrics_per_image.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

summary=[]
for label, method in NAMES.items():
    rr=[r for r in rows if r['label']==label]
    vv=[r for r in rr if r['status']=='ok']
    d=dict(label=label,method=method,n_valid=len(vv),n_total=len(rr),
           reversed_count=sum(r.get('polarity')=='reversed' for r in vv),rankable=len(vv)>=4)
    for k in ['aa_rmse','aa_mae','pearson_abs','spearman_abs','kendall_abs','aa_ssim','aa_psnr']:
        vals=np.asarray([r[k] for r in vv],float)
        d[k+'_mean']=float(np.mean(vals)) if len(vals) else math.nan
        d[k+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else math.nan
    summary.append(d)
with open(OUT/'archived_relative_metrics_summary.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

print('Rankable configurations:', sum(d['rankable'] for d in summary), '/', len(summary))
for d in sorted((d for d in summary if d['rankable']), key=lambda z:z['aa_rmse_mean']):
    print(f"{d['method']}: AA-RMSE={d['aa_rmse_mean']:.4f}, Spearman={d['spearman_abs_mean']:.4f}, AA-SSIM={d['aa_ssim_mean']:.4f}")
