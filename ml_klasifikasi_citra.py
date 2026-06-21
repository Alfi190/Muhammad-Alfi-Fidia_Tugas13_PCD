import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, time, warnings, itertools
warnings.filterwarnings('ignore')

from PIL import Image, ImageDraw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, GridSearchCV, learning_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2hsv, rgb2gray
import scipy.ndimage as ndi

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 100
})

OUTPUT_DIR = 'output'
DATASET_DIR = 'dataset'
IMG_SIZE = 64
N_SAMPLES = 50
CLASSES = ['lingkaran', 'persegi', 'segitiga', 'bintang', 'salib']
COLORS_MAP = ['#4e79a7','#f28e2b','#59a14f','#e15759','#76b7b2']
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

np.random.seed(42)

# =============================================================================
# 1. GENERATE DATASET
# =============================================================================

def add_noise(arr, level=12):
    noise = np.random.randint(-level, level, arr.shape)
    return np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)

def gen_circle(i, size=IMG_SIZE):
    img = Image.new('RGB', (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    r  = np.random.randint(14, 24)
    cx = np.random.randint(r+6, size-r-6)
    cy = np.random.randint(r+6, size-r-6)
    c  = (np.random.randint(30,100), np.random.randint(80,160), np.random.randint(180,255))
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=c, outline=(20,20,20), width=2)
    return add_noise(np.array(img))

def gen_square(i, size=IMG_SIZE):
    img = Image.new('RGB', (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    s   = np.random.randint(14, 23)
    ang = np.random.randint(-20, 20)
    cx  = np.random.randint(s+6, size-s-6)
    cy  = np.random.randint(s+6, size-s-6)
    c   = (np.random.randint(180,255), np.random.randint(50,120), np.random.randint(30,100))
    pts = np.array([[-s,-s],[s,-s],[s,s],[-s,s]], float)
    rad = np.radians(ang)
    rot = np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    pts = pts @ rot.T + [cx, cy]
    draw.polygon([tuple(p) for p in pts], fill=c, outline=(20,20,20), width=2)
    return add_noise(np.array(img))

def gen_triangle(i, size=IMG_SIZE):
    img = Image.new('RGB', (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    s   = np.random.randint(14, 20)
    cx  = np.random.randint(s+6, size-s-6)
    cy  = np.random.randint(s+6, size-s-6)
    off = np.random.randint(0, 360)
    c   = (np.random.randint(30,100), np.random.randint(160,240), np.random.randint(60,130))
    pts = [(cx + s*np.cos(np.radians(off + k*120)),
            cy + s*np.sin(np.radians(off + k*120))) for k in range(3)]
    draw.polygon(pts, fill=c, outline=(20,20,20), width=2)
    return add_noise(np.array(img))

def gen_star(i, size=IMG_SIZE):
    img = Image.new('RGB', (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    ro  = np.random.randint(14, 20)
    ri  = max(5, ro // 2 + np.random.randint(-2, 3))
    cx  = np.random.randint(ro+6, size-ro-6)
    cy  = np.random.randint(ro+6, size-ro-6)
    off = np.random.randint(0, 72)
    c   = (np.random.randint(200,255), np.random.randint(150,210), np.random.randint(0,60))
    pts = []
    for k in range(10):
        r   = ro if k % 2 == 0 else ri
        ang = np.radians(off + k*36 - 90)
        pts.append((cx + r*np.cos(ang), cy + r*np.sin(ang)))
    draw.polygon(pts, fill=c, outline=(20,20,20), width=2)
    return add_noise(np.array(img))

def gen_cross(i, size=IMG_SIZE):
    img = Image.new('RGB', (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    s   = np.random.randint(12, 20)
    w   = np.random.randint(5, 9)
    cx  = np.random.randint(s+6, size-s-6)
    cy  = np.random.randint(s+6, size-s-6)
    c   = (np.random.randint(100,180), np.random.randint(40,100), np.random.randint(160,255))
    draw.rectangle([cx-w, cy-s, cx+w, cy+s], fill=c, outline=(20,20,20), width=2)
    draw.rectangle([cx-s, cy-w, cx+s, cy+w], fill=c, outline=(20,20,20), width=2)
    return add_noise(np.array(img))

GENERATORS = [gen_circle, gen_square, gen_triangle, gen_star, gen_cross]

def generate_dataset():
    print("=" * 60)
    print("  MEMBUAT DATASET CITRA GEOMETRI")
    print("=" * 60)
    images, labels = [], []
    for cls_idx, (cls_name, gen) in enumerate(zip(CLASSES, GENERATORS)):
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(N_SAMPLES):
            arr = gen(i)
            images.append(arr)
            labels.append(cls_idx)
            Image.fromarray(arr).save(os.path.join(cls_dir, f'{cls_name}_{i:03d}.png'))
        print(f"  ✓ {cls_name:12s} : {N_SAMPLES} sampel disimpan")
    print(f"\n  Total: {len(images)} citra | {len(CLASSES)} kelas")
    return np.array(images), np.array(labels)

# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

def extract_hog_features(img_arr):
    gray = rgb2gray(img_arr)
    feat, _ = hog(gray, orientations=8, pixels_per_cell=(8,8),
                  cells_per_block=(2,2), visualize=True)
    return feat

def extract_color_histogram(img_arr, bins=16):
    hsv = rgb2hsv(img_arr)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:,:,ch], bins=bins, range=(0,1))
        feats.extend(hist / (hist.sum() + 1e-8))
    return np.array(feats)

def extract_hu_moments(img_arr):
    from skimage.measure import moments, moments_central, moments_normalized, moments_hu
    gray = rgb2gray(img_arr)
    m    = moments(gray)
    cr   = m[0, 0]
    cx   = m[1, 0] / cr if cr else 0
    cy   = m[0, 1] / cr if cr else 0
    mu   = moments_central(gray, center=(cx, cy))
    nu   = moments_normalized(mu)
    hu   = moments_hu(nu)
    log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return log_hu

def extract_lbp_features(img_arr, P=8, R=1, bins=16):
    gray = (rgb2gray(img_arr) * 255).astype(np.uint8)
    lbp  = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, P+2))
    return hist / (hist.sum() + 1e-8)

def extract_all_features(images):
    print("\n  Mengekstrak fitur...")
    features = []
    for img in images:
        hog_f   = extract_hog_features(img)
        color_f = extract_color_histogram(img)
        hu_f    = extract_hu_moments(img)
        lbp_f   = extract_lbp_features(img)
        combined = np.concatenate([hog_f, color_f, hu_f, lbp_f])
        features.append(combined)
    feat_arr = np.array(features)
    print(f"  ✓ HOG       : {extract_hog_features(images[0]).shape[0]} dimensi")
    print(f"  ✓ Color HSV : {extract_color_histogram(images[0]).shape[0]} dimensi")
    print(f"  ✓ Hu Moments: {extract_hu_moments(images[0]).shape[0]} dimensi")
    print(f"  ✓ LBP       : {extract_lbp_features(images[0]).shape[0]} dimensi")
    print(f"  Total fitur : {feat_arr.shape[1]} dimensi per sampel")
    return feat_arr

# =============================================================================
# 3. VISUALIZATION - DATASET SAMPLES
# =============================================================================

def plot_dataset_samples(images, labels):
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    fig.suptitle('Sampel Dataset – 5 Kelas Geometri (8 sampel/kelas)', fontsize=14, fontweight='bold', y=1.01)
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_imgs = images[labels == cls_idx][:8]
        for j, img in enumerate(cls_imgs):
            ax = axes[cls_idx, j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(cls_name, fontsize=10, fontweight='bold', rotation=90, labelpad=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_dataset_samples.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot sampel dataset disimpan")

# =============================================================================
# 4. KNN ANALYSIS
# =============================================================================

def run_knn_analysis(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  ANALISIS K-NEAREST NEIGHBORS")
    print("="*60)

    K_VALUES  = [1, 3, 5, 7, 9, 11]
    METRICS   = ['euclidean', 'manhattan', 'minkowski']
    results   = {}

    for metric in METRICS:
        results[metric] = {}
        for k in K_VALUES:
            t0 = time.time()
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric, p=3 if metric=='minkowski' else 2)
            clf.fit(X_train, y_train)
            t_train = time.time() - t0

            t0 = time.time()
            y_pred = clf.predict(X_test)
            t_inf  = time.time() - t0

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # CV score
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_full = np.concatenate([X_train, X_test])
            cv_lbl  = np.concatenate([y_train, y_test])
            cv_scores = cross_val_score(clf, cv_full, cv_lbl, cv=skf, scoring='accuracy')

            results[metric][k] = {
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                't_train': t_train, 't_inf': t_inf,
                'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
                'y_pred': y_pred
            }
            print(f"  KNN k={k:2d} | {metric:10s} | Acc={acc:.3f} | F1={f1:.3f} | CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    return results

def plot_knn_results(knn_results, y_test):
    K_VALUES = [1, 3, 5, 7, 9, 11]
    METRICS  = ['euclidean', 'manhattan', 'minkowski']
    MNAMES   = ['Euclidean', 'Manhattan', 'Minkowski (p=3)']
    mcolors  = ['#4e79a7', '#f28e2b', '#59a14f']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Analisis KNN: Pengaruh k dan Metrik Jarak', fontsize=14, fontweight='bold')

    # Accuracy vs k
    ax = axes[0]
    for metric, mname, mc in zip(METRICS, MNAMES, mcolors):
        accs = [knn_results[metric][k]['acc'] for k in K_VALUES]
        ax.plot(K_VALUES, accs, 'o-', color=mc, label=mname, linewidth=2, markersize=7)
    ax.set_xlabel('Nilai k'); ax.set_ylabel('Akurasi')
    ax.set_title('Akurasi vs Nilai k'); ax.legend(); ax.set_xticks(K_VALUES)
    ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3)

    # CV Mean ± std
    ax = axes[1]
    x = np.arange(len(K_VALUES)); w = 0.25
    for i, (metric, mname, mc) in enumerate(zip(METRICS, MNAMES, mcolors)):
        means = [knn_results[metric][k]['cv_mean'] for k in K_VALUES]
        stds  = [knn_results[metric][k]['cv_std']  for k in K_VALUES]
        ax.bar(x + i*w, means, width=w, color=mc, alpha=0.8, label=mname, yerr=stds, capsize=3)
    ax.set_xlabel('Nilai k'); ax.set_ylabel('CV Accuracy (5-fold)')
    ax.set_title('Cross-Validation Accuracy'); ax.legend(fontsize=8)
    ax.set_xticks(x + w); ax.set_xticklabels([f'k={k}' for k in K_VALUES])
    ax.set_ylim(0.5, 1.05); ax.grid(alpha=0.3, axis='y')

    # Overfitting analysis (train CV vs test acc)
    ax = axes[2]
    best_metric = 'euclidean'
    train_accs  = []
    test_accs   = []
    for k in K_VALUES:
        r = knn_results[best_metric][k]
        train_accs.append(r['cv_mean'])
        test_accs.append(r['acc'])
    gap = [abs(tr - te) for tr, te in zip(train_accs, test_accs)]
    ax.plot(K_VALUES, train_accs, 'o-', color='#4e79a7', label='CV Accuracy', linewidth=2, markersize=7)
    ax.plot(K_VALUES, test_accs,  's--', color='#e15759', label='Test Accuracy', linewidth=2, markersize=7)
    ax.fill_between(K_VALUES, train_accs, test_accs, alpha=0.15, color='gray', label='Generalization Gap')
    ax.set_xlabel('Nilai k'); ax.set_ylabel('Akurasi')
    ax.set_title('Overfitting vs Underfitting (Euclidean)')
    ax.legend(); ax.set_xticks(K_VALUES); ax.grid(alpha=0.3)
    ax.annotate('↑ Overfitting (k kecil)', xy=(1, test_accs[0]), xytext=(3, test_accs[0]+0.03),
                fontsize=8, color='#e15759', arrowprops=dict(arrowstyle='->', color='#e15759'))
    ax.annotate('Underfitting (k besar) ↑', xy=(11, test_accs[-1]), xytext=(7, test_accs[-1]-0.06),
                fontsize=8, color='#4e79a7')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_knn_analysis.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot analisis KNN disimpan")

# =============================================================================
# 5. SVM ANALYSIS
# =============================================================================

def run_svm_analysis(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("  ANALISIS SUPPORT VECTOR MACHINE")
    print("="*60)

    # --- Kernel comparison ---
    C_VALUES     = [0.1, 1, 10, 100]
    GAMMA_VALUES = [0.001, 0.01, 0.1, 1]
    KERNELS      = ['linear', 'poly', 'rbf']
    results      = {}

    # Best per kernel (default params)
    for kernel in KERNELS:
        results[kernel] = {}
        for C in C_VALUES:
            if kernel == 'rbf':
                gammas = GAMMA_VALUES
            else:
                gammas = [0.01]
            for gamma in gammas:
                key = f'C={C}_g={gamma}'
                t0 = time.time()
                clf = SVC(kernel=kernel, C=C, gamma=gamma if kernel=='rbf' else 'scale',
                          probability=True, random_state=42)
                clf.fit(X_train, y_train)
                t_train = time.time() - t0

                t0 = time.time()
                y_pred = clf.predict(X_test)
                t_inf  = time.time() - t0

                acc  = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                results[kernel][key] = {
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                    't_train': t_train, 't_inf': t_inf,
                    'C': C, 'gamma': gamma, 'y_pred': y_pred, 'clf': clf
                }
                if kernel == 'rbf':
                    print(f"  SVM {kernel:6s} C={C:5.1f} γ={gamma:.3f} | Acc={acc:.3f} | F1={f1:.3f}")
                elif key == f'C={C}_g=0.01':
                    print(f"  SVM {kernel:6s} C={C:5.1f}         | Acc={acc:.3f} | F1={f1:.3f}")

    return results

def plot_svm_results(svm_results, X_train, X_test, y_train, y_test):
    C_VALUES     = [0.1, 1, 10, 100]
    GAMMA_VALUES = [0.001, 0.01, 0.1, 1]
    KERNELS      = ['linear', 'poly', 'rbf']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analisis Support Vector Machine', fontsize=14, fontweight='bold')

    # 1. Accuracy per kernel vs C
    ax = axes[0, 0]
    kcolors = ['#4e79a7', '#f28e2b', '#59a14f']
    for kernel, kc in zip(KERNELS, kcolors):
        accs = [svm_results[kernel][f'C={C}_g=0.01']['acc'] for C in C_VALUES]
        ax.semilogx(C_VALUES, accs, 'o-', color=kc, label=kernel.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('Nilai C (log scale)'); ax.set_ylabel('Akurasi')
    ax.set_title('Pengaruh Parameter C terhadap Akurasi')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xticks(C_VALUES); ax.set_xticklabels(C_VALUES)

    # 2. RBF heatmap: C vs gamma
    ax = axes[0, 1]
    heat = np.zeros((len(GAMMA_VALUES), len(C_VALUES)))
    for i, gamma in enumerate(GAMMA_VALUES):
        for j, C in enumerate(C_VALUES):
            heat[i, j] = svm_results['rbf'][f'C={C}_g={gamma}']['acc']
    im = ax.imshow(heat, cmap='YlOrRd', aspect='auto', vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(C_VALUES)));     ax.set_xticklabels([f'C={c}' for c in C_VALUES])
    ax.set_yticks(range(len(GAMMA_VALUES))); ax.set_yticklabels([f'γ={g}' for g in GAMMA_VALUES])
    ax.set_title('SVM RBF – Heatmap Akurasi (C × γ)')
    for i in range(len(GAMMA_VALUES)):
        for j in range(len(C_VALUES)):
            ax.text(j, i, f'{heat[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='black' if heat[i,j]<0.8 else 'white', fontweight='bold')

    # 3. Decision boundary PCA 2D (RBF best)
    ax = axes[1, 0]
    pca   = PCA(n_components=2, random_state=42)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    X2d   = pca.fit_transform(X_all)
    X2d_tr, X2d_te = X2d[:len(X_train)], X2d[len(X_train):]

    clf_rbf = SVC(kernel='rbf', C=10, gamma=0.01, random_state=42)
    clf_rbf.fit(X2d_tr, y_train)

    x_min, x_max = X2d[:,0].min()-1, X2d[:,0].max()+1
    y_min, y_max = X2d[:,1].min()-1, X2d[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap='Set1', levels=np.arange(-0.5, 5.5, 1))
    for cls_idx, (cls_name, cc) in enumerate(zip(CLASSES, COLORS_MAP)):
        mask = y_all == cls_idx
        ax.scatter(X2d[mask, 0], X2d[mask, 1], c=cc, s=25, alpha=0.8,
                   edgecolors='white', linewidths=0.5, label=cls_name)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Decision Boundary SVM RBF (PCA 2D)')
    ax.legend(fontsize=8, markerscale=1.5)

    # 4. Training time comparison
    ax = axes[1, 1]
    kernel_times = []
    kernel_accs  = []
    for kernel in KERNELS:
        times = [svm_results[kernel][f'C={C}_g=0.01']['t_train'] for C in C_VALUES]
        accs  = [svm_results[kernel][f'C={C}_g=0.01']['acc']     for C in C_VALUES]
        kernel_times.extend(times)
        kernel_accs.extend(accs)

    x = np.arange(len(C_VALUES)); w = 0.25
    for i, (kernel, kc) in enumerate(zip(KERNELS, kcolors)):
        times = [svm_results[kernel][f'C={C}_g=0.01']['t_train'] for C in C_VALUES]
        ax.bar(x + i*w, times, width=w, color=kc, alpha=0.8, label=kernel.upper())
    ax.set_xlabel('Nilai C'); ax.set_ylabel('Waktu Training (s)')
    ax.set_title('Waktu Training per Kernel dan C')
    ax.legend(); ax.set_xticks(x + w); ax.set_xticklabels([f'C={c}' for c in C_VALUES])
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_svm_analysis.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot analisis SVM disimpan")

# =============================================================================
# 6. CONFUSION MATRIX & EVALUATION
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im  = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30, ha='right')
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    ax.set_title(title)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=120)
    plt.close()

def plot_roc_curves(clf, X_test, y_test, title, filename):
    n_cls = len(CLASSES)
    y_bin = label_binarize(y_test, classes=range(n_cls))
    y_score = clf.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    auc_vals = []
    for i, (cls_name, cc) in enumerate(zip(CLASSES, COLORS_MAP)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        ax.plot(fpr, tpr, color=cc, linewidth=2, label=f'{cls_name} (AUC={roc_auc:.3f})')
    ax.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve – One vs Rest\n{title} | Mean AUC={np.mean(auc_vals):.3f}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=120)
    plt.close()
    return np.mean(auc_vals)

# =============================================================================
# 7. CROSS-VALIDATION & LEARNING CURVE
# =============================================================================

def run_cross_validation(X, y):
    print("\n" + "="*60)
    print("  CROSS-VALIDATION & LEARNING CURVE")
    print("="*60)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Best KNN
    knn_best = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_cv   = cross_val_score(knn_best, X, y, cv=skf, scoring='accuracy')

    # Best SVM
    svm_best = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    svm_cv   = cross_val_score(svm_best, X, y, cv=skf, scoring='accuracy')

    print(f"  KNN (k=5, Euclidean) CV-10: {knn_cv.mean():.3f} ± {knn_cv.std():.3f}")
    print(f"  SVM (RBF, C=10, γ=0.01) CV-10: {svm_cv.mean():.3f} ± {svm_cv.std():.3f}")

    # Learning curves
    train_sizes = np.linspace(0.1, 1.0, 8)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Curve & Cross-Validation', fontsize=14, fontweight='bold')

    for ax, (clf, name, color) in zip(axes, [
        (knn_best, 'KNN (k=5, Euclidean)', '#4e79a7'),
        (svm_best, 'SVM (RBF, C=10, γ=0.01)', '#e15759')
    ]):
        tr_sz, tr_sc, val_sc = learning_curve(
            clf, X, y, train_sizes=train_sizes, cv=skf5, scoring='accuracy',
            shuffle=True, random_state=42, n_jobs=1)

        tr_mean, tr_std  = tr_sc.mean(1), tr_sc.std(1)
        val_mean, val_std = val_sc.mean(1), val_sc.std(1)

        ax.plot(tr_sz, tr_mean, 'o-', color=color, label='Training Accuracy', linewidth=2)
        ax.fill_between(tr_sz, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color=color)
        ax.plot(tr_sz, val_mean, 's--', color='gray', label='Validation Accuracy', linewidth=2)
        ax.fill_between(tr_sz, val_mean-val_std, val_mean+val_std, alpha=0.1, color='gray')
        ax.set_xlabel('Jumlah Sampel Training')
        ax.set_ylabel('Akurasi')
        ax.set_title(f'Learning Curve – {name}')
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_learning_curve.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot learning curve disimpan")
    return knn_cv, svm_cv

# =============================================================================
# 8. HYPERPARAMETER TUNING (GridSearchCV)
# =============================================================================

def run_gridsearch(X_train, y_train):
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # KNN Grid
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {'n_neighbors': [1,3,5,7,9,11], 'metric': ['euclidean','manhattan']},
        cv=skf, scoring='accuracy', n_jobs=1)
    knn_grid.fit(X_train, y_train)
    print(f"  KNN Best Params : {knn_grid.best_params_}")
    print(f"  KNN Best CV Acc : {knn_grid.best_score_:.3f}")

    # SVM Grid
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {'C': [0.1,1,10,100], 'kernel': ['linear','rbf'],
         'gamma': ['scale',0.01,0.1]},
        cv=skf, scoring='accuracy', n_jobs=1)
    svm_grid.fit(X_train, y_train)
    print(f"  SVM Best Params : {svm_grid.best_params_}")
    print(f"  SVM Best CV Acc : {svm_grid.best_score_:.3f}")

    # Plot GridSearch results
    plot_gridsearch(knn_grid, svm_grid)
    return knn_grid.best_estimator_, svm_grid.best_estimator_, knn_grid, svm_grid

def plot_gridsearch(knn_grid, svm_grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hyperparameter Tuning – GridSearchCV', fontsize=14, fontweight='bold')

    # KNN
    ax = axes[0]
    df = knn_grid.cv_results_
    ks  = [p['n_neighbors'] for p in df['params']]
    mts = [p['metric']      for p in df['params']]
    scores = df['mean_test_score']
    for metric, mc in zip(['euclidean','manhattan'], ['#4e79a7','#f28e2b']):
        mask = [m == metric for m in mts]
        k_vals  = [k for k, m in zip(ks, mask) if m]
        sc_vals = [s for s, m in zip(scores, mask) if m]
        ax.plot(k_vals, sc_vals, 'o-', color=mc, label=metric.capitalize(), linewidth=2, markersize=8)
    ax.set_xlabel('k'); ax.set_ylabel('CV Accuracy')
    ax.set_title('KNN GridSearch'); ax.legend(); ax.grid(alpha=0.3)
    ax.axvline(x=knn_grid.best_params_['n_neighbors'], color='red', linestyle='--', alpha=0.6, label='Best k')

    # SVM
    ax = axes[1]
    df2    = svm_grid.cv_results_
    params = df2['params']
    scores2= df2['mean_test_score']
    c_vals = sorted(set(p['C'] for p in params))
    for kernel, kc in zip(['linear','rbf'], ['#4e79a7','#e15759']):
        sc_list = []
        for C in c_vals:
            best_s = max(scores2[i] for i, p in enumerate(params)
                         if p['C']==C and p['kernel']==kernel)
            sc_list.append(best_s)
        ax.semilogx(c_vals, sc_list, 'o-', color=kc, label=kernel.upper(), linewidth=2, markersize=8)
    ax.set_xlabel('C (log scale)'); ax.set_ylabel('CV Accuracy')
    ax.set_title('SVM GridSearch'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_gridsearch.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot GridSearch disimpan")

# =============================================================================
# 9. COMPREHENSIVE COMPARISON
# =============================================================================

def plot_final_comparison(knn_best, svm_best, X_test, y_test):
    methods = []
    accs, precs, recs, f1s, t_trains, t_infs = [], [], [], [], [], []

    for name, clf in [('KNN (k=5)', knn_best), ('SVM (RBF)', svm_best)]:
        t0 = time.time(); y_pred = clf.predict(X_test); t_i = time.time() - t0
        methods.append(name)
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        t_infs.append(t_i)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Perbandingan Komprehensif KNN vs SVM', fontsize=14, fontweight='bold')

    # Metrics bar
    ax = axes[0]
    metrics_names = ['Accuracy','Precision','Recall','F1-Score']
    knn_vals = [accs[0], precs[0], recs[0], f1s[0]]
    svm_vals = [accs[1], precs[1], recs[1], f1s[1]]
    x = np.arange(len(metrics_names)); w = 0.3
    ax.bar(x - w/2, knn_vals, w, color='#4e79a7', label='KNN', alpha=0.85)
    ax.bar(x + w/2, svm_vals, w, color='#e15759', label='SVM', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('Metrik Evaluasi'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for i, (kv, sv) in enumerate(zip(knn_vals, svm_vals)):
        ax.text(i-w/2, kv+0.01, f'{kv:.3f}', ha='center', fontsize=8, fontweight='bold')
        ax.text(i+w/2, sv+0.01, f'{sv:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Per-class F1
    ax = axes[1]
    knn_pred = knn_best.predict(X_test)
    svm_pred = svm_best.predict(X_test)
    knn_f1_cls = f1_score(y_test, knn_pred, average=None, zero_division=0)
    svm_f1_cls = f1_score(y_test, svm_pred, average=None, zero_division=0)
    x = np.arange(len(CLASSES)); w = 0.3
    ax.bar(x - w/2, knn_f1_cls, w, color='#4e79a7', label='KNN', alpha=0.85)
    ax.bar(x + w/2, svm_f1_cls, w, color='#e15759', label='SVM', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=20, ha='right')
    ax.set_ylim(0, 1.15); ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score per Kelas'); ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Speed comparison
    ax = axes[2]
    cats = ['Inference Time (s)']
    knn_speed = [t_infs[0]]
    svm_speed = [t_infs[1]]
    x = np.arange(len(cats)); w = 0.3
    b1 = ax.bar(x - w/2, knn_speed, w, color='#4e79a7', label='KNN', alpha=0.85)
    b2 = ax.bar(x + w/2, svm_speed, w, color='#e15759', label='SVM', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('Waktu (detik)')
    ax.set_title('Perbandingan Kecepatan Inference')
    ax.legend()
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                            f'{bar.get_height():.4f}s', ha='center', fontsize=9, fontweight='bold')
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                            f'{bar.get_height():.4f}s', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_final_comparison.png'), bbox_inches='tight', dpi=120)
    plt.close()
    print("  ✓ Plot perbandingan final disimpan")

# =============================================================================
# 10. SUMMARY REPORT
# =============================================================================

def print_summary(knn_results, svm_results, knn_cv, svm_cv,
                  knn_best, svm_best, X_test, y_test, knn_grid, svm_grid):
    print("\n" + "="*60)
    print("  RINGKASAN HASIL & ANALISIS")
    print("="*60)

    knn_pred = knn_best.predict(X_test)
    svm_pred = svm_best.predict(X_test)

    knn_acc = accuracy_score(y_test, knn_pred)
    svm_acc = accuracy_score(y_test, svm_pred)
    knn_f1  = f1_score(y_test, knn_pred, average='weighted', zero_division=0)
    svm_f1  = f1_score(y_test, svm_pred, average='weighted', zero_division=0)

    print(f"\n  📊 KNN TERBAIK (k=5, Euclidean)")
    print(f"     Accuracy  : {knn_acc:.4f}")
    print(f"     F1-Score  : {knn_f1:.4f}")
    print(f"     CV-10     : {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")
    print(f"     Best Params: {knn_grid.best_params_}")

    print(f"\n  📊 SVM TERBAIK (RBF, C=10, γ=0.01)")
    print(f"     Accuracy  : {svm_acc:.4f}")
    print(f"     F1-Score  : {svm_f1:.4f}")
    print(f"     CV-10     : {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
    print(f"     Best Params: {svm_grid.best_params_}")

    winner = "SVM" if svm_f1 > knn_f1 else "KNN"
    print(f"\n  🏆 METODE TERBAIK: {winner}")
    print(f"\n  ✅ REKOMENDASI:")
    print(f"     • Untuk akurasi tinggi          → SVM dengan kernel RBF")
    print(f"     • Untuk inferensi cepat          → KNN dengan k kecil")
    print(f"     • Untuk dataset kecil/sederhana  → KNN (mudah diimplementasi)")
    print(f"     • Untuk produksi/deployment      → SVM (lebih stabil & generalizable)")

    print(f"\n  📂 File output disimpan di: {OUTPUT_DIR}")
    print("="*60)

    # Classification report
    print("\n  CLASSIFICATION REPORT – KNN:")
    print(classification_report(y_test, knn_pred, target_names=CLASSES))
    print("\n  CLASSIFICATION REPORT – SVM:")
    print(classification_report(y_test, svm_pred, target_names=CLASSES))

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  KLASIFIKASI CITRA: KNN & SVM")
    print("  Dataset: Geometri Sintetis | 5 kelas × 50 sampel")
    print("="*60)

    # 1. Generate dataset
    images, labels = generate_dataset()

    # 2. Plot sample
    plot_dataset_samples(images, labels)

    # 3. Feature extraction
    X = extract_all_features(images)

    # 4. Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.2, random_state=42, stratify=labels)
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # 6. KNN
    knn_results = run_knn_analysis(X_train, X_test, y_train, y_test)
    plot_knn_results(knn_results, y_test)

    # 7. SVM
    svm_results = run_svm_analysis(X_train, X_test, y_train, y_test)
    plot_svm_results(svm_results, X_train, X_test, y_train, y_test)

    # 8. Best models for confusion matrix & ROC
    print("\n  Membuat Confusion Matrix & ROC Curve...")
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_clf.fit(X_train, y_train)
    knn_pred = knn_clf.predict(X_test)

    svm_clf = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_test)

    plot_confusion_matrix(y_test, knn_pred, 'Confusion Matrix – KNN (k=5, Euclidean)',
                          '04_cm_knn.png')
    plot_confusion_matrix(y_test, svm_pred, 'Confusion Matrix – SVM (RBF, C=10)',
                          '05_cm_svm.png')
    print("  ✓ Confusion matrix disimpan")

    auc_knn = plot_roc_curves(knn_clf, X_test, y_test, 'KNN (k=5)', '04b_roc_knn.png')
    auc_svm = plot_roc_curves(svm_clf, X_test, y_test, 'SVM (RBF)', '05b_roc_svm.png')
    print(f"  ✓ ROC curve disimpan | KNN AUC={auc_knn:.3f} | SVM AUC={auc_svm:.3f}")

    # 9. Cross-validation & Learning Curve
    knn_cv, svm_cv = run_cross_validation(X_scaled, labels)

    # 10. GridSearchCV
    knn_best, svm_best, knn_grid, svm_grid = run_gridsearch(X_train, y_train)

    # 11. Final comparison
    knn_best.fit(X_train, y_train)
    svm_best.fit(X_train, y_train)
    plot_final_comparison(knn_best, svm_best, X_test, y_test)

    # 12. Summary
    print_summary(knn_results, svm_results, knn_cv, svm_cv,
                  knn_best, svm_best, X_test, y_test, knn_grid, svm_grid)

if __name__ == '__main__':
    main()
