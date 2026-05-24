
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from skimage.feature import hog, local_binary_pattern
from skimage import io, color, transform, exposure
import time
import os
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = r"C:\Users\Mujgan\Downloads\archive"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_IDX   = {name: i for i, name in enumerate(EMOTION_LABELS)}
IMG_SIZE = 48

# HOG parameters
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP parameters
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
LBP_N_BINS = LBP_POINTS + 2

def load_split(split_dir: str):
    images, labels = [], []
    for emotion in EMOTION_LABELS:
        folder = os.path.join(split_dir, emotion)
        if not os.path.isdir(folder):
            print(f"  [WARNING] Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                img = io.imread(fpath)
                if img.ndim == 3:
                    img = color.rgb2gray(img)
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
                images.append(img)
                labels.append(LABEL_TO_IDX[emotion])
            except Exception as e:
                print(f"  [ERROR] {fpath}: {e}")
    return np.array(images), np.array(labels)


def load_dataset(dataset_path: str):
    print("[INFO] Loading training data...")
    X_train, y_train = load_split(os.path.join(dataset_path, "train"))
    print(f"  Train: {len(X_train)} images")

    print("[INFO] Loading test data...")
    X_test, y_test = load_split(os.path.join(dataset_path, "test"))
    print(f"  Test:  {len(X_test)} images")

    return X_train, y_train, X_test, y_test

def extract_hog_features(image: np.ndarray) -> np.ndarray:
    return hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=False,
    )


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(image, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_BINS + 1),
                           range=(0, LBP_N_BINS))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_all(images: np.ndarray, desc: str = ""):

    hog_feats, lbp_feats = [], []
    n = len(images)
    for i, img in enumerate(images):
        if (i + 1) % 2000 == 0 or i == n - 1:
            print(f"  [{desc}] {i+1}/{n}")
        hog_feats.append(extract_hog_features(img))
        lbp_feats.append(extract_lbp_features(img))

    hog_feats = np.array(hog_feats)
    lbp_feats = np.array(lbp_feats)
    combined  = np.concatenate([hog_feats, lbp_feats], axis=1)
    return hog_feats, lbp_feats, combined



def train_and_evaluate(X_train, y_train, X_test, y_test, label: str):

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    print(f"\n[INFO] Training SVM — {label} (feature dim: {X_train.shape[1]})...")
    t0 = time.time()
    model = SVC(C=10, kernel="rbf", gamma="scale",
                class_weight="balanced", probability=False)
    model.fit(X_tr_sc, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    y_pred = model.predict(X_te_sc)
    acc    = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    prec   = precision_score(y_test, y_pred, average="weighted")
    rec    = recall_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*55}")
    print(f"  [{label}] Accuracy: {acc*100:.2f}%  |  Macro F1: {f1_mac*100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                target_names=EMOTION_LABELS, digits=4))

    return {
        "label":      label,
        "accuracy":   acc,
        "f1_macro":   f1_mac,
        "f1_weighted":f1_w,
        "precision":  prec,
        "recall":     rec,
        "train_time": train_time,
        "y_pred":     y_pred,
        "feat_dim":   X_train.shape[1],
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


def plot_ablation_comparison(results: list, save_path: str = None):

    labels   = [r["label"]       for r in results]
    accuracy = [r["accuracy"]*100 for r in results]
    f1_mac   = [r["f1_macro"]*100 for r in results]
    f1_w     = [r["f1_weighted"]*100 for r in results]

    x    = np.arange(len(labels))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - w,   accuracy, w, label="Accuracy",    color="#4C72B0")
    bars2 = ax.bar(x,       f1_mac,   w, label="Macro F1",    color="#DD8452")
    bars3 = ax.bar(x + w,   f1_w,     w, label="Weighted F1", color="#55A868")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Ablation Study: HOG vs LBP vs HOG+LBP (SVM)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


def plot_per_class_f1(results: list, save_path: str = None):

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    x = np.arange(len(EMOTION_LABELS))
    w = 0.25

    for idx, r in enumerate(results):
        report = classification_report(
            r["y_true"], r["y_pred"],
            target_names=EMOTION_LABELS,
            output_dict=True,
        )
        f1_per_class = [report[e]["f1-score"]*100 for e in EMOTION_LABELS]
        offset = (idx - 1) * w
        ax.bar(x + offset, f1_per_class, w,
               label=r["label"], color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, rotation=15)
    ax.set_ylabel("F1 Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-Class F1 Score: Ablation Study",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


def print_ablation_table(results: list):

    print("\n" + "="*75)
    print(f"  {'Configuration':<20} {'Feat Dim':>8} {'Accuracy':>10} "
          f"{'Macro F1':>10} {'Train Time':>12}")
    print("="*75)
    for r in results:
        print(f"  {r['label']:<20} {r['feat_dim']:>8} "
              f"{r['accuracy']*100:>9.2f}% "
              f"{r['f1_macro']*100:>9.2f}% "
              f"{r['train_time']:>10.1f}s")
    print("="*75)


def visualize_hog_sample(image: np.ndarray, label: int):
    _, hog_img = hog(image, orientations=HOG_ORIENTATIONS,
                     pixels_per_cell=HOG_PIXELS_PER_CELL,
                     cells_per_block=HOG_CELLS_PER_BLOCK,
                     block_norm="L2-Hys", visualize=True)
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 10))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original — {EMOTION_LABELS[label]}")
    axes[0].axis("off")
    axes[1].imshow(hog_img, cmap="gray")
    axes[1].set_title("HOG Visualization")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("hog_sample.png", dpi=150)
    plt.show()


def main():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_PATH}\n"
            "Update the DATASET_PATH variable at the top of the file."
        )

    # 1. Load data
    X_train, y_train, X_test, y_test = load_dataset(DATASET_PATH)

    visualize_hog_sample(X_train[0], y_train[0])

    print("\n[INFO] Feature extraction...")
    t0 = time.time()
    hog_tr, lbp_tr, comb_tr = extract_all(X_train, "Train")
    hog_te, lbp_te, comb_te = extract_all(X_test,  "Test")
    print(f"  HOG dim: {hog_tr.shape[1]} | LBP dim: {lbp_tr.shape[1]} "
          f"| Combined dim: {comb_tr.shape[1]}")
    print(f"  Total extraction time: {time.time()-t0:.1f}s")

    configs = [
        ("HOG only",    hog_tr,  hog_te),
        ("LBP only",    lbp_tr,  lbp_te),
        ("HOG+LBP",     comb_tr, comb_te),
    ]

    results = []
    for name, X_tr, X_te in configs:
        r = train_and_evaluate(X_tr, y_train, X_te, y_test, label=name)
        r["y_true"] = y_test
        results.append(r)

    print_ablation_table(results)

    for r in results:
        plot_confusion_matrix(
            y_test, r["y_pred"],
            title=f"Confusion Matrix — {r['label']} (Normalized)",
            save_path=f"confusion_matrix_{r['label'].replace(' ', '_').replace('+', '')}.png",
        )

    plot_ablation_comparison(results, save_path="ablation_comparison.png")

    plot_per_class_f1(results, save_path="per_class_f1.png")

    print("\n[DONE] Ablation study complete.")
    print("  Saved files:")
    print("    hog_sample.png")
    print("    confusion_matrix_HOG_only.png")
    print("    confusion_matrix_LBP_only.png")
    print("    confusion_matrix_HOGLBP.png")
    print("    ablation_comparison.png")
    print("    per_class_f1.png")


if __name__ == "__main__":
    main()
