from project.utils.train import train_data, save_dir
import numpy as np


import matplotlib.pyplot as plt
from sklearn.metrics import  silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

print("Phân tích không gian tiềm ẩn...")

# Trực quan hóa không gian tiềm ẩn bằng PCA
pca = PCA(n_components=2)
train_latents_pca = pca.fit_transform(train_data['latents'])

plt.figure(figsize=(20, 8))

# Trực quan hóa kết quả PCA
plt.subplot(1, 2, 1)
for i, label in enumerate(['Ung thư', 'Không ung thư']):
    plt.scatter(train_latents_pca[train_data['labels'] == i, 0],
                train_latents_pca[train_data['labels'] == i, 1],
                label=label, alpha=0.5)
plt.title('PCA của không gian tiềm ẩn')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Trực quan hóa kết quả t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
train_latents_tsne = tsne.fit_transform(train_data['latents'])

plt.subplot(1, 2, 2)
for i, label in enumerate(['Ung thư', 'Không ung thư']):
    plt.scatter(train_latents_tsne[train_data['labels'] == i, 0],
                train_latents_tsne[train_data['labels'] == i, 1],
                label=label, alpha=0.5)
plt.title('t-SNE của không gian tiềm ẩn')
plt.xlabel('Chiều t-SNE 1')
plt.ylabel('Chiều t-SNE 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig(f'{save_dir}/latent_space_visualization.png', dpi=300)
plt.show()

def evaluate_clustering(data, labels, clusters, method_name=""):
    """Đánh giá kết quả phân cụm sử dụng nhiều chỉ số đánh giá khác nhau."""
    # Tính toán các chỉ số đánh giá phân cụm
    silhouette = silhouette_score(data, clusters) if len(np.unique(clusters)) > 1 else 0
    calinski = calinski_harabasz_score(data, clusters) if len(np.unique(clusters)) > 1 else 0
    davies = davies_bouldin_score(data, clusters) if len(np.unique(clusters)) > 1 else 0

    # Tính toán độ tinh khiết của cụm và khả năng phục hồi lớp
    n_clusters = len(np.unique(clusters))
    if -1 in np.unique(clusters):  # Xử lý điểm nhiễu của DBSCAN như một cụm riêng
        n_clusters -= 1

    purity = 0
    total_samples = 0

    cluster_distribution = []
    cluster_sizes = []

    # Xử lý từng cụm
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:  # Bỏ qua điểm nhiễu khi tính độ tinh khiết
            continue

        cluster_mask = (clusters == cluster_id)
        cluster_size = np.sum(cluster_mask)

        if cluster_size == 0:
            continue

        cluster_labels = labels[cluster_mask]

        # Xử lý trường hợp hàm mode gặp lỗi
        if len(cluster_labels) == 0:
            continue

        # Sử dụng value_counts hoặc bincount thay vì mode để ổn định hơn
        unique_vals, counts = np.unique(cluster_labels, return_counts=True)
        majority_class = unique_vals[np.argmax(counts)]
        majority_count = np.max(counts)

        purity += majority_count
        total_samples += cluster_size

        # Lưu dữ liệu phân bố để trực quan hóa
        cancer_count = np.sum(cluster_labels == 0)
        no_cancer_count = np.sum(cluster_labels == 1)

        cancer_percent = cancer_count / cluster_size * 100
        no_cancer_percent = no_cancer_count / cluster_size * 100

        cluster_distribution.append([cancer_percent, no_cancer_percent])
        cluster_sizes.append(cluster_size)

        print(f"{method_name} Cụm {cluster_id}: Ung thư: {cancer_count} ({cancer_percent:.1f}%), " +
              f"Không ung thư: {no_cancer_count} ({no_cancer_percent:.1f}%), " +
              f"Tổng: {cluster_size}")

    # Tính toán độ tinh khiết tổng thể
    if total_samples > 0:
        purity = purity / total_samples
    else:
        purity = 0

    print(f"\nChỉ số đánh giá phân cụm {method_name}:")
    print(f"Điểm Silhouette: {silhouette:.4f} (càng cao càng tốt, phạm vi: [-1, 1])")
    print(f"Chỉ số Calinski-Harabasz: {calinski:.4f} (càng cao càng tốt)")
    print(f"Chỉ số Davies-Bouldin: {davies:.4f} (càng thấp càng tốt)")
    print(f"Độ tinh khiết của cụm: {purity:.4f} (càng cao càng tốt, phạm vi: [0, 1])")

    return {
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'purity': purity,
        'distribution': np.array(cluster_distribution) if cluster_distribution else np.array([]),
        'sizes': np.array(cluster_sizes) if cluster_sizes else np.array([])
    }

def visualize_clusters(tsne_data, clusters, true_labels, method_name, save_path):
    """Trực quan hóa kết quả phân cụm sử dụng t-SNE."""
    n_clusters = len(np.unique(clusters))
    has_noise = -1 in np.unique(clusters)

    plt.figure(figsize=(20, 8))

    # Vẽ theo cụm
    plt.subplot(1, 2, 1)

    # Sử dụng bảng màu khác biệt với màu của lớp
    cmap = plt.cm.get_cmap('tab10', n_clusters if not has_noise else n_clusters-1)

    for i, cluster_id in enumerate(sorted(np.unique(clusters))):
        if cluster_id == -1:
            # Vẽ điểm nhiễu bằng màu đen
            mask = clusters == cluster_id
            plt.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                        color='black', marker='x', label='Nhiễu', alpha=0.5)
        else:
            mask = clusters == cluster_id
            plt.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                        color=cmap(i), label=f'Cụm {cluster_id}', alpha=0.6)

    plt.title(f'Các cụm {method_name} (t-SNE)')
    plt.xlabel('Chiều t-SNE 1')
    plt.ylabel('Chiều t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Vẽ theo lớp thực
    plt.subplot(1, 2, 2)
    for i, label_name in enumerate(['Ung thư', 'Không ung thư']):
        plt.scatter(tsne_data[true_labels == i, 0], tsne_data[true_labels == i, 1],
                    label=label_name, alpha=0.6)
    plt.title('Các lớp gốc (t-SNE)')
    plt.xlabel('Chiều t-SNE 1')
    plt.ylabel('Chiều t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def visualize_distribution(distribution, sizes, method_name, save_path):
    """Trực quan hóa phân bố các lớp trong từng cụm."""
    if len(distribution) == 0:
        print(f"Không có cụm hợp lệ để trực quan hóa cho {method_name}")
        return

    n_clusters = len(distribution)

    # Biểu đồ phân bố cụm
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(n_clusters)

    bars1 = plt.bar(index, distribution[:, 0], bar_width, label='Ung thư', color='salmon')
    bars2 = plt.bar(index + bar_width, distribution[:, 1], bar_width, label='Không ung thư', color='skyblue')

    plt.xlabel('Cụm')
    plt.ylabel('Phần trăm (%)')
    plt.title(f'{method_name}: Phân bố lớp trong mỗi cụm')
    plt.xticks(index + bar_width/2, [f'Cụm {i}' for i in range(n_clusters)])
    plt.legend()

    # Thêm phần trăm chính xác phía trên cột
    for i, v in enumerate(distribution[:, 0]):
        plt.text(i - 0.05, v + 1, f'{v:.1f}%', fontsize=9)
    for i, v in enumerate(distribution[:, 1]):
        plt.text(i + bar_width - 0.05, v + 1, f'{v:.1f}%', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_path}_distribution.png', dpi=300)
    plt.show()

    # Biểu đồ kích thước cụm
    plt.figure(figsize=(10, 6))
    plt.bar(index, sizes, color='lightgreen')
    plt.xlabel('Cụm')
    plt.ylabel('Số lượng mẫu')
    plt.title(f'{method_name}: Kích thước các cụm')
    plt.xticks(index, [f'Cụm {i}' for i in range(n_clusters)])

    # Thêm số lượng chính xác phía trên cột
    for i, v in enumerate(sizes):
        plt.text(i - 0.1, v + 1, str(int(v)), fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{save_path}_sizes.png', dpi=300)
    plt.show()

print("\n" + "="*50)
print("PHÂN CỤM VÀ SO SÁNH GIỮA CÁC PHƯƠNG PHÁP PHÂN CỤM")
print("="*50)

# Phần 1: K-Means clustering
print("\n1. Phân cụm K-Means")
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(train_data['latents'])

# Đánh giá và trực quan hóa kết quả phân cụm K-Means
kmeans_results = evaluate_clustering(
    train_data['latents'],
    train_data['labels'],
    kmeans_clusters,
    "K-Means"
)

visualize_clusters(
    train_latents_tsne,
    kmeans_clusters,
    train_data['labels'],
    "K-Means",
    f'{save_dir}/kmeans_clusters.png'
)

if len(kmeans_results['distribution']) > 0:
    visualize_distribution(
        kmeans_results['distribution'],
        kmeans_results['sizes'],
        "K-Means",
        f'{save_dir}/kmeans'
    )

# Lưu kết quả phân cụm K-Means
np.save(f'{save_dir}/kmeans_clusters.npy', kmeans_clusters)

# Phần 2: Mô hình hỗn hợp Gaussian (GMM)
print("\n" + "="*30)
print("2. Mô hình hỗn hợp Gaussian (GMM)")
print("="*30)

# Huấn luyện mô hình GMM với số lượng thành phần giống như K-Means
gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
gmm.fit(train_data['latents'])
gmm_clusters = gmm.predict(train_data['latents'])

# Đánh giá và trực quan hóa kết quả phân cụm GMM
gmm_results = evaluate_clustering(
    train_data['latents'],
    train_data['labels'],
    gmm_clusters,
    "GMM"
)

visualize_clusters(
    train_latents_tsne,
    gmm_clusters,
    train_data['labels'],
    "GMM",
    f'{save_dir}/gmm_clusters.png'
)

if len(gmm_results['distribution']) > 0:
    visualize_distribution(
        gmm_results['distribution'],
        gmm_results['sizes'],
        "GMM",
        f'{save_dir}/gmm'
    )

# Lưu kết quả phân cụm GMM
np.save(f'{save_dir}/gmm_clusters.npy', gmm_clusters)

# Phần 3: Phân cụm DBSCAN
print("\n" + "="*30)
print("3. Phân cụm DBSCAN")
print("="*30)

# Xác định epsilon sử dụng đồ thị k-distance
from sklearn.neighbors import NearestNeighbors
import numpy as np

def find_optimal_eps(data, k=4):
    """Tìm giá trị epsilon tối ưu cho DBSCAN sử dụng đồ thị k-distance."""
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)
    # Sắp xếp khoảng cách theo thứ tự giảm dần
    distances = np.sort(distances[:, k-1])

    # Vẽ đồ thị k-distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Các điểm được sắp xếp theo khoảng cách')
    plt.ylabel(f'Khoảng cách đến láng giềng thứ {k}')
    plt.title('Đồ thị K-distance để chọn Epsilon cho DBSCAN')
    plt.grid(True)
    plt.savefig(f'{save_dir}/dbscan_kdistance.png', dpi=300)
    plt.show()

    # Tìm điểm "khuỷu tay" - có thể tự động hóa, nhưng ở đây chúng ta sẽ trả về một giá trị hợp lý
    # Phương pháp đơn giản: tìm điểm có độ cong tối đa
    diffs = np.diff(distances)
    acceleration = np.diff(diffs)
    max_curve_idx = np.argmax(np.abs(acceleration)) + 2
    optimal_eps = distances[max_curve_idx]

    print(f"Giá trị epsilon tối ưu đề xuất: {optimal_eps}")
    return optimal_eps

# Tìm giá trị epsilon tối ưu cho DBSCAN
optimal_eps = find_optimal_eps(train_data['latents'])

# Chạy DBSCAN với các tham số tối ưu
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
dbscan_clusters = dbscan.fit_predict(train_data['latents'])

print(f"DBSCAN đã xác định {len(np.unique(dbscan_clusters))} cụm (bao gồm cả nhiễu)")
print(f"Số lượng điểm nhiễu: {np.sum(dbscan_clusters == -1)}")

# Đánh giá và trực quan hóa kết quả phân cụm DBSCAN
dbscan_results = evaluate_clustering(
    train_data['latents'],
    train_data['labels'],
    dbscan_clusters,
    "DBSCAN"
)

visualize_clusters(
    train_latents_tsne,
    dbscan_clusters,
    train_data['labels'],
    "DBSCAN",
    f'{save_dir}/dbscan_clusters.png'
)

if len(dbscan_results['distribution']) > 0:
    visualize_distribution(
        dbscan_results['distribution'],
        dbscan_results['sizes'],
        "DBSCAN",
        f'{save_dir}/dbscan'
    )

# Lưu kết quả phân cụm DBSCAN
np.save(f'{save_dir}/dbscan_clusters.npy', dbscan_clusters)

# Phần 4: So sánh tất cả các phương pháp phân cụm
print("\n" + "="*30)
print("4. So sánh giữa các phương pháp phân cụm")
print("="*30)

# Thu thập các chỉ số để so sánh
methods = ['K-Means', 'GMM', 'DBSCAN']
results = [kmeans_results, gmm_results, dbscan_results]

metrics = ['silhouette', 'calinski', 'davies', 'purity']
metric_names = {
    'silhouette': 'Điểm Silhouette\n(càng cao càng tốt)',
    'calinski': 'Chỉ số Calinski-Harabasz\n(càng cao càng tốt)',
    'davies': 'Chỉ số Davies-Bouldin\n(càng thấp càng tốt)',
    'purity': 'Độ tinh khiết của cụm\n(càng cao càng tốt)'
}

# Tạo biểu đồ so sánh
plt.figure(figsize=(18, 12))

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)

    # Xử lý chỉ số Davies-Bouldin khác biệt (càng thấp càng tốt)
    values = [result[metric] for result in results]
    if metric == 'davies':
        # Đảo ngược Davies-Bouldin để trực quan hóa (để cao hơn là tốt hơn cho tất cả các chỉ số)
        max_value = max(values) * 1.1
        values = [max_value - value for value in values]
        plt.title(metric_names[metric])
        plt.axhline(y=max_value, color='r', linestyle='--', alpha=0.3)
        plt.text(len(methods)/2, max_value*1.02, "Càng thấp càng tốt", ha='center')
    else:
        plt.title(metric_names[metric])

    plt.bar(methods, values, color=['skyblue', 'lightgreen', 'salmon'])

    # Thêm nhãn giá trị phía trên các cột
    for j, value in enumerate(values):
        if metric == 'davies':
            # Hiển thị giá trị gốc cho Davies-Bouldin
            original_value = results[j][metric]
            plt.text(j, value/2, f"{original_value:.3f}", ha='center', va='center')
        else:
            plt.text(j, value/2, f"{value:.3f}", ha='center', va='center')

    plt.tight_layout()

plt.savefig(f'{save_dir}/clustering_comparison.png', dpi=300)
plt.show()

# Tạo bảng tóm tắt của tất cả các chỉ số
metrics_table = {
    'Phương pháp': methods,
    'Điểm Silhouette': [r['silhouette'] for r in results],
    'Calinski-Harabasz': [r['calinski'] for r in results],
    'Davies-Bouldin': [r['davies'] for r in results],
    'Độ tinh khiết của cụm': [r['purity'] for r in results]
}

# Hiển thị dưới dạng bảng
from tabulate import tabulate
print("\nTóm tắt các chỉ số đánh giá hiệu suất phân cụm:")
table = tabulate(
    [[m, *[metrics_table[col][i] for col in list(metrics_table.keys())[1:]]]
     for i, m in enumerate(methods)],
    headers=list(metrics_table.keys()),
    tablefmt="grid",
    floatfmt=".4f"
)
print(table)

print("\nKết luận và so sánh:")
print("-"*50)
print("1. K-Means:")
print("   - Ưu điểm: Thuật toán đơn giản, dễ hiểu và triển khai")
print("   - Nhược điểm: Yêu cầu định trước số lượng cụm, nhạy cảm với nhiễu và điểm ngoại lai")
print("   - Phù hợp khi: Các cụm có kích thước tương đương nhau và hình dạng cầu")

print("\n2. GMM (Mô hình hỗn hợp Gaussian):")
print("   - Ưu điểm: Mô hình xác suất, cho phép cụm chồng chéo, linh hoạt hơn với hình dạng cụm")
print("   - Nhược điểm: Phức tạp hơn, vẫn cần định trước số lượng cụm")
print("   - Phù hợp khi: Dữ liệu có phân phối Gaussian, cần phân cụm mềm (soft clustering)")

print("\n3. DBSCAN:")
print("   - Ưu điểm: Không cần định trước số lượng cụm, xử lý tốt điểm ngoại lai, phát hiện được cụm hình dạng tùy ý")
print("   - Nhược điểm: Nhạy cảm với tham số epsilon và min_samples, khó xử lý khi mật độ dữ liệu không đồng đều")
print("   - Phù hợp khi: Dữ liệu có điểm ngoại lai, cụm có hình dạng phức tạp và không biết trước số lượng cụm")

# Xác định phương pháp tốt nhất dựa trên các chỉ số
best_silhouette = methods[np.argmax([r['silhouette'] for r in results])]
best_calinski = methods[np.argmax([r['calinski'] for r in results])]
best_davies = methods[np.argmin([r['davies'] for r in results])]
best_purity = methods[np.argmax([r['purity'] for r in results])]

print("\nPhương pháp phân cụm tốt nhất theo từng tiêu chí:")
print(f"- Điểm Silhouette: {best_silhouette}")
print(f"- Chỉ số Calinski-Harabasz: {best_calinski}")
print(f"- Chỉ số Davies-Bouldin: {best_davies}")
print(f"- Độ tinh khiết của cụm: {best_purity}")

# Kiểm tra xem bản chất cụm có phù hợp với các phân lớp ban đầu không
print("\nTương quan giữa phân cụm và phân lớp ban đầu:")
# Tính ARI (Adjusted Rand Index) để so sánh phân cụm với nhãn thực
from sklearn.metrics import adjusted_rand_score

ari_kmeans = adjusted_rand_score(train_data['labels'], kmeans_clusters)
ari_gmm = adjusted_rand_score(train_data['labels'], gmm_clusters)
ari_dbscan = adjusted_rand_score(train_data['labels'], dbscan_clusters)

print(f"- K-Means ARI: {ari_kmeans:.4f}")
print(f"- GMM ARI: {ari_gmm:.4f}")
print(f"- DBSCAN ARI: {ari_dbscan:.4f}")

print("\nĐã hoàn thành các phân tích và so sánh phương pháp phân cụm!")