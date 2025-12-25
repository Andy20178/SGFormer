import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

class FeatureCollector:
    def __init__(self):
        self.features = []  # 存储所有节点特征 (list of tensors)
        self.labels = []    # 存储对应的真实类别 (list of tensors)
        self.preds = []     # 可选：存储预测类别 (list of tensors)
        self.collected = False  # 标记是否已经拼接过

    def add(self, features, labels, preds=None):
        """
        添加一批特征和标签
        :param features: tensor [N, D]
        :param labels:   tensor [N] or list of int
        :param preds:    tensor [N], optional
        """
        self.features.append(features.cpu().detach())
        self.labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))
        if preds is not None:
            self.preds.append(preds.cpu().detach())
        self.collected = False  # 新增数据后需要重新拼接

    def get_all(self):
        """
        返回拼接后的特征和标签 (numpy arrays)
        """
        if self.collected:
            return self.features_stacked, self.labels_stacked, self.preds_stacked if self.preds else None

        # 拼接所有 tensor
        self.features_stacked = torch.cat(self.features, dim=0).numpy()  # [Total_N, D]
        self.labels_stacked = torch.cat(self.labels, dim=0).numpy()      # [Total_N]
        self.preds_stacked = torch.cat(self.preds, dim=0).numpy() if self.preds else None
        self.collected = True
        return self.features_stacked, self.labels_stacked, self.preds_stacked

    def clear(self):
        self.features = []
        self.labels = []
        self.preds = []
        self.collected = False

    def save_to_file(self, filepath):
        """
        将收集到的原始数据保存到文件（.pt 格式）
        :param filepath: 保存路径，例如 "features_collector_data.pt"
        """
        data = {
            'features': self.features,
            'labels': self.labels,
            'preds': self.preds,
        }
        torch.save(data, filepath)
        print(f"FeatureCollector data saved to {filepath}")

    def load_from_file(self, filepath):
        """
        从文件加载数据，并重置状态
        :param filepath: 加载路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        data = torch.load(filepath, map_location='cpu')
        self.features = data['features']
        self.labels = data['labels']
        self.preds = data.get('preds', [])  # 兼容旧版本没有 preds 的情况
        self.collected = False  # 加载后尚未拼接
        print(f"FeatureCollector data loaded from {filepath}")
        return self

    def plot_tsne(self, num_classes=None, save_path="tsne.png", perplexity=30, n_iter=1000):
        """
        画 t-SNE 图
        """
        features, labels, _ = self.get_all()

        if num_classes is not None:
            mask = labels < num_classes
            features = features[mask]
            labels = labels[mask]

        if features.shape[0] == 0:
            raise ValueError("No data to plot. Please add features first or load from file.")

        print(f"Performing t-SNE on {features.shape[0]} samples with {features.shape[1]} dimensions...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        tsne_result = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab20', s=5)
        plt.colorbar(scatter)
        plt.title("t-SNE Visualization of Node Features")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved to {save_path}")
    def plot_tsne_clear(self, num_classes=None, save_path="tsne.png", perplexity=30, n_iter=1000, class_name=None):
        """
        使用 t-SNE 可视化节点特征，并生成颜色与类别对照图。
        参数:
            num_classes: 只显示前 num_classes 个类别
            save_path: t-SNE 图保存路径
            perplexity: t-SNE 的 perplexity 参数
            n_iter: t-SNE 迭代次数
            class_name: 类别名称列表，如 ['cat', 'dog', ...]，长度应 >= num_classes 或 max(labels)+1
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize
        from matplotlib.patches import Rectangle

        # 获取所有特征和标签
        features, labels, _ = self.get_all()

        # 过滤指定类别的样本
        if num_classes is not None:
            mask = labels < num_classes
            features = features[mask]
            labels = labels[mask]

        if features.shape[0] == 0:
            raise ValueError("No data to plot. Please add features first or load from file.")

        print(f"Performing t-SNE on {features.shape[0]} samples with {features.shape[1]} dimensions...")

        # Step 1: L2 归一化
        print("Normalizing features with L2 norm...")
        features = normalize(features, norm='l2')

        # Step 2: PCA 降维到 50 维（如果原始维度较高）
        original_dim = features.shape[1]
        if original_dim > 50:
            print(f"Applying PCA to reduce dimension from {original_dim} to 50...")
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)

        # Step 3: 调整 perplexity
        effective_perplexity = min(perplexity, max(5, features.shape[0] // 3))
        print(f"Using perplexity = {effective_perplexity}")

        # Step 4: 执行 t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            n_iter=n_iter,
            random_state=42,
            metric='euclidean',
            init='pca',
            learning_rate='auto'
        )
        tsne_result = tsne.fit_transform(features)

        # Step 5: 绘制 t-SNE 图
        plt.figure(figsize=(12, 10))
        unique_labels = np.unique(labels)

        # 选择 colormap
        if len(unique_labels) <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('Spectral', len(unique_labels))

        scatter = plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=labels,
            cmap=cmap,
            s=10,
            alpha=0.8,
            edgecolors='none'
        )
        # 如果提供了 class_name，替换 colorbar 的 tick labels
        if class_name is not None:
            tick_names = [class_name[i] for i in unique_labels]
            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.set_ticklabels(tick_names)
        else:
            plt.colorbar(scatter, ticks=unique_labels)

        plt.title("t-SNE Visualization of Node Features", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved to {save_path}")

        # ============================================================== #
        # 直接绘制并保存：颜色 + 类别名称 + RGB 值 的图例图片（一行多列布局）
        # ============================================================== #

        # 确保 class_name 存在且够长
        if class_name is None:
            class_name = [f"Class {i}" for i in range(len(unique_labels))]

        # 创建图例图像
        legend_save_path = save_path.replace(".png", "_legend.png")
        n_classes = len(unique_labels)
        max_per_row = 10  # 每行最多显示 10 个类别
        n_rows = (n_classes - 1) // max_per_row + 1
        fig_width = 16
        fig_height = 2.5 * n_rows

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, max_per_row)
        ax.set_ylim(0, n_rows)
        ax.axis('off')

        for idx, label in enumerate(unique_labels):
            row = idx // max_per_row
            col = idx % max_per_row

            # 获取颜色
            color = cmap(idx % cmap.N)
            rgb_hex = '#%02x%02x%02x' % (
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )

            # 绘制色块
            rect = Rectangle((col + 0.1, n_rows - row - 0.8), 0.8, 0.8,
                            facecolor=color, edgecolor='black', lw=1.2)
            ax.add_patch(rect)

            # 类别名称（加粗）
            ax.text(col + 0.5, n_rows - row - 0.6, class_name[label],
                    fontsize=10, ha='center', va='bottom', weight='bold', color='black')

            # RGB 值（小字）
            ax.text(col + 0.5, n_rows - row - 0.9, rgb_hex,
                    fontsize=8, ha='center', va='top', color='gray')

        plt.tight_layout()
        plt.savefig(legend_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Legend image with class names and colors saved to {legend_save_path}")