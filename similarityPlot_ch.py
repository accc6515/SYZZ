# 在开头导入seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 设置全局样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
# plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用更美观的字体

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 设置中文字体路径（确保支持中文字符）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)

# 设置为默认字体
plt.rcParams['font.family'] = my_font.get_name()

def plot_enhanced_similarity(dist, best_frame_offset, threshold):
    """
    绘制增强型的相似度折线图
    
    参数:
    dist - 相似度矩阵 (n_frames, n_offsets)
    best_frame_offset - 最佳帧偏移量
    threshold - 分类阈值
    """
    
    # 提取最佳偏移量的相似度序列
    vshift = 15
    best_offset_idx = best_frame_offset + vshift
    y = dist[:, best_offset_idx]
    x = np.arange(len(y))
    cnt_low_threshold = (y < threshold).sum()
    per_low_threshold = round(cnt_low_threshold / len(y) * 100, 2)
    
    # 创建图表
    plt.figure(figsize=(12, 6), dpi=120)
    
    # 绘制平滑曲线（使用高斯滤波）
    from scipy.ndimage import gaussian_filter1d
    smoothed_y = gaussian_filter1d(y, sigma=1.5)
    
    # 主折线图
    plt.plot(x, smoothed_y, 
             color='#1f77b4',  # seaborn默认蓝色
             linewidth=2.5, 
             alpha=0.9,
             label='相似度轨迹')
    
    # 添加原始数据点（透明度处理）
    plt.scatter(x, y, 
                color='#1f77b4', 
                alpha=0.15, 
                s=15, 
                label='原始帧相似度')
    
    # 添加阈值线
    plt.axhline(y=threshold, 
                color='#d62728',  # 红色
                linestyle='--', 
                linewidth=1.8, 
                alpha=0.8, 
                label=f'阈值 ({threshold})')
    
    # 填充阈值下方的区域（红色）
    plt.fill_between(x, y, threshold, 
                     where=(y < threshold), 
                     color='#ff7f0e',  # 橙色
                     alpha=0.25, 
                     label='可能伪影区域')
    
    # 标记重要点
    low_point_idx = y.argmin()
    high_point_idx = y.argmax()
    
    plt.scatter(low_point_idx, y[low_point_idx], 
                color='#d62728', s=100, zorder=5,
                edgecolor='white', linewidth=1.5,
                label=f'最低点 ({y[low_point_idx]:.2f})')
    
    plt.scatter(high_point_idx, y[high_point_idx], 
                color='#2ca02c', s=100, zorder=5,
                edgecolor='white', linewidth=1.5,
                label=f'最高点 ({y[high_point_idx]:.2f})')
    
    # 添加平均线和标注
    mean_y = y.mean()
    plt.axhline(y=mean_y, 
                color='#2ca02c',  # 紫色
                linestyle='-.', 
                linewidth=1.8, 
                alpha=0.8, 
                label=f'平均相似度 ({mean_y:.2f})')
    
    # 添加纵轴标注（主要变化在这里）
    ax = plt.gca()
    
    # 1. 在图表左侧添加阈值和平均值的文本框
    # 阈值标注
    ax.text(-0.03 * len(x),  # x位置 (图表左侧)
            threshold,       # y位置
            f' Th: {threshold}', 
            color='#d62728', 
            fontsize=10,
            weight='bold',
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 平均值标注
    ax.text(-0.03 * len(x),  # x位置 (图表左侧)
            mean_y,          # y位置
            f' Avg: {mean_y:.2f}', 
            color='#2ca02c', 
            fontsize=10,
            weight='bold',
            ha='right',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 添加 cnt_low_threshold 标注
    ax.text(1.03 * len(x),          # x位置 (图表左侧)
        1.2,                     # y位置，图顶端
        # f'Cnt (CosDist < Th): {cnt_low_threshold}', 
        f'相似度低于阈值的百分比: {per_low_threshold}%', 
        color='red', 
        fontsize=16,
        weight='bold',
        ha='right',
        va='top',
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    
    # # 添加视频关键区间标注
    # quarter_length = len(y) // 4
    # for i in range(4):
    #     start = i * quarter_length
    #     end = (i+1) * quarter_length if i < 3 else len(y)
    #     section_y = y[start:end]
        
    #     plt.axvline(x=start, color='gray', alpha=0.4, linestyle=':', linewidth=1)
        
    #     plt.annotate(f'Section {i+1}\n({section_y.mean():.2f})',
    #                 xy=((start + end)/2, max(y) * 0.85),
    #                 ha='center', va='center',
    #                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    # 添加标题和标签
    # plt.title(f'Lip Sync Analysis | Optimal Frame Offset: {best_frame_offset}', 
    #          fontsize=16, pad=15, fontweight='bold')
    plt.title(f'相似度分析', 
             fontsize=16, pad=15, fontweight='bold')
    
    plt.xlabel('视频帧序列', fontsize=13, labelpad=6)
    plt.ylabel('音视频相似度', fontsize=13, labelpad=40)
    
    # 设置坐标轴范围
    plt.ylim(-0.1, 1.05)
    plt.xlim(-5, len(y)+5)
    
    # 添加网格
    plt.grid(True, which='major', axis='both', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', axis='x', linestyle=':', alpha=0.1)
    plt.minorticks_on()
    
    # 添加图例
    plt.legend(loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), 
              fancybox=True, 
              shadow=True, 
              ncol=3, 
              fontsize=11)
    
    # 添加水印和说明
    plt.figtext(0.95, 0.02, 'Deep fake detection system',
               fontsize=10, color='gray',
               ha='right', va='bottom', alpha=0.5)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('enhanced_similarity_plot.png', bbox_inches='tight', dpi=150)
    plt.close()
    return mean_y