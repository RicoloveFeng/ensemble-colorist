import argparse
from bisect import bisect_left
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np


class EnsembleColorist:
    def __init__(self, title: str=''):
        self.title: str = title
        self.theme: dict = {}
        self.data: pd.DataFrame = pd.DataFrame()
        self.cmap = None
        self.norm = None
        
    def set_title(self, title: str):
        self.title = title
    
    def load_data(self, path: str):
        """
        从csv文件中加载数据。数据格式需要遵从data_adapter/README.md的描述
        """
        self.data = pd.read_csv(path)
        # 校验数据格式
        required_columns = ['track', 'sample', 'hours', 'lat', 'lon', 'pressure']
        for column in required_columns:
            assert column in self.data.columns
    
    def load_theme(self, path: str):
        with open(path, 'r') as file:
            self.theme = json.load(file)
            
        color_style = self.theme["color"]
        min_val = 870
        max_val = 1020
        boundaries = [max_val] + color_style["boundaries"] + [min_val]
        boundaries.sort(reverse=True)
        colors = color_style["colors"]
        
        if len(boundaries) - 1 != len(colors):
            raise ValueError("边界数与颜色数不匹配")
        if any(b < min_val or b > max_val for b in boundaries):
            raise ValueError("边界值必须在870-1020之间")
        if any(boundaries[i] <= boundaries[i+1] for i in range(len(boundaries)-1)):
            raise ValueError("边界必须是递减序列")

        color_list = []
        # 遍历每个区间
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            # 生成区间内所有值
            values = range(end, start+1) if i == len(boundaries)-2 else range(end, start)
            values = sorted(values, reverse=True)
            c_spec = colors[i]
            
            # 处理单一颜色
            if isinstance(c_spec, str):
                color_list.extend([c_spec] * len(values))
            # 处理渐变色
            elif isinstance(c_spec, list) and len(c_spec) == 2:
                cmap_seg = LinearSegmentedColormap.from_list(f"seg_{i}", c_spec)
                seg_colors = cmap_seg(np.linspace(0, 1, len(values)))
                color_list.extend([f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" 
                                for r, g, b, _ in seg_colors])
        self.cmap = ListedColormap(color_list[::-1])
        self.norm = plt.Normalize(vmin=870, vmax=1020)
            
    
    def make_fig(self):
        assert self.theme
        assert len(self.data) > 0
        
        # 解析时间变化样式
        decay_style = self.theme["decay"]
        decay_boundaries = decay_style["boundaries"]
        len_decay_diff = len(decay_boundaries) + 1
        decay_size = decay_style.get("size")
        decay_line_width = decay_style["line_width"]
        decay_alpha = decay_style.get("alpha", [1]*len_decay_diff)
        decay_shape = decay_style.get("shape", ['o']*len_decay_diff)
        
        
        
        def make_backgroud():
            # 创建地图
            width, height = self.theme['backgroud']['width'], self.theme['backgroud']['height']
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            # ax = fig.add_axes([0.01, 0.01, 0.9, 0.9], projection=ccrs.PlateCarree())

            # 设置地图范围
            ax.set_extent([
                self.theme['backgroud']['left_longitude'],
                self.theme['backgroud']['right_longitude'],
                self.theme['backgroud']['bottom_latitude'],
                self.theme['backgroud']['top_latitude']
            ], crs=ccrs.PlateCarree())
            ax.set_adjustable('datalim')
            
            # 绘制陆地颜色
            ax.add_feature(cfeature.LAND.with_scale('50m'), color=self.theme['backgroud']['land_color'])

            # 绘制海洋颜色
            ax.add_feature(cfeature.OCEAN.with_scale('50m'), color=self.theme['backgroud']['sea_color'])

            # 绘制海岸线
            if self.theme['backgroud']['enable_coast']:
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor=self.theme['backgroud']['border_color'], linewidth=self.theme['backgroud']['border_width'])

            # 绘制国界
            if self.theme['backgroud']['enable_national_border']:
                ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor=self.theme['backgroud']['border_color'], linewidth=self.theme['backgroud']['border_width'])

            # 添加坐标网格
            if self.theme['backgroud']['enable_coordinate_grid']:
                gl = ax.gridlines(draw_labels={'left': True, 'right': False, 'top': False, 'bottom': True}, linestyle='--', color='gray')
                gl.xlabel_style = {'size': 12}
                gl.ylabel_style = {'size': 12}

            # 添加标题
            if self.theme['backgroud']['enable_title'] and self.title:
                plt.title(self.title, loc='left', fontsize=18)

            # 调整布局参数以缩小白色边框
            fig.tight_layout(pad=1)

            return fig, ax
        
        def make_tracks(fig, ax):
            def get_decay(hour):
                if not decay_style["boundaries"]:
                    return decay_size[0], decay_line_width[0], decay_alpha[0], decay_shape[0]
                else:
                    idx = bisect_left(decay_boundaries, hour)
                    return decay_size[idx], decay_line_width[idx], decay_alpha[idx], decay_shape[idx]
                
            color_style = self.theme["color"]
            track_line_width = color_style["line_width"]
            track_alpha = color_style.get("alpha", 0.5)
            
            # 按track和sample分组
            grouped = self.data.groupby(['track', 'sample'])

            # 先绘制所有路径线
            for (track_id, sample_id), group in grouped:
                # 按hours排序
                group = group.sort_values('hours')
                
                # 检测并处理跨越180度经线的情况
                lons = group['lon'].values
                lats = group['lat'].values
                pressure = group['pressure'].values
                
                # 找出跨越180度经线的位置
                line_segment = [(lons[0], lats[0], pressure[0])]
                
                for i in range(1, len(lons)):
                    prev_lon = lons[i-1]
                    curr_lon = lons[i]
                    if abs(curr_lon - prev_lon) > 180:
                        avg_lat = (lats[i-1] + lats[i]) / 2
                        line_segment.append((prev_lon, avg_lat, pressure[i-1]))
                        line_segment = [(curr_lon, avg_lat, pressure[i-1])]
                    else:
                        line_segment.append((curr_lon, lats[i], pressure[i]))
                
                # 绘制每个段
                if len(line_segment) > 1:
                    segment_lons, segment_lats, segment_pressures = zip(*line_segment)
                    for i in range(len(segment_lons) - 1):
                        color_val = self.cmap(self.norm(segment_pressures[i]))
                        ax.plot([segment_lons[i], segment_lons[i+1]], 
                                [segment_lats[i], segment_lats[i+1]], 
                                linewidth=track_line_width, 
                                alpha=track_alpha,
                                color=color_val if color_style["apply_to_line"] else "grey",
                                transform=ccrs.PlateCarree(),
                                zorder=1)

            # 绘制系统中心的散点
            all_data_sorted = self.data.sort_values('pressure', ascending=False)
            for i in range(len(all_data_sorted)):
                size, line_width, alpha, shape = get_decay(all_data_sorted['hours'].iloc[i])
                scatter = ax.scatter(all_data_sorted['lon'].iloc[i], all_data_sorted['lat'].iloc[i], 
                               s=size, 
                               alpha=alpha,
                               marker=shape,
                               facecolors='none',
                               edgecolors=self.cmap(self.norm(all_data_sorted['pressure'].iloc[i])),
                               linewidth=line_width,
                               transform=ccrs.PlateCarree(),
                               zorder=1030 - all_data_sorted['pressure'].iloc[i])
            
            return fig, ax
            
        def make_provinces(fig, ax):
            if self.theme['backgroud']['enable_province_border']:
                try:
                    ax.add_feature(
                        cfeature.STATES.with_scale('50m'),
                        edgecolor=self.theme['backgroud']['border_color'],
                        linewidth=self.theme['backgroud']['border_width'] * 0.5,
                        linestyle='--',
                        facecolor='none'
                    )
                except:
                    pass
            return fig, ax
        
        def make_legend(fig, ax):

            
            # 创建一个分隔轴对象，用于精确控制colorbar位置
            divider = make_axes_locatable(ax)
            # 在地图右侧创建一个宽度为5%主图宽度，间距为0.1英寸的轴用于放置colorbar
            cax = divider.append_axes("right", size="2%", pad=0.1, axes_class=plt.Axes)
            
            # 使用新创建的轴来放置colorbar
            cbar = fig.colorbar(ScalarMappable(cmap=self.cmap, norm=self.norm), 
                            cax=cax, orientation='vertical')
            cbar.set_ticks(np.arange(870, 1021, 10))  # 每10写一个刻度
            
            # 生成每个 boundary 对应的“时间区间标签”（例如：0-12h、12-24h）
            legend_labels = []
            # 处理边界逻辑：生成 [min_hour, b1, b2, ..., max_hour] 的区间
            if not decay_boundaries:  # 若没有边界，只有1个区间
                legend_labels.append("All Hours")
            else:
                # 补充首尾边界（假设时间从0开始，最大时间取数据中小时的最大值）
                all_boundaries = [0] + sorted(decay_boundaries) + [360]
                # 生成区间标签（如：0-12h、12-24h）
                for i in range(len(all_boundaries)-1):
                    start = all_boundaries[i]
                    end = all_boundaries[i+1]
                    legend_labels.append(f"{int(start)}-{int(end)}h")

            # 手动创建图例元素（每个元素对应一个时间区间的散点样式）
            legend_elements = []
            for i in range(len(legend_labels)):
                elem = mlines.Line2D(
                    [], [],  # 无需实际数据，仅用于样式展示
                    marker=decay_shape[i],        # 散点形状
                    markersize=decay_size[i]/5,  # 图例中标记大小
                    markerfacecolor='none',       # 散点填充透明（与代码中一致）
                    markeredgecolor='gray',       # 散点边框色（用灰色统一，避免与colorbar混淆）
                    markeredgewidth=decay_line_width[i],  # 散点边框宽度
                    alpha=decay_alpha[i],         # 透明度
                    label=legend_labels[i]        # 区间标签
                )
                legend_elements.append(elem)

            # 将散点样式图例添加到地图上（位置可调整，这里放左下角）
            ax.legend(
                handles=legend_elements,
                loc='upper right',  # 位置：右上角
                bbox_to_anchor=(0.98, 0.98),  # 微调偏移（右98%、上98%处，避免贴边）
                frameon=True,
                framealpha=0.8,
                fontsize=10,  # 放大文字（原8→10）
                labelspacing=0.8,  # 增加标签间距（放大垂直方向尺寸）
                handlelength=2.5,  # 增加标记长度（放大水平方向尺寸）
                borderpad=1.2  # 增加图例框外边距（避免与地图元素挤在一起）
            )
            
            # 调整整体布局，减少空白
            fig.tight_layout()
            return fig, ax
        
        fig, ax = make_backgroud()
        fig, ax = make_tracks(fig, ax)
        fig, ax = make_provinces(fig, ax)
        fig, ax = make_legend(fig, ax)
        return fig, ax
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='examples/FNV3_example.csv')
    parser.add_argument('--theme', type=str, default='preset/cma.json')
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--title', type=str, default='FNV3 Ensemble Forecast from Google\nRun: 2025-09-05 18Z')
    args = parser.parse_args()
    ec = EnsembleColorist(args.title)
    ec.load_theme(f'{args.theme}')
    ec.load_data(args.data)
    fig, ax = ec.make_fig()
    plt.savefig(f'{args.output}')