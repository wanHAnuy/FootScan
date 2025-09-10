import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate



class ShoeSizeRecommender:
    """智能鞋码推荐系统 - 基于实际尺码表"""

    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    def __init__(self):
        # 初始化各国尺码对照表
        self.init_size_charts()
        
    def init_size_charts(self):
        """初始化尺码对照表 - 使用实际数据"""
        
        # 男鞋尺码表 (脚长单位: mm)
        self.men_size_chart = {
            'foot_length': [240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310],
            'CN': [38.5, 39, 40, 40.5, 41, 42, 42.5, 43, 44.5, 45, 45.5, 46, 46.5, 47, 47.5],
            'EU': [38.5, 39, 40, 40.5, 41, 42, 42.5, 43, 44.5, 45, 45.5, 46, 46.5, 47, 47.5],
            'US': [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13],
            'UK': [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5],
            'KR': [240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310],
            'JP': [24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31],
            'insole': ['7.00T-7.50T', '', '', '7.00T-7.50T', '8.00T-8.50T', '9.00T-9.50T', '10.00T-10.50T', 
                      '11.00T', '12.00T', '13.00T', '', '', '', '', '']
        }
        
        # 女鞋尺码表 (脚长单位: mm)
        self.women_size_chart = {
            'foot_length': [225, 230, 235, 240, 245, 245, 250, 250, 255, 255],
            'CN': [35, 36, 37, 38, 39, 39.5, 40, 40.5, 41, 41.5],
            'EU': [35, 36, 37, 38, 39, 39.5, 40, 40.5, 41, 41.5],
            'US': [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5],
            'UK': [2.5, 3, 3.5, 4.5, 5, 5.5, 6, 6.5, 7, 7.5],
            'KR': [225, 230, 235, 240, 245, '245-250', 250, '250-255', 255, '255-260'],
            'JP': [22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27]
        }
        
        # 童鞋尺码表 (厘米转毫米)
        self.kids_size_chart = {
            'foot_length': [80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 
                           155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215],
            'CN': [19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 
                   26.5, 27, 27.5, 28, 28.5, 29.5, 30, 31, 31.5, 32, 33, 33.5, 34],
            'EU': [17, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23.5, 24.5, 25, 25.5, 
                   26, 26.5, 27, 27.5, 28, 28.5, 29.5, 30.5, 31, 32, 33, 34, 35],
            'US': [1, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 
                   9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5],
            'UK': [0.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 
                   9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 1, 1.5, 2]
        }
        
        # 脚宽标准 (基于实际描述)
        # 男鞋: S(size)=5.6M, M(size)=7.8, L(size)=9.10, XL(size)=11.12
        # 女鞋: 小码(S)、中码(M)、大码(L)标识的指S=5/6、M=7/8、L=9/10尺码
        self.width_standards = {
            'men': {
                'N': {'label': '窄(N)', 'ratio_min': 0, 'ratio_max': 0.35, 'code': 'AA', 'desc': '瘦脚'},
                'M': {'label': '标准(M)', 'ratio_min': 0.35, 'ratio_max': 0.38, 'code': 'B/C', 'desc': '正常宽度'},
                'W': {'label': '宽(W)', 'ratio_min': 0.38, 'ratio_max': 0.41, 'code': 'D', 'desc': '宽脚/脚背高'},
                'XW': {'label': '加宽(XW)', 'ratio_min': 0.41, 'ratio_max': 1.0, 'code': 'EE', 'desc': '特宽脚'}
            },
            'women': {
                'N': {'label': '窄(N)', 'ratio_min': 0, 'ratio_max': 0.33, 'code': 'AA', 'desc': '瘦脚'},
                'M': {'label': '标准(M)', 'ratio_min': 0.33, 'ratio_max': 0.36, 'code': 'B', 'desc': '正常宽度'},
                'W': {'label': '宽(W)', 'ratio_min': 0.36, 'ratio_max': 0.38, 'code': 'D', 'desc': '宽脚'},
                'XW': {'label': '加宽(XW)', 'ratio_min': 0.38, 'ratio_max': 1.0, 'code': 'EE', 'desc': '特宽脚'}
            }
        }
    
    def determine_age_group(self, foot_length_mm):
        """根据脚长判断年龄组"""
        if foot_length_mm <= 215:
            return 'kids'
        elif foot_length_mm <= 260:
            return 'women_or_youth'
        else:
            return 'men'
    
    def analyze_foot_width(self, foot_length_mm, foot_width_mm, gender='men'):
        """分析脚宽类型"""
        ratio = foot_width_mm / foot_length_mm
        width_std = self.width_standards[gender]
        
        for key, value in width_std.items():
            if value['ratio_min'] <= ratio < value['ratio_max']:
                return {
                    'type': key,
                    'label': value['label'],
                    'code': value['code'],
                    'ratio': ratio,
                    'description': value['desc'],
                    'suggestion': self._get_width_suggestion(key, gender)
                }
        
        return {
            'type': 'XW',
            'label': '加宽(XW)',
            'code': 'EE',
            'ratio': ratio,
            'description': '特宽脚',
            'suggestion': '您的脚型偏宽，建议选择加宽款式或考虑大半码'
        }
    
    def _get_width_suggestion(self, width_type, gender):
        """获取脚宽建议"""
        suggestions = {
            'N': '建议选择窄版鞋款或使用鞋垫调整',
            'M': '大部分常规鞋款都适合',
            'W': '建议选择宽版鞋款或考虑大半码',
            'XW': '建议选择特宽版鞋款或专门的宽脚鞋款'
        }
        return suggestions.get(width_type, '标准鞋款')
    
    def get_size_recommendation(self, foot_length_mm, foot_width_mm, gender='auto'):
        """获取尺码推荐"""
        # 自动判断性别/年龄组
        if gender == 'auto':
            age_group = self.determine_age_group(foot_length_mm)
            if age_group == 'kids':
                return self._get_kids_recommendation(foot_length_mm, foot_width_mm)
            elif age_group == 'women_or_youth':
                # 返回女性和男性两种可能
                return {
                    'women': self._get_adult_recommendation(foot_length_mm, foot_width_mm, 'women'),
                    'men': self._get_adult_recommendation(foot_length_mm, foot_width_mm, 'men'),
                    'suggested': 'women'  # 建议女性尺码
                }
        
        if gender == 'kids':
            return self._get_kids_recommendation(foot_length_mm, foot_width_mm)
        else:
            return self._get_adult_recommendation(foot_length_mm, foot_width_mm, gender)
    
    def _get_adult_recommendation(self, foot_length_mm, foot_width_mm, gender):
        """获取成人尺码推荐"""
        size_chart = self.men_size_chart if gender == 'men' else self.women_size_chart
        df = pd.DataFrame(size_chart)
        
        # 找到最接近的脚长
        foot_lengths = df['foot_length'].values
        closest_idx = np.argmin(np.abs(foot_lengths - foot_length_mm))
        
        # 如果脚长在两个尺码之间，建议选择较大的
        if foot_length_mm > foot_lengths[closest_idx] and closest_idx < len(df) - 1:
            closest_idx += 1
        
        # 获取推荐尺码
        recommendations = {}
        for country in ['CN', 'EU', 'US', 'UK', 'JP']:
            if country in df.columns:
                recommendations[country] = df.loc[closest_idx, country]
        
        # 添加韩国码
        recommendations['KR'] = foot_length_mm
        
        # 分析脚宽
        width_analysis = self.analyze_foot_width(foot_length_mm, foot_width_mm, gender)
        
        # 如果脚宽较宽，建议考虑大半码
        size_adjustment = ""
        if width_analysis['type'] in ['W', 'XW']:
            size_adjustment = "建议考虑大半码"
        
        return {
            'sizes': recommendations,
            'width': width_analysis,
            'adjustment': size_adjustment,
            'foot_length': foot_length_mm,
            'foot_width': foot_width_mm,
            'gender': gender
        }
    
    def _get_kids_recommendation(self, foot_length_mm, foot_width_mm):
        """获取童鞋尺码推荐"""
        df = pd.DataFrame(self.kids_size_chart)
        
        # 找到最接近的脚长
        foot_lengths = df['foot_length'].values
        closest_idx = np.argmin(np.abs(foot_lengths - foot_length_mm))
        
        # 如果脚长在两个尺码之间，建议选择较大的
        if foot_length_mm > foot_lengths[closest_idx] and closest_idx < len(df) - 1:
            closest_idx += 1
        
        # 获取推荐尺码
        recommendations = {}
        for country in ['CN', 'EU', 'US', 'UK']:
            recommendations[country] = df.loc[closest_idx, country]
        
        # 童鞋一般不分宽窄
        width_ratio = foot_width_mm / foot_length_mm
        width_analysis = {
            'type': 'M',
            'label': '标准',
            'code': 'M',
            'ratio': width_ratio,
            'description': '儿童脚型',
            'suggestion': '儿童脚部发育快，建议预留5-10mm生长空间'
        }
        
        return {
            'sizes': recommendations,
            'width': width_analysis,
            'adjustment': '建议选择大半码以预留生长空间',
            'foot_length': foot_length_mm,
            'foot_width': foot_width_mm,
            'gender': 'kids'
        }
    
    def generate_comprehensive_report(self, foot_length_mm, foot_width_mm):
        """生成综合报告表格"""
        # 获取各类推荐
        men_rec = self._get_adult_recommendation(foot_length_mm, foot_width_mm, 'men')
        women_rec = self._get_adult_recommendation(foot_length_mm, foot_width_mm, 'women')
        
        # 判断是否需要童鞋推荐
        kids_rec = None
        if foot_length_mm <= 215:
            kids_rec = self._get_kids_recommendation(foot_length_mm, foot_width_mm)
        
        return {
            'men': men_rec,
            'women': women_rec,
            'kids': kids_rec,
            'measurements': {
                'foot_length_mm': foot_length_mm,
                'foot_width_mm': foot_width_mm,
                'ratio': foot_width_mm / foot_length_mm
            }
        }
    
    def print_recommendation_table(self, foot_length_mm, foot_width_mm):
        """打印推荐表格"""
        report = self.generate_comprehensive_report(foot_length_mm, foot_width_mm)
        
        print("\n" + "="*80)
        print("智能鞋码推荐报告\n\n")
        print("="*80)
        print(f"\n📏 测量数据:")
        print(f"脚长: {foot_length_mm:.1f} mm ({foot_length_mm/10:.1f} cm)")
        print(f"   脚宽: {foot_width_mm:.1f} mm ({foot_width_mm/10:.1f} cm)")
        print(f"\n宽长比: {foot_width_mm/foot_length_mm:.3f}")
        
        print("\n" + "-"*80)
        print("\n国际尺码推荐表:\n\n")
        
        # 创建表格数据
        headers = ['类别', '国家', '推荐尺码', '宽度类型', '特别建议']
        table_data = []
        
        # 男鞋推荐
        if report['men']:
            table_data.append(['男鞋', '中国', f"{report['men']['sizes'].get('CN', '-')}", 
                              report['men']['width']['code'], report['men']['adjustment']])
            table_data.append(['', '欧洲', f"{report['men']['sizes'].get('EU', '-')}", '', ''])
            table_data.append(['', '美国', f"{report['men']['sizes'].get('US', '-')}", '', ''])
            table_data.append(['', '英国', f"{report['men']['sizes'].get('UK', '-')}", '', ''])
            table_data.append(['', '日本', f"{report['men']['sizes'].get('JP', '-')}cm", '', ''])
            table_data.append(['-'*10, '-'*10, '-'*15, '-'*10, '-'*20])
        
        # 女鞋推荐
        if report['women']:
            table_data.append(['女鞋', '中国', f"{report['women']['sizes'].get('CN', '-')}", 
                              report['women']['width']['code'], report['women']['adjustment']])
            table_data.append(['', '欧洲', f"{report['women']['sizes'].get('EU', '-')}", '', ''])
            table_data.append(['', '美国', f"{report['women']['sizes'].get('US', '-')}", '', ''])
            table_data.append(['', '英国', f"{report['women']['sizes'].get('UK', '-')}", '', ''])
            table_data.append(['', '日本', f"{report['women']['sizes'].get('JP', '-')}cm", '', ''])
        
        # 童鞋推荐（如果适用）
        if report['kids']:
            table_data.append(['-'*10, '-'*10, '-'*15, '-'*10, '-'*20])
            table_data.append(['童鞋', '中国', f"{report['kids']['sizes'].get('CN', '-')}", 
                              '标准', report['kids']['adjustment']])
            table_data.append(['', '欧洲', f"{report['kids']['sizes'].get('EU', '-')}", '', ''])
            table_data.append(['', '美国', f"{report['kids']['sizes'].get('US', '-')}", '', ''])
            table_data.append(['', '英国', f"{report['kids']['sizes'].get('UK', '-')}", '', ''])
        
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        
        # 打印脚型分析
        print("\n" + "-"*80)
        print("\n👟 脚型分析详情:\n")
        
        if report['men']:
            print(f"【男性脚型】")
            print(f"   • 类型: {report['men']['width']['label']}")
            print(f"   • 特征: {report['men']['width']['description']}")
            print(f"   • 建议: {report['men']['width']['suggestion']}")
        
        if report['women']:
            print(f"\n【女性脚型】")
            print(f"   • 类型: {report['women']['width']['label']}")
            print(f"   • 特征: {report['women']['width']['description']}")
            print(f"   • 建议: {report['women']['width']['suggestion']}")
        
        if report['kids']:
            print(f"\n【儿童脚型】")
            print(f"   • 建议: {report['kids']['width']['suggestion']}")
        
        print("\n" + "="*80)
        print("💡 温馨提示:")
        print("   1. 不同品牌可能存在尺码差异，建议购买前试穿")
        print("   2. 运动鞋建议预留5-10mm活动空间")
        print("   3. 皮鞋和正装鞋建议选择贴合的尺码")
        print("   4. 脚部会因时间和温度略有变化，建议下午试鞋")
        print("="*80 + "\n")
        
        return report
    
    def visualize_report(self, foot_length_mm, foot_width_mm, save_path="result\shoe_size_report.png"):
        """可视化报告"""
        report = self.generate_comprehensive_report(foot_length_mm, foot_width_mm)

        # 创建图表 - 改为 (3,1) 结构
        fig = plt.figure(figsize=(5,10),dpi=100)
        fig.suptitle(f'智能鞋码推荐报告\n\n 脚长: {foot_length_mm:.1f}mm | 脚宽: {foot_width_mm:.1f}mm | 宽长比: {foot_width_mm/foot_length_mm:.3f} \n\n', 
                        fontsize=12, fontweight='bold')

        # 子图1: 尺码推荐表 + 详细建议
        ax1 = plt.subplot(3, 1, 1)
        ax1.axis('tight')
        ax1.axis('off')

        # 创建简化的表格数据
        table_data = []
        table_data.append(['类别', '中国码', '欧洲码', '美国码', '英国码', '日本码', '宽度'])

        if report['men']:
            table_data.append(['男鞋',
                                f"{report['men']['sizes'].get('CN', '-')}",
                                f"{report['men']['sizes'].get('EU', '-')}",
                                f"{report['men']['sizes'].get('US', '-')}",
                                f"{report['men']['sizes'].get('UK', '-')}",
                                f"{report['men']['sizes'].get('JP', '-')}",
                                report['men']['width']['code']])

        if report['women']:
            table_data.append(['女鞋',
                                f"{report['women']['sizes'].get('CN', '-')}",
                                f"{report['women']['sizes'].get('EU', '-')}",
                                f"{report['women']['sizes'].get('US', '-')}",
                                f"{report['women']['sizes'].get('UK', '-')}",
                                f"{report['women']['sizes'].get('JP', '-')}",
                                report['women']['width']['code']])

        if report['kids']:
            table_data.append(['童鞋',
                                f"{report['kids']['sizes'].get('CN', '-')}",
                                f"{report['kids']['sizes'].get('EU', '-')}",
                                f"{report['kids']['sizes'].get('US', '-')}",
                                f"{report['kids']['sizes'].get('UK', '-')}",
                                '-',
                                '标准'])

        # 创建表格
        table = ax1.table(cellText=table_data,
                            cellLoc='center',
                            loc='upper center',
                            bbox=[0, 0.5, 1, 0.5])  # 表格占上半部分

        table.auto_set_font_size(False)
        table.set_fontsize(9)  # 稍微减小字体
        table.scale(1, 2.0)    # 调整表格高度

        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')

        # 添加详细建议文本到下半部分
        advice_text = "【选鞋建议】\n\n"

        if report['men']:
            advice_text += f"男鞋推荐: {report['men']['sizes'].get('CN', '-')}码 (中国)  "
            advice_text += f"脚型: {report['men']['width']['description']}\n\n"
            advice_text += f"{report['men']['width']['suggestion']}\n\n"

        if report['women']:
            advice_text += f"女鞋推荐: {report['women']['sizes'].get('CN', '-')}码 (中国)  "
            advice_text += f"脚型: {report['women']['width']['description']}\n\n"
            advice_text += f"{report['women']['width']['suggestion']}\n\n"

        if report['kids']:
            advice_text += f"童鞋推荐: {report['kids']['sizes'].get('CN', '-')}码 (中国)  "
            advice_text += f"{report['kids']['width']['suggestion']}\n\n"

        advice_text += "【注意事项】\n\n"
        advice_text += "不同品牌存在差异-建议下午试鞋\n\n运动鞋预留5-10mm-皮鞋选择贴合尺码\n"

        # 将建议文本放在表格下方
        ax1.text(0.5, 0.35, advice_text, transform=ax1.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))

        ax1.set_title('国际尺码对照表 & 选鞋建议', fontsize=10, pad=15)

        # 子图2: 脚型宽度分析
        ax2 = plt.subplot(3, 1, 2)

        # 宽度比例可视化
        ratio = foot_width_mm / foot_length_mm

        # 绘制宽度标准范围
        width_ranges = {
            '窄(N)  AA': (0.30, 0.33),
            '标准(M) B/C': (0.33, 0.36),
            '宽(W)  D': (0.36, 0.39),
            '加宽(XW)  EE': (0.39, 0.42)
        }

        colors = ['lightblue', 'lightgreen', 'yellow', 'orange']
        positions = []

        for i, (label, (min_r, max_r)) in enumerate(width_ranges.items()):
            ax2.barh(0, max_r - min_r, left=min_r, height=0.5, 
                    color=colors[i], alpha=0.6, label=label)
            positions.append((min_r + max_r) / 2)

        # 标记当前脚型
        ax2.scatter([ratio], [0], s=200, c='red', marker='v', zorder=5)
        ax2.text(ratio, -0.3, f'您的脚型\n{ratio:.3f}', 
                ha='center', va='top', fontsize=9, fontweight='bold')

        ax2.set_xlim(0.28, 0.44)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('\n\n\n脚宽/脚长 比例', fontsize=10, labelpad=15)
        ax2.set_title('脚型宽度分析', fontsize=12)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3, axis='x')
        pos = ax2.get_position()        # 先取出原始位置
        ax2.set_position([
            pos.x0,                     # 左边界不变
            pos.y0 - 0.08,              # 下移 0.05（数值越大，下移越多）
            pos.width,                  # 宽度不变
            pos.height                  # 高度不变
        ])

        # 子图3: 尺码范围图
        ax3 = plt.subplot(3, 1, 3)

        # 显示不同尺码的脚长范围
        size_ranges = {
            '童鞋': (80, 215),
            '女鞋': (220, 260),
            '男鞋': (240, 310)
        }

        colors_range = ['#FFE4B5', '#FFB6C1', '#ADD8E6']
        for i, (label, (min_l, max_l)) in enumerate(size_ranges.items()):
            ax3.barh(i, max_l - min_l, left=min_l, height=0.6, 
                    color=colors_range[i], alpha=0.7, label=label)

        # 标记当前脚长
        ax3.axvline(x=foot_length_mm, color='red', linestyle='--', linewidth=1)
        ax3.text(foot_length_mm, 2.5, f'{foot_length_mm}mm', 
                ha='center', fontsize=9, fontweight='bold', color='red')
        

        ax3.set_xlim(50, 320)
        ax3.set_xlabel('脚长 (mm)', fontsize=10)
        ax3.set_yticks(range(3))
        ax3.set_yticklabels(['童鞋', '女鞋', '男鞋'], fontsize=10)
        ax3.set_title('尺码范围对照', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 报告已保存至: {save_path}")

        plt.show()

        return report


# 使用示例
def run_shoe_recommendation(foot_length_mm, foot_width_mm):
    # 初始化推荐器
    recommender = ShoeSizeRecommender()
    
    # 打印表格报告
    report = recommender.print_recommendation_table(foot_length_mm, foot_width_mm)
    
    # 可视化报告
    recommender.visualize_report(foot_length_mm, foot_width_mm)
    
    return report

