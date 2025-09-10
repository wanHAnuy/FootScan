# 足部测量系统
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def process_foot_measurement(image_path="result\warped_a4.png", save_results=True):
    """
    整合的足部测量函数：椭圆修正 + 详细测量
    """
    # 读取透视变换后的图像
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    warped_image = cv2.imread(image_path)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    height, width = warped_image.shape[:2]
    
    # A4纸实际尺寸 (毫米)
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297
    
    # 计算像素到毫米的转换比例
    pixel_to_mm_x = A4_WIDTH_MM / width
    pixel_to_mm_y = A4_HEIGHT_MM / height
    
    # ========== 第一步：检测和修正足部掩膜 ==========
    print("🔍 步骤1: 检测足部...")
    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    _, foot_threshold = cv2.threshold(gray_warped, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学处理
    kernel = np.ones((5,5), np.uint8)
    foot_clean = cv2.morphologyEx(foot_threshold, cv2.MORPH_CLOSE, kernel)
    foot_clean = cv2.morphologyEx(foot_clean, cv2.MORPH_OPEN, kernel)
    
    # 找到足部像素
    foot_pixels = np.where(foot_clean > 0)
    
    if len(foot_pixels[0]) == 0:
        print("❌ 未检测到足部")
        return None
    
    # 计算基本参数
    top_y = np.min(foot_pixels[0])
    bottom_y = np.max(foot_pixels[0])
    foot_length_pixels = bottom_y - top_y
    foot_length_mm = foot_length_pixels * pixel_to_mm_y
    
    print(f"✅ 检测到足部，足长: {foot_length_mm:.1f} mm")
    
    # ========== 第二步：椭圆修正足后跟 ==========
    print("🔧 步骤2: 椭圆修正足后跟区域...")
    
    # 计算足后跟开始线（82%位置）
    heel_start_y = int(top_y + foot_length_pixels * 0.82)
    
    # 在足后跟开始线处找到足部宽度
    foot_pixels_at_heel_start = np.where(foot_clean[heel_start_y, :] > 0)[0]
    if len(foot_pixels_at_heel_start) > 0:
        left_x = np.min(foot_pixels_at_heel_start)
        right_x = np.max(foot_pixels_at_heel_start)
        
        # 椭圆参数
        ellipse_center_x = (left_x + right_x) / 2
        ellipse_center_y = heel_start_y
        ellipse_width = right_x - left_x
        ellipse_height = 2 * (height - 1 - heel_start_y)
        
        # 创建椭圆掩膜
        y_coords, x_coords = np.ogrid[:height, :width]
        ellipse_mask = ((x_coords - ellipse_center_x)**2 / (ellipse_width/2)**2 + 
                       (y_coords - ellipse_center_y)**2 / (ellipse_height/2)**2) <= 1
        
        # 创建足后跟区域掩膜
        heel_region_mask = y_coords >= heel_start_y
        
        # 修改掩膜：足后跟区域只保留椭圆内的部分
        modified_mask = foot_clean.copy()
        heel_outside_ellipse = heel_region_mask & (~ellipse_mask)
        modified_mask[heel_outside_ellipse] = 0
        
        print(f"✅ 椭圆修正完成（足后跟起始位置: {heel_start_y}px）")
    else:
        modified_mask = foot_clean
        print("⚠️ 无法进行椭圆修正，使用原始掩膜")
    
    # ========== 第三步：详细测量 ==========
    print("\n📏 步骤3: 每5mm测量足宽...")
    
    # 重新计算修正后的足部范围
    foot_pixels_modified = np.where(modified_mask > 0)
    top_y = np.min(foot_pixels_modified[0])
    bottom_y = np.max(foot_pixels_modified[0])
    foot_length_pixels = bottom_y - top_y
    foot_length_mm = foot_length_pixels * pixel_to_mm_y
    
    # 测量参数
    measurement_interval_mm = 5  # 5毫米间隔
    measurement_interval_pixels = measurement_interval_mm / pixel_to_mm_y
    num_measurements = int(foot_length_mm / measurement_interval_mm) + 1
    
    # 存储测量结果
    measurement_positions_mm = []
    measurement_widths_mm = []
    measurement_y_pixels = []
    left_edge_points_mm = []
    right_edge_points_mm = []
    center_points_mm = []
    
    print("\n距脚尖距离(mm) | 足宽(mm) | 足宽(cm)")
    print("-" * 40)
    
    for i in range(num_measurements):
        distance_from_top_mm = i * measurement_interval_mm
        distance_from_top_pixels = distance_from_top_mm / pixel_to_mm_y
        current_y = int(top_y + distance_from_top_pixels)
        
        if current_y >= bottom_y:
            break
        
        # 在当前行找足部像素（使用修正后的掩膜）
        foot_pixels_in_row = np.where(modified_mask[current_y, :] > 0)[0]
        
        if len(foot_pixels_in_row) > 0:
            left_x = np.min(foot_pixels_in_row)
            right_x = np.max(foot_pixels_in_row)
            width_pixels = right_x - left_x
            width_mm = width_pixels * pixel_to_mm_x
            
            measurement_positions_mm.append(distance_from_top_mm)
            measurement_widths_mm.append(width_mm)
            measurement_y_pixels.append(current_y)
            
            # 记录左右边缘和中心点（单位：毫米）
            left_edge_mm = left_x * pixel_to_mm_x
            right_edge_mm = right_x * pixel_to_mm_x
            center_mm = ((left_x + right_x) / 2) * pixel_to_mm_x
            left_edge_points_mm.append(left_edge_mm)
            right_edge_points_mm.append(right_edge_mm)
            center_points_mm.append(center_mm)
            
            print(f"{distance_from_top_mm:8.1f}      | {width_mm:7.1f} | {width_mm/10:6.2f}")
    
    # 找到最宽的位置
    if measurement_widths_mm:
        max_width_mm = max(measurement_widths_mm)
        max_width_idx = measurement_widths_mm.index(max_width_mm)
        max_width_position = measurement_positions_mm[max_width_idx]
        
        print(f"\n🎯 最宽位置: 距脚尖 {max_width_position:.1f}mm 处，宽度 {max_width_mm:.1f}mm")

    
    
    # ========== 第四步：可视化结果 ==========

     
    print("\n📊 步骤4: 生成可视化...")
    
    plt.figure(figsize=(8, 8))
    

    
    # 子图2: 原始掩膜
    plt.subplot(2, 3, 1)
    plt.imshow(foot_clean, cmap='gray')
    plt.title("原始足部掩膜")
    plt.axis('off')
    
    # 子图3: 修正后掩膜
    plt.subplot(2, 3, 2)
    plt.imshow(modified_mask, cmap='gray')
    if 'heel_start_y' in locals():
        plt.axhline(y=heel_start_y, color='blue', linewidth=2, linestyle='--')
    plt.title("椭圆修正后掩膜")
    plt.axis('off')
    
    # 子图4:（原始图像 + 测量线）中添加测量点的绘制
    plt.subplot(2, 3, 3)
    plt.imshow(warped_image)

    # 绘制测量线
    colors = plt.cm.rainbow(np.linspace(0, 1, len(measurement_y_pixels)))
    for y_pos, color in zip(measurement_y_pixels, colors):
        plt.axhline(y=y_pos, color=color, alpha=0.5, linewidth=1)

    # 添加这部分：绘制测量点
    for i, (y_pos, left_mm, right_mm, center_mm) in enumerate(zip(measurement_y_pixels, left_edge_points_mm, right_edge_points_mm, center_points_mm)):
        # 将毫米坐标转换回像素坐标
        left_x = left_mm / pixel_to_mm_x
        right_x = right_mm / pixel_to_mm_x
        center_x = center_mm / pixel_to_mm_x
        
        # 绘制左边缘点（蓝色）
        plt.plot(left_x, y_pos, 'bo', markersize=3, alpha=0.8)
        # 绘制右边缘点（红色）
        plt.plot(right_x, y_pos, 'ro', markersize=3, alpha=0.8)
        # 绘制中心点（绿色）
        plt.plot(center_x, y_pos, 'go', markersize=3, alpha=0.8)

    # 添加图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=6, label='左边缘点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=6, label='右边缘点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=6, label='中心点')
    ]
    plt.legend(handles=legend_elements, loc='upper right')   

    
    # 子图5-6: 足宽变化曲线
    plt.subplot(2, 3, (4, 6))
    plt.plot(measurement_positions_mm, measurement_widths_mm, 'b-o', linewidth=2, markersize=4)
    
    if measurement_widths_mm:
        plt.plot(max_width_position, max_width_mm, 'ro', markersize=7, label='最宽处')
        
        # 标记足后跟起始位置
        if 'heel_start_y' in locals():
            heel_start_mm = (heel_start_y - top_y) * pixel_to_mm_y
            plt.axvline(x=heel_start_mm, color='blue', linestyle='--', alpha=0.7, label='足后跟起始')
    
    plt.xlabel('距脚尖距离 (mm)')
    plt.ylabel('足宽 (mm)')
    plt.title('足部宽度变化曲线 (每5mm测量)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    if measurement_widths_mm:
        avg_width = np.mean(measurement_widths_mm)
        plt.text(0.02, 0.98, 
                f'足长: {foot_length_mm:.1f}mm\n'
                f'最大足宽: {max_width_mm:.1f}mm\n'
                f'平均足宽: {avg_width:.1f}mm\n'
                f'测量点数: {len(measurement_widths_mm)}',
                transform=plt.gca().transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()

    foot_measurement_summary_path="result/foot_measurement_summary.png"
    try:
        plt.savefig(foot_measurement_summary_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {foot_measurement_summary_path}")
    except Exception as e:
        print(f"保存失败: {e}")
        print(f"✅ 可视化完成并保存到 {foot_measurement_summary_path}")
    
    # ========== 第五步：保存结果 ==========
    if save_results:
        # 保存修正后的掩膜
        cv2.imwrite("result\modified_foot_mask.png", modified_mask)
        print(f"\n💾 修正后的掩膜已保存到 result\modified_foot_mask.png")
        
        # 保存测量数据
        measurement_data = {
            'positions_mm': measurement_positions_mm,
            'widths_mm': measurement_widths_mm,
            'left_edge_points_mm': left_edge_points_mm,
            'right_edge_points_mm': right_edge_points_mm,
            'center_points_mm': center_points_mm,
            'foot_length_mm': foot_length_mm,
            'max_width_mm': max_width_mm if measurement_widths_mm else 0,
            'max_width_position_mm': max_width_position if measurement_widths_mm else 0,
            'measurement_interval_mm': measurement_interval_mm,
            'heel_correction_applied': 'heel_start_y' in locals()
        }
        
        with open('result\\foot_measurements.json', 'w') as f:
            json.dump(measurement_data, f, indent=2)
        
        print(f"💾 测量数据已保存到 foot_measurements.json")
    
    return measurement_data, foot_measurement_summary_path
