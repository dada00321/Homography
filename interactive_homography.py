"""
點擊圖片選擇四個角落，然後自動進行透視變換
新功能:選擇角落時顯示縮小50%的圖片，座標自動還原到原始尺寸
"""

import cv2
import numpy as np
import os


class InteractiveHomography:
    """互動式 Homography 轉換器（支援縮放顯示）"""
    
    def __init__(self, image_path):
        X = os.getcwd().replace('\\','/')
        self.image_path = f"{X}/interactive/{image_path}"
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        
        # 保存原始圖片和尺寸
        self.original_image = self.image.copy()
        self.original_height, self.original_width = self.image.shape[:2]
        
        # 計算縮小後的尺寸（長寬各縮小一半）
        self.display_width = self.original_width // 2
        self.display_height = self.original_height // 2
        self.scale_factor = 2.0  # 還原時需要乘以的比例
        
        # 創建縮小版本用於顯示和選擇
        self.display_image = cv2.resize(self.image, (self.display_width, self.display_height))
        
        self.corners = []  # 儲存縮小圖片上的座標
        self.original_corners = []  # 儲存還原後的原始座標
        self.window_name = "選擇四個角落 (依序: 左上→右上→右下→左下) [已縮小50%顯示]"
        
        print(f"\n📐 圖片資訊:")
        print(f"   原始尺寸: {self.original_width} x {self.original_height}")
        print(f"   顯示尺寸: {self.display_width} x {self.display_height} (縮小50%)")
        print(f"   座標還原比例: {self.scale_factor}x\n")
        
    def mouse_callback(self, event, x, y, flags, param):
        """滑鼠事件回調函數"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            # 記錄點擊的座標（縮小圖片上的座標）
            self.corners.append([x, y])
            
            # 計算還原後的原始座標
            original_x = int(x * self.scale_factor)
            original_y = int(y * self.scale_factor)
            self.original_corners.append([original_x, original_y])
            
            # 在縮小的圖片上繪製點
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, str(len(self.corners)), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # 顯示原始座標
            print(f"✓ 點 {len(self.corners)}: 顯示座標 ({x}, {y}) → 原始座標 ({original_x}, {original_y})")
            
            # 如果已經有兩個以上的點，繪製連線
            if len(self.corners) > 1:
                cv2.line(self.display_image, 
                        tuple(self.corners[-2]), 
                        tuple(self.corners[-1]), 
                        (0, 0, 255), 2)
            
            # 如果已經選擇了四個點，閉合多邊形
            if len(self.corners) == 4:
                cv2.line(self.display_image, 
                        tuple(self.corners[3]), 
                        tuple(self.corners[0]), 
                        (0, 0, 255), 2)
                print("\n已選擇四個角落！按任意鍵繼續...")
            
            # 更新顯示
            cv2.imshow(self.window_name, self.display_image)
    
    def select_corners(self):
        """讓用戶選擇四個角落"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        print("請依序點擊四個角落：")
        print("   1. 左上角")
        print("   2. 右上角")
        print("   3. 右下角")
        print("   4. 左下角")
        print("\n選擇完成後按任意鍵繼續...\n")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(self.corners) != 4:
            raise ValueError("需要選擇四個角落！")
        
        # 返回還原後的原始座標
        return np.array(self.original_corners, dtype="float32")
    
    def transform(self, output_size=None):
        """執行透視變換（使用原始圖片和還原後的座標）"""
        if len(self.original_corners) != 4:
            raise ValueError("需要先選擇四個角落！")
        
        # 使用還原後的原始座標
        corners = np.array(self.original_corners, dtype="float32")
        
        # 計算輸出尺寸
        if output_size is None:
            # 自動計算（基於原始座標）
            widthA = np.linalg.norm(corners[2] - corners[3])
            widthB = np.linalg.norm(corners[1] - corners[0])
            max_width = max(int(widthA), int(widthB))
            
            heightA = np.linalg.norm(corners[1] - corners[2])
            heightB = np.linalg.norm(corners[0] - corners[3])
            max_height = max(int(heightA), int(heightB))
        else:
            max_width, max_height = output_size
        
        # 定義目標點
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # 計算 Homography 矩陣
        H, status = cv2.findHomography(corners, dst)
        
        # 執行透視變換（使用原始圖片）
        warped = cv2.warpPerspective(self.original_image, H, (max_width, max_height))
        
        print(f"\n輸出資訊:")
        print(f"   變換後尺寸: {max_width} x {max_height}")
        
        return warped, H


def interactive_transform(image_path, output_path=None):
    """
    互動式透視變換的主函數
    
    Args:
        image_path: 輸入圖片路徑
        output_path: 輸出圖片路徑（可選）
        
    Returns:
        warped: 變換後的圖片
        H: Homography 矩陣
    """
    print("="*70)
    print("=== 互動式 Homography 透視變換 v2 ===")
    print("="*70)
    
    # 創建互動式轉換器
    transformer = InteractiveHomography(image_path)
    
    # 讓用戶選擇角落（在縮小的圖片上）
    corners = transformer.select_corners()
    
    print(f"\n選擇的原始座標:")
    for i, corner in enumerate(corners):
        print(f"   點 {i+1}: ({int(corner[0])}, {int(corner[1])})")
    
    # 執行轉換（使用原始圖片和還原的座標）
    warped, H = transformer.transform()
    
    print(f"\nHomography 矩陣:")
    print(H)
    
    # 儲存結果
    if output_path is None:
        output_path = 'interactive/interactive_result.jpg'
    cv2.imwrite(output_path, warped)
    print(f"\n結果已儲存至: {output_path}")
    
    # 顯示結果
    #cv2.imshow("變換結果", warped)
    #print("\n按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return warped, H

if __name__ == "__main__":
    print("="*70)
    print("=== 互動式 Homography 透視變換 v2 ===")
    print("="*70)
    print("\n✨ 新功能:")
    print("   • 選擇角落時顯示縮小50%的圖片（方便大圖操作）")
    print("   • 座標自動還原到原始尺寸")
    print("   • 使用原始高解析度圖片進行變換")
    print("\n使用方式:")
    print(">>> warped, H = interactive_transform('your_image.jpg')")
    print("\n程式會開啟一個視窗讓，你點擊圖片上的四個角落")
    print("依序點擊：左上、右上、右下、左下")
    print("="*70)
    
    # 如果要測試，請取消註解以下程式碼：
    warped, H = interactive_transform('example.jpg')