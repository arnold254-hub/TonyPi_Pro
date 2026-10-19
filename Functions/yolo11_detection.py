"""
实时年龄性别检测程序
使用OpenCV打开摄像头，基于YOLOv11训练的age_gender模型进行实时检测
"""

import cv2
from ultralytics import YOLO
import os
import random
import time
import threading

from speech import speech
import hiwonder.ActionGroupControl as AGC


class AgeGenderRealTimeDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """
        初始化检测器

        Args:
            model_path: 模型路径 (best.pt)
            conf_threshold: 置信度阈值
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # 类别名称
        self.class_names = [
            "female_child",
            "female_youth",
            "female_adult",
            "female_middle",
            "female_elder",
            "male_child",
            "male_youth",
            "male_adult",
            "male_middle",
            "male_elder"
        ]

        # 为不同类别定义不同的颜色，并维护音频路径
        self.colors = {
            "female_child": (255, 192, 203),      # 浅粉色
            "female_youth": (255, 105, 180),      # 热粉色
            "female_adult": (255, 20, 147),       # 深粉色
            "female_middle": (219, 112, 147),     # 苍紫罗兰色
            "female_elder": (186, 85, 211),       # 中紫色
            "male_child": (173, 216, 230),        # 浅蓝色
            "male_youth": (30, 144, 255),         # 道奇蓝
            "male_adult": (0, 0, 255),            # 红色(BGR)
            "male_middle": (0, 100, 255),         # 橙色(BGR)
            "male_elder": (0, 0, 139),            # 深红色(BGR)
        }
        self.audio_map = self._load_greet_audio()
        self.play_lock = threading.Lock()
        self.cooldown_until = 0.0
        speech.set_volume(80)

        print("✓ 模型加载成功\n")

    def draw_detections(self, frame, results):
        """
        在图像上绘制检测结果

        Args:
            frame: 输入帧
            results: YOLO检测结果

        Returns:
            绘制后的帧
        """
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 获取置信度
                conf = float(box.conf[0])

                # 获取类别
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]

                # 获取对应的颜色
                color = self.colors.get(class_name, (0, 255, 0))

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 准备标签文本
                label = f"{class_name} ({conf:.2f})"

                # 获取文本大小
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                # 计算文本背景位置
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + text_size[1] + 10

                # 绘制文本背景
                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0], text_y + 5),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

        return frame
    def _load_greet_audio(self):
        base_audio_dir = os.path.join(os.path.dirname(__file__), 'greet_audio')
        audio_map = {name: [] for name in self.class_names}
        for cls in self.class_names:
            cls_dir = os.path.join(base_audio_dir, cls)
            if os.path.isdir(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.lower().endswith(('.wav', '.mp3', '.ogg')):
                        audio_map[cls].append(os.path.join(cls_dir, file))
        return audio_map

    def _trigger_greeting(self, class_name):
        with self.play_lock:
            now = time.time()
            if now < self.cooldown_until:
                return
            audio_list = self.audio_map.get(class_name) or []
            if not audio_list:
                print(f"[WARN] 没有找到 {class_name} 的问候音频")
                return
            audio_file = random.choice(audio_list)
            print(f"[INFO] 播放问候音频: {audio_file}")

            def play_audio():
                try:
                    speech.play_audio(audio_file)
                except Exception as exc:
                    print(f"[ERROR] 播放音频失败: {exc}")

            def play_action():
                try:
                    AGC.runActionGroup('wave')
                except Exception as exc:
                    print(f"[ERROR] 播放动作失败: {exc}")

            t_audio = threading.Thread(target=play_audio, daemon=True)
            t_action = threading.Thread(target=play_action, daemon=True)
            t_audio.start()
            t_action.start()
            t_audio.join()
            t_action.join()

            self.cooldown_until = time.time() + 10.0

    def run(self, camera_id=0):
        """
        运行实时检测

        Args:
            camera_id: 摄像头ID (默认0为系统默认摄像头)
        """
        print(f"正在打开摄像头 {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"✗ 无法打开摄像头 {camera_id}")
            return

        print("✓ 摄像头已打开")
        print("\n快捷键:")
        print("  q  - 退出程序")
        print("  s  - 保存当前帧")
        print("  p  - 暂停/继续")
        print()

        # 设置摄像头参数以提高性能
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        paused = False
        last_frame = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("✗ 无法读取视频帧")
                    break

                frame_count += 1
                last_frame = frame.copy()

                # 运行检测
                results = self.model(frame, conf=self.conf_threshold, verbose=False)

                # 绘制检测结果
                frame = self.draw_detections(frame, results)
                for result in results:
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_threshold:
                            continue
                        cls_id = int(box.cls[0])
                        class_name = self.class_names[cls_id]
                        self._trigger_greeting(class_name)
            else:
                frame = last_frame.copy()

            # 在左上角显示帧信息
            info_text = f"Frame: {frame_count} | Paused: {paused}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示帧
            cv2.imshow("Age Gender Real-time Detection", frame)

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n✓ 程序已退出")
                break
            elif key == ord('s'):
                filename = f"capture_{frame_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ 帧已保存: {filename}")
            elif key == ord('p'):
                paused = not paused
                pause_status = "已暂停" if paused else "已继续"
                print(f"✓ {pause_status}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    """主函数"""
    print("=" * 60)
    print("OpenCV 实时年龄性别检测程序")
    print("=" * 60)
    print()

    # 使用训练好的最佳模型
    model_path = "./model/best.pt"

    try:
        detector = AgeGenderRealTimeDetector(model_path, conf_threshold=0.3)
        detector.run(camera_id=0)
    except FileNotFoundError as e:
        print(f"✗ 错误: {e}")
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
