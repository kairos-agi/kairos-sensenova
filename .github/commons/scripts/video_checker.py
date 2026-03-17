# -*- coding: utf-8 -*-
import os
import argparse
import math
from typing import Tuple, Dict, Union, List
import cv2
import numpy as np


class VideoConsistencyChecker:
    """
    视频一致性校验器
    功能：
        比较两个视频的 帧率，分辨率，时长，帧数
        比较两个视频的 PSNR, SSIM   
    """
    
    def __init__(self, ssim_threshold:float = 0.93, psnr_threshold:int = 35):
        """
        初始化校验器
        :param ssim_threshold: 一致性阈值（0-1），默认93%
        """
        self.ssim_threshold = ssim_threshold
        self.psnr_threshold = psnr_threshold


    def get_video_metadata(self, video_path: str) -> Dict[str, Union[float, int, Tuple[int, int]]]:
        """
        获取视频元数据（帧率、分辨率、时长、总帧数）
        :param video_path: 视频路径
        :return: 元数据字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        
        # 基础元数据
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 当 total_frames 不可靠时（如某些格式返回0），尝试逐帧计数
        if total_frames <= 0:
            print(f"警告: OpenCV 无法获取 {video_path} 的准确总帧数，将逐帧计数（可能较慢）")
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            # 重置读取位置（后面会重新打开，这里只是计数）
        
        cap.release()
        
        # 重新打开视频以获取时长（因为之前可能已经读完）
        cap = cv2.VideoCapture(video_path)
        duration = total_frames / fps if (fps > 0 and total_frames > 0) else 0
        cap.release()
        
        # 舍入到合理精度
        fps = round(max(fps, 0.0), 2)
        duration = round(max(duration, 0.0), 2)
        total_frames = max(total_frames, 0)
        
        return {
            "fps": fps,
            "resolution": (width, height),
            "duration": duration,
            "total_frames": total_frames
        }

    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算单帧SSIM（结构相似性），越接近1越相似
        :param img1: GT帧（灰度图，uint8）
        :param img2: 生成帧（灰度图，uint8）
        :return: SSIM值
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算单帧PSNR（峰值信噪比），单位dB，越大越相似
        :param img1: GT帧（灰度图，uint8）
        :param img2: 生成帧（灰度图，uint8）
        :return: PSNR值
        """
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255 ** 2) / mse)

    def compare_videos(self, gt_video: str, gen_video: str) -> Dict[str, Union[bool, float, str, List]]:
        """
        核心对比逻辑：先校验元数据，再逐帧计算SSIM（内存中），最后判断一致性
        :param gt_video: GT视频路径
        :param gen_video: 生成视频路径
        :return: 对比结果字典
        """
        # 1. 校验元数据
        print("=== 开始校验视频元数据 ===")
        gt_meta = self.get_video_metadata(gt_video)
        gen_meta = self.get_video_metadata(gen_video)
        
        # 元数据对比（允许微小误差）
        meta_check = {
            "fps": math.isclose(gt_meta["fps"], gen_meta["fps"], abs_tol=0.01),
            "resolution": gt_meta["resolution"] == gen_meta["resolution"],
            "duration": math.isclose(gt_meta["duration"], gen_meta["duration"], abs_tol=0.1),
            "total_frames": math.isclose(gt_meta["total_frames"], gen_meta["total_frames"], abs_tol=1)
        }
        
        if not all(meta_check.values()):
            error_msg = f"元数据不一致：\nGT: {gt_meta}\nGen: {gen_meta}"
            print(f"❌ 帧率，分辨率，时长，帧数校验失败: {error_msg}")
            return {
                "pass": False,
                "consistency_score": 0.0,
                "avg_psnr": 0.0,
                "error": error_msg,
                "meta_check": meta_check
            }
        print("✅ 帧率，分辨率，时长，帧数校验通过")

        # 2. 打开两个视频
        cap_gt = cv2.VideoCapture(gt_video)
        cap_gen = cv2.VideoCapture(gen_video)
        if not cap_gt.isOpened() or not cap_gen.isOpened():
            raise RuntimeError("无法打开视频文件进行帧对比")

        # 3. 逐帧对比（内存中）
        print("=== 开始逐帧计算一致性（SSIM + PSNR） ===")
        ssim_scores = []
        psnr_scores = []
        frame_idx = 0

        while True:
            ret_gt, frame_gt = cap_gt.read()
            ret_gen, frame_gen = cap_gen.read()

            # 任一视频结束即停止
            if not ret_gt or not ret_gen:
                if ret_gt != ret_gen:
                    print(f"⚠️ 视频帧数不一致：GT读取到{frame_idx}帧，生成视频提前结束" if ret_gt else f"生成视频读取到{frame_idx}帧，GT视频提前结束")
                break

            # 转为灰度图（SSIM/PSNR计算基础）
            gray_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)
            gray_gen = cv2.cvtColor(frame_gen, cv2.COLOR_BGR2GRAY)

            # 计算指标
            ssim_val = self.calculate_ssim(gray_gt, gray_gen)
            psnr_val = self.calculate_psnr(gray_gt, gray_gen)

            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)

            # 进度提示（每10帧）
            if frame_idx % 10 == 0:
                print(f"进度: 第 {frame_idx} 帧，SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB")

            frame_idx += 1

        # 释放资源
        cap_gt.release()
        cap_gen.release()

        # 4. 计算整体得分
        if not ssim_scores:
            raise RuntimeError("未读取到任何帧，对比失败")

        avg_ssim = np.mean(ssim_scores)
        # 过滤出有限 PSNR 值
        finite_psnr = [p for p in psnr_scores if p != float('inf')]

        if finite_psnr:
            avg_psnr = np.mean(finite_psnr)
        else:
            avg_psnr = float('inf')   # 所有帧都完美，平均 PSNR 为无穷大

        pass_flag = avg_ssim >= self.ssim_threshold and avg_psnr >= self.psnr_threshold

        # 结果汇总
        result = {
            "pass": pass_flag,
            "consistency_score": round(avg_ssim, 4),
            "avg_psnr": round(avg_psnr, 2) if avg_psnr != float('inf') else float('inf'),
            "ssim_threshold": self.ssim_threshold,
            "meta_check": meta_check,
            "total_frames": frame_idx,
            "error": None
        }

        if pass_flag:
            print(f"✅ 视频一致性校验通过！平均SSIM={avg_ssim:.4f} (阈值={self.ssim_threshold})，平均PSNR={avg_psnr:.2f} dB (阈值={self.psnr_threshold})")
        else:
            print(f"❌ 视频一致性校验失败！平均SSIM={avg_ssim:.4f} (阈值={self.ssim_threshold})，平均PSNR={avg_psnr:.2f} dB (阈值={self.psnr_threshold})")

        return result


def main():
    parser = argparse.ArgumentParser(description="视频一致性校验工具")
    parser.add_argument("--gt", required=True, help="GT视频路径")
    parser.add_argument("--gen", required=True, help="生成视频路径")
    parser.add_argument("--ssim_threshold", type=float, default=0.90, help="一致性阈值（0-1），默认0.93")
    parser.add_argument("--psnr_threshold", type=float, default=30, help="一致性阈值（35～40 dB），默认35dB")
    
    args = parser.parse_args()

    checker = VideoConsistencyChecker(ssim_threshold=args.ssim_threshold, psnr_threshold=args.psnr_threshold)
    try:
        result = checker.compare_videos(args.gt, args.gen)
        exit(0) if result["pass"] else exit(1)
    except Exception as e:
        print(f"❌ 校验过程出错: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()