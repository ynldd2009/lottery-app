
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
福彩专业分析预测系统 - 主应用程序类
"""

import numpy as np
import random
import math
import requests
import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QLabel, QPushButton, QComboBox, QSpinBox, QListWidget, 
                             QCheckBox, QMessageBox, QInputDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QAbstractItemView, 
                             QFileDialog, QTextEdit, QHBoxLayout, QSplitter, QGroupBox,
                             QGridLayout, QScrollArea, QDialog, QLineEdit, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QPainter, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import qrcode
from PIL import Image
import cv2
from pyzbar.pyzbar import decode

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预测模型配置
PREDICTION_MODELS = {
    "1": {"name": "随机森林", "code": "random_forest"},
    "2": {"name": "线性回归", "code": "linear_regression"},
    "3": {"name": "长短记忆(LSTM)", "code": "lstm"},
    "4": {"name": "区间模型", "code": "range_model"},
    "5": {"name": "马尔科夫链模型", "code": "markov_chain"},
    "6": {"name": "贝叶斯概率模型", "code": "bayesian"},
    "7": {"name": "分形模型", "code": "fractal"},
    "8": {"name": "混沌模型", "code": "chaos"},
    "9": {"name": "蝌蚪模型", "code": "tadpole"},
    "10": {"name": "双色球蓝球预测", "code": "ssq_blue"},
    "11": {"name": "AI预测(DeepSeek)", "code": "deepseek_ai"}
}

# 奖金配置
PRIZE_RULES = {
    "双色球": {
        "一等奖": {"base": 5000000, "description": "6+1"},
        "二等奖": {"base": 200000, "description": "6+0"},
        "三等奖": {"base": 3000, "description": "5+1"},
        "四等奖": {"base": 200, "description": "5+0或4+1"},
        "五等奖": {"base": 10, "description": "4+0或3+1"},
        "六等奖": {"base": 5, "description": "2+1或1+1或0+1"}
    },
    "快乐8": {
        "选十中十": {"base": 5000000, "description": "选10中10"},
        "选十中九": {"base": 8000, "description": "选10中9"},
        "选十中八": {"base": 800, "description": "选10中8"},
        "选十中七": {"base": 80, "description": "选10中7"},
        "选十中六": {"base": 5, "description": "选10中6"},
        "选十中五": {"base": 3, "description": "选10中5"},
        "选十中零": {"base": 2, "description": "选10中0"},
        "选九中九": {"base": 300000, "description": "选9中9"},
        "选九中八": {"base": 2000, "description": "选9中8"},
        "选九中七": {"base": 200, "description": "选9中7"},
        "选九中六": {"base": 20, "description": "选9中6"},
        "选九中五": {"base": 5, "description": "选9中5"},
        "选九中四": {"base": 3, "description": "选9中4"}
    },
    "3D": {
        "直选": {"base": 1040, "description": "按位全中"},
        "组选三": {"base": 346, "description": "两个重复号码"},
        "组选六": {"base": 173, "description": "三个不同号码"}
    },
    "七乐彩": {
        "一等奖": {"base": 5000000, "description": "7个基本号码全中"},
        "二等奖": {"base": 10000, "description": "6个基本号码+特别号码"},
        "三等奖": {"base": 1000, "description": "6个基本号码"},
        "四等奖": {"base": 100, "description": "5个基本号码+特别号码"},
        "五等奖": {"base": 50, "description": "5个基本号码"},
        "六等奖": {"base": 10, "description": "4个基本号码+特别号码"},
        "七等奖": {"base": 5, "description": "4个基本号码"}
    }
}

class LotteryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("福彩应用 - 专业彩票分析预测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 初始化代理和用户代理
        self.ua = UserAgent()
        self.proxies = None
        
        # 投注记录存储
        self.betting_history = []
        self.history_data = {}  # 存储历史开奖数据
        self.current_game = ""
        self.api_key = "7c44e685-b52c-4582-addf-0c8ef6a64916"  # 默认API密钥
        self.last_omission_data = {}  # 存储号码遗漏数据
        self.frequency_data = {}     # 存储号码出现频率数据
        self.cold_hot_data = {}      # 存储冷温热数据
        self.auto_update_timer = None  # 自动更新定时器
        
        # 创建主界面
        self.init_ui()
        
        # 加载历史数据和API密钥
        self.load_api_key()
        self.load_history_data()
        self.load_betting_history()
        self.calculate_omission()  # 计算初始遗漏数据
        self.auto_update_data()    # 自动更新数据
    
    def init_ui(self):
        # 主选项卡
        self.tabs = QTabWidget()
        
        # 首页选项卡
        self.home_tab = QWidget()
        self.init_home_tab()
        self.tabs.addTab(self.home_tab, "首页")
        
        # 福彩选项卡
        self.fucai_tab = QWidget()
        self.init_fucai_tab()
        self.tabs.addTab(self.fucai_tab, "福彩投注")
        
        # 数据分析选项卡
        self.analysis_tab = QWidget()
        self.init_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "数据分析")
        
        # 投注记录选项卡
        self.records_tab = QWidget()
        self.init_records_tab()
        self.tabs.addTab(self.records_tab, "投注记录")
        
        # 开奖历史选项卡
        self.history_tab = QWidget()
        self.init_history_tab()
        self.tabs.addTab(self.history_tab, "开奖历史")
        
        # 开奖记录选项卡（网络抓取）
        self.network_tab = QWidget()
        self.init_network_tab()
        self.tabs.addTab(self.network_tab, "开奖记录")
        
        # 号码缩水选项卡
        self.shrink_tab = QWidget()
        self.init_shrink_tab()
        self.tabs.addTab(self.shrink_tab, "号码缩水")
        
        # 矩阵投注选项卡
        self.matrix_tab = QWidget()
        self.init_matrix_tab()
        self.tabs.addTab(self.matrix_tab, "矩阵投注")
        
        # 走势图选项卡
        self.trend_tab = QWidget()
        self.init_trend_tab()
        self.tabs.addTab(self.trend_tab, "走势图")
        
        self.setCentralWidget(self.tabs)
    
    def init_home_tab(self):
        """初始化首页"""
        layout = QVBoxLayout()
        
        # 标题区域
        title_layout = QHBoxLayout()
        
        title_label = QLabel("福彩专业分析预测系统")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #E74C3C; margin: 10px;")
        title_layout.addWidget(title_label)
        
        # 当前时间
        self.current_time_label = QLabel()
        self.current_time_label.setStyleSheet("font-size: 16px; color: #2C3E50;")
        title_layout.addWidget(self.current_time_label)
        
        layout.addLayout(title_layout)
        
        # 滚动公告
        self.marquee_label = QLabel("欢迎使用福彩专业分析预测系统！今日开奖：双色球(21:15) 快乐8(21:30) 3D(21:15) 七乐彩(21:15)")
        self.marquee_label.setStyleSheet("font-size: 16px; color: #E74C3C; background-color: #FFF3CD; padding: 10px; border-radius: 5px;")
        self.marquee_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.marquee_label)
        
        # 快速操作按钮
        quick_btn_layout = QHBoxLayout()
        
        live_btn = QPushButton("开奖直播")
        live_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #E74C3C; color: white;")
        live_btn.clicked.connect(self.show_live_broadcast)
        quick_btn_layout.addWidget(live_btn)
        
        qr_btn = QPushButton("扫码兑奖")
        qr_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #3498DB; color: white;")
        qr_btn.clicked.connect(self.show_qr_scanner)
        quick_btn_layout.addWidget(qr_btn)
        
        trend_btn = QPushButton("走势图")
        trend_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #27AE60; color: white;")
        trend_btn.clicked.connect(self.show_trend_charts)
        quick_btn_layout.addWidget(trend_btn)
        
        predict_btn = QPushButton("号码预测")
        predict_btn.setStyleSheet("font-size: 16px; padding: 10px; background-color: #9B59B6; color: white;")
        predict_btn.clicked.connect(self.show_prediction)
        quick_btn_layout.addWidget(predict_btn)
        
        layout.addLayout(quick_btn_layout)
        
        # 最新开奖信息
        latest_group = QGroupBox("最新开奖信息")
        latest_layout = QVBoxLayout()
        
        self.latest_results_text = QTextEdit()
        self.latest_results_text.setReadOnly(True)
        self.latest_results_text.setStyleSheet("font-size: 14px;")
        latest_layout.addWidget(self.latest_results_text)
        
        latest_group.setLayout(latest_layout)
        layout.addWidget(latest_group)
        
        # 中奖记录
        prize_group = QGroupBox("最近中奖记录")
        prize_layout = QVBoxLayout()
        
        self.prize_records_text = QTextEdit()
        self.prize_records_text.setReadOnly(True)
        self.prize_records_text.setStyleSheet("font-size: 14px;")
        prize_layout.addWidget(self.prize_records_text)
        
        prize_group.setLayout(prize_layout)
        layout.addWidget(prize_group)
        
        # 底部信息
        bottom_layout = QHBoxLayout()
        
        # 截止时间
        self.deadline_label = QLabel()
        self.deadline_label.setStyleSheet("font-size: 14px; color: #E74C3C;")
        bottom_layout.addWidget(self.deadline_label)
        
        # 版本信息
        version_label = QLabel("v2.0 © 2025 福彩分析系统")
        version_label.setStyleSheet("font-size: 12px; color: #7F8C8D;")
        bottom_layout.addWidget(version_label)
        
        layout.addLayout(bottom_layout)
        
        self.home_tab.setLayout(layout)
        
        # 启动定时器
        self.home_timer = QTimer()
        self.home_timer.timeout.connect(self.update_home_info)
        self.home_timer.start(1000)
        
        # 初始化首页信息
        self.update_home_info()
    
    def update_home_info(self):
        """更新首页信息"""
        # 更新时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_time_label.setText(f"当前时间: {current_time}")
        
        # 更新截止时间
        self.update_deadline_info()
        
        # 更新最新开奖信息
        self.update_latest_results()
        
        # 更新中奖记录
        self.update_prize_records()
        
        # 更新滚动公告
        self.update_marquee()
    
    def update_deadline_info(self):
        """更新截止时间信息"""
        now = datetime.now()
        
        # 双色球截止时间（开奖日20:00）
        ssq_deadline = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if now > ssq_deadline:
            ssq_deadline += timedelta(days=1)
        
        # 快乐8截止时间（每日20:00）
        kl8_deadline = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if now > kl8_deadline:
            kl8_deadline += timedelta(days=1)
        
        time_left_ssq = ssq_deadline - now
        time_left_kl8 = kl8_deadline - now
        
        deadline_text = (
            f"双色球截止: {ssq_deadline.strftime('%H:%M')} "
            f"(剩余{time_left_ssq.seconds//3600:02d}:{(time_left_ssq.seconds%3600)//60:02d}) | "
            f"快乐8截止: {kl8_deadline.strftime('%H:%M')} "
            f"(剩余{time_left_kl8.seconds//3600:02d}:{(time_left_kl8.seconds%3600)//60:02d})"
        )
        
        self.deadline_label.setText(deadline_text)
    
    def update_latest_results(self):
        """更新最新开奖信息"""
        latest_text = ""
        
        games = ["双色球", "快乐8", "3D", "七乐彩"]
        for game in games:
            if game in self.history_data and self.history_data[game]:
                latest_record = self.history_data[game][0]
                latest_text += f"{game} 第{latest_record.get('期号', '未知')}期:\n"
                latest_text += f"开奖号码: {latest_record.get('开奖号码', '未知')}\n"
                latest_text += f"开奖时间: {latest_record.get('开奖日期', '未知')}\n\n"
            else:
                latest_text += f"{game}: 暂无数据\n\n"
        
        self.latest_results_text.setText(latest_text)
    
    def update_prize_records(self):
        """更新中奖记录"""
        if not self.betting_history:
            self.prize_records_text.setText("暂无中奖记录")
            return
        
        # 模拟中奖记录（实际应该从投注记录中计算）
        prize_text = "最近中奖记录:\n\n"
        prize_records = [
            "2025-01-01 双色球 五等奖 10元",
            "2024-12-28 快乐8 选十中五 3元", 
            "2024-12-25 3D 组选六 173元",
            "2024-12-20 七乐彩 七等奖 5元"
        ]
        
        for record in prize_records:
            prize_text += f"• {record}\n"
        
        self.prize_records_text.setText(prize_text)
    
    def update_marquee(self):
        """更新滚动公告"""
        now = datetime.now()
        day_of_week = now.weekday()
        
        # 根据星期几显示不同的开奖信息
        if day_of_week in [1, 3, 6]:  # 周二、四、日
            marquee_text = "今日开奖：双色球(21:15) 快乐8(21:30) 3D(21:15) 七乐彩(21:15)"
        else:
            marquee_text = "今日开奖：快乐8(21:30) 3D(21:15) 七乐彩(21:15)"
        
        self.marquee_label.setText(marquee_text)
    
    def show_live_broadcast(self):
        """显示开奖直播"""
        from dialogs import LiveBroadcastDialog
        dialog = LiveBroadcastDialog(self)
        dialog.exec()
    
    def show_qr_scanner(self):
        """显示二维码扫描"""
        from dialogs import QRCodeDialog
        dialog = QRCodeDialog(self)
        dialog.exec()
    
    def show_trend_charts(self):
        """显示走势图"""
        self.tabs.setCurrentWidget(self.trend_tab)
    
    def show_prediction(self):
        """显示预测分析"""
        self.tabs.setCurrentWidget(self.analysis_tab)
    
    def init_trend_tab(self):
        """初始化走势图选项卡"""
        layout = QVBoxLayout()
        
        # 游戏选择
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("选择彩票类型:"))
        
        self.trend_game_combo = QComboBox()
        self.trend_game_combo.addItems(["双色球", "快乐8", "3D", "七乐彩"])
        self.trend_game_combo.currentTextChanged.connect(self.update_trend_chart)
        game_layout.addWidget(self.trend_game_combo)
        
        # 走势图类型
        game_layout.addWidget(QLabel("走势图类型:"))
        self.trend_type_combo = QComboBox()
        self.trend_type_combo.addItems(["基本走势图", "冷热号走势", "遗漏走势", "和值走势"])
        self.trend_type_combo.currentTextChanged.connect(self.update_trend_chart)
        game_layout.addWidget(self.trend_type_combo)
        
        layout.addLayout(game_layout)
        
        # 走势图显示区域
        self.trend_figure = Figure(figsize=(10, 8))
        self.trend_canvas = FigureCanvas(self.trend_figure)
        layout.addWidget(self.trend_canvas)
        
        # 走势图数据
        self.trend_data_text = QTextEdit()
        self.trend_data_text.setReadOnly(True)
        self.trend_data_text.setMaximumHeight(200)
        layout.addWidget(self.trend_data_text)
        
        self.trend_tab.setLayout(layout)
        
        # 初始化走势图
        self.update_trend_chart()
    
    def update_trend_chart(self):
        """更新走势图"""
        game = self.trend_game_combo.currentText()
        trend_type = self.trend_type_combo.currentText()
        
        self.trend_figure.clear()
        
        if game not in self.history_data or not self.history_data[game]:
            ax = self.trend_figure.add_subplot(111)
            ax.text(0.5, 0.5, f"暂无{game}历史数据", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
            self.trend_canvas.draw()
            return
        
        data = self.history_data[game][:50]  # 最近50期数据
        
        if trend_type == "基本走势图":
            self.draw_basic_trend(game, data)
        elif trend_type == "冷热号走势":
            self.draw_cold_hot_trend(game, data)
        elif trend_type == "遗漏走势":
            self.draw_omission_trend(game, data)
        elif trend_type == "和值走势":
            self.draw_sum_trend(game, data)
        
        self.trend_canvas.draw()
        
        # 更新数据文本
        self.update_trend_data_text(game, data)
    
    def draw_basic_trend(self, game, data):
        """绘制基本走势图"""
        ax = self.trend_figure.add_subplot(111)
        
        if game == "双色球":
            # 提取红球号码
            red_numbers = []
            for record in data:
                numbers_str = record["开奖号码"]
                if "+" in numbers_str:
                    red_part = numbers_str.split("+")[0]
                    nums = [int(n) for n in red_part.split()]
                    red_numbers.append(nums)
            
            # 创建走势图数据
            periods = list(range(len(data)))
            for num in range(1, 34):
                positions = []
                for i, nums in enumerate(red_numbers):
                    if num in nums:
                        positions.append(i)
                if positions:
                    ax.scatter([positions], [num] * len(positions), color='red', s=30)
            
            ax.set_xlabel("期数")
            ax.set_ylabel("红球号码")
            ax.set_title("双色球红球基本走势图")
            ax.grid(True)
            
        elif game == "快乐8":
            # 类似处理其他游戏...
            ax.text(0.5, 0.5, f"{game}基本走势图", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16)
    
    def draw_cold_hot_trend(self, game, data):
        """绘制冷热号走势图"""
        ax = self.trend_figure.add_subplot(111)
        ax.set_title(f"{game}冷热号走势图")
        # 实现冷热号走势图绘制逻辑
        ax.text(0.5, 0.5, f"{game}冷热号走势图", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
    
    def draw_omission_trend(self, game, data):
        """绘制遗漏走势图"""
        ax = self.trend_figure.add_subplot(111)
        ax.set_title(f"{game}遗漏走势图")
        # 实现遗漏走势图绘制逻辑
        ax.text(0.5, 0.5, f"{game}遗漏走势图", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
    
    def draw_sum_trend(self, game, data):
        """绘制和值走势图"""
        ax = self.trend_figure.add_subplot(111)
        ax.set_title(f"{game}和值走势图")
        # 实现和值走势图绘制逻辑
        ax.text(0.5, 0.5, f"{game}和值走势图", 
               horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=16)
    
    def update_trend_data_text(self, game, data):
        """更新走势图数据文本"""
        if not data:
            self.trend_data_text.setText("暂无数据")
            return
        
        text = f"{game}走势数据分析 (最近{len(data)}期):\n\n"
        
        if game == "双色球":
            # 统计红球出现次数
            red_count = {}
            for record in data:
                numbers_str = record["开奖号码"]
                if "+" in numbers_str:
                    red_part = numbers_str.split("+")[0]
                    nums = [int(n) for n in red_part.split()]
                    for num in nums:
                        red_count[num] = red_count.get(num, 0) + 1
            
            text += "红球出现次数统计:\n"
            for num in sorted(red_count.keys()):
                text += f"{num:2d}号: {red_count[num]:2d}次\n"
        
        self.trend_data_text.setText(text)

    # 由于代码长度限制，以下只显示部分关键方法
    # 完整代码请查看GitHub仓库

    def generate_ssq_random(self):
        """双色球机选多注"""
        num_bets, ok = QInputDialog.getInt(self, "机选多注", "请输入注数(1-1000):", 5, 1, 1000)
        if not ok:
            return
            
        results = []
        for i in range(num_bets):
            red = sorted(random.sample(range(1, 34), 6))
            blue = random.randint(1, 16)
            results.append(f"第{i+1}注: 红球{', '.join(map(str, red))} 蓝球{blue}")
        
        result_str = "\n\n".join(results)
        QMessageBox.information(self, f"双色球机选{num_bets}注", result_str)
        
        # 添加到投注记录
        for res in results:
            self.add_betting_record("双色球", "机选", res, 1, 2.0)

    def generate_ssq_complex(self):
        """双色球机选复式 - 可选择号码数量"""
        red_count, ok1 = QInputDialog.getInt(self, "复式投注", "红球选择数量(7-33):", 7, 7, 33)
        blue_count, ok2 = QInputDialog.getInt(self, "复式投注", "蓝球选择数量(1-16):", 1, 1, 16)
        
        if ok1 and ok2:
            try:
                red = sorted(random.sample(range(1, 34), red_count))
                blue = sorted(random.sample(range(1, 17), blue_count))
                result = f"红球({red_count}个):\n{', '.join(map(str, red))}\n\n蓝球({blue_count}个):\n{', '.join(map(str, blue))}"
                
                # 计算注数和金额
                red_comb = math.comb(red_count, 6)
                blue_comb = math.comb(blue_count, 1)
                bets = red_comb * blue_comb
                amount = bets * 2.0
                
                # 显示号码概率
                prob_text = self.calculate_probability("双色球", red, blue)
                
                QMessageBox.information(self, "双色球复式投注", 
                                      f"{result}\n\n{prob_text}\n\n注数: {bets}注\n金额: ¥{amount:.2f}")
                self.add_betting_record("双色球", "复式", result, bets, amount)
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))

    def calculate_probability(self, game_type, numbers, extra_numbers=None):
        """计算号码概率"""
        if game_type == "双色球":
            # 计算红球概率
            red_prob = {}
            for num in numbers:
                omission = self.get_omission("双色球", num, "red")
                frequency = self.get_frequency("双色球", num, "red")
                cold_hot = self.get_cold_hot("双色球", num, "red")
                red_prob[num] = {
                    "omission": omission,
                    "frequency": frequency,
                    "status": cold_hot
                }
            
            # 计算蓝球概率
            blue_prob = {}
            if extra_numbers:
                for num in extra_numbers:
                    omission = self.get_omission("双色球", num, "blue")
                    frequency = self.get_frequency("双色球", num, "blue")
                    cold_hot = self.get_cold_hot("双色球", num, "blue")
                    blue_prob[num] = {
                        "omission": omission,
                        "frequency": frequency,
                        "status": cold_hot
                    }
            
            # 生成概率文本
            prob_text = "号码概率分析:\n"
            prob_text += "红球:\n"
            for num, info in red_prob.items():
                status_text = {"hot": "热", "warm": "温", "cold": "冷", "default": "普"}[info["status"]]
                prob_text += f"{num:2d}号: 遗漏{info['omission']:2d}期 出现{info['frequency']:2d}次 {status_text}号\n"
            
            if blue_prob:
                prob_text += "\n蓝球:\n"
                for num, info in blue_prob.items():
                    status_text = {"hot": "热", "warm": "温", "cold": "冷", "default": "普"}[info["status"]]
                    prob_text += f"{num:2d}号: 遗漏{info['omission']:2d}期 出现{info['frequency']:2d}次 {status_text}号\n"
            
            return prob_text
        
        return "概率分析功能开发中"

    # 其他原有方法...
    # 由于代码长度限制，这里只显示关键修改部分

    def auto_update_data(self):
        """自动更新所有玩法的最新50期开奖数据"""
        try:
            games = ["双色球", "快乐8", "3D", "七乐彩"]
            for game in games:
                self.update_game_data(game)
            
            # 设置定时器每30分钟自动更新一次
            if not self.auto_update_timer:
                self.auto_update_timer = QTimer()
                self.auto_update_timer.timeout.connect(self.auto_update_data)
                self.auto_update_timer.start(1800000)  # 30分钟
                
            logger.info("自动更新数据完成")
        except Exception as e:
            logger.error(f"自动更新数据失败: {str(e)}")

    def load_history_data(self):
        """加载历史数据"""
        game_types = ["双色球", "快乐8", "3D", "七乐彩"]
        for game in game_types:
            filename = f"{game}_history.json"
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 只保留2025开头的期号
                        data = [record for record in data if str(record.get("期号", "")).startswith("2025")]
                        self.history_data[game] = data
                    print(f"Loaded {len(self.history_data[game])} records for {game}")
                except Exception as e:
                    print(f"加载历史数据失败: {str(e)}")
                    self.history_data[game] = []

    # 其他方法...
    # 完整代码请查看GitHub仓库
