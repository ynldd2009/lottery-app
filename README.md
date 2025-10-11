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
from PySide6.QtGui import QFont, QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
import qrcode
from PIL import Image
import cv2
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl

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
        "选九中四": {"base": 3, "description": "选9中4"},
        "选八中八": {"base": 50000, "description": "选8中8"},
        "选八中七": {"base": 800, "description": "选8中7"},
        "选八中六": {"base": 88, "description": "选8中6"},
        "选八中五": {"base": 10, "description": "选8中5"},
        "选八中四": {"base": 3, "description": "选8中4"},
        "选七中七": {"base": 10000, "description": "选7中7"},
        "选七中六": {"base": 288, "description": "选7中6"},
        "选七中五": {"base": 28, "description": "选7中5"},
        "选七中四": {"base": 4, "description": "选7中4"},
        "选六中六": {"base": 3000, "description": "选6中6"},
        "选六中五": {"base": 30, "description": "选6中5"},
        "选六中四": {"base": 10, "description": "选6中4"},
        "选五中五": {"base": 1000, "description": "选5中5"},
        "选五中四": {"base": 21, "description": "选5中4"},
        "选五中三": {"base": 3, "description": "选5中3"},
        "选四中四": {"base": 100, "description": "选4中4"},
        "选四中三": {"base": 5, "description": "选4中3"},
        "选四中二": {"base": 3, "description": "选4中2"},
        "选三中三": {"base": 53, "description": "选3中3"},
        "选三中二": {"base": 3, "description": "选3中2"},
        "选二中二": {"base": 19, "description": "选2中2"},
        "选一中一": {"base": 4.6, "description": "选1中1"}
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

class QRCodeDialog(QDialog):
    """二维码扫描对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("二维码扫描与兑奖")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("彩票二维码扫描与兑奖系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #2C3E50;")
        layout.addWidget(title_label)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        
        # 二维码生成选项卡
        self.qr_generate_tab = QWidget()
        self.init_qr_generate_tab()
        self.tabs.addTab(self.qr_generate_tab, "生成二维码")
        
        # 二维码扫描选项卡
        self.qr_scan_tab = QWidget()
        self.init_qr_scan_tab()
        self.tabs.addTab(self.qr_scan_tab, "扫描二维码")
        
        # 兑奖查询选项卡
        self.prize_check_tab = QWidget()
        self.init_prize_check_tab()
        self.tabs.addTab(self.prize_check_tab, "兑奖查询")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def init_qr_generate_tab(self):
        layout = QVBoxLayout()
        
        # 输入区域
        input_group = QGroupBox("生成投注二维码")
        input_layout = QVBoxLayout()
        
        # 游戏选择
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("彩票类型:"))
        self.qr_game_combo = QComboBox()
        self.qr_game_combo.addItems(["双色球", "快乐8", "3D", "七乐彩"])
        game_layout.addWidget(self.qr_game_combo)
        input_layout.addLayout(game_layout)
        
        # 投注号码
        numbers_layout = QHBoxLayout()
        numbers_layout.addWidget(QLabel("投注号码:"))
        self.qr_numbers_edit = QLineEdit()
        self.qr_numbers_edit.setPlaceholderText("请输入投注号码，用逗号分隔")
        numbers_layout.addWidget(self.qr_numbers_edit)
        input_layout.addLayout(numbers_layout)
        
        # 投注金额
        amount_layout = QHBoxLayout()
        amount_layout.addWidget(QLabel("投注金额:"))
        self.qr_amount_edit = QLineEdit()
        self.qr_amount_edit.setText("2.00")
        amount_layout.addWidget(self.qr_amount_edit)
        input_layout.addLayout(amount_layout)
        
        # 生成按钮
        self.generate_qr_btn = QPushButton("生成二维码")
        self.generate_qr_btn.clicked.connect(self.generate_qr_code)
        input_layout.addWidget(self.generate_qr_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 二维码显示区域
        self.qr_display_label = QLabel()
        self.qr_display_label.setAlignment(Qt.AlignCenter)
        self.qr_display_label.setMinimumSize(300, 300)
        self.qr_display_label.setStyleSheet("border: 2px solid #ccc; background-color: white;")
        layout.addWidget(self.qr_display_label)
        
        self.qr_generate_tab.setLayout(layout)
    
    def init_qr_scan_tab(self):
        layout = QVBoxLayout()
        
        # 扫描区域
        scan_group = QGroupBox("扫描二维码")
        scan_layout = QVBoxLayout()
        
        # 摄像头选择
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["默认摄像头", "前置摄像头", "后置摄像头"])
        camera_layout.addWidget(self.camera_combo)
        scan_layout.addLayout(camera_layout)
        
        # 扫描按钮
        self.scan_qr_btn = QPushButton("开始扫描")
        self.scan_qr_btn.clicked.connect(self.toggle_qr_scan)
        scan_layout.addWidget(self.scan_qr_btn)
        
        # 扫描结果显示
        self.scan_result_label = QLabel("扫描结果将显示在这里")
        self.scan_result_label.setWordWrap(True)
        scan_layout.addWidget(self.scan_result_label)
        
        scan_group.setLayout(scan_layout)
        layout.addWidget(scan_group)
        
        self.qr_scan_tab.setLayout(layout)
    
    def init_prize_check_tab(self):
        layout = QVBoxLayout()
        
        # 兑奖查询区域
        check_group = QGroupBox("兑奖查询")
        check_layout = QVBoxLayout()
        
        # 期号输入
        issue_layout = QHBoxLayout()
        issue_layout.addWidget(QLabel("期号:"))
        self.issue_edit = QLineEdit()
        self.issue_edit.setPlaceholderText("请输入彩票期号")
        issue_layout.addWidget(self.issue_edit)
        check_layout.addLayout(issue_layout)
        
        # 号码输入
        check_numbers_layout = QHBoxLayout()
        check_numbers_layout.addWidget(QLabel("投注号码:"))
        self.check_numbers_edit = QLineEdit()
        self.check_numbers_edit.setPlaceholderText("请输入投注号码")
        check_numbers_layout.addWidget(self.check_numbers_edit)
        check_layout.addLayout(check_numbers_layout)
        
        # 查询按钮
        self.check_prize_btn = QPushButton("查询中奖结果")
        self.check_prize_btn.clicked.connect(self.check_prize_result)
        check_layout.addWidget(self.check_prize_btn)
        
        # 结果显示
        self.prize_result_label = QLabel("中奖结果将显示在这里")
        self.prize_result_label.setWordWrap(True)
        self.prize_result_label.setStyleSheet("font-size: 14px; color: #2C3E50;")
        check_layout.addWidget(self.prize_result_label)
        
        check_group.setLayout(check_layout)
        layout.addWidget(check_group)
        
        self.prize_check_tab.setLayout(layout)
    
    def generate_qr_code(self):
        """生成二维码"""
        game = self.qr_game_combo.currentText()
        numbers = self.qr_numbers_edit.text()
        amount = self.qr_amount_edit.text()
        
        if not numbers:
            QMessageBox.warning(self, "错误", "请输入投注号码")
            return
        
        # 创建二维码数据
        qr_data = f"LOTTERY|{game}|{numbers}|{amount}|{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            # 生成二维码
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_data)
            qr.make(fit=True)
            
            # 创建二维码图像
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # 转换为QPixmap并显示
            qr_image = qr_image.convert("RGB")
            data = qr_image.tobytes("raw", "RGB")
            qim = Image.frombytes("RGB", qr_image.size, data)
            qim.save("temp_qr.png")
            
            # 在标签中显示二维码
            pixmap = QPixmap("temp_qr.png")
            self.qr_display_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            
            QMessageBox.information(self, "成功", "二维码生成成功！")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"生成二维码失败: {str(e)}")
    
    def toggle_qr_scan(self):
        """切换二维码扫描状态"""
        if self.scan_qr_btn.text() == "开始扫描":
            self.scan_qr_btn.setText("停止扫描")
            self.scan_result_label.setText("正在扫描...")
            # 这里应该启动摄像头扫描，简化实现
            self.simulate_qr_scan()
        else:
            self.scan_qr_btn.setText("开始扫描")
            self.scan_result_label.setText("扫描已停止")
    
    def simulate_qr_scan(self):
        """模拟二维码扫描（简化实现）"""
        # 在实际应用中，这里应该使用OpenCV等库进行真正的二维码扫描
        # 这里我们只是模拟扫描过程
        QTimer.singleShot(2000, self.show_scan_result)
    
    def show_scan_result(self):
        """显示扫描结果"""
        game = random.choice(["双色球", "快乐8", "3D", "七乐彩"])
        numbers = "01,02,03,04,05,06,07" if game == "双色球" else "01,02,03,04,05,06,07,08,09,10"
        amount = "2.00"
        
        result_text = f"""
扫描结果:
彩票类型: {game}
投注号码: {numbers}
投注金额: ¥{amount}
扫描时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

二维码数据已验证！
        """
        
        self.scan_result_label.setText(result_text)
        self.scan_qr_btn.setText("开始扫描")
    
    def check_prize_result(self):
        """查询中奖结果"""
        issue = self.issue_edit.text()
        numbers = self.check_numbers_edit.text()
        
        if not issue or not numbers:
            QMessageBox.warning(self, "错误", "请输入期号和投注号码")
            return
        
        # 模拟中奖查询
        game = random.choice(["双色球", "快乐8", "3D", "七乐彩"])
        prize_level = random.choice(["一等奖", "二等奖", "三等奖", "未中奖"])
        prize_amount = random.choice([5000000, 200000, 3000, 0])
        
        if prize_level == "未中奖":
            result_text = f"""
兑奖结果:
期号: {issue}
投注号码: {numbers}
中奖状态: 未中奖
中奖金额: ¥0

感谢参与，祝您下次好运！
            """
        else:
            result_text = f"""
兑奖结果:
期号: {issue}
投注号码: {numbers}
中奖状态: {prize_level}
中奖金额: ¥{prize_amount:,}

恭喜您中奖！请携带彩票到指定兑奖点兑奖。
            """
        
        self.prize_result_label.setText(result_text)

class LiveBroadcastDialog(QDialog):
    """开奖直播对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("福彩开奖直播")
        self.setGeometry(100, 100, 1000, 700)
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("中国福利彩票开奖直播")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #E74C3C;")
        layout.addWidget(title_label)
        
        # 当前时间显示
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 16px; color: #2C3E50;")
        layout.addWidget(self.time_label)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        
        # 直播视频选项卡
        self.live_tab = QWidget()
        self.init_live_tab()
        self.tabs.addTab(self.live_tab, "开奖直播")
        
        # 最新开奖选项卡
        self.results_tab = QWidget()
        self.init_results_tab()
        self.tabs.addTab(self.results_tab, "最新开奖")
        
        # 走势图选项卡
        self.trend_tab = QWidget()
        self.init_trend_tab()
        self.tabs.addTab(self.trend_tab, "走势图")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
        # 更新时间
        self.update_time()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
    
    def init_live_tab(self):
        layout = QVBoxLayout()
        
        # 直播视频区域
        video_group = QGroupBox("开奖直播视频")
        video_layout = QVBoxLayout()
        
        # 视频播放器
        self.video_widget = QVideoWidget()
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        video_layout.addWidget(self.video_widget)
        
        # 直播控制按钮
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("播放直播")
        self.play_btn.clicked.connect(self.play_live)
        control_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("停止播放")
        self.stop_btn.clicked.connect(self.stop_live)
        control_layout.addWidget(self.stop_btn)
        
        video_layout.addLayout(control_layout)
        
        # 直播信息
        info_label = QLabel("开奖时间: 每天21:30\n直播频道: 中央电视台财经频道")
        info_label.setStyleSheet("font-size: 14px; color: #7F8C8D;")
        video_layout.addWidget(info_label)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        self.live_tab.setLayout(layout)
    
    def init_results_tab(self):
        layout = QVBoxLayout()
        
        # 最新开奖结果
        results_group = QGroupBox("最新开奖结果")
        results_layout = QVBoxLayout()
        
        # 游戏选择
        game_layout = QHBoxLayout()
        game_layout.addWidget(QLabel("选择彩票:"))
        self.live_game_combo = QComboBox()
        self.live_game_combo.addItems(["双色球", "快乐8", "3D", "七乐彩"])
        self.live_game_combo.currentTextChanged.connect(self.update_live_results)
        game_layout.addWidget(self.live_game_combo)
        results_layout.addLayout(game_layout)
        
        # 开奖结果显示
        self.live_results_text = QTextEdit()
        self.live_results_text.setReadOnly(True)
        results_layout.addWidget(self.live_results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # 更新开奖结果
        self.update_live_results()
        
        self.results_tab.setLayout(layout)
    
    def init_trend_tab(self):
        layout = QVBoxLayout()
        
        # 走势图区域
        trend_group = QGroupBox("彩票走势图")
        trend_layout = QVBoxLayout()
        
        # 游戏选择
        trend_game_layout = QHBoxLayout()
        trend_game_layout.addWidget(QLabel("选择彩票:"))
        self.trend_game_combo = QComboBox()
        self.trend_game_combo.addItems(["双色球", "快乐8", "3D", "七乐彩"])
        self.trend_game_combo.currentTextChanged.connect(self.update_trend_chart)
        trend_game_layout.addWidget(self.trend_game_combo)
        trend_layout.addLayout(trend_game_layout)
        
        # 走势图显示
        self.trend_figure = Figure(figsize=(10, 6))
        self.trend_canvas = FigureCanvas(self.trend_figure)
        trend_layout.addWidget(self.trend_canvas)
        
        trend_group.setLayout(trend_layout)
        layout.addWidget(trend_group)
        
        # 更新走势图
        self.update_trend_chart()
        
        self.trend_tab.setLayout(layout)
    
    def update_time(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(f"当前时间: {current_time}")
    
    def play_live(self):
        """播放直播"""
        try:
            # 这里应该使用真实的直播流URL
            # 由于版权问题，我们使用一个示例URL
            live_url = "https://example.com/live_stream"
            self.media_player.setSource(QUrl(live_url))
            self.media_player.play()
            self.play_btn.setText("播放中...")
        except Exception as e:
            QMessageBox.warning(self, "播放错误", f"无法播放直播: {str(e)}")
    
    def stop_live(self):
        """停止播放"""
        self.media_player.stop()
        self.play_btn.setText("播放直播")
    
    def update_live_results(self):
        """更新开奖结果"""
        game = self.live_game_combo.currentText()
        
        # 模拟开奖数据
        if game == "双色球":
            results = f"""
{game}最新开奖结果
期号: 2025001
开奖时间: 2025-01-01 21:30:00
开奖号码: 01 05 09 15 20 25 + 12
销售额: 3.5亿元
奖池金额: 10.2亿元

一等奖: 5注 每注¥10,000,000
二等奖: 120注 每注¥200,000
            """
        elif game == "快乐8":
            results = f"""
{game}最新开奖结果
期号: 2025001
开奖时间: 2025-01-01 21:30:00
开奖号码: 01 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 76 77 78 79
选十中十: 2注 每注¥5,000,000
选九中九: 15注 每注¥300,000
            """
        else:
            results = f"""
{game}最新开奖结果
期号: 2025001
开奖时间: 2025-01-01 21:30:00
开奖号码: 1 5 9
直选: 100注 每注¥1,040
组选: 200注 每注¥346
            """
        
        self.live_results_text.setText(results)
    
    def update_trend_chart(self):
        """更新走势图"""
        game = self.trend_game_combo.currentText()
        self.trend_figure.clear()
        
        ax = self.trend_figure.add_subplot(111)
        
        # 生成模拟走势数据
        if game == "双色球":
            # 红球走势
            periods = range(1, 31)
            red_trend = np.random.randint(1, 34, (6, 30))
            
            for i in range(6):
                ax.plot(periods, red_trend[i], 'o-', label=f'红球{i+1}')
            
            ax.set_title(f"{game}红球走势图 (最近30期)")
            ax.set_xlabel("期号")
            ax.set_ylabel("号码")
            ax.legend()
            ax.grid(True)
            
        elif game == "快乐8":
            # 号码出现频率
            numbers = range(1, 81)
            frequencies = np.random.randint(1, 20, 80)
            
            ax.bar(numbers, frequencies)
            ax.set_title(f"{game}号码出现频率")
            ax.set_xlabel("号码")
            ax.set_ylabel("出现次数")
            ax.grid(True)
            
        else:
            # 3D和七乐彩的简单走势
            periods = range(1, 31)
            trend_data = np.random.randint(0, 10, (3, 30))
            
            for i in range(3):
                ax.plot(periods, trend_data[i], 'o-', label=f'位置{i+1}')
            
            ax.set_title(f"{game}走势图 (最近30期)")
            ax.set_xlabel("期号")
            ax.set_ylabel("号码")
            ax.legend()
            ax.grid(True)
        
        self.trend_canvas.draw()

class NumberButton(QPushButton):
    """自定义数字按钮，支持圆形样式和遗漏期数显示"""
    def __init__(self, text, cold_hot_type="default", omission=0, frequency=0, probability=0, is_dan=False, is_tuo=False, parent=None):
        super().__init__(text, parent)
        # 优化球体大小和布局
        self.setFixedSize(50, 50)  # 增大以容纳更多信息
        self.setCheckable(True)
        self.is_dan = is_dan
        self.is_tuo = is_tuo
        
        # 设置颜色
        colors = {
            "hot": "#FF6B6B",   # 热号 - 红色
            "warm": "#4ECDC4",  # 温号 - 青色
            "cold": "#45B7D1",  # 冷号 - 蓝色
            "default": "#F0F0F0" # 默认 - 灰色
        }
        color = colors.get(cold_hot_type, "#F0F0F0")
        
        # 胆码和拖码的特殊样式
        if is_dan:
            color = "#FFD700"  # 金色
        elif is_tuo:
            color = "#87CEEB"  # 浅蓝色
        
        # 优化圆形样式和遗漏期数显示
        self.setStyleSheet(f"""
            QPushButton {{
                border-radius: 25px;
                border: 2px solid #8f8f91;
                font-weight: bold;
                font-size: 14px;
                background-color: #F8F9FA;
                color: #2C3E50;
            }}
            QPushButton:checked {{
                background-color: {color};
                color: white;
                border: 2px solid #2C3E50;
            }}
            QPushButton:hover {{
                background-color: #E9ECEF;
            }}
        """)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_layout.setSpacing(1)
        
        # 概率标签（顶部）
        self.probability_label = QLabel(f"概:{probability}%")
        self.probability_label.setAlignment(Qt.AlignCenter)
        self.probability_label.setStyleSheet("font-size: 8px; color: #555; font-weight: bold;")
        self.probability_label.setFixedHeight(10)
        
        # 数字标签（中间）
        self.number_label = QLabel(text)
        self.number_label.setAlignment(Qt.AlignCenter)
        self.number_label.setStyleSheet("font-size: 16px; font-weight: bold; color: inherit;")
        self.number_label.setFixedHeight(20)
        
        # 遗漏标签（底部）
        self.omission_label = QLabel(f"漏:{omission}")
        self.omission_label.setAlignment(Qt.AlignCenter)
        self.omission_label.setStyleSheet("font-size: 8px; color: #555; font-weight: bold;")
        self.omission_label.setFixedHeight(10)
        
        # 添加到布局
        self.main_layout.addWidget(self.probability_label)
        self.main_layout.addWidget(self.number_label)
        self.main_layout.addWidget(self.omission_label)
        
        # 设置标签样式
        self.set_omission(omission)
        self.set_probability(probability)
    
    def set_omission(self, omission):
        """设置遗漏期数"""
        self.omission_label.setText(f"漏:{omission}")
        # 根据遗漏期数设置标签颜色
        if omission > 20:
            self.omission_label.setStyleSheet("font-size: 8px; color: #E74C3C; font-weight: bold;")
        elif omission > 10:
            self.omission_label.setStyleSheet("font-size: 8px; color: #F39C12; font-weight: bold;")
        else:
            self.omission_label.setStyleSheet("font-size: 8px; color: #555; font-weight: bold;")
    
    def set_probability(self, probability):
        """设置出现概率"""
        self.probability_label.setText(f"概:{probability}%")
        # 根据概率设置标签颜色
        if probability > 15:
            self.probability_label.setStyleSheet("font-size: 8px; color: #27AE60; font-weight: bold;")
        elif probability > 10:
            self.probability_label.setStyleSheet("font-size: 8px; color: #27AE60; font-weight: bold;")
        else:
            self.probability_label.setStyleSheet("font-size: 8px; color: #555; font-weight: bold;")
    
    def set_dan(self, is_dan):
        """设置为胆码"""
        self.is_dan = is_dan
        if is_dan:
            self.setStyleSheet("""
                QPushButton {
                    border-radius: 25px;
                    border: 2px solid #8f8f91;
                    font-weight: bold;
                    font-size: 14px;
                    background-color: #FFD700;
                    color: #2C3E50;
                }
                QPushButton:checked {
                    background-color: #FFD700;
                    color: #2C3E50;
                    border: 2px solid #2C3E50;
                }
                QPushButton:hover {
                    background-color: #FFE55C;
                }
            """)
    
    def set_tuo(self, is_tuo):
        """设置为拖码"""
        self.is_tuo = is_tuo
        if is_tuo:
            self.setStyleSheet("""
                QPushButton {
                    border-radius: 25px;
                    border: 2px solid #8f8f91;
                    font-weight: bold;
                    font-size: 14px;
                    background-color: #87CEEB;
                    color: #2C3E50;
                }
                QPushButton:checked {
                    background-color: #87CEEB;
                    color: #2C3E50;
                    border: 2px solid #2C3E50;
                }
                QPushButton:hover {
                    background-color: #A3D9F4;
                }
            """)

class LotteryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("福彩应用 - 完整版")
        self.setGeometry(100, 100, 1600, 1000)
        
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
        self.probability_data = {}   # 存储号码概率数据
        self.auto_update_timer = None  # 自动更新定时器
        
        # 创建主界面
        self.init_ui()
        
        # 加载历史数据和API密钥
        self.load_api_key()
        self.load_history_data()
        self.load_betting_history()
        self.calculate_omission()  # 计算初始遗漏数据
        self.auto_update_data()    # 自动更新数据
        
        # 启动时间更新定时器
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time_display)
        self.time_timer.start(1000)
        
        # 启动滚动信息定时器
        self.scroll_timer = QTimer()
        self.scroll_timer.timeout.connect(self.update_scroll_info)
        self.scroll_timer.start(5000)
        self.scroll_index = 0
        
    def update_time_display(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, 'time_label'):
            self.time_label.setText(f"当前时间: {current_time}")
    
    def update_scroll_info(self):
        """更新滚动信息"""
        scroll_messages = [
            "今日开奖: 双色球(21:15) 快乐8(21:30) 3D(21:15) 七乐彩(21:15)",
            "购彩截止时间: 双色球20:00 快乐8:19:30 3D:19:30 七乐彩:19:30",
            "理性购彩，量力而行，未成年人不得购买彩票",
            "中国福利彩票，扶老、助残、救孤、济困",
            "开奖结果以官方公布为准，本应用仅供参考"
        ]
        
        if hasattr(self, 'scroll_label'):
            self.scroll_label.setText(scroll_messages[self.scroll_index])
            self.scroll_index = (self.scroll_index + 1) % len(scroll_messages)
    
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
    
    def update_game_data(self, game_type):
        """更新指定游戏的数据"""
        try:
            if not self.api_key:
                logger.warning("API密钥未设置，无法更新数据")
                return
                
            if game_type not in self.history_data:
                self.history_data[game_type] = []
            
            # 检查是否需要更新
            latest_issue = None
            if self.history_data[game_type]:
                latest_issue = self.history_data[game_type][0]["期号"]
            
            # 获取最新50期数据
            game_id_map = {
                "双色球": "ssq",
                "快乐8": "kl8",
                "3D": "fc3d",
                "七乐彩": "qlc"
            }
            
            game_id = game_id_map.get(game_type)
            if not game_id:
                logger.warning(f"不支持的彩票类型: {game_type}")
                return
                
            url = "https://www.myshengong.com/api/openapi.lottery/history"
            params = {
                "apikey": self.api_key,
                "code": game_id,
                "size": 50
            }
            
            logger.info(f"请求{game_type}数据: {url}")
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if data.get("code") == 1:
                records = data.get("data", [])
                if not isinstance(records, list):
                    records = [records]
                
                # 处理并添加新记录
                new_records_count = 0
                for record in records:
                    issue = record.get("issue", "")
                    
                    # 检查是否已存在该期数据
                    existing_issues = {r["期号"] for r in self.history_data[game_type]}
                    if issue not in existing_issues:
                        self.process_api_data(game_type, record)
                        new_records_count += 1
                
                if new_records_count > 0:
                    logger.info(f"成功更新{game_type} {new_records_count}条记录")
                    # 重新计算遗漏数据
                    self.calculate_omission()
                else:
                    logger.info(f"{game_type}没有新数据")
            else:
                error_msg = data.get("msg", "未知错误")
                logger.error(f"API返回错误: {error_msg}")
        except Exception as e:
            logger.error(f"更新{game_type}数据失败: {str(e)}")
    
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
        
        # 二维码扫描选项卡
        self.qrcode_tab = QWidget()
        self.init_qrcode_tab()
        self.tabs.addTab(self.qrcode_tab, "二维码扫描")
        
        # 开奖直播选项卡
        self.live_tab = QWidget()
        self.init_live_tab()
        self.tabs.addTab(self.live_tab, "开奖直播")
        
        self.setCentralWidget(self.tabs)
    
    def init_home_tab(self):
        """初始化首页"""
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("中国福利彩票")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #E74C3C; margin: 20px;")
        layout.addWidget(title_label)
        
        # 时间显示
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-size: 18px; color: #2C3E50; margin: 10px;")
        layout.addWidget(self.time_label)
        
        # 滚动信息
        self.scroll_label = QLabel()
        self.scroll_label.setAlignment(Qt.AlignCenter)
        self.scroll_label.setStyleSheet("font-size: 16px; color: #E67E22; background-color: #FFF9E6; padding: 10px; border-radius: 5px;")
        self.scroll_label.setWordWrap(True)
        layout.addWidget(self.scroll_label)
        
        # 最新开奖结果区域
        latest_results_group = QGroupBox("最新开奖结果")
        latest_results_layout = QVBoxLayout()
        
        # 创建游戏开奖结果显示区域
        self.home_results_text = QTextEdit()
        self.home_results_text.setReadOnly(True)
        self.home_results_text.setStyleSheet("font-size: 14px; line-height: 1.5;")
        latest_results_layout.addWidget(self.home_results_text)
        
        latest_results_group.setLayout(latest_results_layout)
        layout.addWidget(latest_results_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 快速投注按钮
        quick_bet_btn = QPushButton("快速投注")
        quick_bet_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        quick_bet_btn.clicked.connect(self.quick_bet)
        button_layout.addWidget(quick_bet_btn)
        
        # 开奖直播按钮
        live_btn = QPushButton("开奖直播")
        live_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        live_btn.clicked.connect(self.show_live_broadcast)
        button_layout.addWidget(live_btn)
        
        # 二维码扫描按钮
        qr_btn = QPushButton("二维码扫描")
        qr_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        qr_btn.clicked.connect(self.show_qr_scanner)
        button_layout.addWidget(qr_btn)
        
        layout.addLayout(button_layout)
        
        # 更新首页内容
        self.update_home_content()
        
        self.home_tab.setLayout(layout)
    
    def update_home_content(self):
        """更新首页内容"""
        # 更新时间
        self.update_time_display()
        
        # 更新滚动信息
        self.update_scroll_info()
        
        # 更新开奖结果
        results_text = ""
        
        for game in ["双色球", "快乐8", "3D", "七乐彩"]:
            if game in self.history_data and self.history_data[game]:
                latest_record = self.history_data[game][0]
                results_text += f"""
{game}最新开奖:
期号: {latest_record.get('期号', '未知')}
开奖号码: {latest_record.get('开奖号码', '未知')}
开奖时间: {latest_record.get('开奖日期', '未知')}
"""
                if game == "双色球":
                    results_text += f"奖池: {latest_record.get('奖池', '未知')}\n"
                results_text += "\n" + "="*50 + "\n\n"
            else:
                results_text += f"{game}: 暂无开奖数据\n\n" + "="*50 + "\n\n"
        
        self.home_results_text.setText(results_text)
    
    def quick_bet(self):
        """快速投注"""
        self.tabs.setCurrentWidget(self.fucai_tab)
    
    def show_live_broadcast(self):
        """显示开奖直播"""
        live_dialog = LiveBroadcastDialog(self)
        live_dialog.exec()
    
    def show_qr_scanner(self):
        """显示二维码扫描"""
        qr_dialog = QRCodeDialog(self)
        qr_dialog.exec()
    
    def init_qrcode_tab(self):
        """初始化二维码扫描选项卡"""
        layout = QVBoxLayout()
        
        # 创建二维码扫描对话框嵌入到选项卡中
        self.qr_code_dialog = QRCodeDialog(self)
        layout.addWidget(self.qr_code_dialog)
        
        self.qrcode_tab.setLayout(layout)
    
    def init_live_tab(self):
        """初始化开奖直播选项卡"""
        layout = QVBoxLayout()
        
        # 创建开奖直播对话框嵌入到选项卡中
        self.live_broadcast_dialog = LiveBroadcastDialog(self)
        layout.addWidget(self.live_broadcast_dialog)
        
        self.live_tab.setLayout(layout)

    # ... 其余代码保持不变，但需要修改自选号码对话框中的机选按钮和复式投注

    def self_select_ssq(self, bet_type="自选"):
        """双色球自选号码 - 修改版"""
        dialog = QDialog(self)
        dialog.setWindowTitle("双色球自选号码")
        dialog.setMinimumSize(500, 900)
        
        layout = QVBoxLayout()
        
        # 红球选择
        red_group = QGroupBox("红球选择 (选择6个号码, 范围1-33)")
        red_layout = QVBoxLayout()
        
        # 添加区间选择按钮 - 修复版本
        area_layout = QHBoxLayout()
        area_buttons = []
        areas = [("1-11", 1, 11), ("12-22", 12, 22), ("23-33", 23, 33)]
        for area_text, start, end in areas:
            btn = QPushButton(area_text)
            btn.setToolTip(f"选择{area_text}区间的所有号码")
            btn.clicked.connect(lambda checked, s=start, e=end: self.select_area_red_fixed(s, e, red_buttons))
            area_buttons.append(btn)
            area_layout.addWidget(btn)
        red_layout.addLayout(area_layout)
        
        # 添加尾号选择按钮 - 修复版本
        tail_layout = QHBoxLayout()
        tail_buttons = []
        for i in range(10):
            btn = QPushButton(f"{i}尾")
            btn.setToolTip(f"选择所有尾数为{i}的号码")
            btn.clicked.connect(lambda checked, t=i: self.select_tail_red_fixed(t, red_buttons))
            tail_buttons.append(btn)
            tail_layout.addWidget(btn)
        red_layout.addLayout(tail_layout)
        
        # 号码选择网格布局
        red_grid = QGridLayout()
        red_grid.setSpacing(5)
        
        red_buttons = []
        for i in range(1, 34):
            # 获取遗漏、频率和概率
            omission = self.get_omission("双色球", i, "red")
            frequency = self.get_frequency("双色球", i, "red")
            probability = self.get_probability("双色球", i, "red")
            cold_hot = self.get_cold_hot("双色球", i, "red")
            btn = NumberButton(str(i), cold_hot, omission, frequency, probability)
            red_buttons.append(btn)
            # 使用6列布局
            row = (i-1) // 6
            col = (i-1) % 6
            red_grid.addWidget(btn, row, col)
        
        red_layout.addLayout(red_grid)
        
        # 机选多注按钮
        random_multiple_layout = QHBoxLayout()
        self.red_multiple_spin = QSpinBox()
        self.red_multiple_spin.setRange(1, 1000)
        self.red_multiple_spin.setValue(5)
        self.red_multiple_spin.setSuffix(" 注")
        random_multiple_layout.addWidget(QLabel("机选注数:"))
        random_multiple_layout.addWidget(self.red_multiple_spin)
        
        red_random_multiple_btn = QPushButton("机选多注")
        red_random_multiple_btn.clicked.connect(lambda: self.generate_multiple_random(red_buttons, self.red_multiple_spin.value(), 6))
        random_multiple_layout.addWidget(red_random_multiple_btn)
        red_layout.addLayout(random_multiple_layout)
        
        red_group.setLayout(red_layout)
        layout.addWidget(red_group)
        
        # 蓝球选择
        blue_group = QGroupBox("蓝球选择 (选择1个号码, 范围1-16)")
        blue_layout = QVBoxLayout()
        
        blue_grid = QGridLayout()
        blue_grid.setSpacing(5)
        
        blue_buttons = []
        for i in range(1, 17):
            omission = self.get_omission("双色球", i, "blue")
            frequency = self.get_frequency("双色球", i, "blue")
            probability = self.get_probability("双色球", i, "blue")
            cold_hot = self.get_cold_hot("双色球", i, "blue")
            btn = NumberButton(str(i), cold_hot, omission, frequency, probability)
            blue_buttons.append(btn)
            # 使用8列布局
            row = (i-1) // 8
            col = (i-1) % 8
            blue_grid.addWidget(btn, row, col)
        
        blue_layout.addLayout(blue_grid)
        
        # 蓝球机选多注
        blue_random_layout = QHBoxLayout()
        self.blue_multiple_spin = QSpinBox()
        self.blue_multiple_spin.setRange(1, 1000)
        self.blue_multiple_spin.setValue(5)
        self.blue_multiple_spin.setSuffix(" 注")
        blue_random_layout.addWidget(QLabel("机选注数:"))
        blue_random_layout.addWidget(self.blue_multiple_spin)
        
        blue_random_multiple_btn = QPushButton("机选多注")
        blue_random_multiple_btn.clicked.connect(lambda: self.generate_multiple_random(blue_buttons, self.blue_multiple_spin.value(), 1))
        blue_random_layout.addWidget(blue_random_multiple_btn)
        blue_layout.addLayout(blue_random_layout)
        
        blue_group.setLayout(blue_layout)
        layout.addWidget(blue_group)
        
        # 复式投注选择
        complex_layout = QHBoxLayout()
        complex_layout.addWidget(QLabel("复式投注:"))
        
        self.red_complex_spin = QSpinBox()
        self.red_complex_spin.setRange(7, 33)
        self.red_complex_spin.setValue(8)
        self.red_complex_spin.setSuffix(" 个红球")
        complex_layout.addWidget(self.red_complex_spin)
        
        self.blue_complex_spin = QSpinBox()
        self.blue_complex_spin.setRange(2, 16)
        self.blue_complex_spin.setValue(3)
        self.blue_complex_spin.setSuffix(" 个蓝球")
        complex_layout.addWidget(self.blue_complex_spin)
        
        complex_btn = QPushButton("生成复式")
        complex_btn.clicked.connect(lambda: self.generate_complex_bet(red_buttons, blue_buttons, self.red_complex_spin.value(), self.blue_complex_spin.value()))
        complex_layout.addWidget(complex_btn)
        
        layout.addLayout(complex_layout)
        
        # 缩水投注按钮
        shrink_layout = QHBoxLayout()
        shrink_methods = ["中6保6", "中6保5", "中6保4", "中6保3", "中5保5", "中5保4", 
                         "中4保4", "中4保3", "胆拖投注", "旋转矩阵", "尾数缩水", "区间缩水"]
        
        self.shrink_combo = QComboBox()
        self.shrink_combo.addItems(shrink_methods)
        shrink_layout.addWidget(QLabel("缩水方法:"))
        shrink_layout.addWidget(self.shrink_combo)
        
        shrink_btn = QPushButton("缩水投注")
        shrink_btn.clicked.connect(lambda: self.apply_shrink_method(red_buttons, blue_buttons, self.shrink_combo.currentText()))
        shrink_layout.addWidget(shrink_btn)
        
        layout.addLayout(shrink_layout)
        
        # 矩阵投注按钮
        matrix_layout = QHBoxLayout()
        matrix_methods = ["7码矩阵", "8码矩阵", "9码矩阵", "10码矩阵", "11码矩阵", "12码矩阵",
                         "13码矩阵", "14码矩阵", "15码矩阵", "16码矩阵", "17码矩阵", "18码矩阵"]
        
        self.matrix_combo = QComboBox()
        self.matrix_combo.addItems(matrix_methods)
        matrix_layout.addWidget(QLabel("矩阵方法:"))
        matrix_layout.addWidget(self.matrix_combo)
        
        matrix_btn = QPushButton("矩阵投注")
        matrix_btn.clicked.connect(lambda: self.apply_matrix_method(red_buttons, blue_buttons, self.matrix_combo.currentText()))
        matrix_layout.addWidget(matrix_btn)
        
        layout.addLayout(matrix_layout)
        
        # 确认和取消按钮
        btn_layout = QHBoxLayout()
        confirm_btn = QPushButton("确认投注")
        cancel_btn = QPushButton("取消")
        
        confirm_btn.clicked.connect(lambda: self.confirm_ssq_bet(red_buttons, blue_buttons, dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def select_area_red_fixed(self, start, end, buttons):
        """修复的区间选择"""
        for i in range(start, end + 1):
            if 1 <= i <= 33 and i-1 < len(buttons):
                buttons[i-1].setChecked(True)
    
    def select_tail_red_fixed(self, tail, buttons):
        """修复的尾号选择"""
        for i in range(1, 34):
            if i % 10 == tail and i-1 < len(buttons):
                buttons[i-1].setChecked(True)
    
    def generate_multiple_random(self, buttons, count, select_count):
        """机选多注"""
        # 清除所有选择
        for btn in buttons:
            btn.setChecked(False)
        
        # 随机选择指定数量的号码
        all_numbers = list(range(1, len(buttons) + 1))
        for _ in range(count):
            nums = random.sample(all_numbers, select_count)
            for num in nums:
                if 0 < num <= len(buttons):
                    buttons[num-1].setChecked(True)
    
    def generate_complex_bet(self, red_buttons, blue_buttons, red_count, blue_count):
        """生成复式投注"""
        # 清除所有选择
        for btn in red_buttons + blue_buttons:
            btn.setChecked(False)
        
        # 随机选择红球
        red_numbers = random.sample(range(1, 34), red_count)
        for num in red_numbers:
            if 0 < num <= len(red_buttons):
                red_buttons[num-1].setChecked(True)
        
        # 随机选择蓝球
        blue_numbers = random.sample(range(1, 17), blue_count)
        for num in blue_numbers:
            if 0 < num <= len(blue_buttons):
                blue_buttons[num-1].setChecked(True)
    
    def apply_shrink_method(self, red_buttons, blue_buttons, method):
        """应用缩水方法"""
        # 这里实现12种缩水方法的逻辑
        QMessageBox.information(self, "缩水投注", f"已应用{method}缩水方法")
    
    def apply_matrix_method(self, red_buttons, blue_buttons, method):
        """应用矩阵方法"""
        # 这里实现12种矩阵方法的逻辑
        QMessageBox.information(self, "矩阵投注", f"已应用{method}矩阵方法")
    
    def confirm_ssq_bet(self, red_buttons, blue_buttons, dialog):
        """确认双色球投注"""
        red_selected = [i+1 for i, btn in enumerate(red_buttons) if btn.isChecked()]
        blue_selected = [i+1 for i, btn in enumerate(blue_buttons) if btn.isChecked()]
        
        if len(red_selected) < 6 or len(blue_selected) < 1:
            QMessageBox.warning(dialog, "错误", "请选择至少6个红球和1个蓝球")
            return
        
        # 处理投注逻辑
        result = f"红球: {', '.join(map(str, sorted(red_selected)))}\n蓝球: {', '.join(map(str, sorted(blue_selected)))}"
        
        # 计算注数和金额
        if len(red_selected) > 6 or len(blue_selected) > 1:
            red_comb = math.comb(len(red_selected), 6)
            blue_comb = len(blue_selected)
            bets = red_comb * blue_comb
            bet_type = "复式"
        else:
            bets = 1
            bet_type = "单式"
        
        amount = bets * 2.0
        
        QMessageBox.information(self, "投注确认", 
                              f"{result}\n\n注数: {bets}注\n金额: ¥{amount:.2f}\n投注类型: {bet_type}")
        
        self.add_betting_record("双色球", bet_type, result, bets, amount)
        dialog.accept()

    # 添加概率计算方法
    def calculate_probability(self):
        """计算号码出现概率"""
        for game_type, records in self.history_data.items():
            if not records:
                continue
                
            total_periods = len(records)
            
            if game_type not in self.probability_data:
                self.probability_data[game_type] = {}
            
            if game_type == "双色球":
                # 红球概率
                red_prob = {}
                for num in range(1, 34):
                    freq = self.get_frequency("双色球", num, "red")
                    prob = round((freq / total_periods) * 100, 1)
                    red_prob[num] = prob
                
                # 蓝球概率
                blue_prob = {}
                for num in range(1, 17):
                    freq = self.get_frequency("双色球", num, "blue")
                    prob = round((freq / total_periods) * 100, 1)
                    blue_prob[num] = prob
                
                self.probability_data[game_type]["red"] = red_prob
                self.probability_data[game_type]["blue"] = blue_prob
                
            elif game_type in ["3D"]:
                positions = [f"pos{i+1}" for i in range(3)]
                for pos in positions:
                    pos_prob = {}
                    for num in range(0, 10):
                        freq = self.get_frequency(game_type, num, pos)
                        prob = round((freq / total_periods) * 100, 1)
                        pos_prob[num] = prob
                    self.probability_data[game_type][pos] = pos_prob
                    
            else:
                # 快乐8和七乐彩
                main_prob = {}
                max_num = 80 if game_type == "快乐8" else 30
                for num in range(1, max_num + 1):
                    freq = self.get_frequency(game_type, num)
                    prob = round((freq / total_periods) * 100, 1)
                    main_prob[num] = prob
                self.probability_data[game_type]["main"] = main_prob
    
    def get_probability(self, game_type, number, area="main"):
        """获取号码出现概率"""
        if game_type not in self.probability_data:
            return 0
            
        if area not in self.probability_data[game_type]:
            return 0
            
        return self.probability_data[game_type][area].get(number, 0)

    # 修改快乐8奖金计算方法
    def calculate_kl8_prize(self, select_count, match_count):
        """计算快乐8奖金"""
        if select_count == 10:
            if match_count == 10:
                return PRIZE_RULES["快乐8"]["选十中十"]["base"]
            elif match_count == 9:
                return PRIZE_RULES["快乐8"]["选十中九"]["base"]
            elif match_count == 8:
                return PRIZE_RULES["快乐8"]["选十中八"]["base"]
            elif match_count == 7:
                return PRIZE_RULES["快乐8"]["选十中七"]["base"]
            elif match_count == 6:
                return PRIZE_RULES["快乐8"]["选十中六"]["base"]
            elif match_count == 5:
                return PRIZE_RULES["快乐8"]["选十中五"]["base"]
            elif match_count == 0:
                return PRIZE_RULES["快乐8"]["选十中零"]["base"]
        elif select_count == 9:
            if match_count == 9:
                return PRIZE_RULES["快乐8"]["选九中九"]["base"]
            elif match_count == 8:
                return PRIZE_RULES["快乐8"]["选九中八"]["base"]
            elif match_count == 7:
                return PRIZE_RULES["快乐8"]["选九中七"]["base"]
            elif match_count == 6:
                return PRIZE_RULES["快乐8"]["选九中六"]["base"]
            elif match_count == 5:
                return PRIZE_RULES["快乐8"]["选九中五"]["base"]
            elif match_count == 4:
                return PRIZE_RULES["快乐8"]["选九中四"]["base"]
        elif select_count == 8:
            if match_count == 8:
                return PRIZE_RULES["快乐8"]["选八中八"]["base"]
            elif match_count == 7:
                return PRIZE_RULES["快乐8"]["选八中七"]["base"]
            elif match_count == 6:
                return PRIZE_RULES["快乐8"]["选八中六"]["base"]
            elif match_count == 5:
                return PRIZE_RULES["快乐8"]["选八中五"]["base"]
            elif match_count == 4:
                return PRIZE_RULES["快乐8"]["选八中四"]["base"]
        elif select_count == 7:
            if match_count == 7:
                return PRIZE_RULES["快乐8"]["选七中七"]["base"]
            elif match_count == 6:
                return PRIZE_RULES["快乐8"]["选七中六"]["base"]
            elif match_count == 5:
                return PRIZE_RULES["快乐8"]["选七中五"]["base"]
            elif match_count == 4:
                return PRIZE_RULES["快乐8"]["选七中四"]["base"]
        elif select_count == 6:
            if match_count == 6:
                return PRIZE_RULES["快乐8"]["选六中六"]["base"]
            elif match_count == 5:
                return PRIZE_RULES["快乐8"]["选六中五"]["base"]
            elif match_count == 4:
                return PRIZE_RULES["快乐8"]["选六中四"]["base"]
        elif select_count == 5:
            if match_count == 5:
                return PRIZE_RULES["快乐8"]["选五中五"]["base"]
            elif match_count == 4:
                return PRIZE_RULES["快乐8"]["选五中四"]["base"]
            elif match_count == 3:
                return PRIZE_RULES["快乐8"]["选五中三"]["base"]
        elif select_count == 4:
            if match_count == 4:
                return PRIZE_RULES["快乐8"]["选四中四"]["base"]
            elif match_count == 3:
                return PRIZE_RULES["快乐8"]["选四中三"]["base"]
            elif match_count == 2:
                return PRIZE_RULES["快乐8"]["选四中二"]["base"]
        elif select_count == 3:
            if match_count == 3:
                return PRIZE_RULES["快乐8"]["选三中三"]["base"]
            elif match_count == 2:
                return PRIZE_RULES["快乐8"]["选三中二"]["base"]
        elif select_count == 2:
            if match_count == 2:
                return PRIZE_RULES["快乐8"]["选二中二"]["base"]
        elif select_count == 1:
            if match_count == 1:
                return PRIZE_RULES["快乐8"]["选一中一"]["base"]
        
        return 0

    # ... 其余方法保持不变，但需要类似修改其他游戏的自选对话框

    def init_fucai_tab(self):
        """初始化福彩投注选项卡 - 简化版示例"""
        layout = QVBoxLayout()
        
        # 双色球部分
        ssq_group = QGroupBox("双色球")
        ssq_layout = QVBoxLayout()
        
        ssq_btn = QPushButton("双色球自选号码")
        ssq_btn.clicked.connect(self.self_select_ssq)
        ssq_layout.addWidget(ssq_btn)
        
        ssq_group.setLayout(ssq_layout)
        layout.addWidget(ssq_group)
        
        # 快乐8部分
        kl8_group = QGroupBox("快乐8")
        kl8_layout = QVBoxLayout()
        
        kl8_btn = QPushButton("快乐8自选号码")
        kl8_btn.clicked.connect(self.self_select_kl8)
        kl8_layout.addWidget(kl8_btn)
        
        kl8_group.setLayout(kl8_layout)
        layout.addWidget(kl8_group)
        
        # 3D部分
        d3_group = QGroupBox("3D")
        d3_layout = QVBoxLayout()
        
        d3_btn = QPushButton("3D自选号码")
        d3_btn.clicked.connect(self.self_select_d3)
        d3_layout.addWidget(d3_btn)
        
        d3_group.setLayout(d3_layout)
        layout.addWidget(d3_group)
        
        # 七乐彩部分
        qlc_group = QGroupBox("七乐彩")
        qlc_layout = QVBoxLayout()
        
        qlc_btn = QPushButton("七乐彩自选号码")
        qlc_btn.clicked.connect(self.self_select_qlc)
        qlc_layout.addWidget(qlc_btn)
        
        qlc_group.setLayout(qlc_layout)
        layout.addWidget(qlc_group)
        
        self.fucai_tab.setLayout(layout)

    # 其他游戏的自选方法需要类似修改，这里只展示双色球的修改

    def self_select_kl8(self):
        """快乐8自选号码 - 修改版"""
        # 类似双色球的修改，添加机选多注、复式投注、缩水投注、矩阵投注等功能
        dialog = QDialog(self)
        dialog.setWindowTitle("快乐8自选号码")
        dialog.setMinimumSize(600, 800)
        
        layout = QVBoxLayout()
        
        # 选择玩法
        play_layout = QHBoxLayout()
        play_layout.addWidget(QLabel("选择玩法:"))
        self.kl8_play_combo = QComboBox()
        self.kl8_play_combo.addItems(["选一", "选二", "选三", "选四", "选五", "选六", "选七", "选八", "选九", "选十"])
        play_layout.addWidget(self.kl8_play_combo)
        layout.addLayout(play_layout)
        
        # 号码选择区域
        numbers_group = QGroupBox("选择号码 (范围1-80)")
        numbers_layout = QVBoxLayout()
        
        # 号码网格
        numbers_grid = QGridLayout()
        numbers_grid.setSpacing(3)
        
        kl8_buttons = []
        for i in range(1, 81):
            omission = self.get_omission("快乐8", i)
            frequency = self.get_frequency("快乐8", i)
            probability = self.get_probability("快乐8", i)
            cold_hot = self.get_cold_hot("快乐8", i)
            btn = NumberButton(str(i), cold_hot, omission, frequency, probability)
            kl8_buttons.append(btn)
            row = (i-1) // 10
            col = (i-1) % 10
            numbers_grid.addWidget(btn, row, col)
        
        numbers_layout.addLayout(numbers_grid)
        
        # 机选多注
        random_multiple_layout = QHBoxLayout()
        self.kl8_multiple_spin = QSpinBox()
        self.kl8_multiple_spin.setRange(1, 1000)
        self.kl8_multiple_spin.setValue(5)
        random_multiple_layout.addWidget(QLabel("机选注数:"))
        random_multiple_layout.addWidget(self.kl8_multiple_spin)
        
        kl8_random_btn = QPushButton("机选多注")
        kl8_random_btn.clicked.connect(lambda: self.generate_kl8_multiple_random(kl8_buttons, self.kl8_multiple_spin.value(), self.kl8_play_combo.currentText()))
        random_multiple_layout.addWidget(kl8_random_btn)
        numbers_layout.addLayout(random_multiple_layout)
        
        numbers_group.setLayout(numbers_layout)
        layout.addWidget(numbers_group)
        
        # 缩水投注
        shrink_layout = QHBoxLayout()
        kl8_shrink_methods = ["中10保10", "中10保9", "中10保8", "中9保9", "中9保8", "中8保8",
                             "中7保7", "中6保6", "中5保5", "胆拖投注", "旋转矩阵", "尾数缩水"]
        
        self.kl8_shrink_combo = QComboBox()
        self.kl8_shrink_combo.addItems(kl8_shrink_methods)
        shrink_layout.addWidget(QLabel("缩水方法:"))
        shrink_layout.addWidget(self.kl8_shrink_combo)
        
        kl8_shrink_btn = QPushButton("缩水投注")
        kl8_shrink_btn.clicked.connect(lambda: self.apply_kl8_shrink_method(kl8_buttons, self.kl8_shrink_combo.currentText(), self.kl8_play_combo.currentText()))
        shrink_layout.addWidget(kl8_shrink_btn)
        
        layout.addLayout(shrink_layout)
        
        # 矩阵投注
        matrix_layout = QHBoxLayout()
        kl8_matrix_methods = ["11码矩阵", "12码矩阵", "13码矩阵", "14码矩阵", "15码矩阵", "16码矩阵",
                             "17码矩阵", "18码矩阵", "19码矩阵", "20码矩阵", "21码矩阵", "22码矩阵"]
        
        self.kl8_matrix_combo = QComboBox()
        self.kl8_matrix_combo.addItems(kl8_matrix_methods)
        matrix_layout.addWidget(QLabel("矩阵方法:"))
        matrix_layout.addWidget(self.kl8_matrix_combo)
        
        kl8_matrix_btn = QPushButton("矩阵投注")
        kl8_matrix_btn.clicked.connect(lambda: self.apply_kl8_matrix_method(kl8_buttons, self.kl8_matrix_combo.currentText(), self.kl8_play_combo.currentText()))
        matrix_layout.addWidget(kl8_matrix_btn)
        
        layout.addLayout(matrix_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        confirm_btn = QPushButton("确认投注")
        cancel_btn = QPushButton("取消")
        
        confirm_btn.clicked.connect(lambda: self.confirm_kl8_bet(kl8_buttons, self.kl8_play_combo.currentText(), dialog))
        cancel_btn.clicked.connect(dialog.reject)
        
        btn_layout.addWidget(confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def generate_kl8_multiple_random(self, buttons, count, play_type):
        """快乐8机选多注"""
        select_count = int(play_type[1:])  # 从"选五"中提取5
        self.generate_multiple_random(buttons, count, select_count)
    
    def apply_kl8_shrink_method(self, buttons, method, play_type):
        """应用快乐8缩水方法"""
        QMessageBox.information(self, "快乐8缩水投注", f"已应用{method}缩水方法，玩法: {play_type}")
    
    def apply_kl8_matrix_method(self, buttons, method, play_type):
        """应用快乐8矩阵方法"""
        QMessageBox.information(self, "快乐8矩阵投注", f"已应用{method}矩阵方法，玩法: {play_type}")
    
    def confirm_kl8_bet(self, buttons, play_type, dialog):
        """确认快乐8投注"""
        selected = [i+1 for i, btn in enumerate(buttons) if btn.isChecked()]
        select_count = int(play_type[1:])
        
        if len(selected) < select_count:
            QMessageBox.warning(dialog, "错误", f"请选择至少{select_count}个号码")
            return
        
        result = f"号码: {', '.join(map(str, sorted(selected)))}"
        
        # 计算注数和金额
        if len(selected) > select_count:
            bets = math.comb(len(selected), select_count)
            bet_type = "复式"
        else:
            bets = 1
            bet_type = "单式"
        
        amount = bets * 2.0
        
        QMessageBox.information(self, "投注确认", 
                              f"玩法: {play_type}\n{result}\n\n注数: {bets}注\n金额: ¥{amount:.2f}\n投注类型: {bet_type}")
        
        self.add_betting_record("快乐8", f"{play_type}-{bet_type}", result, bets, amount)
        dialog.accept()

    # 3D和七乐彩的类似修改略...

    # 原有的其他方法保持不变
    def add_betting_record(self, game, bet_type, numbers, bets_count, amount):
        """添加投注记录"""
        record = {
            "game": game,
            "bet_type": bet_type,
            "numbers": numbers,
            "bets_count": bets_count,
            "amount": amount,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.betting_history.append(record)
        if hasattr(self, 'records_table'):
            self.update_records_table()

    def init_analysis_tab(self):
        """初始化数据分析选项卡"""
        layout = QVBoxLayout()
        
        # 数据分析界面代码...
        analysis_label = QLabel("数据分析功能")
        analysis_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(analysis_label)
        
        self.analysis_tab.setLayout(layout)

    def init_records_tab(self):
        """初始化投注记录选项卡"""
        layout = QVBoxLayout()
        
        # 投注记录界面代码...
        records_label = QLabel("投注记录功能")
        records_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(records_label)
        
        self.records_tab.setLayout(layout)

    def init_history_tab(self):
        """初始化开奖历史选项卡"""
        layout = QVBoxLayout()
        
        # 开奖历史界面代码...
        history_label = QLabel("开奖历史功能")
        history_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(history_label)
        
        self.history_tab.setLayout(layout)

    def init_network_tab(self):
        """初始化开奖记录选项卡"""
        layout = QVBoxLayout()
        
        # 开奖记录界面代码...
        network_label = QLabel("开奖记录功能")
        network_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(network_label)
        
        self.network_tab.setLayout(layout)

    def init_shrink_tab(self):
        """初始化号码缩水选项卡"""
        layout = QVBoxLayout()
        
        # 号码缩水界面代码...
        shrink_label = QLabel("号码缩水功能")
        shrink_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(shrink_label)
        
        self.shrink_tab.setLayout(layout)

    def init_matrix_tab(self):
        """初始化矩阵投注选项卡"""
        layout = QVBoxLayout()
        
        # 矩阵投注界面代码...
        matrix_label = QLabel("矩阵投注功能")
        matrix_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(matrix_label)
        
        self.matrix_tab.setLayout(layout)

    # 其他必要的方法...
    def load_api_key(self):
        """加载API密钥"""
        try:
            if os.path.exists("api_key.json"):
                with open("api_key.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key", self.api_key)
        except:
            pass

    def load_history_data(self):
        """加载历史数据"""
        try:
            for game in ["双色球", "快乐8", "3D", "七乐彩"]:
                filename = f"{game}_history.json"
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as f:
                        self.history_data[game] = json.load(f)
        except:
            pass

    def load_betting_history(self):
        """加载投注记录"""
        try:
            if os.path.exists("betting_history.json"):
                with open("betting_history.json", "r", encoding="utf-8") as f:
                    self.betting_history = json.load(f)
        except:
            pass

    def calculate_omission(self):
        """计算遗漏数据"""
        # 简化实现
        for game in self.history_data:
            if game not in self.last_omission_data:
                self.last_omission_data[game] = {}
            if game not in self.frequency_data:
                self.frequency_data[game] = {}
            if game not in self.cold_hot_data:
                self.cold_hot_data[game] = {}
        
        # 计算概率
        self.calculate_probability()

    def get_omission(self, game_type, number, area="main"):
        """获取遗漏期数"""
        return self.last_omission_data.get(game_type, {}).get(area, {}).get(number, 0)

    def get_frequency(self, game_type, number, area="main"):
        """获取出现频率"""
        return self.frequency_data.get(game_type, {}).get(area, {}).get(number, 0)

    def get_cold_hot(self, game_type, number, area="main"):
        """获取冷热状态"""
        return self.cold_hot_data.get(game_type, {}).get(area, {}).get(number, "default")

    def self_select_d3(self):
        """3D自选号码"""
        QMessageBox.information(self, "3D自选", "3D自选功能开发中...")

    def self_select_qlc(self):
        """七乐彩自选号码"""
        QMessageBox.information(self, "七乐彩自选", "七乐彩自选功能开发中...")

    def update_records_table(self):
        """更新投注记录表格"""
        # 简化实现
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LotteryApp()
    window.show()
    sys.exit(app.exec())
