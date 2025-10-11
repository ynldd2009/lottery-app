# lottery-app
福彩应用 -包含投注、分析、二维码扫描和开奖直播功能
lottery-app/
├── main.py                    # 主程序入口
├── requirements.txt           # 依赖包列表
├── README.md                  # 项目说明文档
├── src/                       # 源代码目录
│   ├── __init__.py
│   ├── lottery_app.py         # 主应用类（已提供部分）
│   ├── qr_scanner.py          # 二维码扫描模块
│   ├── live_broadcast.py      # 开奖直播模块
│   ├── analysis_tools.py      # 分析工具模块
│   └── utils.py               # 工具函数
├── data/                      # 数据目录
│   ├── api_key.json
│   ├── betting_history.json
│   └── lottery_history/
└── images/                    # 图片资源
    └── icons/
