# 基于分类相似律的缩尺拟静力试验非完全相似比确定工具

本版本以原拟静力 `app.py` 为页面主体，保留原有蓝色标题区、Sidebar、四个原页签、五类内置截面、三种内置计算方法、结果卡片和导出方式，仅在末尾增加“扩展组件管理”页签。

## 目录结构

```text
quasi_static_modified/
├─ app.py
├─ requirements.txt
├─ README.md
├─ similarity_engine/
│  ├─ __init__.py
│  ├─ api.py
│  ├─ definitions.py
│  ├─ models.py
│  ├─ safe_expr.py
│  └─ ui/
│     ├─ __init__.py
│     ├─ component_manager.py
│     └─ dynamic_form.py
├─ user_components/
│  ├─ sections/
│  └─ methods/
└─ tests/
   └─ test_regression.py
```

`user_components` 用于保存经过校验的用户 YAML/JSON 配置。首次启动时若目录不存在，程序会自动创建。

## 安装与启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 新增功能

- 上传 `.yaml`、`.yml`、`.json` 自定义截面或计算方法；
- 安装前校验组件 ID、参数、默认值、几何约束和数学表达式；
- 用户截面采用原型构件/缩尺模型两列动态输入；
- 用户方法可生成附加参数输入；
- 用户组件可列出、删除并下载模板；
- 内置组件与用户组件统一显示“软件内置”或“用户扩展”；
- 配置表达式由 AST 白名单解释器计算，不使用 `eval()`、`exec()` 或动态导入；
- 软件内置组件不可删除，也不能被同 ID 用户组件覆盖。

## 计算路径

- 内置截面 + 内置方法：继续调用原 `calculate_similarity()`，保证计算结果不变；
- 任一用户扩展组件：调用 `SimilarityAPI`，再转换为原拟静力结果字典格式。

## 回归测试

```bash
python -m unittest tests.test_regression -v
```

测试包括：

- 默认空心方钢在 DA、CSL-Y、CSL-N 下的 14 个核心结果；
- 矩形用户截面的安装、计算与删除；
- 用户计算方法的附加参数和输出；
- `.py` 文件与危险表达式拒绝测试。
