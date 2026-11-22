"""
自动钓鱼(星渚游艇码头) — 带自动传送寻路的 ok-nte 单文件自定义任务

单文件自包含: 传送锚点等模板图以内嵌 base64 随脚本携带, 运行时解码并按捕获分辨率
自动缩放匹配, 无需再部署 assets/ 模板文件或 LabelEnum.py。

导入方式(二选一):
  A. 把本文件复制到 D:\Game\ok-nte\data\apps\ok-nte\working\ok_tasks\ 然后重启 ok-nte;
  B. ok-nte 任务编辑页新建脚本, 粘贴本文件全文并保存。

在原版"自动钓鱼"基础上增加导航前置流程: 大世界任意位置
→ (开启"都市活力为0时跳过"时) M 开图读取右上角都市活力左侧计数, 为0则直接结束任务
→ F1 界域页"经验及甲硬币"页签, 点最上方的"前往"(合订本/万花筒/硬笔记任一都会传送
  到胡迪尼的魔术舞台, 位于钓鱼点附近)
→ M 开大地图(确认打开, 最多按3次), 优先在地图右上区域找锚点(找不到回退全屏)
→ 点击锚点偏左下位置(实测可直接打开传送确认面板, 不触发多选项对话框), 确认面板点传送
→ 按录制路径盲走 → 点按A微调逼近交互点(按下立即松开, 最多10次, 每次核对提示文字
  含"钓鱼", 超限自动停止任务)
→ 进入原版钓鱼流程(抛竿/控条/补饵卖鱼/轮次零改动)。

任务自停四重保险: 都市活力为0(开关开启时) / 寻路重试N次失败 / 钓鱼失败轮数达N /
任务总超时(分钟), 到限自动干净停止, 无需手动点停止。

前提: 大地图缩放档位为 Mid(模板档位; 已确认重开游戏后自动回归 Mid)。
      若同一局游戏内手动改过缩放, 请调回 Mid 或重启游戏。
"""
import base64
import os
import re
import sys
import time

# 保证能 import 宿主程序的 src 包 (脚本位于 working\ok_tasks\ 下; 粘贴导入时工作目录已在 path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from ok import TaskDisabledException, WaitFailedException

from src.tasks.FishingTask import FishingTask
from src.ui.icons import task_icon

# ===================== 标定区(换钓点/调参改这里) =====================

# -- 锚点 --
# 大地图上传送锚点的匹配名(仅内部使用, 模板已内嵌, 无需 assets 文件;
# 勿与 GUI 导出的注册特征 ReroRero 重名, 以确保走内嵌模板匹配路径)
ANCHOR_FEATURE = "xingzhu_marina_tp"
# 传送后锚点在大地图(以角色为中心)上的优先搜索区(右上角), 找不到回退全屏;
# 以及锚点搜索等待时长(秒)
ANCHOR_REGION = (0.45, 0.02, 0.98, 0.62)
ANCHOR_TIMEOUT = 12
# 点击锚点的偏移(相对锚点框的比例 x, y): 偏左下可直接打开传送确认面板,
# 不会触发"电话亭/猫刊亭"多选项对话框
ANCHOR_CLICK_OFFSET = (0.25, 0.75)

# -- 地图打开判据 --
# 大地图右上角的关闭按钮(×), 从运行截图实测裁剪; 框架内置 Labels.close_button
# 与 NTE 地图的关闭按钮不匹配, 故弃用
MAP_OPEN_FEATURE = "nav_map_close_btn"
MAP_OPEN_BOX = (0.90, 0.01, 1.00, 0.13)  # × 搜索区(归一化: 左, 上, 右, 下)

# -- 中转界域 --
# 经 F1 界域页"经验及甲硬币"页签传送: 合订本/万花筒/硬笔记任一点"前往"都会
# 传送到胡迪尼的魔术舞台(位于钓鱼点附近), 定位方式照搬 AnomalyTask
HOUDINI_TAB = "exp"                              # 页签(PositionMap 属性名)
HOUDINI_BTN_BOX = (0.925, 0.190, 0.982, 0.760)   # 右侧前往按钮列

# -- 盲走路径 --
# 传送落地后的行走路径(来自录制器: 传送后录制的操作.txt); hold=按住秒数, after=松开后等待秒数
WAYPOINTS = [
    {"key": "w", "hold": 5.14, "after": 1.33},
    {"key": "a", "hold": 4.71, "after": 0.87},
    {"key": "w", "hold": 15.53, "after": 1.32},
    {"key": "a", "hold": 13.59, "after": 0.50},
]

# -- 落点微调与交互核对 --
# 盲走落点距交互点差几步时点按微调(按下立即松开), 最多 N 次, 超限截图留证并停止任务;
# 每次核对 F 提示旁文字含 INTERAC_TEXT("钓鱼")才算到达, 排除其他可互动物品
APPROACH_TAP_KEY = "a"
APPROACH_TAP_MAX = 10
APPROACH_TAP_SETTLE = 0.6  # 每次点按后等待提示出现的秒数
INTERAC_TEXT = "钓鱼"

# -- 都市活力检查 --
# 大地图右上角计数器 OCR 区域内取左侧数字(当前活力), 为0视为用完 → 跳过任务;
# 区域不含上限数字, 上限变化无需改代码
VITALITY_BOX = (0.66, 0.02, 0.82, 0.10)
VITALITY_TIMEOUT = 3

# -- 自停与超时参数 --
NAV_TIMEOUT = 240    # 寻路全流程超时(秒), 超时后放弃寻路并按原版逻辑继续
NAV_MAX_RETRY = 2    # 寻路连续失败达该次数后自动停止任务
FAIL_LIMIT = 5       # 钓鱼失败轮数达到后自动停止任务(0=不限制)

# ================================================================

# ---- 内嵌模板(勿动) ----
# 锚点: ReroRero_1920x1080_1139_223.png (46x40, 胡迪尼传送后右上视角)
ANCHOR_TEMPLATE_B64 = (
    "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"
)

# 地图关闭按钮(×): map_close_btn_1920x1080_1802_35.png (64x63)
MAP_OPEN_TEMPLATE_B64 = (
    "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"
    "SUVORK5CYII="
)

TEMPLATE_BASE_WIDTH = 1920  # 模板截图分辨率宽, 用于跨分辨率缩放


def _decode_b64_png(data, what):
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(what + "解码失败")
    return image


ANCHOR_TEMPLATE = _decode_b64_png(ANCHOR_TEMPLATE_B64, "锚点模板")
MAP_OPEN_TEMPLATE = _decode_b64_png(MAP_OPEN_TEMPLATE_B64, "地图关闭按钮模板")


class FishingXingzhuTask(FishingTask):
    """自动传送至星渚游艇码头钓鱼点并完成钓鱼; 在原版自动钓鱼上仅增加导航前置流程。"""

    CONF_VITALITY_SKIP = "都市活力为0时跳过"
    CONF_TASK_TIMEOUT = "任务总超时(分钟)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "自动钓鱼-星渚码头"
        self.description = "自动传送至星渚游艇码头钓鱼点, 并完成一轮或多轮钓鱼"
        self.group_name = "都市闲趣"
        self.group_icon = task_icon("city_fun")
        self._nav_attempts = 0
        self._anchor_tpl_cache = {}
        self._map_open_tpl_cache = {}
        self._task_finished = True
        self.default_config.update(
            {
                self.CONF_VITALITY_SKIP: True,
                self.CONF_TASK_TIMEOUT: 30,
            }
        )
        self.config_description.update(
            {
                self.CONF_VITALITY_SKIP: "开始任务前先开图读取右上角都市活力左侧计数, "
                "为0视为本周用完, 直接结束任务; 关闭则不检查直接开始流程",
                self.CONF_TASK_TIMEOUT: "整个任务的最长运行时间(分钟), 到时自动停止; 0=不限制",
            }
        )

    # ---- 生命周期: 入口与自停保险 ----

    def run(self):
        self._nav_attempts = 0
        self._task_finished = False
        self._start_task_watchdog()
        try:
            super().run()
        finally:
            self._task_finished = True

    def _start_task_watchdog(self):
        """总超时看门狗: 到时请求停止轮次, 流程在下一个循环干净退出。"""
        try:
            minutes = max(0, int(self.config.get(self.CONF_TASK_TIMEOUT, 30)))
        except (TypeError, ValueError):
            minutes = 30
        if minutes <= 0:
            return
        deadline = time.time() + minutes * 60

        def watchdog():
            if self._task_finished:
                return False
            if time.time() >= deadline:
                self.stop_rounds(f"任务总超时({minutes}分钟)")
                return False
            return True

        self.submit_periodic_task(5, watchdog)

    def has_remaining_rounds(self):
        """钓鱼失败轮数达到 FAIL_LIMIT 时请求停止任务。"""
        if not super().has_remaining_rounds():
            return False
        if FAIL_LIMIT > 0 and self._round_state.failed_count >= FAIL_LIMIT:
            self.stop_rounds(f"失败次数已达 {FAIL_LIMIT}")
            return False
        return True

    # ---- 接入点: 钓鱼 ENTER guard 的动作(大世界 → 钓鱼界面) ----
    # guard 只在 is_in_team() 为真时触发; 已在钓鱼界面时不会进到这里。

    def _enter_fishing_from_interaction(self):
        if not self.has_remaining_rounds():
            return  # 已请求停止(超时/失败达限), 直接结束流程
        if self.config.get(self.CONF_VITALITY_SKIP, True):
            self._check_vitality_and_maybe_skip()
            if not self.has_remaining_rounds():
                return  # 都市活力已用完, 任务结束
        self._navigate_then_ready()
        super()._enter_fishing_from_interaction()

    # ---- 都市活力检查 ----

    def _check_vitality_and_maybe_skip(self):
        self._set_stage("检查都市活力")
        self.ensure_main(time_out=30)
        if not self._open_map_until_ready():
            self.log_warning("检查都市活力: 大地图未能打开, 跳过检查继续任务流程")
            return
        results = self.wait_ocr(
            *VITALITY_BOX,
            time_out=VITALITY_TIMEOUT,
            raise_if_not_found=False,
        )
        vital = -1
        if results:
            m = re.search(r"\d+", "".join(b.name for b in results))
            if m:
                vital = int(m.group(0))
        if vital < 0:
            self.log_warning("未读取到都市活力计数, 继续任务流程", notify=True)
        else:
            self.info_set("都市活力", str(vital))
            if vital == 0:
                self.log_info("都市活力已用完(0), 按配置跳过任务", notify=True)
                self.stop_rounds("都市活力已用完(0)")
                return
            self.log_info(f"都市活力剩余 {vital}, 继续任务流程")
        self.ensure_main(time_out=15)  # ESC 关闭地图回大世界

    # ---- 寻路主流程 ----

    def _navigate_then_ready(self):
        if not self.is_in_team():
            self.log_info("当前不在大世界, 跳过自动寻路")
            return
        if self._at_fishing_interac():
            self.log_info("已在钓鱼交互点旁(F提示含'钓鱼'), 跳过自动寻路")
            return
        if self._nav_attempts >= NAV_MAX_RETRY:
            self.stop_rounds("自动寻路多次失败")
            self.log_warning(f"自动寻路失败已达 {NAV_MAX_RETRY} 次, 任务即将停止", notify=True)
            return
        self._nav_attempts += 1
        self.log_round_info(f"自动寻路开始 (第 {self._nav_attempts}/{NAV_MAX_RETRY} 次)")
        timer = self.clock.timer(timeout=NAV_TIMEOUT)
        try:
            self._go_to_houdini_stage(timer)
            self._teleport_to_anchor(timer)
            self._walk_recorded_path(timer)
            self._approach_until_interac()
            self.log_round_info("自动寻路完成, 已到达钓鱼交互点")
        except TaskDisabledException:
            raise
        except Exception as e:
            if not self.has_remaining_rounds():
                return  # 已请求停止(如微调超限), 不再降级重试
            self.screenshot("fishing_nav_failed")
            self.log_error("自动寻路失败, 改按原版逻辑进入钓鱼", e, notify=True)
            try:
                self.ensure_main(time_out=15)
            except Exception:
                pass

    # ---- 中转段: 经 F1 界域页"经验及甲硬币"页签传送到钓鱼点附近 ----

    def _go_to_houdini_stage(self, timer):
        """传送经验及甲硬币页签任一关卡(合订本/万花筒/硬笔记), 均到胡迪尼的魔术舞台。

        定位方式照搬 AnomalyTask: 找右侧按钮列(0.925~0.982)最上方的确认按钮点击。
        """
        self._set_stage("前往胡迪尼的魔术舞台")
        self.ensure_main(time_out=30)
        self.open_f1_domain()
        self._check_nav_timer(timer)

        tab = getattr(self.pos.panels.f1.domain, HOUDINI_TAB, None)
        if tab is not None:
            self.operate_click(*tab)
            self.sleep(0.5)

        btns = self.wait_until(
            lambda: self.find_confirms(self.box_of_screen(*HOUDINI_BTN_BOX)),
            time_out=10,
            raise_if_not_found=False,
        )
        if not btns:
            self.screenshot("nav_houdini_button_missing")
            raise WaitFailedException("界域页未找到前往按钮")
        target = min(btns, key=lambda b: b.y) if isinstance(btns, list) else btns
        self.log_info(f"点击前往按钮: {target}")
        self.operate_click(target, action_name="nav_click_houdini_go")
        self._check_nav_timer(timer)
        if not self.click_traval_button(raise_if_not_found=False):
            self.screenshot("nav_houdini_travel_missing")
            raise WaitFailedException("界域传送按钮未出现")
        self.wait_in_team(time_out=300, settle_time=0.25)
        self.sleep(1)

    # ---- 传送段: 开大地图 → 找锚点 → 偏左下点击 → 传送 → 等加载 ----

    def _teleport_to_anchor(self, timer):
        self._set_stage("传送至钓鱼点")
        self.ensure_main(time_out=30)
        self._check_nav_timer(timer)

        if not self._open_map_until_ready():
            self.screenshot("nav_map_not_open")
            raise WaitFailedException("大地图未能打开(已按3次M)")
        self._check_nav_timer(timer)

        anchor = self.wait_until(
            self._find_anchor_on_map,
            time_out=ANCHOR_TIMEOUT,
            raise_if_not_found=False,
        )
        if not anchor:
            self.screenshot("nav_anchor_not_found")
            raise WaitFailedException(f"大地图上未找到传送锚点: {ANCHOR_FEATURE}")
        self.log_info(f"找到传送锚点: {anchor}")

        # 点击锚点偏左下位置: 实测可直接打开传送确认面板(不触发多选项对话框)
        offset_x = int(anchor.x + anchor.width * ANCHOR_CLICK_OFFSET[0])
        offset_y = int(anchor.y + anchor.height * ANCHOR_CLICK_OFFSET[1])
        self.log_info(f"点击锚点偏左下位置: ({offset_x}, {offset_y})")
        self.operate_click(offset_x, offset_y, action_name="nav_click_anchor_bl")
        self._check_nav_timer(timer)

        if not self.click_traval_button(raise_if_not_found=False):
            self.screenshot("nav_travel_button_missing")
            raise WaitFailedException("点击锚点后未出现传送按钮")
        self.wait_in_team(time_out=300, settle_time=0.25)
        self.sleep(1)

    def _open_map_until_ready(self):
        """按M打开大地图并确认已打开(首次按键可能被游戏丢弃, 最多按3次)。"""
        if self._map_is_open():
            self.sleep(0.8)
            return True
        for _ in range(3):
            self.send_key("m")
            if self.wait_until(self._map_is_open, time_out=2.5, raise_if_not_found=False):
                self.sleep(0.8)  # 等开图动画稳定后再搜索/点击
                return True
        return False

    def _map_is_open(self):
        """地图打开判据: 右上角关闭按钮(内嵌模板)。"""
        box = self.box_of_screen(*MAP_OPEN_BOX, name="nav_map_close_btn")
        hit = self.find_one(
            MAP_OPEN_FEATURE,
            template=self._map_open_template(),
            box=box,
            threshold=0.7,
        )
        return bool(hit)

    def _find_anchor_on_map(self):
        """三级搜索: 右上定向 → 全屏 → 全屏放宽阈值。"""
        template = self._anchor_template()
        region = self.box_of_screen(*ANCHOR_REGION, name="nav_anchor_region")
        for box, threshold in ((region, 0.7), (None, 0.7), (None, 0.65)):
            hit = self.find_one(ANCHOR_FEATURE, template=template, box=box, threshold=threshold)
            if hit:
                return hit
        return None

    # ---- 模板工具 ----

    def _scaled_template(self, template, cache):
        """按当前捕获分辨率缩放内嵌模板(模板基于 TEMPLATE_BASE_WIDTH 宽截图)。"""
        scale = self.width / float(TEMPLATE_BASE_WIDTH)
        if abs(scale - 1.0) < 0.02:
            return template
        key = round(scale, 3)
        cached = cache.get(key)
        if cached is None:
            cached = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            cache[key] = cached
        return cached

    def _anchor_template(self):
        return self._scaled_template(ANCHOR_TEMPLATE, self._anchor_tpl_cache)

    def _map_open_template(self):
        return self._scaled_template(MAP_OPEN_TEMPLATE, self._map_open_tpl_cache)

    # ---- 行走段: 回正镜头 + 按录制路径盲走 ----

    def _walk_recorded_path(self, timer):
        self._set_stage("前往钓鱼交互点")
        self.log_info("播放传送后行走路径")
        self.middle_click(after_sleep=1.9)
        total = len(WAYPOINTS)
        for i, wp in enumerate(WAYPOINTS, 1):
            self._check_nav_timer(timer)
            key = wp.get("key", "w")
            hold = float(wp.get("hold", 0.5))
            after = float(wp.get("after", 0.3))
            self.log_info(f"路径段 {i}/{total}: {key} 按住{hold:.2f}s")
            self.send_key(key, down_time=hold, after_sleep=after)

    # ---- 落点微调与交互核对 ----

    def _at_fishing_interac(self):
        """F提示可见且其旁文字含 INTERAC_TEXT("钓鱼")才视为到达, 排除其他可互动物品。"""
        prompts = self.find_interac()
        if not prompts:
            return False
        prompt = prompts[0]
        left = max(0.02, (prompt.x - self.width_of_screen(0.10)) / self.width)
        top = max(0.02, (prompt.y - self.height_of_screen(0.08)) / self.height)
        right = min(0.98, (prompt.x + prompt.width + self.width_of_screen(0.10)) / self.width)
        bottom = min(0.98, (prompt.y + prompt.height + self.height_of_screen(0.08)) / self.height)
        pattern = re.compile(re.escape(INTERAC_TEXT))
        hit = self.wait_ocr(
            left, top, right, bottom,
            match=[pattern],
            time_out=1.5,
            raise_if_not_found=False,
        )
        return bool(hit)

    def _approach_until_interac(self):
        if self._at_fishing_interac():
            return
        for attempt in range(1, APPROACH_TAP_MAX + 1):
            self.log_info(
                f"未见钓鱼F提示, 第 {attempt}/{APPROACH_TAP_MAX} 次点按{APPROACH_TAP_KEY}键微调"
            )
            self.send_key(APPROACH_TAP_KEY, down_time=0.2)
            if self.wait_until(
                self._at_fishing_interac, time_out=APPROACH_TAP_SETTLE, raise_if_not_found=False
            ):
                self.log_round_info("点按微调后检测到钓鱼F提示, 已到达交互点")
                return
        self.screenshot("nav_approach_taps_exhausted")
        self.stop_rounds(f"点按{APPROACH_TAP_KEY} {APPROACH_TAP_MAX} 次仍未出现钓鱼F提示")
        raise WaitFailedException("微调超限: 未找到钓鱼交互点, 任务停止")

    def _check_nav_timer(self, timer):
        if timer.expired():
            raise WaitFailedException("寻路总超时")
