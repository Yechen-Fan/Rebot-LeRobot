//! RobotArm —— 机械臂控制句柄，直接持有 motorbridge vendor 控制器（Rust 原生）。
//!
//! Python 接口与 `reBotArm_control_py/actuator/arm.py` 的 `RobotArm` 对齐。
//! 运行时状态置于 `Arc<Inner>` 的内部可变容器中，使所有 pymethod 取 `&self`，
//! 从而允许后台控制循环线程回调本对象时多重不可变借用并存（避免 PyO3 重入借用冲突）。

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::config::{parse_arm_config, JointCfg};
use crate::vendor::{UniController, UniMotor};

fn sleep_s(s: f64) {
    if s > 0.0 {
        thread::sleep(Duration::from_secs_f64(s));
    }
}

/// 控制频率上界（Hz）：防止 NaN/inf/超大 rate 导致 0 周期 tight loop 或总线打满。
const MAX_CONTROL_HZ: f64 = 10_000.0;

/// 校验可选输入向量长度必须等于关节数，避免 RT 线程中按 i 索引 panic。
fn check_len(name: &str, v: &Option<Vec<f64>>, n: usize) -> PyResult<()> {
    if let Some(v) = v {
        if v.len() != n {
            return Err(PyValueError::new_err(format!(
                "{name} 长度 {} 必须等于关节数 {}",
                v.len(),
                n
            )));
        }
    }
    Ok(())
}

/// 校验控制频率：必须有限、>0、且 <= MAX_CONTROL_HZ。
fn validate_rate(rate: f64) -> PyResult<()> {
    if !rate.is_finite() || rate <= 0.0 || rate > MAX_CONTROL_HZ {
        return Err(PyValueError::new_err(format!(
            "rate={rate} 非法，必须为有限值且 0 < rate <= {MAX_CONTROL_HZ}"
        )));
    }
    Ok(())
}

/// 尽力为当前线程申请 SCHED_FIFO 实时调度与 CPU 亲和性。
#[cfg(target_os = "linux")]
fn try_set_realtime(priority: i32, cpu: Option<i32>) {
    unsafe {
        if priority > 0 {
            let max = libc::sched_get_priority_max(libc::SCHED_FIFO);
            let prio = priority.min(max);
            let param = libc::sched_param { sched_priority: prio };
            let _ = libc::sched_setscheduler(0, libc::SCHED_FIFO, &param);
        }
        if let Some(c) = cpu {
            // 边界校验：CPU_SET 在越界索引上是未定义行为，必须先检查。
            if c >= 0 && (c as usize) < libc::CPU_SETSIZE as usize {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut set);
                libc::CPU_SET(c as usize, &mut set);
                let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn try_set_realtime(_priority: i32, _cpu: Option<i32>) {}

/// RT-native 控制循环的目标快照（无锁交换）。
struct Targets {
    mode: u8, // 1=MIT, 2=POS_VEL, 3=VEL
    pos: Vec<f32>,
    vel: Vec<f32>,
    kp: Vec<f32>,
    kd: Vec<f32>,
    tau: Vec<f32>,
    vlim: Vec<f32>,
    force_pos: Vec<bool>,
    force_pos_torque_ratio: Vec<f32>,
}

impl Targets {
    fn zeros(n: usize) -> Self {
        Targets {
            mode: 1,
            pos: vec![0.0; n],
            vel: vec![0.0; n],
            kp: vec![0.0; n],
            kd: vec![0.0; n],
            tau: vec![0.0; n],
            vlim: vec![0.0; n],
            force_pos: vec![false; n],
            force_pos_torque_ratio: vec![0.0; n],
        }
    }
}

/// 运行时内部状态（内部可变）。
struct Inner {
    channel: String,
    rate: f64,
    order: Vec<String>,
    joints: Vec<JointCfg>,

    ctrls: Mutex<HashMap<String, Arc<UniController>>>, // vendor -> controller
    motors: Mutex<HashMap<String, Arc<UniMotor>>>,     // name   -> motor

    mode: Mutex<String>,
    mit_kp: Mutex<Vec<f64>>,
    mit_kd: Mutex<Vec<f64>>,
    pv_vlim: Mutex<Vec<f64>>,

    running: AtomicBool,
    targets: ArcSwap<Targets>,
    targets_set: AtomicBool, // 用户是否显式调用过 set_targets
    thread: Mutex<Option<JoinHandle<()>>>,
    feedback_thread: Mutex<Option<JoinHandle<()>>>,
    ctrl_rate: Mutex<f64>,
    send_overruns: AtomicU64, // 控制发送线程错过截止的次数
    read_overruns: AtomicU64, // 反馈读取线程错过截止的次数
}

impl Inner {
    fn num_joints(&self) -> usize {
        self.order.len()
    }

    fn motor(&self, name: &str) -> Option<Arc<UniMotor>> {
        self.motors.lock().unwrap().get(name).cloned()
    }

    fn loop_active(&self) -> bool {
        self.running.load(Ordering::Acquire)
            && (self.thread.lock().unwrap().is_some()
                || self.feedback_thread.lock().unwrap().is_some())
    }

    /// 构建/重建控制器与电机（按 vendor 去重控制器）。
    fn build(&self) -> PyResult<()> {
        let mut ctrls = self.ctrls.lock().unwrap();
        let mut motors = self.motors.lock().unwrap();
        ctrls.clear();
        motors.clear();
        for jc in &self.joints {
            if !ctrls.contains_key(&jc.vendor) {
                let c = UniController::new(&self.channel, &jc.vendor)
                    .map_err(PyValueError::new_err)?;
                ctrls.insert(jc.vendor.clone(), Arc::new(c));
            }
            let ctrl = ctrls.get(&jc.vendor).unwrap();
            let m = ctrl
                .add_motor(jc.motor_id, jc.feedback_id, &jc.model)
                .map_err(PyValueError::new_err)?;
            motors.insert(jc.name.clone(), Arc::new(m));
        }
        Ok(())
    }

    fn poll_all(&self) {
        for c in self.ctrls.lock().unwrap().values() {
            let _ = c.poll_feedback_once();
        }
    }

    fn request_and_poll(&self) {
        let motors = self.motors.lock().unwrap();
        for name in &self.order {
            if let Some(m) = motors.get(name) {
                let _ = m.request_feedback();
            }
        }
        drop(motors);
        self.poll_all();
    }

    /// 读取所有关节状态（pos, vel, torq）。缺反馈时返回 NaN；状态码单独读取。
    fn snapshot(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let motors = self.motors.lock().unwrap();
        let n = self.order.len();
        let (mut pos, mut vel, mut torq) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for name in &self.order {
            match motors.get(name).and_then(|m| m.get_state()) {
                Some(s) => {
                    pos.push(s.pos);
                    vel.push(s.vel);
                    torq.push(s.torq);
                }
                _ => {
                    pos.push(f64::NAN);
                    vel.push(f64::NAN);
                    torq.push(f64::NAN);
                }
            }
        }
        (pos, vel, torq)
    }

    /// 读取当前状态码；255 表示尚无反馈。
    fn status_codes(&self) -> Vec<u8> {
        let motors = self.motors.lock().unwrap();
        self.order
            .iter()
            .map(|name| {
                motors
                    .get(name)
                    .and_then(|motor| motor.get_state())
                    .map_or(u8::MAX, |state| state.status_code)
            })
            .collect()
    }

    /// 阻塞读取当前关节位置（多次请求+轮询等反馈到达）。全部电机有反馈才返回 Some。
    fn read_positions_blocking(&self, attempts: usize) -> Option<Vec<f64>> {
        for _ in 0..attempts {
            self.request_and_poll();
            let motors = self.motors.lock().unwrap();
            let mut pos = Vec::with_capacity(self.order.len());
            let mut all = true;
            for name in &self.order {
                match motors.get(name).and_then(|m| m.get_state()) {
                    Some(s) => pos.push(s.pos),
                    None => {
                        all = false;
                        break;
                    }
                }
            }
            drop(motors);
            if all {
                return Some(pos);
            }
            sleep_s(0.005);
        }
        None
    }

    fn ensure_mode_one(&self, name: &str, mode: u32) -> bool {
        match self.motor(name) {
            Some(m) => match m.ensure_mode(mode, 1000) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[ensure_mode/{name}] 跳过: {e}");
                    false
                }
            },
            None => false,
        }
    }

    fn ensure_mode_one_timeout(&self, name: &str, mode: u32, timeout_ms: u32) -> Result<(), String> {
        match self.motor(name) {
            Some(m) => m.ensure_mode(mode, timeout_ms),
            None => Err(format!("Unknown joint: {name}")),
        }
    }

    fn stop_loop(&self, py: Python<'_>) {
        self.running.store(false, Ordering::Release);
        let handle = self.thread.lock().unwrap().take();
        let feedback_handle = self.feedback_thread.lock().unwrap().take();
        if let Some(h) = handle {
            // 必须释放 GIL 再 join：兼容模式下循环线程可能正卡在 with_gil。
            py.allow_threads(move || {
                let _ = h.join();
            });
        }
        if let Some(h) = feedback_handle {
            py.allow_threads(move || {
                let _ = h.join();
            });
        }
    }

    fn enable(&self, vendor: Option<String>, delay_per_motor: f64, retries: i64, poll_interval: f64) {
        let vendors: Vec<String> = match vendor {
            Some(v) => vec![v],
            None => self.ctrls.lock().unwrap().keys().cloned().collect(),
        };
        for v in &vendors {
            if let Some(c) = self.ctrls.lock().unwrap().get(v) {
                if let Err(e) = c.enable_all() {
                    eprintln!("[enable/{v}] 调用 enable_all() 失败: {e}");
                }
            }
            sleep_s(delay_per_motor);
        }
        if retries <= 0 {
            return;
        }
        let mut enabled: HashMap<String, bool> =
            self.order.iter().map(|n| (n.clone(), false)).collect();
        let mut last_status: HashMap<String, Option<u8>> =
            self.order.iter().map(|n| (n.clone(), None)).collect();
        for _ in 0..retries {
            self.request_and_poll();
            let mut all_done = true;
            for name in &self.order {
                if enabled[name] {
                    continue;
                }
                match self.motor(name).and_then(|m| m.get_state()) {
                    Some(s) if s.status_code == 1 => {
                        enabled.insert(name.clone(), true);
                    }
                    Some(s) => {
                        last_status.insert(name.clone(), Some(s.status_code));
                        all_done = false;
                    }
                    None => all_done = false,
                }
            }
            if all_done {
                break;
            }
            sleep_s(poll_interval);
        }
        let failed: Vec<&String> = self.order.iter().filter(|n| !enabled[*n]).collect();
        for name in &failed {
            eprintln!(
                "[enable] {name} 使能确认失败（已重试 {retries} 次，last_status={:?}）",
                last_status.get(*name).copied().flatten()
            );
        }
        if !failed.is_empty() {
            eprintln!("[enable] 未就绪电机: {failed:?}");
        }
    }

    fn disable(
        &self,
        py: Python<'_>,
        vendor: Option<String>,
        delay_per_motor: f64,
        retries: i64,
        poll_interval: f64,
    ) {
        if self.loop_active() {
            self.stop_loop(py);
        }
        let vendors: Vec<String> = match vendor {
            Some(v) => vec![v],
            None => self.ctrls.lock().unwrap().keys().cloned().collect(),
        };
        for v in &vendors {
            if let Some(c) = self.ctrls.lock().unwrap().get(v) {
                if let Err(e) = c.disable_all() {
                    eprintln!("[disable/{v}] 调用 disable_all() 失败: {e}");
                }
            }
            sleep_s(delay_per_motor);
        }
        if retries <= 0 {
            return;
        }
        let mut disabled: HashMap<String, bool> =
            self.order.iter().map(|n| (n.clone(), false)).collect();
        let mut last_status: HashMap<String, Option<u8>> =
            self.order.iter().map(|n| (n.clone(), None)).collect();
        for _ in 0..retries {
            self.request_and_poll();
            let mut all = true;
            for name in &self.order {
                if disabled[name] {
                    continue;
                }
                match self.motor(name).and_then(|m| m.get_state()) {
                    Some(s) if s.status_code == 0 => {
                        disabled.insert(name.clone(), true);
                    }
                    Some(s) => {
                        last_status.insert(name.clone(), Some(s.status_code));
                        all = false;
                    }
                    None => all = false,
                }
            }
            if all {
                break;
            }
            sleep_s(poll_interval);
        }
        let failed: Vec<&String> = self.order.iter().filter(|n| !disabled[*n]).collect();
        if !failed.is_empty() {
            for name in &failed {
                eprintln!(
                    "[disable] {name} 失能确认失败（已重试 {retries} 次，last_status={:?}）",
                    last_status.get(*name).copied().flatten()
                );
            }
            eprintln!("[disable] 未确认失能的电机: {failed:?}");
        }
    }
}

/// 机械臂控制句柄。
#[pyclass(subclass)]
pub struct RobotArm {
    #[pyo3(get)]
    name: String,
    inner: Arc<Inner>,
}

#[pymethods]
impl RobotArm {
    #[new]
    #[pyo3(signature = (cfg_path=None))]
    fn new(cfg_path: Option<String>) -> PyResult<Self> {
        let path = cfg_path.unwrap_or_else(|| "config/arm.yaml".to_string());
        let cfg = parse_arm_config(&path)?;
        let order: Vec<String> = cfg.joints.iter().map(|j| j.name.clone()).collect();
        let mit_kp: Vec<f64> = cfg.joints.iter().map(|j| j.kp).collect();
        let mit_kd: Vec<f64> = cfg.joints.iter().map(|j| j.kd).collect();
        let pv_vlim: Vec<f64> = cfg.joints.iter().map(|j| j.vlim).collect();
        let n = cfg.joints.len();

        let inner = Arc::new(Inner {
            channel: cfg.channel,
            rate: cfg.rate,
            order,
            joints: cfg.joints,
            ctrls: Mutex::new(HashMap::new()),
            motors: Mutex::new(HashMap::new()),
            mode: Mutex::new("mit".to_string()),
            mit_kp: Mutex::new(mit_kp),
            mit_kd: Mutex::new(mit_kd),
            pv_vlim: Mutex::new(pv_vlim),
            running: AtomicBool::new(false),
            targets: ArcSwap::from_pointee(Targets::zeros(n.max(1))),
            targets_set: AtomicBool::new(false),
            thread: Mutex::new(None),
            feedback_thread: Mutex::new(None),
            ctrl_rate: Mutex::new(cfg.rate),
            send_overruns: AtomicU64::new(0),
            read_overruns: AtomicU64::new(0),
        });
        inner.build()?;
        Ok(RobotArm {
            name: cfg.name,
            inner,
        })
    }

    // ---------------- 属性 ----------------

    #[getter]
    fn num_joints(&self) -> usize {
        self.inner.num_joints()
    }

    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.inner.order.clone()
    }

    #[getter]
    fn mode(&self) -> String {
        self.inner.mode.lock().unwrap().clone()
    }

    #[getter]
    fn control_loop_active(&self) -> bool {
        self.inner.loop_active()
    }

    /// 与示例兼容：`arm._joints[i].kp`
    #[getter]
    fn _joints(&self) -> Vec<JointCfg> {
        self.inner.joints.clone()
    }

    /// 与示例兼容：`arm._rate`
    #[getter]
    fn _rate(&self) -> f64 {
        self.inner.rate
    }

    // ---------------- 连接 / 断开 ----------------

    fn connect(&self) {
        // Controller 已在 __init__ 建立；保留扩展口。
    }

    #[pyo3(signature = (disable=true))]
    fn disconnect(&self, py: Python<'_>, disable: bool) {
        self.inner.stop_loop(py);
        if disable {
            self.inner.disable(py, None, 0.05, 0, 0.1);
            sleep_s(0.5);
        }
        let ctrls: Vec<Arc<UniController>> =
            self.inner.ctrls.lock().unwrap().values().cloned().collect();
        for c in &ctrls {
            let _ = c.shutdown();
            sleep_s(0.1);
            let _ = c.close_bus();
        }
        self.inner.ctrls.lock().unwrap().clear();
        self.inner.motors.lock().unwrap().clear();
        // 断开后清除 hold 目标标志：下次 start_rt_loop 将重新读取当前位置。
        self.inner.targets_set.store(false, Ordering::Release);
    }

    #[pyo3(signature = (init_delay=1.0, post_setup_delay=0.5))]
    fn reconnect(&self, py: Python<'_>, init_delay: f64, post_setup_delay: f64) -> PyResult<()> {
        self.disconnect(py, true);
        sleep_s(init_delay);
        self.inner.build()?;
        sleep_s(post_setup_delay);
        eprintln!("[reconnect] 控制器和电机已重新初始化");
        Ok(())
    }

    // ---------------- 使能 / 失能 ----------------

    #[pyo3(signature = (vendor=None, delay_per_motor=0.05, retries=0, poll_interval=0.1))]
    fn enable(&self, vendor: Option<String>, delay_per_motor: f64, retries: i64, poll_interval: f64) {
        self.inner.enable(vendor, delay_per_motor, retries, poll_interval);
    }

    #[pyo3(signature = (vendor=None, delay_per_motor=0.05, retries=0, poll_interval=0.1))]
    fn disable(
        &self,
        py: Python<'_>,
        vendor: Option<String>,
        delay_per_motor: f64,
        retries: i64,
        poll_interval: f64,
    ) {
        self.inner.disable(py, vendor, delay_per_motor, retries, poll_interval);
    }

    // ---------------- 零点 ----------------

    #[pyo3(signature = (poll_max=200, poll_interval=0.05, set_zero_delay=0.1))]
    fn set_zero(&self, py: Python<'_>, poll_max: i64, poll_interval: f64, set_zero_delay: f64) {
        self.inner.disable(py, None, 0.05, 10, 0.1);
        sleep_s(0.3);
        for name in &self.inner.order {
            let mut ready = false;
            for _ in 0..poll_max {
                self.inner.request_and_poll();
                if let Some(s) = self.inner.motor(name).and_then(|m| m.get_state()) {
                    if s.status_code == 0 {
                        ready = true;
                    }
                }
                if ready {
                    break;
                }
                sleep_s(poll_interval);
            }
            if !ready {
                eprintln!("[set_zero] {name}: 等待状态就绪超时，跳过");
                sleep_s(set_zero_delay);
                continue;
            }
            match self.inner.motor(name).map(|m| m.set_zero()) {
                Some(Ok(())) => eprintln!("[set_zero] {name}: OK"),
                Some(Err(e)) => eprintln!("[set_zero] {name}: {e}"),
                None => {}
            }
            sleep_s(set_zero_delay);
        }
    }

    #[pyo3(signature = (name, poll_max=200, poll_interval=0.05))]
    fn set_zero_single(
        &self,
        py: Python<'_>,
        name: String,
        poll_max: i64,
        poll_interval: f64,
    ) -> PyResult<bool> {
        if self.inner.motor(&name).is_none() {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "Unknown joint: {name}"
            )));
        }
        self.inner.disable(py, None, 0.05, 10, 0.1);
        sleep_s(0.3);
        for _ in 0..poll_max {
            self.inner.request_and_poll();
            if let Some(s) = self.inner.motor(&name).and_then(|m| m.get_state()) {
                if s.status_code == 0 {
                    break;
                }
            }
            sleep_s(poll_interval);
        }
        match self.inner.motor(&name).map(|m| m.set_zero()) {
            Some(Ok(())) => Ok(true),
            Some(Err(e)) => {
                eprintln!("[set_zero] {name}: {e}");
                Ok(false)
            }
            None => Ok(false),
        }
    }

    // ---------------- 状态读取 ----------------

    #[pyo3(signature = (request=false))]
    fn get_state<'py>(
        &self,
        py: Python<'py>,
        request: bool,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        if request {
            self.inner.request_and_poll();
        } else {
            self.inner.poll_all();
        }
        let (pos, vel, torq) = self.inner.snapshot();
        (
            pos.into_pyarray(py),
            vel.into_pyarray(py),
            torq.into_pyarray(py),
        )
    }

    #[pyo3(signature = (request=false))]
    fn get_status_codes<'py>(&self, py: Python<'py>, request: bool) -> Bound<'py, PyArray1<u8>> {
        if request {
            self.inner.request_and_poll();
        } else {
            self.inner.poll_all();
        }
        self.inner.status_codes().into_pyarray(py)
    }

    #[pyo3(signature = (request=false))]
    fn get_positions<'py>(&self, py: Python<'py>, request: bool) -> Bound<'py, PyArray1<f64>> {
        if request {
            self.inner.request_and_poll();
        } else {
            self.inner.poll_all();
        }
        self.inner.snapshot().0.into_pyarray(py)
    }

    #[pyo3(signature = (attempts=20))]
    fn get_positions_blocking<'py>(&self, py: Python<'py>, attempts: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self.inner.read_positions_blocking(attempts) {
            Some(pos) => Ok(pos.into_pyarray(py)),
            None => Err(PyRuntimeError::new_err(
                "无法读取全部关节位置；未建立安全初始目标，请检查电机反馈和串口/CAN 通信",
            )),
        }
    }

    #[pyo3(signature = (request=false))]
    fn get_velocities<'py>(&self, py: Python<'py>, request: bool) -> Bound<'py, PyArray1<f64>> {
        if request {
            self.inner.request_and_poll();
        } else {
            self.inner.poll_all();
        }
        self.inner.snapshot().1.into_pyarray(py)
    }

    #[pyo3(signature = (request=false))]
    fn get_torques<'py>(&self, py: Python<'py>, request: bool) -> Bound<'py, PyArray1<f64>> {
        if request {
            self.inner.request_and_poll();
        } else {
            self.inner.poll_all();
        }
        self.inner.snapshot().2.into_pyarray(py)
    }

    // ---------------- 模式切换 ----------------

    #[pyo3(signature = (kp=None, kd=None, stabilize_delay=0.2))]
    fn mode_mit(&self, kp: Option<Vec<f64>>, kd: Option<Vec<f64>>, stabilize_delay: f64) -> PyResult<bool> {
        if self.inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环运行中不可切换模式，请先 stop_control_loop()",
            ));
        }
        let n = self.inner.num_joints();
        check_len("kp", &kp, n)?;
        check_len("kd", &kd, n)?;
        *self.inner.mode.lock().unwrap() = "mit".to_string();
        let kp = kp.unwrap_or_else(|| self.inner.joints.iter().map(|j| j.kp).collect());
        let kd = kd.unwrap_or_else(|| self.inner.joints.iter().map(|j| j.kd).collect());
        *self.inner.mit_kp.lock().unwrap() = kp;
        *self.inner.mit_kd.lock().unwrap() = kd;
        let mut ok = true;
        for name in &self.inner.order {
            if !self.inner.ensure_mode_one(name, 1) {
                ok = false;
            }
            sleep_s(0.05);
        }
        sleep_s(stabilize_delay);
        Ok(ok)
    }

    #[pyo3(signature = (vlim=None, stabilize_delay=0.2))]
    fn mode_pos_vel(&self, vlim: Option<Vec<f64>>, stabilize_delay: f64) -> PyResult<bool> {
        if self.inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环运行中不可切换模式，请先 stop_control_loop()",
            ));
        }
        check_len("vlim", &vlim, self.inner.num_joints())?;
        *self.inner.mode.lock().unwrap() = "pos_vel".to_string();
        let vlim = vlim.unwrap_or_else(|| self.inner.joints.iter().map(|j| j.vlim).collect());
        *self.inner.pv_vlim.lock().unwrap() = vlim;
        let mut ok = true;
        for jc in &self.inner.joints {
            if let Some(m) = self.inner.motor(&jc.name) {
                if jc.vel_kp > 0.0 {
                    let _ = m.write_register_f32(25, jc.vel_kp as f32); // KP_ASR
                }
                if jc.vel_ki > 0.0 {
                    let _ = m.write_register_f32(26, jc.vel_ki as f32); // KI_ASR
                }
                if jc.pos_kp > 0.0 {
                    let _ = m.write_register_f32(27, jc.pos_kp as f32); // KP_APR
                }
                if jc.pos_ki > 0.0 {
                    let _ = m.write_register_f32(28, jc.pos_ki as f32); // KI_APR
                }
                if jc.vel_kp > 0.0 || jc.vel_ki > 0.0 || jc.pos_kp > 0.0 || jc.pos_ki > 0.0 {
                    sleep_s(0.02);
                }
            }
            if !self.inner.ensure_mode_one(&jc.name, 2) {
                ok = false;
            }
            sleep_s(0.05);
        }
        sleep_s(stabilize_delay);
        Ok(ok)
    }

    #[pyo3(signature = (stabilize_delay=0.2))]
    fn mode_vel(&self, stabilize_delay: f64) -> PyResult<bool> {
        if self.inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环运行中不可切换模式，请先 stop_control_loop()",
            ));
        }
        *self.inner.mode.lock().unwrap() = "vel".to_string();
        let mut ok = true;
        for name in &self.inner.order {
            if !self.inner.ensure_mode_one(name, 3) {
                ok = false;
            }
            sleep_s(0.05);
        }
        sleep_s(stabilize_delay);
        Ok(ok)
    }

    #[pyo3(signature = (name, mode, timeout_ms=1000))]
    fn ensure_mode(&self, name: String, mode: u32, timeout_ms: u32) -> PyResult<bool> {
        if self.inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环运行中不可切换模式，请先 stop_control_loop()",
            ));
        }

        self.inner
            .ensure_mode_one_timeout(&name, mode, timeout_ms)
            .map(|()| true)
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (name, rid, timeout_ms=200))]
    fn get_damiao_register_f32(&self, name: String, rid: u8, timeout_ms: u64) -> PyResult<f32> {
        self.inner
            .motor(&name)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown joint: {name}")))?
            .get_register_f32(rid, timeout_ms)
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (timeout_ms=200))]
    fn get_pos_vel_gains(&self, timeout_ms: u64) -> PyResult<HashMap<String, (f32, f32, f32, f32)>> {
        let mut gains = HashMap::new();
        for name in &self.inner.order {
            let motor = self
                .inner
                .motor(name)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown joint: {name}")))?;
            let vel_kp = motor.get_register_f32(25, timeout_ms).map_err(|e| {
                PyRuntimeError::new_err(format!("read {name}.vel_kp register 25 failed: {e}"))
            })?;
            let vel_ki = motor.get_register_f32(26, timeout_ms).map_err(|e| {
                PyRuntimeError::new_err(format!("read {name}.vel_ki register 26 failed: {e}"))
            })?;
            let pos_kp = motor.get_register_f32(27, timeout_ms).map_err(|e| {
                PyRuntimeError::new_err(format!("read {name}.pos_kp register 27 failed: {e}"))
            })?;
            let pos_ki = motor.get_register_f32(28, timeout_ms).map_err(|e| {
                PyRuntimeError::new_err(format!("read {name}.pos_ki register 28 failed: {e}"))
            })?;
            gains.insert(name.clone(), (vel_kp, vel_ki, pos_kp, pos_ki));
        }
        Ok(gains)
    }

    // ---------------- 控制命令 ----------------

    #[pyo3(signature = (pos, vel=None, kp=None, kd=None, tau=None, request_feedback=true))]
    fn mit(
        &self,
        pos: Vec<f64>,
        vel: Option<Vec<f64>>,
        kp: Option<Vec<f64>>,
        kd: Option<Vec<f64>>,
        tau: Option<Vec<f64>>,
        request_feedback: bool,
    ) -> PyResult<()> {
        let n = self.inner.num_joints();
        if pos.len() != n {
            return Err(PyValueError::new_err(format!(
                "pos 长度 {} 必须等于关节数 {}",
                pos.len(),
                n
            )));
        }
        check_len("vel", &vel, n)?;
        check_len("kp", &kp, n)?;
        check_len("kd", &kd, n)?;
        check_len("tau", &tau, n)?;
        let vel = vel.unwrap_or_else(|| vec![0.0; n]);
        let tau = tau.unwrap_or_else(|| vec![0.0; n]);
        let kp = kp.unwrap_or_else(|| self.inner.mit_kp.lock().unwrap().clone());
        let kd = kd.unwrap_or_else(|| self.inner.mit_kd.lock().unwrap().clone());

        let motors = self.inner.motors.lock().unwrap();
        for (i, name) in self.inner.order.iter().enumerate() {
            if let Some(m) = motors.get(name) {
                let _ = m.send_mit(
                    pos[i] as f32,
                    vel[i] as f32,
                    kp[i] as f32,
                    kd[i] as f32,
                    tau[i] as f32,
                );
            }
        }
        if request_feedback {
            for name in &self.inner.order {
                if let Some(m) = motors.get(name) {
                    let _ = m.request_feedback();
                }
            }
            drop(motors);
            self.inner.poll_all();
        }
        Ok(())
    }

    #[pyo3(signature = (pos, vlim=None))]
    fn pos_vel(&self, pos: Vec<f64>, vlim: Option<Vec<f64>>) -> PyResult<()> {
        let n = self.inner.num_joints();
        if pos.len() != n {
            return Err(PyValueError::new_err(format!(
                "pos 长度 {} 必须等于关节数 {}",
                pos.len(),
                n
            )));
        }
        check_len("vlim", &vlim, n)?;
        let vlim = vlim.unwrap_or_else(|| self.inner.pv_vlim.lock().unwrap().clone());
        let motors = self.inner.motors.lock().unwrap();
        for (i, name) in self.inner.order.iter().enumerate() {
            if let Some(m) = motors.get(name) {
                let _ = m.send_pos_vel(pos[i] as f32, vlim[i] as f32);
            }
        }
        Ok(())
    }

    fn set_vel(&self, vel: Vec<f64>) -> PyResult<()> {
        let n = self.inner.num_joints();
        if vel.len() != n {
            return Err(PyValueError::new_err(format!(
                "vel 长度 {} 必须等于关节数 {}",
                vel.len(),
                n
            )));
        }
        let motors = self.inner.motors.lock().unwrap();
        for (i, name) in self.inner.order.iter().enumerate() {
            if let Some(m) = motors.get(name) {
                let _ = m.send_vel(vel[i] as f32);
            }
        }
        Ok(())
    }

    fn estop(&self, py: Python<'_>) {
        self.inner.disable(py, None, 0.05, 0, 0.1);
    }

    /// 设置电机侧 CAN 超时看门狗（毫秒）：若超过该时间未收到控制帧，电机固件自动停机。
    /// 受 HighTorque SDK set_timeout 启发——RT 控制进程崩溃/卡死时的关键安全兜底。
    /// timeout_ms=0 禁用。返回成功设置的电机数（Damiao / RobStride 支持）。
    #[pyo3(signature = (timeout_ms))]
    fn set_can_timeout_ms(&self, timeout_ms: u32) -> usize {
        let motors = self.inner.motors.lock().unwrap();
        let mut ok = 0usize;
        for name in &self.inner.order {
            if let Some(m) = motors.get(name) {
                match m.set_can_timeout_ms(timeout_ms) {
                    Ok(()) => ok += 1,
                    Err(e) => eprintln!("[set_can_timeout_ms/{name}] {e}"),
                }
            }
        }
        ok
    }

    // ---------------- RT-native 目标设置 ----------------

    /// 设置 RT 控制循环的目标（按当前 mode 解释）。供 start_rt_loop 使用。
    #[pyo3(signature = (
        pos,
        kp=None,
        kd=None,
        vel=None,
        tau=None,
        vlim=None,
        force_pos=None,
        force_pos_torque_ratio=None
    ))]
    fn set_targets(
        &self,
        pos: Vec<f64>,
        kp: Option<Vec<f64>>,
        kd: Option<Vec<f64>>,
        vel: Option<Vec<f64>>,
        tau: Option<Vec<f64>>,
        vlim: Option<Vec<f64>>,
        force_pos: Option<Vec<bool>>,
        force_pos_torque_ratio: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let n = self.inner.num_joints();
        if pos.len() != n {
            return Err(PyValueError::new_err(format!(
                "pos 长度 {} 必须等于关节数 {}",
                pos.len(),
                n
            )));
        }
        check_len("kp", &kp, n)?;
        check_len("kd", &kd, n)?;
        check_len("vel", &vel, n)?;
        check_len("tau", &tau, n)?;
        check_len("vlim", &vlim, n)?;
        if let Some(v) = &force_pos {
            if v.len() != n {
                return Err(PyValueError::new_err(format!(
                    "force_pos 长度 {} 必须等于关节数 {}",
                    v.len(),
                    n
                )));
            }
        }
        check_len("force_pos_torque_ratio", &force_pos_torque_ratio, n)?;
        let f32v = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let mode = match self.inner.mode.lock().unwrap().as_str() {
            "pos_vel" => 2u8,
            "vel" => 3u8,
            _ => 1u8,
        };
        let t = Targets {
            mode,
            pos: f32v(pos),
            vel: f32v(vel.unwrap_or_else(|| vec![0.0; n])),
            kp: f32v(kp.unwrap_or_else(|| self.inner.mit_kp.lock().unwrap().clone())),
            kd: f32v(kd.unwrap_or_else(|| self.inner.mit_kd.lock().unwrap().clone())),
            tau: f32v(tau.unwrap_or_else(|| vec![0.0; n])),
            vlim: f32v(vlim.unwrap_or_else(|| self.inner.pv_vlim.lock().unwrap().clone())),
            force_pos: force_pos.unwrap_or_else(|| vec![false; n]),
            force_pos_torque_ratio: f32v(force_pos_torque_ratio.unwrap_or_else(|| vec![0.0; n])),
        };
        self.inner.targets.store(Arc::new(t));
        self.inner.targets_set.store(true, Ordering::Release);
        Ok(())
    }

    // ---------------- 控制循环 ----------------

    /// 兼容模式：后台线程每 tick 回调 Python `control_fn(arm, dt)`（持 GIL）。
    #[pyo3(signature = (control_fn, rate=None))]
    fn start_control_loop(slf: Bound<'_, Self>, control_fn: PyObject, rate: Option<f64>) -> PyResult<()> {
        let inner = slf.borrow().inner.clone();
        if inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环已在运行，请先调用 stop_control_loop()",
            ));
        }
        let rate_val = rate.unwrap_or(inner.rate);
        validate_rate(rate_val)?;
        *inner.ctrl_rate.lock().unwrap() = rate_val;
        inner.running.store(true, Ordering::Release);

        let arm_obj: Py<Self> = slf.clone().unbind();
        let run_inner = inner.clone();
        let handle = thread::spawn(move || {
            let dt = 1.0 / rate_val;
            while run_inner.running.load(Ordering::Acquire) {
                let t0 = Instant::now();
                Python::with_gil(|py| {
                    let arm_ref = arm_obj.clone_ref(py);
                    if let Err(e) = control_fn.call1(py, (arm_ref, dt)) {
                        e.print(py);
                    }
                });
                let s = dt - t0.elapsed().as_secs_f64();
                sleep_s(s);
            }
        });
        *inner.thread.lock().unwrap() = Some(handle);
        Ok(())
    }

    /// RT-native 模式：后台线程全程释放 GIL，直接按 set_targets 的目标驱动电机。
    ///
    /// 安全：若从未调用过 set_targets，则启动前读取当前关节位置作为 hold 目标（绝不下发全 0）。
    /// 反馈解耦：控制以 `rate` Hz 发送。
    /// motorbridge 默认有后台 polling 线程，会消费控制帧/请求帧返回的反馈并更新缓存。
    /// `request_feedback=true` 时额外启动反馈请求线程，按 `feedback_rate` Hz 主动请求每个电机状态。
    /// 调度：绝对时间节拍（无漂移）+ 超时计数；`rt_priority`>0 时尽力申请 SCHED_FIFO。
    /// 注意：这是 Rust 软实时（std thread + sleep），非 Flexiv 式硬实时 SDK。
    #[pyo3(signature = (rate=None, feedback_rate=100.0, rt_priority=0, cpu=None, command_gap_us=0, request_feedback=false))]
    fn start_rt_loop(
        &self,
        rate: Option<f64>,
        feedback_rate: f64,
        rt_priority: i32,
        cpu: Option<i32>,
        command_gap_us: i64,
        request_feedback: bool,
    ) -> PyResult<()> {
        let inner = self.inner.clone();
        if inner.loop_active() {
            return Err(PyRuntimeError::new_err(
                "控制循环已在运行，请先调用 stop_control_loop()",
            ));
        }
        let rate_val = rate.unwrap_or(inner.rate);
        validate_rate(rate_val)?;
        if feedback_rate.is_nan() {
            return Err(PyValueError::new_err("feedback_rate 不能为 NaN"));
        }
        if command_gap_us < 0 {
            return Err(PyValueError::new_err("command_gap_us 必须 >= 0"));
        }

        // 安全：未显式设过目标时，用当前关节位置建立 hold 目标，避免拉向 0 位姿。
        if !inner.targets_set.load(Ordering::Acquire) {
            match inner.read_positions_blocking(20) {
                Some(pos) => {
                    let n = inner.num_joints();
                    let f32v =
                        |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
                    let mode = match inner.mode.lock().unwrap().as_str() {
                        "pos_vel" => 2u8,
                        "vel" => 3u8,
                        _ => 1u8,
                    };
                    let hold = Targets {
                        mode,
                        pos: f32v(pos),
                        vel: vec![0.0; n],
                        kp: f32v(inner.mit_kp.lock().unwrap().clone()),
                        kd: f32v(inner.mit_kd.lock().unwrap().clone()),
                        tau: vec![0.0; n],
                        vlim: f32v(inner.pv_vlim.lock().unwrap().clone()),
                        force_pos: vec![false; n],
                        force_pos_torque_ratio: vec![0.0; n],
                    };
                    // VEL 模式下 hold=零速度，已是 vel 全 0，安全。
                    inner.targets.store(Arc::new(hold));
                    eprintln!("[rt] 未设置目标，已用当前关节位置建立 hold 目标");
                }
                None => {
                    return Err(PyRuntimeError::new_err(
                        "无法读取当前关节位置以建立 hold 目标；请先 enable() 并确认有反馈，\
                         或先调用 set_targets() 再 start_rt_loop()",
                    ));
                }
            }
        }

        *inner.ctrl_rate.lock().unwrap() = rate_val;
        inner.send_overruns.store(0, Ordering::Release);
        inner.read_overruns.store(0, Ordering::Release);
        inner.running.store(true, Ordering::Release);

        // 启动前捕获电机/控制器句柄（避免循环中加锁）。
        let motors: Vec<Arc<UniMotor>> = {
            let m = inner.motors.lock().unwrap();
            inner.order.iter().filter_map(|n| m.get(n).cloned()).collect()
        };
        let run_inner = inner.clone();
        let control_motors = motors.clone();
        let handle = thread::spawn(move || {
            try_set_realtime(rt_priority, cpu);
            let dt = Duration::from_secs_f64(1.0 / rate_val);
            let command_gap = Duration::from_micros(command_gap_us as u64);
            let mut next = Instant::now() + dt;
            let mut tick: u64 = 0;
            while run_inner.running.load(Ordering::Acquire) {
                let t = run_inner.targets.load();
                for (i, m) in control_motors.iter().enumerate() {
                    match t.mode {
                        2 => {
                            if t.force_pos.get(i).copied().unwrap_or(false) {
                                let ratio = t.force_pos_torque_ratio.get(i).copied().unwrap_or(0.0);
                                let _ = m.send_force_pos(t.pos[i], t.vlim[i], ratio);
                            } else {
                                let _ = m.send_pos_vel(t.pos[i], t.vlim[i]);
                            }
                        }
                        3 => {
                            let _ = m.send_vel(t.vel[i]);
                        }
                        _ => {
                            let _ = m.send_mit(t.pos[i], t.vel[i], t.kp[i], t.kd[i], t.tau[i]);
                        }
                    }
                    if !command_gap.is_zero() && i + 1 < control_motors.len() {
                        thread::sleep(command_gap);
                    }
                }
                tick = tick.wrapping_add(1);
                // 绝对时间节拍：无漂移；错过截止则计一次 overrun 并重新对齐。
                let now = Instant::now();
                if now < next {
                    thread::sleep(next - now);
                    next += dt;
                } else {
                    run_inner.send_overruns.fetch_add(1, Ordering::Relaxed);
                    next = now + dt;
                }
            }
        });
        *inner.thread.lock().unwrap() = Some(handle);

        if feedback_rate > 0.0 && request_feedback {
            let feedback_inner = inner.clone();
            let fb_motors = motors.clone();
            let feedback_handle = thread::spawn(move || {
                let dt = Duration::from_secs_f64(1.0 / feedback_rate);
                let request_gap = Duration::from_micros(command_gap_us as u64);
                let mut next = Instant::now() + dt;
                while feedback_inner.running.load(Ordering::Acquire) {
                    for (i, m) in fb_motors.iter().enumerate() {
                        let _ = m.request_feedback();
                        if i + 1 < fb_motors.len() && !request_gap.is_zero() {
                            thread::sleep(request_gap);
                        }
                    }
                    let now = Instant::now();
                    if now < next {
                        thread::sleep(next - now);
                        next += dt;
                    } else {
                        feedback_inner.read_overruns.fetch_add(1, Ordering::Relaxed);
                        next = now + dt;
                    }
                }
            });
            *inner.feedback_thread.lock().unwrap() = Some(feedback_handle);
        }
        Ok(())
    }

    /// 最近一次 RT 循环运行期间控制发送线程错过截止的次数。
    #[getter]
    fn rt_send_overruns(&self) -> u64 {
        self.inner.send_overruns.load(Ordering::Relaxed)
    }

    /// 最近一次 RT 循环运行期间反馈读取线程错过截止的次数。
    #[getter]
    fn rt_read_overruns(&self) -> u64 {
        self.inner.read_overruns.load(Ordering::Relaxed)
    }

    /// 兼容旧接口：等价于 rt_send_overruns。
    #[getter]
    fn rt_overruns(&self) -> u64 {
        self.inner.send_overruns.load(Ordering::Relaxed)
    }

    fn stop_control_loop(&self, py: Python<'_>) {
        self.inner.stop_loop(py);
        sleep_s(0.05);
    }

    // ---------------- 上下文管理器 ----------------

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.disconnect(py, true);
    }

    fn __repr__(&self) -> String {
        format!(
            "RobotArm({:?}, joints={}, mode={}, rate={}Hz)",
            self.name,
            self.inner.num_joints(),
            self.inner.mode.lock().unwrap(),
            *self.inner.ctrl_rate.lock().unwrap()
        )
    }
}
