//! Gripper —— 夹爪控制句柄（单电机），直接持有 motorbridge vendor 控制器。
//! Python 接口与 `reBotArm_control_py/actuator/gripper.py` 的 `Gripper` 对齐。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::config::{parse_gripper_config, GripperCfg};
use crate::vendor::{UniController, UniMotor};

fn sleep_s(s: f64) {
    if s > 0.0 {
        thread::sleep(Duration::from_secs_f64(s));
    }
}

struct Inner {
    cfg: GripperCfg,
    ctrl: Mutex<Option<Arc<UniController>>>,
    motor: Mutex<Option<Arc<UniMotor>>>,
    mode: Mutex<String>,
    mit_kp: Mutex<f64>,
    mit_kd: Mutex<f64>,
    running: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
    rate: Mutex<f64>,
    channel: String,
}

impl Inner {
    fn motor(&self) -> Option<Arc<UniMotor>> {
        self.motor.lock().unwrap().clone()
    }
    fn ctrl(&self) -> Option<Arc<UniController>> {
        self.ctrl.lock().unwrap().clone()
    }
    fn loop_active(&self) -> bool {
        self.running.load(Ordering::Acquire) && self.thread.lock().unwrap().is_some()
    }
    fn build(&self) -> PyResult<()> {
        let c = UniController::new(&self.channel, &self.cfg.vendor)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let c = Arc::new(c);
        let m = c
            .add_motor(self.cfg.motor_id, self.cfg.feedback_id, &self.cfg.model)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        *self.ctrl.lock().unwrap() = Some(c);
        *self.motor.lock().unwrap() = Some(Arc::new(m));
        Ok(())
    }
    fn request(&self) {
        if let Some(m) = self.motor() {
            let _ = m.request_feedback();
        }
    }
    fn poll(&self) {
        if let Some(c) = self.ctrl() {
            let _ = c.poll_feedback_once();
        }
    }
    fn stop_loop(&self, py: Python<'_>) {
        self.running.store(false, Ordering::Release);
        let h = self.thread.lock().unwrap().take();
        if let Some(h) = h {
            py.allow_threads(move || {
                let _ = h.join();
            });
        }
    }
}

#[pyclass(subclass)]
pub struct Gripper {
    inner: Arc<Inner>,
}

#[pymethods]
impl Gripper {
    #[new]
    #[pyo3(signature = (cfg_path=None))]
    fn new(cfg_path: Option<String>) -> PyResult<Self> {
        let path = cfg_path.unwrap_or_else(|| "config/gripper.yaml".to_string());
        let cfg = parse_gripper_config(&path)?;
        let inner = Arc::new(Inner {
            mit_kp: Mutex::new(cfg.gripper.kp),
            mit_kd: Mutex::new(cfg.gripper.kd),
            channel: cfg.channel,
            cfg: cfg.gripper,
            ctrl: Mutex::new(None),
            motor: Mutex::new(None),
            mode: Mutex::new("mit".to_string()),
            running: AtomicBool::new(false),
            thread: Mutex::new(None),
            rate: Mutex::new(100.0),
        });
        inner.build()?;
        Ok(Gripper { inner })
    }

    #[getter]
    fn mode(&self) -> String {
        self.inner.mode.lock().unwrap().clone()
    }

    fn connect(&self) {}

    // ---------------- 使能 / 失能 ----------------

    #[pyo3(signature = (retries=0, poll_interval=0.1))]
    fn enable(&self, retries: i64, poll_interval: f64) -> bool {
        let Some(c) = self.inner.ctrl() else {
            return false;
        };
        if let Err(e) = c.enable_all() {
            eprintln!("[enable] 调用 enable_all() 失败: {e}");
            return false;
        }
        if retries <= 0 {
            return true;
        }
        let mut last_status = None;
        for _ in 0..retries {
            self.inner.poll();
            if let Some(s) = self.inner.motor().and_then(|m| m.get_state()) {
                if s.status_code == 1 {
                    return true;
                }
                last_status = Some(s.status_code);
            }
            sleep_s(poll_interval);
        }
        eprintln!("[enable] 使能确认失败（已重试 {retries} 次，last_status={last_status:?}）");
        false
    }

    #[pyo3(signature = (retries=0, poll_interval=0.1))]
    fn disable(&self, py: Python<'_>, retries: i64, poll_interval: f64) -> bool {
        self.inner.stop_loop(py);
        let Some(c) = self.inner.ctrl() else {
            return false;
        };
        if let Err(e) = c.disable_all() {
            eprintln!("[disable] 调用 disable_all() 失败: {e}");
            return false;
        }
        if retries <= 0 {
            return true;
        }
        let mut last_status = None;
        for _ in 0..retries {
            self.inner.poll();
            if let Some(s) = self.inner.motor().and_then(|m| m.get_state()) {
                if s.status_code == 0 {
                    return true;
                }
                last_status = Some(s.status_code);
            }
            sleep_s(poll_interval);
        }
        eprintln!("[disable] 失能确认失败（已重试 {retries} 次，last_status={last_status:?}）");
        false
    }

    // ---------------- 零点 ----------------

    #[pyo3(signature = (poll_max=200, poll_interval=0.05))]
    fn set_zero(&self, py: Python<'_>, poll_max: i64, poll_interval: f64) -> bool {
        self.disable(py, 10, 0.1);
        sleep_s(0.3);
        let mut ready = false;
        for _ in 0..poll_max {
            self.inner.poll();
            if let Some(s) = self.inner.motor().and_then(|m| m.get_state()) {
                if s.status_code == 0 {
                    ready = true;
                    break;
                }
            }
            sleep_s(poll_interval);
        }
        if !ready {
            eprintln!("[set_zero] 等待状态就绪超时");
            return false;
        }
        match self.inner.motor().map(|m| m.set_zero()) {
            Some(Ok(())) => {
                eprintln!("[set_zero] OK");
                true
            }
            Some(Err(e)) => {
                eprintln!("[set_zero] {e}");
                false
            }
            None => false,
        }
    }

    // ---------------- 状态 ----------------

    #[pyo3(signature = (request=true))]
    fn get_state(&self, request: bool) -> (f64, f64, f64) {
        if request {
            self.inner.request();
            self.inner.poll();
        }
        self.inner.poll();
        match self.inner.motor().and_then(|m| m.get_state()) {
            Some(s) => (s.pos, s.vel, s.torq),
            None => (0.0, 0.0, 0.0),
        }
    }

    #[pyo3(signature = (request=true))]
    fn get_position(&self, request: bool) -> f64 {
        self.get_state(request).0
    }
    #[pyo3(signature = (request=true))]
    fn get_velocity(&self, request: bool) -> f64 {
        self.get_state(request).1
    }
    #[pyo3(signature = (request=true))]
    fn get_torque(&self, request: bool) -> f64 {
        self.get_state(request).2
    }

    // ---------------- 模式切换 ----------------

    #[pyo3(signature = (kp=None, kd=None, stabilize_delay=0.2))]
    fn mode_mit(&self, kp: Option<f64>, kd: Option<f64>, stabilize_delay: f64) -> bool {
        *self.inner.mode.lock().unwrap() = "mit".to_string();
        if let Some(kp) = kp {
            *self.inner.mit_kp.lock().unwrap() = kp;
        }
        if let Some(kd) = kd {
            *self.inner.mit_kd.lock().unwrap() = kd;
        }
        let ok = self
            .inner
            .motor()
            .map(|m| match m.ensure_mode(1, 1000) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("[ensure_mode] {e}");
                    false
                }
            })
            .unwrap_or(false);
        sleep_s(stabilize_delay);
        ok
    }

    #[pyo3(signature = (stabilize_delay=0.2))]
    fn mode_pos_vel(&self, stabilize_delay: f64) -> bool {
        *self.inner.mode.lock().unwrap() = "pos_vel".to_string();
        if let Some(m) = self.inner.motor() {
            let _ = m.write_register_f32(25, self.inner.cfg.vel_kp as f32);
            let _ = m.write_register_f32(26, self.inner.cfg.vel_ki as f32);
            let _ = m.write_register_f32(27, self.inner.cfg.pos_kp as f32);
            let _ = m.write_register_f32(28, self.inner.cfg.pos_ki as f32);
            sleep_s(0.02);
        }
        let ok = self
            .inner
            .motor()
            .map(|m| m.ensure_mode(2, 1000).is_ok())
            .unwrap_or(false);
        sleep_s(stabilize_delay);
        ok
    }

    #[pyo3(signature = (stabilize_delay=0.2))]
    fn mode_vel(&self, stabilize_delay: f64) -> bool {
        *self.inner.mode.lock().unwrap() = "vel".to_string();
        let ok = self
            .inner
            .motor()
            .map(|m| m.ensure_mode(3, 1000).is_ok())
            .unwrap_or(false);
        sleep_s(stabilize_delay);
        ok
    }

    // ---------------- 控制命令 ----------------

    #[pyo3(signature = (pos, vel=0.0, kp=None, kd=None, tau=0.0))]
    fn mit(&self, pos: f64, vel: f64, kp: Option<f64>, kd: Option<f64>, tau: f64) {
        let kp = kp.unwrap_or_else(|| *self.inner.mit_kp.lock().unwrap());
        let kd = kd.unwrap_or_else(|| *self.inner.mit_kd.lock().unwrap());
        if let Some(m) = self.inner.motor() {
            let _ = m.send_mit(pos as f32, vel as f32, kp as f32, kd as f32, tau as f32);
        }
        self.inner.request();
        self.inner.poll();
    }

    #[pyo3(signature = (pos, vlim=None))]
    fn pos_vel(&self, pos: f64, vlim: Option<f64>) {
        let vlim = vlim.unwrap_or(self.inner.cfg.vlim);
        if let Some(m) = self.inner.motor() {
            let _ = m.send_pos_vel(pos as f32, vlim as f32);
        }
        self.inner.request();
        self.inner.poll();
    }

    fn set_vel(&self, vel: f64) {
        if let Some(m) = self.inner.motor() {
            let _ = m.send_vel(vel as f32);
        }
        self.inner.request();
        self.inner.poll();
    }

    // ---------------- 控制循环 ----------------

    #[pyo3(signature = (controller, rate=100.0))]
    fn start_control_loop(slf: Bound<'_, Self>, controller: PyObject, rate: f64) -> PyResult<()> {
        let inner = slf.borrow().inner.clone();
        if inner.loop_active() {
            return Err(PyRuntimeError::new_err("控制循环已在运行"));
        }
        if !rate.is_finite() || rate <= 0.0 || rate > 10_000.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rate={rate} 非法，必须为有限值且 0 < rate <= 10000"
            )));
        }
        *inner.rate.lock().unwrap() = rate;
        inner.running.store(true, Ordering::Release);
        let obj: Py<Self> = slf.clone().unbind();
        let run_inner = inner.clone();
        let handle = thread::spawn(move || {
            let dt = 1.0 / rate;
            let mut last = Instant::now();
            while run_inner.running.load(Ordering::Acquire) {
                let elapsed = last.elapsed().as_secs_f64();
                if elapsed >= dt {
                    last = Instant::now();
                    Python::with_gil(|py| {
                        let g = obj.clone_ref(py);
                        if let Err(e) = controller.call1(py, (g, elapsed)) {
                            e.print(py);
                        }
                    });
                } else {
                    sleep_s(1e-4);
                }
            }
        });
        *inner.thread.lock().unwrap() = Some(handle);
        Ok(())
    }

    fn stop_control_loop(&self, py: Python<'_>) {
        self.inner.stop_loop(py);
    }

    // ---------------- 连接 / 断开 ----------------

    fn disconnect(&self, py: Python<'_>) {
        self.inner.stop_loop(py);
        self.disable(py, 0, 0.1);
        sleep_s(0.5);
        if let Some(c) = self.inner.ctrl() {
            let _ = c.shutdown();
            let _ = c.close_bus();
        }
        *self.inner.ctrl.lock().unwrap() = None;
        *self.inner.motor.lock().unwrap() = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.disconnect(py);
    }

    fn __repr__(&self) -> String {
        format!(
            "Gripper({:?}, mode={})",
            self.inner.cfg.name,
            self.inner.mode.lock().unwrap()
        )
    }
}
