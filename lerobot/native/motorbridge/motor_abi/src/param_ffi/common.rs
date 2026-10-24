use crate::{set_last_error, MotorHandle, MotorHandleInner};

pub(crate) fn ffi_get<T, F>(motor: *mut MotorHandle, out_value: *mut T, f: F) -> i32
where
    F: FnOnce(&MotorHandleInner) -> Result<T, String>,
{
    if motor.is_null() || out_value.is_null() {
        set_last_error("motor or out_value is null");
        return -1;
    }
    let motor = unsafe { &*motor };
    let inner = match motor.inner.lock() {
        Ok(inner) => inner,
        Err(_) => {
            set_last_error("motor handle lock poisoned");
            return -1;
        }
    };
    let out = unsafe { &mut *out_value };
    match f(&inner) {
        Ok(v) => {
            *out = v;
            0
        }
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}

pub(crate) fn ffi_run<F>(motor: *mut MotorHandle, f: F) -> i32
where
    F: FnOnce(&MotorHandleInner) -> Result<(), String>,
{
    if motor.is_null() {
        set_last_error("motor is null");
        return -1;
    }
    let motor = unsafe { &*motor };
    let inner = match motor.inner.lock() {
        Ok(inner) => inner,
        Err(_) => {
            set_last_error("motor handle lock poisoned");
            return -1;
        }
    };
    match f(&inner) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}
