use crate::commands::{cmd_scan_robstride_progress, parse_vendor_in_msg};
use crate::model::Vendor;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::time::Duration;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio::net::TcpStream;
use tokio::time;
use tokio_tungstenite::{
    accept_hdr_async,
    tungstenite::{
        handshake::server::{ErrorResponse, Request, Response},
        protocol::Message,
    },
};

use crate::model::ServerConfig;
use crate::session::SessionCtx;

mod dispatch;
mod handlers;
pub(crate) mod stream;

use stream::ParamStream;

async fn send_json<S>(tx: &mut S, obj: Value) -> Result<(), String>
where
    S: futures_util::Sink<Message, Error = tokio_tungstenite::tungstenite::Error> + Unpin,
{
    tx.send(Message::Text(obj.to_string()))
        .await
        .map_err(|e| e.to_string())
}

fn decode_query_value(value: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = bytes[i + 1] as char;
                let lo = bytes[i + 2] as char;
                let hex = [hi, lo].iter().collect::<String>();
                let decoded = u8::from_str_radix(&hex, 16).ok()?;
                out.push(decoded);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

fn query_token(uri: &Uri) -> Option<String> {
    uri.query()?.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        if key == "motorbridge_ws_token" {
            decode_query_value(value)
        } else {
            None
        }
    })
}

fn request_token(req: &Request) -> Option<String> {
    req.headers()
        .get("x-motorbridge-token")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned)
        .or_else(|| {
            req.headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::to_owned)
        })
        .or_else(|| query_token(req.uri()))
}

#[allow(clippy::result_large_err)]
pub(crate) async fn handle_socket(stream: TcpStream, cfg: ServerConfig) -> Result<(), String> {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let expected_token = std::env::var("MOTORBRIDGE_WS_TOKEN").ok();
    if std::env::var("MOTORBRIDGE_WS_DEBUG").is_ok() {
        eprintln!("[ws_gateway] websocket handshake start: {peer}");
    }
    let ws = accept_hdr_async(stream, move |req: &Request, response: Response| {
        if let Some(token) = expected_token.as_deref() {
            let provided = request_token(req);
            if provided.as_deref() != Some(token) {
                let err_resp: ErrorResponse =
                    tokio_tungstenite::tungstenite::http::Response::builder()
                        .status(tokio_tungstenite::tungstenite::http::StatusCode::UNAUTHORIZED)
                        .header("content-type", "text/plain; charset=utf-8")
                        .body(Some("unauthorized websocket client".to_string()))
                        .expect("build unauthorized response");
                return Err(err_resp);
            }
        }
        Ok(response)
    })
    .await
    .map_err(|e| e.to_string())?;
    if std::env::var("MOTORBRIDGE_WS_DEBUG").is_ok() {
        eprintln!("[ws_gateway] websocket handshake ok: {peer}");
    }
    let (mut tx, mut rx) = ws.split();

    let mut ctx = SessionCtx::new(cfg.target.clone());
    let _ = send_json(
        &mut tx,
        json!({
            "type":"event",
            "event":"connected",
            "data": {
                "peer": peer,
                "router_mode": "standby",
                "connected_bus": false,
                "default_target": {
                    "vendor": ctx.target.vendor.as_str(),
                    "transport": ctx.target.transport.as_str(),
                    "channel": ctx.target.channel,
                    "model": ctx.target.model
                }
            }
        }),
    )
    .await;

    let mut ticker = time::interval(Duration::from_millis(cfg.dt_ms));
    let mut state_stream_enabled: bool = false;
    let mut state_tick_counter: u64 = 0;
    let state_tick_div: u64 = 5;
    let mut param_stream = ParamStream::default();
    loop {
        tokio::select! {
            maybe_msg = rx.next() => {
                let msg = match maybe_msg {
                    Some(Ok(m)) => m,
                    Some(Err(e)) => return Err(format!("ws recv error: {e}")),
                    None => break,
                };

                match msg {
                    Message::Text(text) => {
                        let v: Value = match serde_json::from_str(&text) {
                            Ok(x) => x,
                            Err(e) => {
                                send_json(&mut tx, json!({"ok":false, "error": format!("invalid json: {e}")})).await?;
                                continue;
                            }
                        };
                        let op = v.get("op").and_then(Value::as_str).unwrap_or("").to_lowercase();
                        let req_id = v.get("req_id").cloned();

                        if op == "scan"
                            && parse_vendor_in_msg(&v, ctx.target.vendor).ok()
                                == Some(Vendor::Robstride)
                        {
                            dispatch::release_session_before_scan(
                                &v,
                                &mut ctx,
                                &mut state_stream_enabled,
                                &mut param_stream,
                            );
                            let target = ctx.target.clone();
                            let req = v.clone();
                            let (progress_tx, mut progress_rx) =
                                tokio::sync::mpsc::unbounded_channel::<Value>();
                            let mut task = tokio::task::spawn_blocking(move || {
                                let mut emit = |event: Value| {
                                    let _ = progress_tx.send(event);
                                };
                                cmd_scan_robstride_progress(&req, &target, &mut emit)
                            });

                            let result = loop {
                                tokio::select! {
                                    Some(mut event) = progress_rx.recv() => {
                                        if let Some(id) = req_id.clone() {
                                            if let Some(obj) = event.as_object_mut() {
                                                obj.insert("req_id".to_string(), id);
                                            }
                                        }
                                        send_json(&mut tx, event).await?;
                                    }
                                    joined = &mut task => {
                                        break joined.map_err(|e| e.to_string())?;
                                    }
                                }
                            };
                            while let Ok(mut event) = progress_rx.try_recv() {
                                if let Some(id) = req_id.clone() {
                                    if let Some(obj) = event.as_object_mut() {
                                        obj.insert("req_id".to_string(), id);
                                    }
                                }
                                send_json(&mut tx, event).await?;
                            }

                            match result {
                                Ok(data) => {
                                    let mut resp = json!({"ok": true, "op": op, "data": data});
                                    if let Some(id) = req_id.clone() {
                                        if let Some(obj) = resp.as_object_mut() {
                                            obj.insert("req_id".to_string(), id);
                                        }
                                    }
                                    send_json(&mut tx, resp).await?
                                }
                                Err(err) => {
                                    let mut resp = json!({"ok": false, "op": op, "error": err});
                                    if let Some(id) = req_id.clone() {
                                        if let Some(obj) = resp.as_object_mut() {
                                            obj.insert("req_id".to_string(), id);
                                        }
                                    }
                                    send_json(&mut tx, resp).await?
                                }
                            }
                            continue;
                        }

                        let result = dispatch::dispatch_op(
                            &op,
                            &v,
                            &mut ctx,
                            &mut state_stream_enabled,
                            &mut param_stream,
                            cfg.dt_ms,
                        );
                        match result {
                            Ok(data) => {
                                let mut resp = json!({"ok": true, "op": op, "data": data});
                                if let Some(id) = req_id.clone() {
                                    if let Some(obj) = resp.as_object_mut() {
                                        obj.insert("req_id".to_string(), id);
                                    }
                                }
                                send_json(&mut tx, resp).await?
                            }
                            Err(err) => {
                                let mut resp = json!({"ok": false, "op": op, "error": err});
                                if let Some(id) = req_id.clone() {
                                    if let Some(obj) = resp.as_object_mut() {
                                        obj.insert("req_id".to_string(), id);
                                    }
                                }
                                send_json(&mut tx, resp).await?
                            }
                        }
                    }
                    Message::Ping(payload) => {
                        tx.send(Message::Pong(payload)).await.map_err(|e| e.to_string())?;
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            _ = ticker.tick() => {
                if ctx.active.is_some() {
                    let apply_rc = tokio::task::block_in_place(|| ctx.apply_active());
                    if let Err(e) = apply_rc {
                        ctx.active = None;
                        send_json(&mut tx, json!({"ok": false, "op": "active_tick", "error": e})).await?;
                    }
                }
                if state_stream_enabled && ctx.motor.is_some() {
                    state_tick_counter = state_tick_counter.wrapping_add(1);
                    if state_tick_counter.is_multiple_of(state_tick_div) {
                        let snapshot = tokio::task::block_in_place(|| ctx.build_state_snapshot());
                        match snapshot {
                            Ok(st) => send_json(&mut tx, json!({"type":"state", "data": st})).await?,
                            Err(err) => send_json(&mut tx, json!({"ok": false, "op":"state_tick","error": err})).await?,
                        }
                    }
                }
                if param_stream.enabled && ctx.motor.is_some() {
                    param_stream.tick_counter = param_stream.tick_counter.wrapping_add(1);
                    if param_stream
                        .tick_counter
                        .is_multiple_of(param_stream.tick_div)
                    {
                        let snapshot = tokio::task::block_in_place(|| {
                            ctx.build_param_snapshot(&param_stream.params, param_stream.timeout_ms)
                        });
                        match snapshot {
                            Ok(st) => {
                                let frame_type = match st.get("vendor").and_then(Value::as_str) {
                                    Some("damiao") => "damiao_params",
                                    Some("robstride") => "robstride_params",
                                    _ => "motor_params",
                                };
                                send_json(&mut tx, json!({"type": frame_type, "data": st})).await?
                            }
                            Err(err) => {
                                send_json(
                                    &mut tx,
                                    json!({"ok": false, "op":"param_tick","error": err}),
                                )
                                .await?
                            }
                        }
                    }
                }
            }
        }
    }

    ctx.disconnect(false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_tungstenite::tungstenite::http::Request as HttpRequest;

    #[test]
    fn query_token_decodes_percent_and_plus() {
        let uri: Uri = "ws://host:9002/?motorbridge_ws_token=a%2Bb+c".parse().expect("uri");
        assert_eq!(query_token(&uri).as_deref(), Some("a+b c"));
    }

    #[test]
    fn request_token_prefers_header_then_bearer_then_query() {
        let req = HttpRequest::builder()
            .uri("ws://host:9002/?motorbridge_ws_token=query-token")
            .header("authorization", "Bearer bearer-token")
            .header("x-motorbridge-token", "header-token")
            .body(())
            .expect("request");
        assert_eq!(request_token(&req).as_deref(), Some("header-token"));

        let req = HttpRequest::builder()
            .uri("ws://host:9002/?motorbridge_ws_token=query-token")
            .header("authorization", "Bearer bearer-token")
            .body(())
            .expect("request");
        assert_eq!(request_token(&req).as_deref(), Some("bearer-token"));

        let req = HttpRequest::builder()
            .uri("ws://host:9002/?motorbridge_ws_token=query-token")
            .body(())
            .expect("request");
        assert_eq!(request_token(&req).as_deref(), Some("query-token"));
    }
}
