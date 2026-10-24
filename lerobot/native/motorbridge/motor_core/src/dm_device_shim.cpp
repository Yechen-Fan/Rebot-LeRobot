#include "dmcan.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

extern "C" {

struct mb_dm_frame {
    uint32_t can_id;
    uint8_t data[8];
    uint8_t dlc;
    uint8_t channel;
    uint8_t ext;
    uint8_t canfd;
};

} // extern "C"

namespace {

using context_create_fn = void (*)(dmcan_context**);
using find_devices_with_type_fn = int (*)(dmcan_context*, dmcan_device_type_t);
using device_get_fn = bool (*)(dmcan_context*, dmcan_device_handle**, int);
using device_open_fn = bool (*)(dmcan_device_handle*);
using device_enable_channel_fn = bool (*)(dmcan_device_handle*, uint8_t);
using device_get_channel_baudrate_fn =
    bool (*)(dmcan_device_handle*, uint8_t, dmcan_channel_can_info_t*);
using device_set_channel_baudrate_fn =
    bool (*)(dmcan_device_handle*, uint8_t, dmcan_channel_can_info_t);
using device_hook_recv_callback_fn =
    void (*)(dmcan_device_handle*, dev_recv_callback);
using device_send_can_fn =
    bool (*)(dmcan_device_handle*, uint8_t, uint32_t, bool, bool, bool, bool, uint8_t, uint8_t*);
using utils_get_dlc_from_len_fn = int (*)(int);

struct Api {
    void* lib = nullptr;
    context_create_fn context_create = nullptr;
    find_devices_with_type_fn find_devices_with_type = nullptr;
    device_get_fn device_get = nullptr;
    device_open_fn device_open = nullptr;
    device_enable_channel_fn device_enable_channel = nullptr;
    device_get_channel_baudrate_fn device_get_channel_baudrate = nullptr;
    device_set_channel_baudrate_fn device_set_channel_baudrate = nullptr;
    device_hook_recv_callback_fn device_hook_recv_callback = nullptr;
    device_send_can_fn device_send_can = nullptr;
    utils_get_dlc_from_len_fn utils_get_dlc_from_len = nullptr;
};

struct mb_dm_handle {
    Api api;
    dmcan_context* ctx = nullptr;
    dmcan_device_handle* dev = nullptr;
    int device_type = -1;
    uint8_t selected_channel = 0;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<mb_dm_frame> queue;
    bool stopped = false;
};

std::mutex g_registry_mutex;
std::unordered_map<dmcan_device_handle*, mb_dm_handle*> g_registry;
std::mutex g_persistent_mutex;
mb_dm_handle* g_persistent_handle = nullptr;

void set_err(char* err_buf, size_t err_len, const std::string& msg)
{
    if (!err_buf || err_len == 0) {
        return;
    }
    const size_t n = msg.size() < err_len - 1 ? msg.size() : err_len - 1;
    std::memcpy(err_buf, msg.data(), n);
    err_buf[n] = '\0';
}

#if defined(_WIN32)
void* open_library(const char* path)
{
    return reinterpret_cast<void*>(LoadLibraryA(path));
}

void* load_symbol(void* lib, const char* name)
{
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
}
#else
void* open_library(const char* path)
{
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void* load_symbol(void* lib, const char* name)
{
    return dlsym(lib, name);
}
#endif

template <typename T>
bool load_required(Api& api, T& dst, const char* name, char* err_buf, size_t err_len)
{
    void* sym = load_symbol(api.lib, name);
    if (!sym) {
        set_err(err_buf, err_len, std::string("load symbol failed: ") + name);
        return false;
    }
    dst = reinterpret_cast<T>(sym);
    return true;
}

bool load_api(Api& api, const char* library_path, char* err_buf, size_t err_len)
{
    api.lib = open_library(library_path);
    if (!api.lib) {
        set_err(err_buf, err_len, std::string("load DM_Device SDK failed: ") + library_path);
        return false;
    }

    return load_required(api, api.context_create, "dmcan_context_create", err_buf, err_len) &&
           load_required(api, api.find_devices_with_type, "dmcan_find_devices_with_type", err_buf, err_len) &&
           load_required(api, api.device_get, "dmcan_device_get", err_buf, err_len) &&
           load_required(api, api.device_open, "dmcan_device_open", err_buf, err_len) &&
           load_required(api, api.device_enable_channel, "dmcan_device_enable_channel", err_buf, err_len) &&
           load_required(api, api.device_get_channel_baudrate, "dmcan_device_get_channel_baudrate", err_buf, err_len) &&
           load_required(api, api.device_set_channel_baudrate, "dmcan_device_set_channel_baudrate", err_buf, err_len) &&
           load_required(api, api.device_hook_recv_callback, "dmcan_device_hook_recv_callback", err_buf, err_len) &&
           load_required(api, api.device_send_can, "dmcan_device_send_can", err_buf, err_len) &&
           load_required(api, api.utils_get_dlc_from_len, "dmcan_utils_get_dlc_from_len", err_buf, err_len);
}

bool configure_channel(mb_dm_handle* h, uint8_t channel, uint32_t can_baudrate,
                       uint32_t canfd_baudrate, char* err_buf, size_t err_len)
{
    bool enabled = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (h->api.device_enable_channel(h->dev, channel)) {
            enabled = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (!enabled) {
        set_err(err_buf, err_len, "dmcan_device_enable_channel(" + std::to_string(channel) + ") failed");
        return false;
    }

    dmcan_channel_can_info_t info{};
    if (!h->api.device_get_channel_baudrate(h->dev, channel, &info)) {
        set_err(err_buf, err_len, "dmcan_device_get_channel_baudrate(" + std::to_string(channel) + ") failed");
        return false;
    }

    info.channel = channel;
    info.canfd = true;
    info.can_baudrate = can_baudrate;
    info.canfd_baudrate = canfd_baudrate;
    info.can_sp = 0.75f;
    info.canfd_sp = 0.75f;

    if (!h->api.device_set_channel_baudrate(h->dev, channel, info)) {
        set_err(err_buf, err_len, "dmcan_device_set_channel_baudrate(" + std::to_string(channel) + ") failed");
        return false;
    }
    return true;
}

void recv_callback(dmcan_device_handle* dev, usb_rx_frame_t* frame)
{
    if (!dev || !frame) {
        return;
    }

    mb_dm_handle* h = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_registry.find(dev);
        if (it == g_registry.end()) {
            return;
        }
        h = it->second;
    }

    if (frame->head.channel != h->selected_channel || frame->head.rtr) {
        return;
    }

    int len = h->api.utils_get_dlc_from_len(frame->head.dlc);
    if (len < 0) {
        len = 0;
    }
    if (len > 8) {
        len = 8;
    }

    mb_dm_frame out{};
    out.can_id = frame->head.can_id;
    out.dlc = static_cast<uint8_t>(len);
    out.channel = frame->head.channel;
    out.ext = static_cast<uint8_t>(frame->head.ext ? 1 : 0);
    out.canfd = static_cast<uint8_t>(frame->head.canfd ? 1 : 0);
    if (len > 0) {
        std::memcpy(out.data, frame->payload, static_cast<size_t>(len));
    }

    {
        std::lock_guard<std::mutex> lock(h->queue_mutex);
        if (h->queue.size() >= 512) {
            h->queue.pop_front();
        }
        h->queue.push_back(out);
    }
    h->queue_cv.notify_one();
}

} // namespace

extern "C" int mb_dm_open(const char* library_path, int device_type, uint8_t selected_channel,
                          uint32_t can_baudrate, uint32_t canfd_baudrate, mb_dm_handle** out,
                          char* err_buf, size_t err_len)
{
    if (!library_path || !out) {
        set_err(err_buf, err_len, "invalid mb_dm_open argument");
        return -1;
    }

    {
        std::lock_guard<std::mutex> persistent_lock(g_persistent_mutex);
        if (g_persistent_handle && g_persistent_handle->device_type == device_type &&
            g_persistent_handle->dev) {
            auto* h = g_persistent_handle;
            h->selected_channel = selected_channel;
            {
                std::lock_guard<std::mutex> queue_lock(h->queue_mutex);
                h->stopped = false;
                h->queue.clear();
            }
            {
                std::lock_guard<std::mutex> registry_lock(g_registry_mutex);
                g_registry[h->dev] = h;
            }
            h->api.device_hook_recv_callback(h->dev, recv_callback);
            *out = h;
            return 0;
        }
    }

    auto* h = new mb_dm_handle();
    h->device_type = device_type;
    h->selected_channel = selected_channel;
    if (!load_api(h->api, library_path, err_buf, err_len)) {
        delete h;
        return -1;
    }

    h->api.context_create(&h->ctx);
    if (!h->ctx) {
        set_err(err_buf, err_len, "dmcan_context_create failed");
        delete h;
        return -1;
    }

    int count = 0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        count = h->api.find_devices_with_type(h->ctx, static_cast<dmcan_device_type_t>(device_type));
        if (count > 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (count <= 0) {
        set_err(err_buf, err_len, "no DM_Device SDK device found");
        delete h;
        return -1;
    }

    if (!h->api.device_get(h->ctx, &h->dev, 0) || !h->dev) {
        set_err(err_buf, err_len, "dmcan_device_get(index=0) failed");
        delete h;
        return -1;
    }

    bool opened = false;
    for (int attempt = 0; attempt < 5; ++attempt) {
        if (h->api.device_open(h->dev)) {
            opened = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (!opened) {
        set_err(err_buf, err_len, "dmcan_device_open failed");
        delete h;
        return -1;
    }

    if (device_type == USB2CANFD_DUAL) {
        if (!configure_channel(h, 0, can_baudrate, canfd_baudrate, err_buf, err_len) ||
            !configure_channel(h, 1, can_baudrate, canfd_baudrate, err_buf, err_len)) {
            delete h;
            return -1;
        }
    } else if (device_type == LINKX4C) {
        for (uint8_t channel = 0; channel < 4; ++channel) {
            if (!configure_channel(h, channel, can_baudrate, canfd_baudrate, err_buf, err_len)) {
                delete h;
                return -1;
            }
        }
    } else if (!configure_channel(h, selected_channel, can_baudrate, canfd_baudrate, err_buf, err_len)) {
        delete h;
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        g_registry[h->dev] = h;
    }
    h->api.device_hook_recv_callback(h->dev, recv_callback);

    {
        std::lock_guard<std::mutex> persistent_lock(g_persistent_mutex);
        g_persistent_handle = h;
    }

    *out = h;
    return 0;
}

extern "C" int mb_dm_send(mb_dm_handle* handle, uint32_t can_id, uint8_t ext, uint8_t dlc,
                          const uint8_t* data, char* err_buf, size_t err_len)
{
    if (!handle || !data || dlc > 8) {
        set_err(err_buf, err_len, "invalid mb_dm_send argument");
        return -1;
    }

    uint8_t payload[8]{};
    std::memcpy(payload, data, dlc);
    if (!handle->api.device_send_can(handle->dev, handle->selected_channel, can_id, false,
                                     ext != 0, false, false, dlc, payload)) {
        set_err(err_buf, err_len, "dmcan_device_send_can failed");
        return -1;
    }
    return 0;
}

extern "C" int mb_dm_recv(mb_dm_handle* handle, mb_dm_frame* out, uint32_t timeout_ms,
                          char* err_buf, size_t err_len)
{
    if (!handle || !out) {
        set_err(err_buf, err_len, "invalid mb_dm_recv argument");
        return -1;
    }

    std::unique_lock<std::mutex> lock(handle->queue_mutex);
    if (handle->queue.empty() && timeout_ms > 0) {
        handle->queue_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&]() {
            return handle->stopped || !handle->queue.empty();
        });
    }
    if (handle->queue.empty()) {
        return 0;
    }
    *out = handle->queue.front();
    handle->queue.pop_front();
    return 1;
}

extern "C" void mb_dm_shutdown(mb_dm_handle* handle)
{
    if (!handle) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        g_registry.erase(handle->dev);
    }
    {
        std::lock_guard<std::mutex> lock(handle->queue_mutex);
        handle->stopped = true;
    }
    handle->queue_cv.notify_all();
    // Deliberately do not call dmcan_device_close/dmcan_context_destroy. This
    // matches the known-good diagnostic tool and avoids the SDK/libusb teardown
    // race observed on Linux.
}
