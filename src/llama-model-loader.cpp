#include "llama-model-loader.h"

#include "ggml-alloc.h"
#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif
#include "ggml.h"
#include "gguf.h"
#include "llama-hparams.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <future>
#include <mutex>
#include <regex>
#include <thread>
#if defined(__linux__)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

struct llama_psi_totals {
    uint64_t some = 0;
    uint64_t full = 0;
};

static llama_psi_totals llama_memory_psi_totals() {
    std::ifstream psi("/proc/pressure/memory");
    std::string line;
    llama_psi_totals result;
    while (std::getline(psi, line)) {
        const bool is_some = line.rfind("some ", 0) == 0;
        const bool is_full = line.rfind("full ", 0) == 0;
        if (!is_some && !is_full) {
            continue;
        }
        const std::string key = "total=";
        const size_t pos = line.find(key);
        if (pos == std::string::npos) {
            continue;
        }
        try {
            uint64_t value = std::stoull(line.substr(pos + key.size()));
            if (is_some) {
                result.some = value;
            } else {
                result.full = value;
            }
        } catch (...) {
            continue;
        }
    }
    return result;
}

static uint64_t llama_mem_available_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t value_kib = 0;
    std::string unit;
    while (meminfo >> key >> value_kib >> unit) {
        if (key == "MemAvailable:") {
            return value_kib * 1024;
        }
    }
    return 0;
}

#if defined(__linux__)
static uint64_t llama_htonll(uint64_t v) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (static_cast<uint64_t>(htonl(static_cast<uint32_t>(v))) << 32) | htonl(static_cast<uint32_t>(v >> 32));
#else
    return v;
#endif
}

static uint64_t llama_ntohll(uint64_t v) {
    return llama_htonll(v);
}

static bool llama_socket_send_all(int fd, const void * data, size_t size) {
    const char * p = static_cast<const char *>(data);
    while (size > 0) {
        ssize_t n = ::send(fd, p, size, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static bool llama_socket_recv_all(int fd, void * data, size_t size) {
    char * p = static_cast<char *>(data);
    while (size > 0) {
        ssize_t n = ::recv(fd, p, size, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}

static bool llama_parse_ipv4_endpoint(const char * endpoint, std::string & host, int & port) {
    if (endpoint == nullptr) {
        return false;
    }
    std::string value(endpoint);
    const size_t sep = value.rfind(':');
    if (sep == std::string::npos || sep == 0 || sep + 1 >= value.size()) {
        return false;
    }
    host = value.substr(0, sep);
    try {
        port = std::stoi(value.substr(sep + 1));
    } catch (...) {
        return false;
    }
    return port > 0 && port <= 65535;
}

static bool llama_read_from_odirect_stream(
        const char * endpoint,
        const char * path,
        uint64_t file_offset,
        void * data,
        size_t size) {
    std::string host;
    int port = 0;
    if (!llama_parse_ipv4_endpoint(endpoint, host, port) || path == nullptr || data == nullptr) {
        return false;
    }

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1 ||
            ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return false;
    }

    const uint32_t path_len = (uint32_t) strlen(path);
    const uint32_t path_len_net = htonl(path_len);
    const uint64_t offset_net = llama_htonll(file_offset);
    const uint64_t size_net = llama_htonll(size);
    if (!llama_socket_send_all(fd, &path_len_net, sizeof(path_len_net)) ||
            !llama_socket_send_all(fd, &offset_net, sizeof(offset_net)) ||
            !llama_socket_send_all(fd, &size_net, sizeof(size_net)) ||
            !llama_socket_send_all(fd, path, path_len)) {
        ::close(fd);
        return false;
    }

    uint64_t response_size_net = 0;
    if (!llama_socket_recv_all(fd, &response_size_net, sizeof(response_size_net))) {
        ::close(fd);
        return false;
    }
    const uint64_t response_size = llama_ntohll(response_size_net);
    if (response_size != size) {
        ::close(fd);
        return false;
    }

    const bool ok = llama_socket_recv_all(fd, data, size);
    ::close(fd);
    return ok;
}

struct llama_aligned_free {
    void operator()(void * p) const {
        free(p);
    }
};

static bool llama_read_local_odirect(
        int fd,
        int tail_fd,
        uint64_t file_size,
        uint64_t file_offset,
        size_t size,
        void * aligned_buffer,
        size_t aligned_buffer_size,
        void ** data) {
    constexpr size_t alignment = 4096;
    if (fd < 0 || aligned_buffer == nullptr || data == nullptr ||
            file_offset > file_size || size > file_size - file_offset) {
        return false;
    }

    const uint64_t requested = size;
    const uint64_t aligned_offset = file_offset & ~(static_cast<uint64_t>(alignment) - 1);
    const uint64_t end = file_offset + requested;
    const uint64_t direct_end = end & ~(static_cast<uint64_t>(alignment) - 1);
    const uint64_t direct_size = direct_end > aligned_offset ? direct_end - aligned_offset : 0;
    if (direct_size > aligned_buffer_size) {
        return false;
    }

    char * out = static_cast<char *>(aligned_buffer);
    *data = out;

    uint64_t copied = 0;
    uint64_t direct_done = 0;
    while (direct_done < direct_size) {
        ssize_t n = ::pread(fd,
                static_cast<char *>(aligned_buffer) + direct_done,
                direct_size - direct_done,
                static_cast<off_t>(aligned_offset + direct_done));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        direct_done += static_cast<uint64_t>(n);
    }

    if (direct_size > 0) {
        const uint64_t copy_start = std::max<uint64_t>(file_offset, aligned_offset);
        const uint64_t copy_end = std::min<uint64_t>(end, aligned_offset + direct_size);
        if (copy_end > copy_start) {
            const uint64_t copy_size = copy_end - copy_start;
            memmove(out, static_cast<char *>(aligned_buffer) + (copy_start - aligned_offset), copy_size);
            copied += copy_size;
        }
    }

    if (copied < requested) {
        if (tail_fd < 0) {
            return false;
        }
        while (copied < requested) {
            const size_t tail_size = static_cast<size_t>(requested - copied);
            ssize_t n = ::pread(tail_fd, out + copied, tail_size, static_cast<off_t>(file_offset + copied));
            if (n <= 0) {
                if (n < 0 && errno == EINTR) {
                    continue;
                }
                return false;
            }
            ::posix_fadvise(tail_fd, static_cast<off_t>(file_offset + copied), n, POSIX_FADV_DONTNEED);
            copied += static_cast<uint64_t>(n);
        }
    }

    return copied == requested;
}

static double llama_env_nonnegative_double(const char * name, double fallback, double max_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    if (value[0] == '-' || value[0] == '+') {
        throw std::runtime_error(format("%s: invalid non-negative value '%s'", name, value));
    }
    errno = 0;
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0 || parsed > max_value) {
        throw std::runtime_error(format("%s: invalid non-negative value '%s'", name, value));
    }
    return parsed;
}

static bool llama_env_equals(const char * name, const char * expected) {
    const char * value = std::getenv(name);
    return value != nullptr && strcmp(value, expected) == 0;
}

static bool llama_odirect_scheduler_weighted_enabled() {
    const char * value = std::getenv("GGML_ODIRECT_READ_SCHEDULER");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "none") == 0) {
        return false;
    }
    if (strcmp(value, "weighted") == 0) {
        return true;
    }
    throw std::runtime_error(format("GGML_ODIRECT_READ_SCHEDULER: invalid value '%s'", value));
}

static void llama_odirect_rate_limit_impl(size_t bytes, double limit_mibps, std::mutex & mutex, std::chrono::steady_clock::time_point & next_time) {
    if (limit_mibps <= 0.0 || bytes == 0) {
        return;
    }
    const double seconds = (bytes / 1024.0 / 1024.0) / limit_mibps;
    std::unique_lock<std::mutex> lock(mutex);
    const auto now = std::chrono::steady_clock::now();
    if (next_time < now) {
        next_time = now;
    }
    const auto wait_until = next_time;
    next_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
    lock.unlock();
    std::this_thread::sleep_until(wait_until);
}

static bool llama_odirect_rate_limit_weighted(bool rpc, size_t bytes) {
    static const bool enabled = llama_odirect_scheduler_weighted_enabled();
    static const double total_mibps = llama_env_nonnegative_double("GGML_ODIRECT_READ_RATE_LIMIT_MIBPS", 0.0, 100000.0);
    static const double rpc_weight = llama_env_nonnegative_double("GGML_RPC_ODIRECT_READ_WEIGHT", 1.0, 100000.0);
    static const double local_weight = llama_env_nonnegative_double("GGML_LOCAL_ODIRECT_READ_WEIGHT", 1.0, 100000.0);
    static const double weight_sum = std::max(0.0, rpc_weight) + std::max(0.0, local_weight);
    static std::mutex global_mutex;
    static std::mutex rpc_mutex;
    static std::mutex local_mutex;
    static auto global_next_time = std::chrono::steady_clock::now();
    static auto rpc_next_time = std::chrono::steady_clock::now();
    static auto local_next_time = std::chrono::steady_clock::now();

    if (!enabled) {
        return false;
    }
    if (total_mibps <= 0.0 || weight_sum <= 0.0) {
        throw std::runtime_error("GGML_ODIRECT_READ_SCHEDULER=weighted requires positive total rate limit and at least one positive path weight");
    }

    const double path_weight = rpc ? std::max(0.0, rpc_weight) : std::max(0.0, local_weight);
    llama_odirect_rate_limit_impl(bytes, total_mibps, global_mutex, global_next_time);
    if (path_weight <= 0.0) {
        return true;
    }

    const double path_mibps = total_mibps * path_weight / weight_sum;
    llama_odirect_rate_limit_impl(bytes, path_mibps, rpc ? rpc_mutex : local_mutex, rpc ? rpc_next_time : local_next_time);
    return true;
}

static void llama_odirect_rate_limit_rpc(size_t bytes) {
    if (llama_odirect_rate_limit_weighted(true, bytes)) {
        return;
    }
    static const double limit_mibps = llama_env_nonnegative_double(
            "GGML_RPC_ODIRECT_READ_RATE_LIMIT_MIBPS",
            llama_env_nonnegative_double("GGML_ODIRECT_READ_RATE_LIMIT_MIBPS", 0.0, 100000.0),
            100000.0);
    static std::mutex mutex;
    static auto next_time = std::chrono::steady_clock::now();
    llama_odirect_rate_limit_impl(bytes, limit_mibps, mutex, next_time);
}

static void llama_odirect_rate_limit_local(size_t bytes) {
    if (llama_odirect_rate_limit_weighted(false, bytes)) {
        return;
    }
    static const double limit_mibps = llama_env_nonnegative_double(
            "GGML_LOCAL_ODIRECT_READ_RATE_LIMIT_MIBPS",
            llama_env_nonnegative_double("GGML_ODIRECT_READ_RATE_LIMIT_MIBPS", 0.0, 100000.0),
            100000.0);
    static std::mutex mutex;
    static auto next_time = std::chrono::steady_clock::now();
    llama_odirect_rate_limit_impl(bytes, limit_mibps, mutex, next_time);
}

struct llama_rpc_odirect_read_state {
    int fd = -1;
    int tail_fd = -1;
    uint64_t file_size = 0;
    uint64_t file_offset = 0;
    uint64_t copied = 0;
    void * aligned_buffer = nullptr;
    size_t aligned_buffer_size = 0;
    double read_ms = 0.0;
    size_t chunks = 0;
};

static bool llama_rpc_odirect_read_callback(void * user_data, void * data, size_t size) {
    auto * state = static_cast<llama_rpc_odirect_read_state *>(user_data);
    if (state == nullptr) {
        return false;
    }
    void * data_ptr = nullptr;
    llama_odirect_rate_limit_rpc(size);
    const auto time_before = std::chrono::steady_clock::now();
    const bool ok = llama_read_local_odirect(
            state->fd,
            state->tail_fd,
            state->file_size,
            state->file_offset + state->copied,
            size,
            state->aligned_buffer,
            state->aligned_buffer_size,
            &data_ptr);
    if (!ok || data_ptr == nullptr) {
        return false;
    }
    memcpy(data, data_ptr, size);
    const auto time_after = std::chrono::steady_clock::now();
    state->read_ms += std::chrono::duration<double, std::milli>(time_after - time_before).count();
    state->copied += size;
    state->chunks++;
    return true;
}
#endif

const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q1_0:     return "Q1_0";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return "MXFP4 MoE";
        case LLAMA_FTYPE_MOSTLY_NVFP4:    return "NVFP4";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                gguf_get_arr_n(ctx, k),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(const std::string & key, T & result, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(enum llm_kv kid, T & result, bool required) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);
    template std::enable_if<std::is_integral<uint32_t>::value, bool>::type
    llama_model_loader::get_arr_n<uint32_t>(const std::string & key, uint32_t & result, bool required);

    template<typename T>
    bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);
            result.clear();

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result.emplace_back(value);
            }
        } else {
            result.resize(arr_info.length);
            result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
        }

        return true;
    }

    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_BOOL:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if (arr_info.length > N_MAX) {
            throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result[i] = value;
            }
        } else {
            if (arr_info.gt == GGUF_TYPE_BOOL) {
                const int8_t * values = (const int8_t *) arr_info.data;
                std::transform(values, values + arr_info.length, result.begin(), [](int8_t x) {
                    return static_cast<T>(x != 0);
                });
            } else {
                std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
            }
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
        return get_arr(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr<std::vector<std::string>>(enum llm_kv kid, std::vector<std::string> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<int32_t, 512>>(enum llm_kv kid, std::array<int32_t, 512> & result, bool required);
    template bool llama_model_loader::get_arr<std::vector<int32_t>>(enum llm_kv kid, std::vector<int32_t> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<uint32_t, LLAMA_MAX_LAYERS>>(enum llm_kv kid, std::array<uint32_t, LLAMA_MAX_LAYERS> & result, bool required);

    template<typename T>
    bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(metadata, key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
        return get_key(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
    template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
    template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
    template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

    template<>
    bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
        uint32_t tmp;
        const bool found = get_key(kid, tmp, required);
        if (found) {
            result = (enum llama_pooling_type) tmp;
        } else {
            result = LLAMA_POOLING_TYPE_UNSPECIFIED;
        }
        return found;
    }

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        if (n > N_MAX) {
            throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", n, (uint32_t) N_MAX, key.c_str()));
        }

        if (gguf_get_kv_type(metadata, kid) == GGUF_TYPE_ARRAY) {
            struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);

            if (n != arr_info.length) {
                throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
            }

            return get_arr(key, result, required);
        }

        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required) {
        const std::string key = llm_kv(kid);

        const int id = gguf_find_key(metadata, key.c_str());

        if (id < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        // throw and error if type is an array
        if (gguf_get_kv_type(metadata, id) == GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("expected scalar, found array for key: %s", key.c_str()));
            }
            return false;
        }

        return get_key(key, result, required);
    }

    // TODO: this is not very clever - figure out something better
    template bool llama_model_loader::get_key_or_arr<std::array<int,      4>>  (enum llm_kv kid, std::array<int,      4>   & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<float,    512>>(enum llm_kv kid, std::array<float,    512> & result, uint32_t n, bool required);


llama_model_loader::llama_model_loader(
        struct gguf_context * meta,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits,
        FILE * file,
        bool use_mmap,
        bool use_direct_io,
        bool uma_loader_safe,
        uint32_t uma_loader_slice_mib,
        uint32_t uma_loader_psi_gate,
        uint32_t uma_loader_min_available_gib,
        uint32_t uma_loader_buffer_slice_layers,
        uint32_t uma_loader_upload_chunk_mib,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p)
        : metadata(meta), set_tensor_data(set_tensor_data), set_tensor_data_ud(set_tensor_data_ud) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    if (!fname.empty()) {
        // Load the main GGUF
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file(fname.c_str(), params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(fname.c_str(), "rb", use_direct_io));
        contexts.emplace_back(ctx);

        if (use_mmap && use_direct_io) {
            if (files.back()->has_direct_io()) {
                LLAMA_LOG_WARN("%s: direct I/O is enabled, disabling mmap\n", __func__);
                use_mmap = false;
            } else {
                LLAMA_LOG_WARN("%s: direct I/O is not available, using mmap\n", __func__);
                use_direct_io = false;

                // reopen file using std::fopen for mmap
                files.pop_back();
                files.emplace_back(new llama_file(fname.c_str(), "rb", false));
            }
        }

        // Save tensors data offset of the main file.
        // For subsidiary files, `meta` tensor data offset must not be used,
        // so we build a unified tensors index for weights.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
        uint16_t n_split = 0;
        get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

        // Load additional GGML contexts
        if (n_split > 1) {
            // make sure the main file is loaded first
            uint16_t idx = 0;
            const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
            get_key(kv_split_no, idx);
            if (idx != 0) {
                throw std::runtime_error(format("illegal split file idx: %d (file: %s), model must be loaded with the first split", idx, fname.c_str()));
            }

            // generate list of splits if needed
            if (splits.empty()) {
                splits = llama_get_list_splits(fname, idx, n_split);
            }

            // in case user give a custom list of splits, check if it matches the expected number
            if (n_split != (uint16_t)splits.size()) {
                throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", splits.size(), n_split));
            }

            if (trace > 0) {
                LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
            }

            // load other splits
            for (idx = 1; idx < n_split; idx++) {
                const char * fname_split = splits[idx].c_str();

                struct gguf_init_params split_params = {
                    /*.no_alloc = */ true,
                    /*.ctx      = */ &ctx,
                };
                gguf_context_ptr ctx_gguf { gguf_init_from_file(fname_split, split_params) };
                if (!ctx_gguf) {
                    throw std::runtime_error(format("%s: failed to load GGUF split from %s", __func__, fname_split));
                }

                // check idx
                {
                    const int kid = gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
                    if (kid < 0) {
                        throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
                    }
                    int idx_gguf = gguf_get_val_u16(ctx_gguf.get(), kid);
                    if (idx_gguf != idx) {
                        throw std::runtime_error(format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
                    }
                }

                files.emplace_back(new llama_file(fname_split, "rb", use_direct_io));
                contexts.emplace_back(ctx);

                // Save tensors data offset info of the shard.
                for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                    std::string tensor_name = std::string(cur->name);
                    // make sure there is no duplicated tensor names
                    if (weights_map.find(tensor_name) != weights_map.end()) {
                        throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
                    }
                    n_elements += ggml_nelements(cur);
                    n_bytes    += ggml_nbytes(cur);
                    weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
                }
            }

            get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

            // sanity check
            {
                const int n_tensors_loaded = (int) weights_map.size();
                if (n_tensors != n_tensors_loaded) {
                    throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
                }
            }

            LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
        }
    } else if (file != nullptr) {
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file_ptr(file, params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from file pointer", __func__));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(file));
        contexts.emplace_back(ctx);

        // Save tensors data offset info of the main file.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
    } else {
        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    }

    n_kv      = gguf_get_n_kv(metadata);
    n_tensors = weights_map.size();
    file_read_mutexes.clear();
    file_read_mutexes.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        file_read_mutexes.emplace_back(new std::mutex());
    }

    fver = (enum llama_fver) gguf_get_version(metadata);

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.empty() ? "(file*)" : fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const ggml_tensor * tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(metadata, i);
            const enum gguf_type type   = gguf_get_kv_type(metadata, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(metadata, i)), gguf_get_arr_n(metadata, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(metadata, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }

    if (uma_loader_safe && use_mmap) {
        LLAMA_LOG_WARN("%s: UMA loader safe mode disables mmap to reduce page cache pressure\n", __func__);
        use_mmap = false;
    }

    this->use_mmap = use_mmap;
    this->use_direct_io = use_direct_io;
    this->uma_loader_safe = uma_loader_safe;
    this->uma_loader_slice_mib = uma_loader_slice_mib;
    this->uma_loader_psi_gate = uma_loader_psi_gate;
    this->uma_loader_min_available_gib = uma_loader_min_available_gib;
    this->uma_loader_buffer_slice_layers = uma_loader_buffer_slice_layers;
    this->uma_loader_upload_chunk_mib = uma_loader_upload_chunk_mib;
    this->check_tensors = check_tensors;
    this->no_alloc = no_alloc;
}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const std::string & name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name.c_str());
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }
    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                const int n_expert_used = hparams.n_expert_used;
                GGML_ASSERT(n_expert_used > 0);
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                const int n_expert_used = hparams.n_expert_used;
                GGML_ASSERT(n_expert_used > 0);
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                const int n_embd_head = hparams.n_embd_head_v();
                const int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd_inp = hparams.n_embd_inp();
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_inp, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t * buft_list) {
    GGML_ASSERT(!buft_list->empty());
    for (const auto & cur : *buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

struct ggml_tensor * llama_model_loader::create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft, int32_t slice) -> ggml_context * {
        const ctx_key key { buft, slice };
        auto it = ctx_map.find(key);
        if (it == ctx_map.end()) {
            // one ggml context per buffer type, or per buffer slice when explicitly requested
            int max_n_tensors = n_tensors;
            max_n_tensors += 1;                   // duplicated output tensor
            max_n_tensors += hparams.n_layer()*2; // duplicated rope freq tensors
            if (files.empty()) {
                max_n_tensors += hparams.n_layer()*256; // this should be well above what any model actually uses
            }
            const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map.emplace(key, ctx);

            return ctx;
        }
        return it->second.get();
    };

    auto buffer_slice_for_tensor = [&](const llm_tensor_info & info) -> int32_t {
        if (uma_loader_buffer_slice_layers == 0) {
            return 0;
        }

        switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                return -1;
            case LLM_TENSOR_LAYER_OUTPUT:
                return INT32_MAX;
            case LLM_TENSOR_LAYER_REPEATING:
                GGML_ASSERT(tn.bid >= 0);
                return tn.bid / (int32_t) uma_loader_buffer_slice_layers;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
        }
    };

    int32_t selected_buffer_slice = 0;
    auto buft_for_tensor = [&](ggml_tensor * t_meta) -> ggml_backend_buffer_type_t {
        if (!t_meta) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
        }

        // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
        // the tensor is duplicated
        // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
        llm_tensor tn_tensor = tn.tensor;
        if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && (flags & TENSOR_DUPLICATED)) {
            tn_tensor = LLM_TENSOR_OUTPUT;
        }

        llm_tensor_info info;
        try {
            info = llm_tensor_info_for(tn_tensor);
        } catch (const std::out_of_range & e) {
            throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
        }

        // skip unused tensors
        if (info.op == GGML_OP_NONE || (flags & TENSOR_SKIP)) {
            const size_t nbytes = ggml_nbytes(t_meta);
            LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

            size_data -= nbytes;
            n_created++;

            return nullptr;
        }

        // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
        ggml_op op;
        bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
        if (bias) {
            if (info.op == GGML_OP_MUL_MAT_ID) {
                op = GGML_OP_ADD_ID;
            } else {
                op = GGML_OP_ADD;
            }
        } else {
            op = info.op;
        }

        // sanity checks
        if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
            if (tn.bid != -1) {
                GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
            }
        } else {
            if (tn.bid == -1) {
                GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
            }
        }

        selected_buffer_slice = buffer_slice_for_tensor(info);

        // select the buffer type for this tensor
        const buft_list_t * buft_list;
        switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                buft_list = buft_list_input;
                break;
            case LLM_TENSOR_LAYER_OUTPUT:
                buft_list = buft_list_output;
                break;
            case LLM_TENSOR_LAYER_REPEATING:
                GGML_ASSERT(buft_list_layer != nullptr);
                buft_list = buft_list_layer;
                break;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
        }

        ggml_backend_buffer_type_t buft = nullptr;

        // check overrides
        if (tensor_buft_overrides) {
            std::string tensor_name = tn.str();
            for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                std::regex pattern(overrides->pattern);
                if (std::regex_search(tensor_name, pattern)) {
                    if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                        // when overriding to a CPU buffer, consider the extra buffer types
                        buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
                        if (use_mmap) {
                            static std::once_flag once;
                            std::call_once(once, [] {
                                LLAMA_LOG_WARN("llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance\n");
                            });
                        }
                    } else {
                        buft = overrides->buft;
                    }

                    LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
                            tensor_name.c_str(),
                            ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                            ggml_backend_buft_name(buft));
                    break;
                }
            }
        }

        if (!buft) {
            buft = select_weight_buft(hparams, t_meta, op, buft_list);
            if (!buft) {
                throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }
        }

        // avoid using a host buffer when using mmap
        auto * buft_dev = ggml_backend_buft_get_device(buft);
        if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
            auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu_dev) {
                throw std::runtime_error("no CPU backend found");
            }
            buft = ggml_backend_dev_buffer_type(cpu_dev);
        }

        if (buft != buft_list->front().second) {
            if (n_tensors_moved == 0) {
                first_tensor_moved_name = t_meta->name;
                first_tensor_moved_type_name = ggml_type_name(t_meta->type);
                first_moved_from_buft = buft_list->front().second;
                first_moved_to_buft   = buft;
            }
            n_tensors_moved++;
        }

        return buft;
    };

    if (files.empty()) {
        if (flags & TENSOR_SKIP_IF_VIRTUAL) {
            return nullptr;
        }
        ggml_type type = GGML_TYPE_F32;
        const int64_t tid = gguf_find_tensor(metadata, tn.str().c_str());
        if (tid != -1) {
            type = gguf_get_tensor_type(metadata, tid);
        }

        // for tensors that are not required some of the dimensions can be invalid:
        if (flags & TENSOR_NOT_REQUIRED) {
            for (size_t dim = 0; dim < ne.size(); dim++) {
                if (ne.begin()[dim] <= 0) {
                    return nullptr;
                }
            }
        }

        ggml_tensor t_meta;
        memset(&t_meta, 0, sizeof(ggml_tensor));
        t_meta.type = type;
        for (size_t dim = 0; dim < GGML_MAX_DIMS; dim++) {
            t_meta.ne[dim] = dim < ne.size() ? ne.begin()[dim] : 1;
            GGML_ASSERT(t_meta.ne[dim] >= 1);
            t_meta.nb[dim] = dim == 0 ? ggml_type_size(type) : t_meta.ne[dim-1]*t_meta.nb[dim-1];
            GGML_ASSERT(t_meta.nb[dim] >= 1);
        }
        ggml_set_name(&t_meta, tn.str().c_str());

        ggml_backend_buffer_type_t buft = buft_for_tensor(&t_meta);
        GGML_ASSERT(buft != nullptr);
        ggml_context * ctx = ctx_for_buft(buft, selected_buffer_slice);
        ggml_tensor * ret = ggml_dup_tensor(ctx, &t_meta);
        ggml_set_name(ret, tn.str().c_str());
        return ret;
    }

    ggml_tensor * t_meta = get_tensor_meta(tn.str().c_str());
    ggml_backend_buffer_type_t buft = buft_for_tensor(t_meta);
    if (buft == nullptr) {
        return nullptr; // return type is ggml_tensor *
    }
    ggml_context * ctx = ctx_for_buft(buft, selected_buffer_slice);

    // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
    if (flags & TENSOR_DUPLICATED) {
        ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
        if (t) {
            return t;
        }
    }

    LLAMA_LOG_DEBUG("%s: loading tensor %s\n", __func__, tn.str().c_str());
    const struct ggml_tensor * cur = check_tensor_dims(tn.str(), ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;
}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne.begin()[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
                                    dims[0], dims[1], dims[2], dims[3],
                                    cur->nb[1], cur->nb[2], cur->nb[3],
                                    offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors(bool partial) const {
    if (n_created > n_tensors) {
        throw std::runtime_error(format("%s: too many tensors created; expected %d, got %d", __func__, n_tensors, n_created));
    }
    if (n_created < n_tensors) {
        if (!partial) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
        LLAMA_LOG_INFO("%s: partial load — used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n",
                __func__, n_created, n_tensors);
    }
    if (n_tensors_moved > 0) {
        LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %zu others) cannot be used with preferred buffer type %s, using %s instead\n",
            __func__, first_tensor_moved_name.c_str(), first_tensor_moved_type_name.c_str(), n_tensors_moved - 1,
            ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            bool is_numa = false;

            auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                auto * is_numa_fn = (decltype(ggml_is_numa) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
                if (is_numa_fn) {
                    is_numa = is_numa_fn();
                }
            }

            std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa, uma_loader_safe);
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += ggml_nbytes(it.second.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = get_weight(ggml_get_name(tensor));
        if (!weight || weight->idx != idx) {
            continue;
        }
        *first = std::min(*first, weight->offs);
        *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
    }
}

void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
        if (uma_loader_safe) {
            file->advise_dontneed(w.offs, ggml_nbytes(cur));
        }
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}

bool llama_model_loader::load_all_data(
        struct ggml_context * ctx,
        llama_buf_map & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    if (files.empty()) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            set_tensor_data(t, set_tensor_data_ud);
        }
        return true;
    }
    if (cancelled.load()) {
        return false;
    }
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;

    size_t alignment = 1;
    for (const auto & file : files) {
        alignment = std::max(file->read_alignment(), alignment);
        if (uma_loader_safe) {
            file->advise_noreuse(0, file->size());
        }
    }

    // Buffer size: balance between memory usage and I/O efficiency.
    // Upstream defaults to 1 MiB for ordinary reads; UMA safe loads can override this
    // to better match large local NVMe reads without enabling O_DIRECT globally.
    const size_t default_buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;
    const size_t requested_buffer_size =
        (uma_loader_safe && uma_loader_upload_chunk_mib > 0) ? (size_t) uma_loader_upload_chunk_mib * MiB : 0;
    const size_t buffer_size = requested_buffer_size > 0 ?
        (alignment != 1 ? requested_buffer_size + 2 * alignment : requested_buffer_size) :
        default_buffer_size;

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t> events;
    std::vector<void *> host_ptrs;
    size_t buffer_idx = 0; // buffer to use for async loads
    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);

            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            auto * event = ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    struct async_upload_resources {
        std::vector<ggml_backend_buffer_t> & host_buffers;
        std::vector<ggml_backend_event_t> & events;
        ggml_backend_t & upload_backend;

        ~async_upload_resources() {
            cleanup();
        }

        void cleanup() {
            for (auto * event : events) {
                ggml_backend_event_synchronize(event);
                ggml_backend_event_free(event);
            }
            events.clear();

            for (auto * buf : host_buffers) {
                ggml_backend_buffer_free(buf);
            }
            host_buffers.clear();

            if (upload_backend != nullptr) {
                ggml_backend_free(upload_backend);
                upload_backend = nullptr;
            }
        }
    } async_upload_cleanup { host_buffers, events, upload_backend };

    if (upload_backend == nullptr && (!host_buffers.empty() || !events.empty())) {
        async_upload_cleanup.cleanup();
    }

    if (upload_backend) {
        LLAMA_LOG_WARN("%s: using async uploads for device %s, buffer type %s, backend %s, staging buffers = %zu x %.2f MiB\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
            ggml_backend_name(upload_backend),
            n_buffers, buffer_size / 1024.0 / 1024.0);
    }
    bool logged_chunked_tensor_set = false;
    bool logged_rpc_odirect_stream = false;
    bool logged_local_odirect_stream = false;
    bool logged_local_odirect_direct = false;
    const char * rpc_odirect_stream_endpoint = uma_loader_safe ? std::getenv("GGML_RPC_ODIRECT_STREAM_ENDPOINT") : nullptr;
    const char * rpc_odirect_stream_mode = uma_loader_safe ? std::getenv("GGML_RPC_ODIRECT_STREAM_MODE") : nullptr;
    const char * local_odirect_stream_endpoint = uma_loader_safe ? std::getenv("GGML_LOCAL_ODIRECT_STREAM_ENDPOINT") : nullptr;
    const char * local_odirect_stream_mode = uma_loader_safe ? std::getenv("GGML_LOCAL_ODIRECT_STREAM_MODE") : nullptr;
    auto mode_is = [](const char * mode, const char * value) {
        return mode != nullptr && strcmp(mode, value) == 0;
    };
    if (uma_loader_safe && rpc_odirect_stream_mode != nullptr && rpc_odirect_stream_mode[0] != '\0' &&
            !mode_is(rpc_odirect_stream_mode, "stream") && !mode_is(rpc_odirect_stream_mode, "local")) {
        throw std::runtime_error(format("%s: invalid GGML_RPC_ODIRECT_STREAM_MODE=%s", __func__, rpc_odirect_stream_mode));
    }
    if (uma_loader_safe && local_odirect_stream_endpoint != nullptr && local_odirect_stream_endpoint[0] != '\0' &&
            local_odirect_stream_mode != nullptr && local_odirect_stream_mode[0] != '\0' &&
            !mode_is(local_odirect_stream_mode, "direct") && !mode_is(local_odirect_stream_mode, "local") &&
            !mode_is(local_odirect_stream_mode, "async")) {
        throw std::runtime_error(format("%s: invalid GGML_LOCAL_ODIRECT_STREAM_MODE=%s", __func__, local_odirect_stream_mode));
    }
    if (uma_loader_safe && mode_is(rpc_odirect_stream_mode, "stream") &&
            (rpc_odirect_stream_endpoint == nullptr || rpc_odirect_stream_endpoint[0] == '\0')) {
        throw std::runtime_error(format("%s: GGML_RPC_ODIRECT_STREAM_MODE=stream requires GGML_RPC_ODIRECT_STREAM_ENDPOINT", __func__));
    }
    const bool rpc_odirect_process_direct =
        mode_is(rpc_odirect_stream_mode, "local");
    const bool local_odirect_stream_direct =
        local_odirect_stream_endpoint != nullptr && local_odirect_stream_endpoint[0] != '\0' &&
        (local_odirect_stream_mode == nullptr || local_odirect_stream_mode[0] == '\0' || mode_is(local_odirect_stream_mode, "direct"));
    const bool local_odirect_process_direct =
        local_odirect_stream_endpoint != nullptr && local_odirect_stream_endpoint[0] != '\0' &&
        mode_is(local_odirect_stream_mode, "local");
    const bool local_odirect_stream_async =
        local_odirect_stream_endpoint != nullptr && local_odirect_stream_endpoint[0] != '\0' &&
        mode_is(local_odirect_stream_mode, "async");

    auto read_file_raw_locked = [&](int file_idx, size_t offset, void * data, size_t size, bool unsafe) {
        if (file_idx < 0 || (size_t) file_idx >= files.size() || (size_t) file_idx >= file_read_mutexes.size()) {
            throw std::runtime_error(format("%s: invalid file index %d for shared read", __func__, file_idx));
        }
        std::lock_guard<std::mutex> lock(*file_read_mutexes[file_idx]);
        auto & file = files.at(file_idx);
        file->seek(offset, SEEK_SET);
        if (unsafe) {
            file->read_raw_unsafe(data, size);
        } else {
            file->read_raw(data, size);
        }
    };

    const size_t slice_bytes = (uma_loader_safe && uma_loader_slice_mib > 0) ? (size_t) uma_loader_slice_mib * MiB : 0;
    const uint64_t min_available_bytes =
        (uma_loader_safe && uma_loader_min_available_gib > 0) ? (uint64_t) uma_loader_min_available_gib * GiB : 0;
    size_t slice_done = 0;
    llama_psi_totals last_mem_psi_total = llama_memory_psi_totals();
    llama_psi_totals phase_start_psi = last_mem_psi_total;
    uint64_t phase_start_available = llama_mem_available_bytes();
    const char * phase_buft_name = nullptr;
    size_t local_direct_chunks = 0;
    size_t local_direct_bytes = 0;
    double local_direct_read_ms = 0.0;
    double local_direct_set_ms = 0.0;
#if defined(__linux__)
    std::vector<int> local_odirect_fds(files.size(), -1);
    std::vector<int> local_odirect_tail_fds(files.size(), -1);
    struct local_odirect_fd_resources {
        std::vector<int> & fds;
        std::vector<int> & tail_fds;

        ~local_odirect_fd_resources() {
            cleanup();
        }

        static void close_all(std::vector<int> & values) {
            for (int & fd : values) {
                if (fd != -1) {
                    ::close(fd);
                    fd = -1;
                }
            }
        }

        void cleanup() {
            close_all(fds);
            close_all(tail_fds);
        }
    } local_odirect_fd_cleanup { local_odirect_fds, local_odirect_tail_fds };

    auto get_local_odirect_fd = [&](int file_idx, bool tail) -> int {
        auto & fds = tail ? local_odirect_tail_fds : local_odirect_fds;
        if (file_idx < 0 || (size_t) file_idx >= fds.size()) {
            return -1;
        }
        if (fds[file_idx] == -1) {
            fds[file_idx] = ::open(files.at(file_idx)->path().c_str(), tail ? O_RDONLY : (O_RDONLY | O_DIRECT));
        }
        return fds[file_idx];
    };
#endif

    auto uma_slice_gate = [&]() {
        if (slice_bytes == 0 || slice_done < slice_bytes) {
            return;
        }

        for (auto * event : events) {
            ggml_backend_event_synchronize(event);
        }

        const llama_psi_totals psi_before = last_mem_psi_total;
        llama_psi_totals psi_now = llama_memory_psi_totals();
        last_mem_psi_total = psi_now;
        uint64_t available_now = llama_mem_available_bytes();

        const bool psi_some_moved = psi_now.some > psi_before.some;
        const bool psi_full_moved = psi_now.full > psi_before.full;
        const bool psi_moved = psi_some_moved || psi_full_moved;
        const bool low_available = min_available_bytes > 0 && (available_now == 0 || available_now < min_available_bytes);

        if (uma_loader_psi_gate > 0 && (psi_moved || low_available)) {
            LLAMA_LOG_WARN("%s: UMA loader slice gate: psi some %" PRIu64 " -> %" PRIu64 ", full %" PRIu64 " -> %" PRIu64 ", MemAvailable %.2f GiB, waiting up to %u s%s%s%s\n",
                    __func__, psi_before.some, psi_now.some, psi_before.full, psi_now.full,
                    available_now / 1024.0 / 1024.0 / 1024.0, uma_loader_psi_gate,
                    psi_some_moved ? " [psi_some]" : "",
                    psi_full_moved ? " [psi_full]" : "",
                    low_available ? " [available]" : "");

            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(uma_loader_psi_gate);
            bool gate_ok = false;
            do {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                const llama_psi_totals psi_next = llama_memory_psi_totals();
                const uint64_t available_next = llama_mem_available_bytes();
                const bool psi_stable = psi_next.some == psi_now.some && psi_next.full == psi_now.full;
                const bool available_ok = min_available_bytes == 0 || (available_next != 0 && available_next >= min_available_bytes);

                if (psi_stable && available_ok) {
                    gate_ok = true;
                    break;
                }
                psi_now = psi_next;
                last_mem_psi_total = psi_now;
                available_now = available_next;
            } while (std::chrono::steady_clock::now() < deadline);
            if (!gate_ok && !llama_env_equals("LLAMA_UMA_LOADER_GATE_FAIL_OPEN", "1")) {
                cancelled.store(true);
                throw std::runtime_error(format(
                            "%s: UMA loader slice gate timed out after %u s with psi some/full=%" PRIu64 "/%" PRIu64
                            " and MemAvailable=%.2f GiB",
                            __func__, uma_loader_psi_gate, psi_now.some, psi_now.full,
                            available_now / 1024.0 / 1024.0 / 1024.0));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        slice_done = 0;
    };

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        if (cancelled.load()) {
            return false;
        }
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (phase_buft_name == nullptr && cur->buffer != nullptr) {
            phase_buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer));
            LLAMA_LOG_WARN("%s: UMA loader phase begin: buffer type = %s, psi some/full = %" PRIu64 "/%" PRIu64 ", MemAvailable %.2f GiB\n",
                    __func__, phase_buft_name, phase_start_psi.some, phase_start_psi.full,
                    phase_start_available / 1024.0 / 1024.0 / 1024.0);
        }

        if (progress_callback) {
            size_t current_size_done = 0;
            {
                std::lock_guard<std::mutex> lock(size_done_mutex);
                current_size_done = size_done;
            }
            std::lock_guard<std::mutex> lock(progress_callback_mutex);
            if (!progress_callback((float) current_size_done / size_data, progress_callback_user_data)) {
                cancelled.store(true);
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);
        bool slice_counted_in_chunks = false;

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);

            if (ggml_backend_buffer_is_host(cur->buffer)) {
                if (uma_loader_safe) {
                    const bool use_host_odirect =
                        local_odirect_process_direct || local_odirect_stream_direct || local_odirect_stream_async;
                    if (!use_host_odirect) {
                        cancelled.store(true);
                        throw std::runtime_error(format(
                                    "%s: UMA safe loader requires local O_DIRECT direct/local/async mode for host tensor %s",
                                    __func__, ggml_get_name(cur)));
                    }
                    slice_counted_in_chunks = true;
#if defined(__linux__)
                    const size_t chunk_size = 64 * MiB;
                    if (!logged_local_odirect_direct) {
                        LLAMA_LOG_WARN("%s: UMA loader host O_DIRECT tensor-read path enabled, mode = %s, endpoint = %s, chunk = %.2f MiB\n",
                                __func__, local_odirect_process_direct ? "local" :
                                    (local_odirect_stream_async ? "async" : "direct"),
                                local_odirect_stream_endpoint, chunk_size / 1024.0 / 1024.0);
                        logged_local_odirect_direct = true;
                    }
                    read_buf.resize(std::min(n_size, chunk_size));
                    std::unique_ptr<void, llama_aligned_free> aligned_read_buf;
                    void * aligned_read_raw = nullptr;
                    const size_t aligned_read_size = chunk_size + 4096;
                    if (local_odirect_process_direct) {
                        const int ret = posix_memalign(&aligned_read_raw, 4096, aligned_read_size);
                        if (ret != 0) {
                            throw std::runtime_error(format("%s: posix_memalign failed with error %d", __func__, ret));
                        }
                        aligned_read_buf.reset(aligned_read_raw);
                    }

                    size_t data_read = 0;
                    while (data_read < n_size) {
                        if (cancelled.load()) {
                            return false;
                        }
                        const size_t data_to_copy = std::min(chunk_size, n_size - data_read);
                        void * data_ptr = read_buf.data();
                        if (local_odirect_process_direct) {
                            const int fd = get_local_odirect_fd(weight->idx, false);
                            const int tail_fd = get_local_odirect_fd(weight->idx, true);
                            llama_odirect_rate_limit_local(data_to_copy);
                            if (!llama_read_local_odirect(fd, tail_fd, file->size(), weight->offs + data_read, data_to_copy,
                                        aligned_read_buf.get(), aligned_read_size, &data_ptr)) {
                                throw std::runtime_error(format("%s: failed to read host tensor from local O_DIRECT file %s",
                                            __func__, file->path().c_str()));
                            }
                        } else {
                            if (!llama_read_from_odirect_stream(local_odirect_stream_endpoint, file->path().c_str(),
                                        weight->offs + data_read, read_buf.data(), data_to_copy)) {
                                throw std::runtime_error(format("%s: failed to read host tensor from local O_DIRECT stream endpoint %s",
                                            __func__, local_odirect_stream_endpoint));
                            }
                        }
                        memcpy(static_cast<uint8_t *>(cur->data) + data_read, data_ptr, data_to_copy);
                        data_read += data_to_copy;
                        slice_done += data_to_copy;
                        uma_slice_gate();
                    }
#else
                    throw std::runtime_error(format("%s: host O_DIRECT tensor-read path is unavailable on this platform", __func__));
#endif
                } else {
                    read_file_raw_locked(weight->idx, weight->offs, cur->data, n_size, false);
                    if (uma_loader_safe) {
                        file->advise_dontneed(weight->offs, n_size);
                    }
                }
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                const char * cur_buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer));
                const bool cur_is_rpc_buffer = strncmp(cur_buft_name, "RPC", 3) == 0;
                if (!check_tensors && (local_odirect_stream_direct || local_odirect_process_direct) && !cur_is_rpc_buffer) {
                    slice_counted_in_chunks = true;
#if defined(__linux__)
                    const size_t chunk_size = 64 * MiB;
                    if (!logged_local_odirect_direct) {
                        LLAMA_LOG_WARN("%s: UMA loader local O_DIRECT direct tensor-set path enabled, mode = %s, endpoint = %s, chunk = %.2f MiB, buffer type = %s\n",
                                __func__, local_odirect_process_direct ? "local" : "stream",
                                local_odirect_stream_endpoint, chunk_size / 1024.0 / 1024.0,
                                cur_buft_name);
                        logged_local_odirect_direct = true;
                    }
                    read_buf.resize(std::min(n_size, chunk_size));
                    std::unique_ptr<void, llama_aligned_free> aligned_read_buf;
                    void * aligned_read_raw = nullptr;
                    const size_t aligned_read_size = chunk_size + 4096;
                    if (local_odirect_process_direct) {
                        const int ret = posix_memalign(&aligned_read_raw, 4096, aligned_read_size);
                        if (ret != 0) {
                            throw std::runtime_error(format("%s: posix_memalign failed with error %d", __func__, ret));
                        }
                        aligned_read_buf.reset(aligned_read_raw);
                    }
                    size_t data_read = 0;

                    while (data_read < n_size) {
                        if (cancelled.load()) {
                            return false;
                        }
                        const size_t data_to_copy = std::min(chunk_size, n_size - data_read);
                        const llama_psi_totals psi_before_read = llama_memory_psi_totals();
                        const uint64_t available_before_read = llama_mem_available_bytes();
                        const auto time_before_read = std::chrono::steady_clock::now();
                        void * data_ptr = read_buf.data();
                        if (local_odirect_process_direct) {
                            const int fd = get_local_odirect_fd(weight->idx, false);
                            const int tail_fd = get_local_odirect_fd(weight->idx, true);
                            llama_odirect_rate_limit_local(data_to_copy);
                            if (!llama_read_local_odirect(fd, tail_fd, file->size(), weight->offs + data_read, data_to_copy,
                                        aligned_read_buf.get(), aligned_read_size, &data_ptr)) {
                                throw std::runtime_error(format(
                                            "%s: failed to read local O_DIRECT file %s",
                                            __func__, file->path().c_str()));
                            }
                        } else {
                            if (!llama_read_from_odirect_stream(local_odirect_stream_endpoint, file->path().c_str(),
                                        weight->offs + data_read, read_buf.data(), data_to_copy)) {
                                throw std::runtime_error(format(
                                            "%s: failed to read from local O_DIRECT stream endpoint %s",
                                            __func__, local_odirect_stream_endpoint));
                            }
                        }
                        const auto time_after_read = std::chrono::steady_clock::now();
                        const llama_psi_totals psi_after_read = llama_memory_psi_totals();
                        const uint64_t available_after_read = llama_mem_available_bytes();

                        const auto time_before_set = std::chrono::steady_clock::now();
                        ggml_backend_tensor_set(cur, data_ptr, data_read, data_to_copy);
                        const auto time_after_set = std::chrono::steady_clock::now();
                        const llama_psi_totals psi_after_set = llama_memory_psi_totals();
                        const uint64_t available_after_set = llama_mem_available_bytes();
                        const double read_ms = std::chrono::duration<double, std::milli>(time_after_read - time_before_read).count();
                        const double set_ms = std::chrono::duration<double, std::milli>(time_after_set - time_before_set).count();
                        local_direct_chunks++;
                        local_direct_bytes += data_to_copy;
                        local_direct_read_ms += read_ms;
                        local_direct_set_ms += set_ms;

                        const bool read_psi_moved =
                            psi_after_read.some > psi_before_read.some || psi_after_read.full > psi_before_read.full;
                        const bool set_psi_moved =
                            psi_after_set.some > psi_after_read.some || psi_after_set.full > psi_after_read.full;
                        if (read_psi_moved || set_psi_moved) {
                            LLAMA_LOG_WARN(
                                    "%s: UMA local direct chunk PSI: tensor=%s buffer=%s file_idx=%d tensor_off=%zu file_off=%zu size=%.2f MiB "
                                    "read_ms=%.3f set_ms=%.3f "
                                    "read_psi some/full +%" PRIu64 "/+%" PRIu64 " set_psi some/full +%" PRIu64 "/+%" PRIu64 " "
                                    "MemAvailable %.2f -> %.2f -> %.2f GiB%s%s\n",
                                    __func__, ggml_get_name(cur), cur_buft_name, weight->idx, data_read,
                                    weight->offs + data_read, data_to_copy / 1024.0 / 1024.0,
                                    read_ms, set_ms,
                                    psi_after_read.some >= psi_before_read.some ? psi_after_read.some - psi_before_read.some : 0,
                                    psi_after_read.full >= psi_before_read.full ? psi_after_read.full - psi_before_read.full : 0,
                                    psi_after_set.some >= psi_after_read.some ? psi_after_set.some - psi_after_read.some : 0,
                                    psi_after_set.full >= psi_after_read.full ? psi_after_set.full - psi_after_read.full : 0,
                                    available_before_read / 1024.0 / 1024.0 / 1024.0,
                                    available_after_read / 1024.0 / 1024.0 / 1024.0,
                                    available_after_set / 1024.0 / 1024.0 / 1024.0,
                                    read_psi_moved ? " [read]" : "",
                                    set_psi_moved ? " [set]" : "");
                        }

                        data_read += data_to_copy;
                        slice_done += data_to_copy;
                        uma_slice_gate();
                    }
#else
                    throw std::runtime_error(format("%s: local O_DIRECT stream path is unavailable on this platform", __func__));
#endif
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                } else if (upload_backend) {
                    slice_counted_in_chunks = true;
                    size_t offset = weight->offs;
                    alignment = file->read_alignment();
                    size_t aligned_offset = offset & ~(alignment - 1);
                    size_t offset_from_alignment = offset - aligned_offset;

                    // Calculate aligned read boundaries
                    size_t read_start = aligned_offset;
                    size_t read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);

                    size_t bytes_read = 0;
                    size_t data_read = 0;  // Actual tensor data copied (excluding padding)

                    while (bytes_read < read_end - read_start) {
                        if (cancelled.load()) {
                            return false;
                        }
                        size_t read_size = std::min<size_t>(buffer_size, read_end - read_start - bytes_read);

                        // Align the destination pointer within the pinned buffer
                        uintptr_t ptr_dest_aligned = (reinterpret_cast<uintptr_t>(host_ptrs[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        // Wait for previous upload to complete before reusing buffer
                        ggml_backend_event_synchronize(events[buffer_idx]);

                        // Read aligned chunk from file.
                        const bool use_local_odirect_stream = local_odirect_stream_async;
                        if (use_local_odirect_stream) {
#if defined(__linux__)
                            if (!logged_local_odirect_stream) {
                                LLAMA_LOG_WARN("%s: UMA loader local O_DIRECT stream read path enabled, endpoint = %s\n",
                                        __func__, local_odirect_stream_endpoint);
                                logged_local_odirect_stream = true;
                            }
                            if (!llama_read_from_odirect_stream(local_odirect_stream_endpoint, file->path().c_str(),
                                        read_start + bytes_read, reinterpret_cast<void *>(ptr_dest_aligned), read_size)) {
                                throw std::runtime_error(format(
                                            "%s: failed to read from local O_DIRECT stream endpoint %s",
                                            __func__, local_odirect_stream_endpoint));
                            }
#else
                            throw std::runtime_error(format("%s: local O_DIRECT stream path is unavailable on this platform", __func__));
#endif
                        } else {
                            if (uma_loader_safe) {
                                throw std::runtime_error(format(
                                            "%s: UMA safe loader refuses buffered async-upload read fallback for tensor %s",
                                            __func__, ggml_get_name(cur)));
                            }
                            read_file_raw_locked(weight->idx, read_start + bytes_read,
                                    reinterpret_cast<void *>(ptr_dest_aligned), read_size, true);
                        }

                        // Calculate actual data portion (excluding alignment padding)
                        uintptr_t ptr_data = ptr_dest_aligned;
                        size_t data_to_copy = read_size;

                        // Skip alignment padding at start of first chunk
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }

                        // Trim alignment padding at end of last chunk
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        // Async upload actual data to GPU
                        ggml_backend_tensor_set_async(upload_backend, cur,
                                                      reinterpret_cast<void *>(ptr_data), data_read, data_to_copy);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;
                        slice_done += data_to_copy;
                        uma_slice_gate();

                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    bool use_rpc_odirect_process_direct = false;
#if defined(__linux__)
                    if (!check_tensors && rpc_odirect_process_direct && cur_is_rpc_buffer) {
#ifdef GGML_USE_RPC
                        const size_t chunk_size = 64 * MiB;
                        void * aligned_read_raw = nullptr;
                        const size_t aligned_read_size = chunk_size + 4096;
                        const int ret = posix_memalign(&aligned_read_raw, 4096, aligned_read_size);
                        if (ret != 0) {
                            throw std::runtime_error(format("%s: posix_memalign failed with error %d", __func__, ret));
                        }
                        std::unique_ptr<void, llama_aligned_free> aligned_read_buf(aligned_read_raw);

                        llama_rpc_odirect_read_state state {
                            /*.fd                  =*/ get_local_odirect_fd(weight->idx, false),
                            /*.tail_fd             =*/ get_local_odirect_fd(weight->idx, true),
                            /*.file_size           =*/ file->size(),
                            /*.file_offset         =*/ weight->offs,
                            /*.copied              =*/ 0,
                            /*.aligned_buffer      =*/ aligned_read_buf.get(),
                            /*.aligned_buffer_size =*/ aligned_read_size,
                            /*.read_ms             =*/ 0.0,
                            /*.chunks              =*/ 0,
                        };
                        if (state.fd < 0 || state.tail_fd < 0) {
                            throw std::runtime_error(format("%s: failed to open local O_DIRECT file %s", __func__, file->path().c_str()));
                        }
                        if (!logged_rpc_odirect_stream) {
                            LLAMA_LOG_WARN("%s: UMA loader RPC in-process O_DIRECT stream path enabled, chunk = %.2f MiB, buffer type = %s\n",
                                    __func__, chunk_size / 1024.0 / 1024.0,
                                    ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)));
                            logged_rpc_odirect_stream = true;
                        }

                        const auto time_before_rpc = std::chrono::steady_clock::now();
                        if (!ggml_backend_rpc_buffer_set_tensor_from_callback(
                                    cur->buffer, cur, 0, n_size, &state, llama_rpc_odirect_read_callback)) {
                            throw std::runtime_error(format("%s: failed to stream local O_DIRECT file %s to RPC tensor",
                                        __func__, file->path().c_str()));
                        }
                        const auto time_after_rpc = std::chrono::steady_clock::now();
                        const double total_ms = std::chrono::duration<double, std::milli>(time_after_rpc - time_before_rpc).count();
                        LLAMA_LOG_WARN("%s: UMA RPC in-process O_DIRECT tensor timing: tensor=%s chunks=%zu bytes=%.2f GiB read=%.3f s send_set_total=%.3f s effective=%.3f GiB/s\n",
                                __func__, ggml_get_name(cur), state.chunks, n_size / 1024.0 / 1024.0 / 1024.0,
                                state.read_ms / 1000.0, total_ms / 1000.0,
                                total_ms > 0.0 ? (n_size / 1024.0 / 1024.0 / 1024.0) / (total_ms / 1000.0) : 0.0);

                        use_rpc_odirect_process_direct = true;
                        slice_counted_in_chunks = true;
                        slice_done += n_size;
                        uma_slice_gate();
#else
                        throw std::runtime_error(format("%s: RPC in-process O_DIRECT path requires GGML_USE_RPC", __func__));
#endif
                    }
#else
                    if (rpc_odirect_process_direct) {
                        throw std::runtime_error(format("%s: RPC in-process O_DIRECT path is unavailable on this platform", __func__));
                    }
#endif
                    bool use_rpc_odirect_stream = false;
                    if (!check_tensors &&
                            !use_rpc_odirect_process_direct &&
                            !rpc_odirect_process_direct &&
                            rpc_odirect_stream_endpoint != nullptr &&
                            rpc_odirect_stream_endpoint[0] != '\0' &&
                            cur_is_rpc_buffer) {
#ifdef GGML_USE_RPC
                        use_rpc_odirect_stream = ggml_backend_rpc_buffer_set_tensor_from_file(
                                cur->buffer, cur, rpc_odirect_stream_endpoint, file->path().c_str(), weight->offs, 0, n_size);
                        if (!use_rpc_odirect_stream) {
                            throw std::runtime_error(format("%s: RPC O_DIRECT stream endpoint was requested but failed for tensor %s",
                                        __func__, ggml_get_name(cur)));
                        }
#else
                        throw std::runtime_error(format("%s: RPC O_DIRECT stream endpoint was requested but GGML_USE_RPC is disabled", __func__));
#endif
                    }
                    if (use_rpc_odirect_process_direct) {
                        // already streamed through the existing RPC connection above
                    } else if (use_rpc_odirect_stream) {
                        slice_counted_in_chunks = true;
                        if (!logged_rpc_odirect_stream) {
                            LLAMA_LOG_WARN("%s: UMA loader RPC O_DIRECT stream path enabled, endpoint = %s, buffer type = %s\n",
                                    __func__, rpc_odirect_stream_endpoint,
                                    ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)));
                            logged_rpc_odirect_stream = true;
                        }
                        slice_done += n_size;
                        uma_slice_gate();
                    } else if (check_tensors) {
                        if (uma_loader_safe) {
                            cancelled.store(true);
                            throw std::runtime_error(format("%s: UMA safe loader does not allow buffered --check-tensors fallback", __func__));
                        }
                        if (local_odirect_stream_async && !cur_is_rpc_buffer) {
                            cancelled.store(true);
                            throw std::runtime_error(format("%s: GGML_LOCAL_ODIRECT_STREAM_MODE=async cannot be used with tensor checking fallback", __func__));
                        }
                        read_buf.resize(n_size);
                        read_file_raw_locked(weight->idx, weight->offs, read_buf.data(), n_size, false);
                        ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                        if (!ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                            throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                        }
                    } else {
                        if (uma_loader_safe) {
                            cancelled.store(true);
                            throw std::runtime_error(format(
                                        "%s: UMA safe loader refuses buffered chunked read fallback for tensor %s",
                                        __func__, ggml_get_name(cur)));
                        }
                        if (local_odirect_stream_async && !cur_is_rpc_buffer) {
                            cancelled.store(true);
                            throw std::runtime_error(format("%s: GGML_LOCAL_ODIRECT_STREAM_MODE=async would fall back to buffered chunked reads", __func__));
                        }
                        slice_counted_in_chunks = true;
                        const size_t chunk_size = std::max<size_t>(buffer_size, 64 * MiB);
                        if (!logged_chunked_tensor_set) {
                            LLAMA_LOG_WARN("%s: UMA loader chunked backend tensor set fallback enabled, chunk = %.2f MiB, buffer type = %s\n",
                                    __func__, chunk_size / 1024.0 / 1024.0,
                                    ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)));
                            logged_chunked_tensor_set = true;
                        }
                        read_buf.resize(std::min(n_size, chunk_size));
                        size_t data_read = 0;

                        while (data_read < n_size) {
                            if (cancelled.load()) {
                                return false;
                            }
                            const size_t data_to_copy = std::min(chunk_size, n_size - data_read);
                            read_file_raw_locked(weight->idx, weight->offs + data_read,
                                    read_buf.data(), data_to_copy, false);

                            ggml_backend_tensor_set(cur, read_buf.data(), data_read, data_to_copy);

                            data_read += data_to_copy;
                            slice_done += data_to_copy;
                            uma_slice_gate();
                        }
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(size_done_mutex);
            size_done += n_size;
        }
        if (!slice_counted_in_chunks) {
            slice_done += n_size;
            uma_slice_gate();
        }
    }

#if defined(__linux__)
    local_odirect_fd_cleanup.cleanup();
#endif

    async_upload_cleanup.cleanup();

    const llama_psi_totals phase_end_psi = llama_memory_psi_totals();
    const uint64_t phase_end_available = llama_mem_available_bytes();
    if (phase_buft_name != nullptr) {
        LLAMA_LOG_WARN("%s: UMA loader phase end: buffer type = %s, psi some/full delta = +%" PRIu64 "/+%" PRIu64 ", MemAvailable %.2f -> %.2f GiB\n",
                __func__, phase_buft_name,
                phase_end_psi.some >= phase_start_psi.some ? phase_end_psi.some - phase_start_psi.some : 0,
                phase_end_psi.full >= phase_start_psi.full ? phase_end_psi.full - phase_start_psi.full : 0,
                phase_start_available / 1024.0 / 1024.0 / 1024.0,
                phase_end_available / 1024.0 / 1024.0 / 1024.0);
        if (local_direct_chunks > 0) {
            const double total_ms = local_direct_read_ms + local_direct_set_ms;
            LLAMA_LOG_WARN("%s: UMA local direct phase timing: buffer type = %s, chunks = %zu, bytes = %.2f GiB, read = %.3f s, tensor_set = %.3f s, total = %.3f s, effective = %.3f GiB/s\n",
                    __func__, phase_buft_name, local_direct_chunks,
                    local_direct_bytes / 1024.0 / 1024.0 / 1024.0,
                    local_direct_read_ms / 1000.0,
                    local_direct_set_ms / 1000.0,
                    total_ms / 1000.0,
                    total_ms > 0.0 ? (local_direct_bytes / 1024.0 / 1024.0 / 1024.0) / (total_ms / 1000.0) : 0.0);
        }
    }

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    bool all_data_loaded = false;
    {
        std::lock_guard<std::mutex> lock(size_done_mutex);
        all_data_loaded = size_done >= size_data;
    }
    if (all_data_loaded) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            std::lock_guard<std::mutex> lock(progress_callback_mutex);
            if (progress_final_emitted) {
                return true;
            }
            progress_final_emitted = true;
            if (!progress_callback(1.0f, progress_callback_user_data)) {
                cancelled.store(true);
                return false;
            }
            return true;
        }
    }

    return true;
}

std::string llama_model_loader::ftype_name() const {
    return llama_model_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_model_ftype_name(ftype).c_str());
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}
