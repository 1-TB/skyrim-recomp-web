// xex-decrypt: writes a decrypted, decompressed copy of an XEX2 file, which is
// the form the runtime loads.
//
// Built by `cargo xtask decrypt` against the user's local XenonRecomp checkout:
// the AES key comes from XenonUtils/xex.h, and AES/LZX/SHA-1 from its
// thirdparty sources. No key material lives in this repository.
//
// The decryption and decompression steps follow XenonRecomp's Xex2LoadImage
// (MIT), minus its import-thunk rewriting: the output keeps the original
// header and an untouched image so import records can still be read.
//
// Usage:
//   xex-decrypt IN.xex OUT.xex
//   xex-decrypt --encrypt IN.xex OUT.xex   (test fixtures: re-encrypt an
//                                           unencrypted XEX with a fixed key)

#include <xex.h>
#include <xex_patcher.h>

#include <aes.hpp>
#include <TinySHA1.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kKeySize = 16;
constexpr uint8_t kDevkitKey[kKeySize] = {};

bool readFile(const char* path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

bool writeFile(const char* path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(f);
}

struct Xex {
    std::vector<uint8_t> bytes;
    uint32_t headerSize = 0;
    uint32_t imageSize = 0;
    Xex2SecurityInfo* security = nullptr;
    Xex2OptFileFormatInfo* format = nullptr;

    bool parse(const char* path) {
        if (!readFile(path, bytes) || bytes.size() < sizeof(Xex2Header)) {
            std::fprintf(stderr, "%s: cannot read\n", path);
            return false;
        }
        auto* header = reinterpret_cast<Xex2Header*>(bytes.data());
        if (std::memcmp(bytes.data(), "XEX2", 4) != 0) {
            std::fprintf(stderr, "%s: not an XEX2 file\n", path);
            return false;
        }
        headerSize = header->headerSize;
        if (headerSize > bytes.size() || header->securityOffset + sizeof(Xex2SecurityInfo) > headerSize) {
            std::fprintf(stderr, "%s: truncated header\n", path);
            return false;
        }
        security = reinterpret_cast<Xex2SecurityInfo*>(bytes.data() + header->securityOffset);
        imageSize = security->imageSize;
        format = const_cast<Xex2OptFileFormatInfo*>(
            static_cast<const Xex2OptFileFormatInfo*>(getOptHeaderPtr(bytes.data(), XEX_HEADER_FILE_FORMAT_INFO)));
        return true;
    }

    size_t payloadSize() const { return bytes.size() - headerSize; }
};

// Session key = AES-CBC-decrypt(root key, security.aesKey); payload decrypted in place.
void decryptPayload(const uint8_t* encryptedSessionKey, const uint8_t* rootKey, std::vector<uint8_t>& payload) {
    uint8_t sessionKey[kKeySize];
    std::memcpy(sessionKey, encryptedSessionKey, kKeySize);
    AES_ctx ctx;
    AES_init_ctx_iv(&ctx, rootKey, AESBlankIV);
    AES_CBC_decrypt_buffer(&ctx, sessionKey, kKeySize);
    AES_init_ctx_iv(&ctx, sessionKey, AESBlankIV);
    AES_CBC_decrypt_buffer(&ctx, payload.data(), payload.size() & ~size_t(15));
}

bool decompress(const Xex& xex, const std::vector<uint8_t>& payload, std::vector<uint8_t>& image) {
    image.assign(xex.imageSize, 0);
    const uint16_t compression = xex.format ? uint16_t(xex.format->compressionType) : XEX_COMPRESSION_NONE;
    switch (compression) {
    case XEX_COMPRESSION_NONE:
        std::memcpy(image.data(), payload.data(), std::min<size_t>(payload.size(), image.size()));
        return true;

    case XEX_COMPRESSION_BASIC: {
        auto* blocks = reinterpret_cast<const Xex2FileBasicCompressionBlock*>(xex.format + 1);
        const size_t count = xex.format->infoSize / sizeof(Xex2FileBasicCompressionInfo) - 1;
        size_t src = 0, dst = 0;
        for (size_t i = 0; i < count; i++) {
            const size_t dataSize = blocks[i].dataSize, zeroSize = blocks[i].zeroSize;
            if (src + dataSize > payload.size() || dst + dataSize + zeroSize > image.size()) return false;
            std::memcpy(image.data() + dst, payload.data() + src, dataSize);
            src += dataSize;
            dst += dataSize + zeroSize;
        }
        return true;
    }

    case XEX_COMPRESSION_NORMAL: {
        // A chain of SHA-1-verified blocks, each holding length-prefixed LZX
        // chunks; the chunks concatenated form one LZX stream.
        auto* info = reinterpret_cast<const Xex2FileNormalCompressionInfo*>(xex.format + 1);
        const Xex2CompressedBlockInfo* block = &info->firstBlock;
        std::vector<uint8_t> stream;
        stream.reserve(payload.size());
        const uint8_t* p = payload.data();
        const uint8_t* end = payload.data() + payload.size();
        uint8_t digest[20];
        sha1::SHA1 sha;
        while (block->blockSize) {
            if (p + block->blockSize > end) return false;
            sha.reset();
            sha.processBytes(p, block->blockSize);
            sha.finalize(digest);
            if (std::memcmp(digest, block->blockHash, sizeof(digest)) != 0) return false;
            const uint8_t* next = p + block->blockSize;
            const auto* nextBlock = reinterpret_cast<const Xex2CompressedBlockInfo*>(p);
            const uint8_t* q = p + sizeof(Xex2CompressedBlockInfo);
            while (q + 2 <= next) {
                const size_t chunk = (size_t(q[0]) << 8) | q[1];
                q += 2;
                if (chunk == 0) break;
                if (q + chunk > next) return false;
                stream.insert(stream.end(), q, q + chunk);
                q += chunk;
            }
            p = next;
            block = nextBlock;
        }
        return lzxDecompress(stream.data(), stream.size(), image.data(), image.size(), info->windowSize, nullptr, 0) == 0;
    }

    default:
        std::fprintf(stderr, "unsupported compression type %u (delta patches are applied by XenonRecomp)\n", compression);
        return false;
    }
}

bool looksLikePe(const std::vector<uint8_t>& image) {
    return image.size() >= 2 && image[0] == 'M' && image[1] == 'Z';
}

int decryptCommand(const char* in, const char* out) {
    Xex xex;
    if (!xex.parse(in)) return 1;
    std::vector<uint8_t> payload(xex.bytes.begin() + xex.headerSize, xex.bytes.end());
    std::vector<uint8_t> image;
    const char* how = "unencrypted";

    if (xex.format && xex.format->encryptionType == XEX_ENCRYPTION_NORMAL) {
        struct { const uint8_t* key; const char* name; } keys[] = {{Xex2RetailKey, "retail"}, {kDevkitKey, "devkit"}};
        bool ok = false;
        for (auto& k : keys) {
            std::vector<uint8_t> attempt = payload;
            decryptPayload(reinterpret_cast<const uint8_t*>(xex.security->aesKey), k.key, attempt);
            if (decompress(xex, attempt, image) && looksLikePe(image)) {
                how = k.name;
                ok = true;
                break;
            }
        }
        if (!ok) {
            std::fprintf(stderr, "%s: neither the retail nor the devkit key produced a valid image\n", in);
            return 1;
        }
    } else if (!decompress(xex, payload, image) || !looksLikePe(image)) {
        std::fprintf(stderr, "%s: decompression did not produce a PE image\n", in);
        return 1;
    }

    std::vector<uint8_t> output(xex.bytes.begin(), xex.bytes.begin() + xex.headerSize);
    if (xex.format) {
        auto* format = reinterpret_cast<Xex2OptFileFormatInfo*>(output.data() + (reinterpret_cast<uint8_t*>(xex.format) - xex.bytes.data()));
        format->encryptionType = XEX_ENCRYPTION_NONE;
        format->compressionType = XEX_COMPRESSION_NONE;
    }
    output.insert(output.end(), image.begin(), image.end());
    if (!writeFile(out, output)) {
        std::fprintf(stderr, "%s: cannot write\n", out);
        return 1;
    }
    std::printf("%s -> %s: %s, %u-byte image\n", in, out, how, xex.imageSize);
    return 0;
}

int encryptCommand(const char* in, const char* out) {
    Xex xex;
    if (!xex.parse(in)) return 1;
    if (!xex.format || xex.format->encryptionType != XEX_ENCRYPTION_NONE) {
        std::fprintf(stderr, "%s: input must be unencrypted with a file format header\n", in);
        return 1;
    }
    // Fixed session key, stored encrypted under the retail key as on a real disc.
    uint8_t sessionKey[kKeySize];
    for (uint32_t i = 0; i < kKeySize; i++) sessionKey[i] = uint8_t(0x5A ^ (i * 17));
    uint8_t encryptedKey[kKeySize];
    std::memcpy(encryptedKey, sessionKey, kKeySize);
    AES_ctx ctx;
    AES_init_ctx_iv(&ctx, Xex2RetailKey, AESBlankIV);
    AES_CBC_encrypt_buffer(&ctx, encryptedKey, kKeySize);

    std::vector<uint8_t> output = xex.bytes;
    output.resize(xex.headerSize + ((xex.payloadSize() + 15) & ~size_t(15)), 0);
    auto* header = reinterpret_cast<Xex2Header*>(output.data());
    auto* security = reinterpret_cast<Xex2SecurityInfo*>(output.data() + header->securityOffset);
    std::memcpy(security->aesKey, encryptedKey, kKeySize);
    auto* format = reinterpret_cast<Xex2OptFileFormatInfo*>(output.data() + (reinterpret_cast<uint8_t*>(xex.format) - xex.bytes.data()));
    format->encryptionType = XEX_ENCRYPTION_NORMAL;
    AES_init_ctx_iv(&ctx, sessionKey, AESBlankIV);
    AES_CBC_encrypt_buffer(&ctx, output.data() + xex.headerSize, output.size() - xex.headerSize);
    if (!writeFile(out, output)) {
        std::fprintf(stderr, "%s: cannot write\n", out);
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--encrypt") return encryptCommand(argv[2], argv[3]);
    if (argc == 3) return decryptCommand(argv[1], argv[2]);
    std::fprintf(stderr, "usage: xex-decrypt IN.xex OUT.xex\n       xex-decrypt --encrypt IN.xex OUT.xex\n");
    return 2;
}
