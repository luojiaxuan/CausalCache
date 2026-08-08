"""Small HTTP-range ZIP reader for fetching selected public benchmark members."""

from __future__ import annotations

import binascii
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    flag: int
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def parse_end_of_central_directory(payload: bytes, *, absolute_start: int) -> tuple[int, int, int]:
    signature = b"PK\x05\x06"
    offset = payload.rfind(signature)
    if offset < 0 or len(payload) - offset < 22:
        raise ValueError("ZIP end-of-central-directory record not found")
    values = struct.unpack_from("<4s4H2LH", payload, offset)
    _, disk, central_disk, disk_entries, total_entries, size, central_offset, comment = values
    if disk != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("multi-disk ZIP archives are unsupported")
    if offset + 22 + comment > len(payload):
        raise ValueError("truncated ZIP comment")
    if central_offset + size > absolute_start + offset:
        raise ValueError("invalid central-directory bounds")
    return central_offset, size, total_entries


def parse_central_directory(payload: bytes, *, expected_entries: int) -> dict[str, RemoteZipEntry]:
    entries: dict[str, RemoteZipEntry] = {}
    offset = 0
    for _ in range(expected_entries):
        if len(payload) - offset < 46:
            raise ValueError("truncated central-directory entry")
        values = struct.unpack_from("<4s6H3L5H2L", payload, offset)
        if values[0] != b"PK\x01\x02":
            raise ValueError("invalid central-directory signature")
        flag, compression, crc32 = values[3], values[4], values[7]
        compressed_size, uncompressed_size = values[8], values[9]
        name_size, extra_size, comment_size = values[10], values[11], values[12]
        local_offset = values[16]
        start = offset + 46
        end = start + name_size
        extra = payload[end : end + extra_size]
        if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
            zip64 = None
            extra_offset = 0
            while extra_offset + 4 <= len(extra):
                header_id, data_size = struct.unpack_from("<HH", extra, extra_offset)
                data_start = extra_offset + 4
                data_end = data_start + data_size
                if data_end > len(extra):
                    raise ValueError("truncated ZIP extra field")
                if header_id == 0x0001:
                    zip64 = extra[data_start:data_end]
                    break
                extra_offset = data_end
            if zip64 is None:
                raise ValueError("ZIP64 sentinel lacks a ZIP64 extra field")
            zip64_offset = 0
            resolved = []
            for value in (uncompressed_size, compressed_size, local_offset):
                if value == 0xFFFFFFFF:
                    if zip64_offset + 8 > len(zip64):
                        raise ValueError("truncated ZIP64 extra field")
                    value = struct.unpack_from("<Q", zip64, zip64_offset)[0]
                    zip64_offset += 8
                resolved.append(value)
            uncompressed_size, compressed_size, local_offset = resolved
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = payload[start:end].decode(encoding)
        entries[name] = RemoteZipEntry(
            name=name,
            flag=flag,
            compression=compression,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_offset,
        )
        offset = end + extra_size + comment_size
    if offset != len(payload):
        raise ValueError("central-directory payload has trailing or missing bytes")
    return entries


def decompress_entry(entry: RemoteZipEntry, payload: bytes) -> bytes:
    if len(payload) != entry.compressed_size:
        raise ValueError("compressed ZIP member size mismatch")
    if entry.compression == 0:
        result = payload
    elif entry.compression == 8:
        result = zlib.decompress(payload, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression method: {entry.compression}")
    if len(result) != entry.uncompressed_size:
        raise ValueError("uncompressed ZIP member size mismatch")
    if binascii.crc32(result) & 0xFFFFFFFF != entry.crc32:
        raise ValueError("ZIP member CRC32 mismatch")
    return result


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        value = response.headers.get("Content-Length")
    if value is None or int(value) <= 0:
        raise ValueError("remote ZIP lacks a positive Content-Length")
    return int(value)


def _http_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", None)
        payload = response.read()
    expected = end - start + 1
    if status != 206 or len(payload) != expected:
        raise ValueError(f"HTTP server did not honor byte range {start}-{end}")
    return payload


def remote_zip_catalog(url: str) -> dict[str, RemoteZipEntry]:
    size = _remote_size(url)
    tail_size = min(size, 128 * 1024)
    tail_start = size - tail_size
    tail = _http_range(url, tail_start, size - 1)
    central_offset, central_size, total_entries = parse_end_of_central_directory(
        tail,
        absolute_start=tail_start,
    )
    central = _http_range(url, central_offset, central_offset + central_size - 1)
    return parse_central_directory(central, expected_entries=total_entries)


def download_remote_zip_member(
    url: str,
    entry: RemoteZipEntry,
    destination: Path,
) -> None:
    if destination.exists() and destination.stat().st_size == entry.uncompressed_size:
        return
    local_header = _http_range(
        url,
        entry.local_header_offset,
        entry.local_header_offset + 29,
    )
    values = struct.unpack("<4s5H3L2H", local_header)
    if values[0] != b"PK\x03\x04" or values[3] != entry.compression:
        raise ValueError("invalid or inconsistent local ZIP header")
    data_start = entry.local_header_offset + 30 + values[9] + values[10]
    compressed = _http_range(
        url,
        data_start,
        data_start + entry.compressed_size - 1,
    )
    result = decompress_entry(entry, compressed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(result)
    partial.replace(destination)


__all__ = [
    "RemoteZipEntry",
    "decompress_entry",
    "download_remote_zip_member",
    "parse_central_directory",
    "parse_end_of_central_directory",
    "remote_zip_catalog",
]
