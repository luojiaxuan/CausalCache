import binascii
import io
import struct
import zipfile

from causalcache.remote_zip import (
    decompress_entry,
    parse_central_directory,
    parse_end_of_central_directory,
)


def test_parse_and_decompress_zip_member() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/val/ep_1.mp4", b"video-bytes" * 20)
    payload = stream.getvalue()
    central_offset, central_size, entries = parse_end_of_central_directory(
        payload,
        absolute_start=0,
    )
    catalog = parse_central_directory(
        payload[central_offset : central_offset + central_size],
        expected_entries=entries,
    )
    entry = catalog["videos/val/ep_1.mp4"]
    local = struct.unpack_from("<4s5H3L2H", payload, entry.local_header_offset)
    data_start = entry.local_header_offset + 30 + local[9] + local[10]
    compressed = payload[data_start : data_start + entry.compressed_size]
    result = decompress_entry(entry, compressed)
    assert result == b"video-bytes" * 20
    assert binascii.crc32(result) & 0xFFFFFFFF == entry.crc32


def test_parse_zip64_local_offset() -> None:
    name = b"videos/val/ep_1.mp4"
    actual_offset = 5_000_000_000
    extra = struct.pack("<HHQ", 0x0001, 8, actual_offset)
    fixed = struct.pack(
        "<4s6H3L5H2L",
        b"PK\x01\x02",
        45,
        45,
        0,
        8,
        0,
        0,
        0,
        10,
        20,
        len(name),
        len(extra),
        0,
        0,
        0,
        0,
        0xFFFFFFFF,
    )
    entry = parse_central_directory(fixed + name + extra, expected_entries=1)[name.decode()]
    assert entry.local_header_offset == actual_offset
