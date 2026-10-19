"""Built-in camera presets for the Metadata tab ("Camera data").

Each preset mimics the EXIF a real camera writes: Make / Model exactly as the camera spells them, the lens string,
a typical exposure, and the ranges used by the optional per-image variation. Values are typical, not measured from
any particular photo. Serial numbers, owner names, image ids and GPS are deliberately absent.

Strings marked "verify" are written from general knowledge of the camera's EXIF output and have not been checked
against a file from that camera; correct them if you have one. Firmware strings change with every update: the
ones here are plausible examples.

Preset keys: the ``core.metadata.CameraData`` fields, plus ``shutter_range`` / ``iso_range`` / ``f_range`` (the
``Variation`` ranges) and ``note`` (one line shown in the UI).
"""
from __future__ import annotations

from dataclasses import fields

from core.metadata import CameraData

# EXIF codes: exposure_program 1 manual / 2 program / 3 aperture priority; metering 2 center-weighted /
# 5 multi-segment; flash 16 off (compulsory) / 24 off (auto); white balance 0 auto.


def _p(note: str, shutter: str, iso_range: str, f_range: str, **kw) -> dict:
    d = {"exposure_bias": "0", "white_balance": "0", "metering_mode": "5", "flash": "16",
         "exposure_program": "3"}
    d.update(kw)
    d.update(shutter_range=shutter, iso_range=iso_range, f_range=f_range, note=note)
    return d


CAMERA_PRESETS: dict[str, dict] = {
    "Canon EOS R5 + RF24-70mm F2.8": _p(
        "Full-frame mirrorless, 45 MP. Standard zoom at 50 mm.", "1/2000-1/60", "100-3200", "2.8-8",
        make="Canon", model="Canon EOS R5", lens_make="Canon",  # verify: lens make
        lens_model="RF24-70mm F2.8 L IS USM", software="Firmware Version 1.8.2",  # verify
        f_number="2.8", exposure_time="1/250", iso="200", focal_length="50", focal_length_35mm="50"),
    "Canon EOS 5D Mark IV + EF 50mm f/1.8 STM": _p(
        "Full-frame DSLR, 30 MP. Fast 'nifty fifty'.", "1/4000-1/80", "100-3200", "1.8-5.6",
        make="Canon", model="Canon EOS 5D Mark IV", lens_model="EF50mm f/1.8 STM",
        # no Software tag: the camera does not write one (verify)
        f_number="2.2", exposure_time="1/200", iso="400", focal_length="50", focal_length_35mm="50"),
    "Sony a7 IV (ILCE-7M4) + FE 24-70mm F2.8 GM II": _p(
        "Full-frame mirrorless, 33 MP.", "1/2000-1/60", "100-6400", "2.8-8",
        make="SONY", model="ILCE-7M4", lens_make="Sony",  # verify: lens make
        lens_model="FE 24-70mm F2.8 GM II", software="ILCE-7M4 v1.10",  # verify
        f_number="2.8", exposure_time="1/250", iso="200", focal_length="35", focal_length_35mm="35"),
    "Sony a7R V (ILCE-7RM5) + FE 85mm F1.4 GM": _p(
        "High-resolution full-frame, 61 MP. Portrait prime.", "1/3200-1/100", "100-3200", "1.4-4",
        make="SONY", model="ILCE-7RM5", lens_make="Sony",  # verify: lens make
        lens_model="FE 85mm F1.4 GM", software="ILCE-7RM5 v2.00",  # verify
        f_number="1.8", exposure_time="1/500", iso="100", focal_length="85", focal_length_35mm="85"),
    "Nikon Z 8 + NIKKOR Z 24-120mm f/4 S": _p(
        "Full-frame mirrorless, 45.7 MP. Travel zoom.", "1/2000-1/60", "64-6400", "4-8",
        make="NIKON CORPORATION", model="NIKON Z 8", lens_make="NIKON",  # verify: lens make
        lens_model="NIKKOR Z 24-120mm f/4 S", software="Ver.02.00",  # verify
        f_number="5.6", exposure_time="1/320", iso="100", focal_length="70", focal_length_35mm="70"),
    "Nikon D850 + AF-S 50mm f/1.8G": _p(
        "Full-frame DSLR, 45.7 MP.", "1/4000-1/80", "64-3200", "1.8-5.6",
        make="NIKON CORPORATION", model="NIKON D850", lens_make="NIKON",  # verify: lens make
        lens_model="AF-S Nikkor 50mm f/1.8G", software="Ver.1.10",  # verify
        f_number="2.5", exposure_time="1/250", iso="200", focal_length="50", focal_length_35mm="50"),
    "Fujifilm X-T5 + XF33mm F1.4": _p(
        "APS-C mirrorless, 40 MP. 33 mm = 50 mm equivalent.", "1/2000-1/60", "125-3200", "1.4-5.6",
        make="FUJIFILM", model="X-T5", lens_make="FUJIFILM",  # verify: lens make
        lens_model="XF33mmF1.4 R LM WR", software="Digital Camera X-T5 Ver1.01",  # verify
        f_number="2", exposure_time="1/250", iso="160", focal_length="33", focal_length_35mm="50"),
    "Fujifilm X100VI (fixed 23mm)": _p(
        "APS-C compact, 40 MP, fixed 23 mm f/2 lens (35 mm equivalent).", "1/4000-1/60", "125-3200", "2-8",
        make="FUJIFILM", model="X100VI", lens_make="FUJIFILM",  # verify: lens make
        lens_model="23mm F2",  # verify: exact lens string
        software="Digital Camera X100VI Ver1.00",  # verify
        f_number="4", exposure_time="1/500", iso="160", focal_length="23", focal_length_35mm="35"),
    "Panasonic Lumix S5 II (DC-S5M2)": _p(
        "Full-frame mirrorless, 24 MP, with the 20-60 mm kit zoom.", "1/2000-1/60", "100-6400", "3.5-8",
        make="Panasonic", model="DC-S5M2", lens_make="Panasonic",  # verify: lens make
        lens_model="LUMIX S 20-60/F3.5-5.6",  # verify: exact lens string
        software="Ver.2.0",  # verify
        f_number="5.6", exposure_time="1/250", iso="200", focal_length="35", focal_length_35mm="35"),
    "OM System OM-1 + M.Zuiko 12-40mm F2.8 PRO": _p(
        "Micro Four Thirds, 20 MP. 12-40 mm = 24-80 mm equivalent.", "1/2000-1/80", "200-6400", "2.8-8",
        make="OM Digital Solutions", model="OM-1", lens_make="OM Digital Solutions",  # verify: lens make
        lens_model="OLYMPUS M.12-40mm F2.8 PRO",  # verify: exact lens string
        software="Version 1.6",  # verify
        f_number="4", exposure_time="1/320", iso="200", focal_length="25", focal_length_35mm="50"),
    "Leica Q3 (fixed 28mm)": _p(
        "Full-frame compact, 60 MP, fixed Summilux 28 mm f/1.7.", "1/4000-1/60", "100-6400", "1.7-8",
        make="LEICA CAMERA AG", model="LEICA Q3", lens_make="LEICA CAMERA AG",  # verify: lens make
        lens_model="LEICA SUMMILUX 28 f/1.7 ASPH.",  # verify: exact lens string
        software="6.1.0.0",  # verify
        f_number="2.8", exposure_time="1/500", iso="100", focal_length="28", focal_length_35mm="28"),
    "Ricoh GR IIIx (fixed 26.1mm)": _p(
        "APS-C pocket camera, 24 MP, fixed 26.1 mm f/2.8 (40 mm equivalent).", "1/2500-1/60", "100-3200", "2.8-8",
        make="RICOH IMAGING COMPANY, LTD.", model="GR IIIx",
        software="GR IIIx Ver 1.60",  # verify
        f_number="5.6", exposure_time="1/500", iso="200", focal_length="26.1", focal_length_35mm="40",
        exposure_program="2"),
    "Apple iPhone 15 Pro (main camera)": _p(
        "Phone main camera: fixed f/1.78, 24 mm equivalent.", "1/8000-1/60", "50-1600", "",
        make="Apple", model="iPhone 15 Pro", lens_make="Apple",
        lens_model="iPhone 15 Pro back triple camera 6.765mm f/1.78", software="17.2.1",
        f_number="1.78", exposure_time="1/120", iso="64", focal_length="6.765", focal_length_35mm="24",
        exposure_program="2", flash="24"),
    "Google Pixel 8 Pro (main camera)": _p(
        "Phone main camera: fixed f/1.68, 25 mm equivalent.", "1/8000-1/60", "50-1600", "",
        make="Google", model="Pixel 8 Pro", software="HDR+ 1.0",  # verify: Pixel software string
        f_number="1.68", exposure_time="1/100", iso="55", focal_length="6.9", focal_length_35mm="25",
        exposure_program="2", flash="24"),
    "Samsung Galaxy S24 Ultra (main camera)": _p(
        "Phone main camera: fixed f/1.7, 24 mm equivalent.", "1/8000-1/60", "50-1600", "",
        make="samsung", model="SM-S928B", software="S928BXXU1AXA1",  # verify: firmware string
        f_number="1.7", exposure_time="1/120", iso="64", focal_length="6.3", focal_length_35mm="24",
        exposure_program="2", flash="24"),
    "DJI Mavic 3 (drone, Hasselblad camera)": _p(
        "Drone 4/3 camera: 12.29 mm = 24 mm equivalent.", "1/8000-1/100", "100-800", "2.8-11",
        make="Hasselblad", model="L2D-20c",  # verify: DJI writes the camera module name
        software="v01.00.0300",  # verify
        f_number="5.6", exposure_time="1/800", iso="100", focal_length="12.29", focal_length_35mm="24",
        exposure_program="1"),
}


def preset_fields(preset: dict) -> dict:
    """Only the CameraData fields of a preset dict (drops note and ranges)."""
    names = {f.name for f in fields(CameraData)}
    return {k: str(v) for k, v in preset.items() if k in names}


def to_camera(preset: dict) -> CameraData:
    return CameraData(**preset_fields(preset))
