# -*- coding: utf-8 -*-
"""
根据“精确到日期”的历史日食记录反演 Delta T 区间

改进点:
1. 输入精确到日期: year + month + day
2. 不再全年扫描候选日食，而是在给定日期附近搜索一次局部合朔/候选日食
3. 太阳/月球位置改为视位置: SPICE spkpos(..., "CN+S", ...)
4. 地面站 ITRS/ECEF <-> 惯性系变换改为 IAU 2006/2000A:
   使用 pyerfa.erfa.c2t06a(TT, UT1, xp, yp)
5. 保留原有“是否在该地点出现日全食”的 Delta T 区间求解框架

依赖:
    pip install numpy spiceypy pyerfa

输入:
    - JPL DE441 SPICE 星历: de441.bsp
    - 历史日期 year/month/day
    - 地点经纬度、高程
    - Delta T 搜索区间 [dt_min, dt_max]，单位秒

    .venv/bin/python exo-deltaT.py 
    --kernel "files/de441_part-1.bsp" 
    --year 1542 
    --month 8 
    --day 11 
    --lat "36 11 16.51N" 
    --lon "113 07 57.30E" 
    --elev-m 0 
    --delta-t-min -1000 
    --delta-t-max 1200 
    --delta-t-step 5 
    --boundary-tol 0.5 
    --calendar julian

输出:
    - 该日期附近候选日食时刻（TT）
    - 满足“该地点会发生日全食”的历史 Delta T 区间
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import spiceypy as sp
import erfa
import re
from pathlib import Path


# ----------------------------
# 常数
# ----------------------------

J2000_JD_TT = 2451545.0
DAY_S = 86400.0

SUN_RADIUS_KM = 695700.0
MOON_RADIUS_KM = 1737.4
MIN_MAGNITUDE = 1.0

WGS84_A_KM = 6378.137  #椭球长半轴
WGS84_F = 1.0 / 298.257223563 #椭球扁率
WGS84_E2 = WGS84_F * (2.0 - WGS84_F) 


@dataclass
class Location:
    lat_deg: float
    lon_deg: float
    elev_m: float = 0.0


@dataclass # "匀速转"模型下预测的时间和日月差角
class EclipseCandidate:
    jd_tt: float
    sep_deg: float

@dataclass # 日全食起止时间
class TotalityTimeRange:
    delta_t_start_s: float
    delta_t_end_s: float
    ut1_start_jd: float
    ut1_end_jd: float
    tt_start_jd: float
    tt_end_jd: float
    best_margin_rad: float


# ----------------------------
# 历法与 JD
# ----------------------------

def infer_totality_time_range_for_interval(
    event_jd_tt: float,
    loc: Location,
    dt_start_s: float,
    dt_end_s: float,
    sample_step_s: float = 5.0,
    search_half_window_hours: float = 6.0,
) -> TotalityTimeRange:
    """
    对一个 Delta T 相容区间做采样，推断该地点食甚时刻的时间范围。

    返回:
        - 该区间对应的 UT1 时间范围
        - 该区间对应的 TT 时间范围
        - 采样过程中遇到的最大 totality margin
    """
    if dt_end_s < dt_start_s:
        raise ValueError("dt_end_s must be >= dt_start_s")
    if sample_step_s <= 0.0:
        raise ValueError("sample_step_s must be > 0")

    if dt_end_s - dt_start_s <= sample_step_s:
        dt_samples = np.array([dt_start_s, dt_end_s], dtype=float)
    else:
        dt_samples = np.arange(dt_start_s, dt_end_s + 0.5 * sample_step_s, sample_step_s, dtype=float)
        if dt_samples[-1] < dt_end_s:
            dt_samples = np.append(dt_samples, dt_end_s)
        dt_samples[0] = dt_start_s
        dt_samples[-1] = dt_end_s

    ut1_times = []
    tt_times = []
    best_margin = -1e99

    for dt_s in dt_samples:
        ok, margin, best_jd_ut1 = is_total_at_some_time_near_event(
            dt_s,
            event_jd_tt,
            loc,
            search_half_window_hours=search_half_window_hours,
        )
        if not ok:
            continue

        best_jd_tt = best_jd_ut1 + dt_s / DAY_S
        ut1_times.append(best_jd_ut1)
        tt_times.append(best_jd_tt)
        if margin > best_margin:
            best_margin = margin

    if not ut1_times:
        raise ValueError("该 Delta T 区间在采样时未找到任何全食时刻，请减小 sample_step_s")

    return TotalityTimeRange(
        delta_t_start_s=dt_start_s,
        delta_t_end_s=dt_end_s,
        ut1_start_jd=min(ut1_times),
        ut1_end_jd=max(ut1_times),
        tt_start_jd=min(tt_times),
        tt_end_jd=max(tt_times),
        best_margin_rad=best_margin,
    )

def infer_totality_time_ranges(
    event_jd_tt: float,
    loc: Location,
    intervals: List[Tuple[float, float]],
    sample_step_s: float = 5.0,
    search_half_window_hours: float = 6.0,
) -> List[TotalityTimeRange]:
    """
    对多个 Delta T 相容区间做采样，推断该地点食甚时刻的时间范围。
    """
    ranges: List[TotalityTimeRange] = []

    for dt_start_s, dt_end_s in intervals:
        tr = infer_totality_time_range_for_interval(
            event_jd_tt=event_jd_tt,
            loc=loc,
            dt_start_s=dt_start_s,
            dt_end_s=dt_end_s,
            sample_step_s=sample_step_s,
            search_half_window_hours=search_half_window_hours,
        )
        ranges.append(tr)

    return ranges

def calendar_to_jd(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
    calendar: str = "auto",
) -> float:
    """
    支持天文学年份编号:
        year = 0  -> 1 BCE
        year = -1 -> 2 BCE
    将公历/儒略历日期转换为 Julian Date (JD)，其中 day 可以是小数以包含时间部分。
    """
    y = year
    m = month
    d = day + (hour + (minute + second / 60.0) / 60.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    if calendar == "auto":
        calendar = "gregorian" if year >= 1583 else "julian"

    if calendar == "gregorian":
        a = math.floor(y / 100)
        b = 2 - a + math.floor(a / 4)
    elif calendar == "julian":
        b = 0
    else:
        raise ValueError("calendar must be one of: auto, julian, gregorian")

    jd = (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + d
        + b
        - 1524.5
    )
    return jd


def jd_to_calendar(jd: float, calendar: str = "auto") -> Tuple[int, int, int, int, int, float]:
    """
    将 Julian Date (JD) 转换为公历/儒略历日期，返回 (year, month, day, hour, minute, second)，其中 day 可以是小数以包含时间部分。
    """
    z = math.floor(jd + 0.5)
    f = (jd + 0.5) - z

    if calendar == "auto":
        calendar = "gregorian" if jd >= 2299160.5 else "julian"

    if calendar == "gregorian":
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    elif calendar == "julian":
        a = z
    else:
        raise ValueError("calendar must be one of: auto, julian, gregorian")

    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)

    day_float = b - d - math.floor(30.6001 * e) + f
    day = int(math.floor(day_float))
    frac_day = day_float - day

    if e < 14:
        month = e - 1
    else:
        month = e - 13

    if month > 2:
        year = c - 4716
    else:
        year = c - 4715

    hour_float = frac_day * 24.0
    hour = int(math.floor(hour_float))
    minute_float = (hour_float - hour) * 60.0
    minute = int(math.floor(minute_float))
    second = (minute_float - minute) * 60.0

    return year, month, day, hour, minute, second


def format_jd(jd: float, calendar: str = "auto") -> str:
    """
    书写固定格式的日期时间字符串，格式为 "YYYY-MM-DD HH:MM:SS.sss"，其中秒部分保留三位小数。
    """
    y, m, d, hh, mm, ss = jd_to_calendar(jd, calendar=calendar)
    return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:06.3f}"


def jd_to_two_part(jd: float) -> Tuple[float, float]:
    """
    gulpy 和 erfa 都要求将 Julian Date 分成两部分，以提高数值精度。
    """
    return 2400000.5, jd - 2400000.5


# ----------------------------
# 几何与坐标
# ----------------------------

def jd_tt_to_et_seconds(jd_tt: float) -> float:
    # 这里仍把 TDB - TT 的毫秒级差异忽略
    # 你可以查看Park_2021_AJ_161_105的公式（3）来看看这两个值之间如何进行修正。这能在个人电脑（PC）上完成吗？
    # TDB考虑了一些微小的周期项，主要是由于地球绕太阳的运动引起的。这些周期项的振幅通常在几毫秒范围内，因此对于大多数应用来说，直接使用 TT 作为 ET 的近似是足够的。
    return (jd_tt - J2000_JD_TT) * DAY_S


def norm(v: np.ndarray) -> float:
    # 这个函数计算一个向量的长度，并返回一个标量值。
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    # 单位向量
    n = norm(v)
    if n == 0.0:
        raise ValueError("zero vector")
    return v / n


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    # 计算两个向量之间的夹角（弧度）
    c = float(np.dot(unit(v1), unit(v2)))
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, elev_m: float) -> np.ndarray:
    """
    将地理坐标（纬度、经度、高程）转换为地心地固坐标（ECEF），单位为千米。
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    h = elev_m / 1000.0

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + h) * cos_lat * math.cos(lon)
    y = (n + h) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_enu(vec_ecef: np.ndarray, lat_deg: float, lon_deg: float) -> np.ndarray:
    """
    将地心地固坐标（ECEF）转换为东-北-天（ENU）坐标系。
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    east = -sin_lon * vec_ecef[0] + cos_lon * vec_ecef[1]
    north = (
        -sin_lat * cos_lon * vec_ecef[0]
        - sin_lat * sin_lon * vec_ecef[1]
        + cos_lat * vec_ecef[2]
    )
    up = (
        cos_lat * cos_lon * vec_ecef[0]
        + cos_lat * sin_lon * vec_ecef[1]
        + sin_lat * vec_ecef[2]
    )
    return np.array([east, north, up], dtype=float)


def itrs_to_gcrs_matrix(jd_ut1: float, jd_tt: float, xp_arcsec: float = 0.0, yp_arcsec: float = 0.0) -> np.ndarray:
    """
    返回 ITRS -> GCRS 旋转矩阵。
    erfa.c2t06a 返回的是 GCRS -> ITRS，因此这里取其转置。
    """
    uta, utb = jd_to_two_part(jd_ut1)
    tta, ttb = jd_to_two_part(jd_tt)

    xp = math.radians(xp_arcsec / 3600.0)
    yp = math.radians(yp_arcsec / 3600.0)

    rc2t = np.array(erfa.c2t06a(tta, ttb, uta, utb, xp, yp), dtype=float)
    return rc2t.T


def gcrs_to_itrs_matrix(jd_ut1: float, jd_tt: float, xp_arcsec: float = 0.0, yp_arcsec: float = 0.0) -> np.ndarray:
    """
    返回 GCRS -> ITRS 旋转矩阵。
    erfa.c2t06a 返回的就是 GCRS -> ITRS，因此直接使用即可。
    """
    uta, utb = jd_to_two_part(jd_ut1)
    tta, ttb = jd_to_two_part(jd_tt)

    xp = math.radians(xp_arcsec / 3600.0)
    yp = math.radians(yp_arcsec / 3600.0)

    return np.array(erfa.c2t06a(tta, ttb, uta, utb, xp, yp), dtype=float)


def ecef_to_gcrs(vec_ecef: np.ndarray, jd_ut1: float, delta_t_s: float) -> np.ndarray:
    """
    这个函数将地心地固坐标（ECEF）转换为惯性坐标系（GCRS），需要提供 UT1 时间和 Delta T 来计算 TT 时间。
    """
    jd_tt = jd_ut1 + delta_t_s / DAY_S
    rot = itrs_to_gcrs_matrix(jd_ut1, jd_tt)
    return rot @ vec_ecef


def gcrs_to_ecef(vec_gcrs: np.ndarray, jd_ut1: float, delta_t_s: float) -> np.ndarray:
    """
    这个函数将惯性坐标系（GCRS）转换为地心地固坐标（ECEF），需要提供 UT1 时间和 Delta T 来计算 TT 时间。
    """
    jd_tt = jd_ut1 + delta_t_s / DAY_S
    rot = gcrs_to_itrs_matrix(jd_ut1, jd_tt)
    return rot @ vec_gcrs


def altitude_rad_from_topo_gcrs(topo_gcrs: np.ndarray, jd_ut1: float, delta_t_s: float, loc: Location) -> float:
    """
    计算地平高度角（弧度），输入是地面站到天体的向量在 GCRS 中的表示，以及 UT1 时间和 Delta T 来计算 TT 时间。
    """
    topo_ecef = gcrs_to_ecef(topo_gcrs, jd_ut1, delta_t_s)
    enu = ecef_to_enu(topo_ecef, loc.lat_deg, loc.lon_deg)
    horiz = math.hypot(enu[0], enu[1])
    return math.atan2(enu[2], horiz)


# ----------------------------
# SPICE 位置计算
# ----------------------------
# SPICE's turn！
def body_app_pos_km(body: str, jd_tt: float) -> np.ndarray:
    """
    视位置:
    CN+S = converged Newtonian light-time + stellar aberration
    """
    et = jd_tt_to_et_seconds(jd_tt)
    pos, _ = sp.spkpos(body, et, "J2000", "CN+S", "EARTH")
    return np.array(pos, dtype=float)


def geocentric_sun_moon_sep_rad(jd_tt: float) -> float:
    """
    计算地心位置下的日月角距离（弧度），输入是 TT 时间。
    """
    sun = body_app_pos_km("SUN", jd_tt)
    moon = body_app_pos_km("MOON", jd_tt)
    return angle_between(sun, moon)


# ----------------------------
# 全食判据
# ----------------------------

def totality_margin_rad(jd_ut1: float, delta_t_s: float, loc: Location) -> float:
    """
    利用日月角距离和太阳高度角，判断此刻是否可能发生全食
    """
    jd_tt = jd_ut1 + delta_t_s / DAY_S

    sun_geo = body_app_pos_km("SUN", jd_tt)
    moon_geo = body_app_pos_km("MOON", jd_tt)

    site_ecef = geodetic_to_ecef_km(loc.lat_deg, loc.lon_deg, loc.elev_m)
    site_gcrs = ecef_to_gcrs(site_ecef, jd_ut1, delta_t_s)

    sun_topo = sun_geo - site_gcrs
    moon_topo = moon_geo - site_gcrs

    sun_alt = altitude_rad_from_topo_gcrs(sun_topo, jd_ut1, delta_t_s, loc)
    if sun_alt <= 0.0:
        return -1e9

    rho = angle_between(sun_topo, moon_topo)
    alpha_s = math.asin(min(1.0, SUN_RADIUS_KM / norm(sun_topo)))
    alpha_m = math.asin(min(1.0, MOON_RADIUS_KM / norm(moon_topo)))

    magnitude = (alpha_s + alpha_m - rho) / (2.0 * alpha_s)

    return magnitude - MIN_MAGNITUDE


def is_total_at_some_time_near_event(
    delta_t_s: float,
    event_jd_tt: float,
    loc: Location,
    search_half_window_hours: float = 6.0,
) -> Tuple[bool, float, float]:
    """
    返回:
        (是否全食, 最大 margin[rad], 最优 jd_ut1)
    """
    center_ut1 = event_jd_tt - delta_t_s / DAY_S
    half = search_half_window_hours / 24.0
    a = center_ut1 - half
    b = center_ut1 + half

    f = lambda jd: totality_margin_rad(jd, delta_t_s, loc)
    best_jd, best_margin = golden_maximize(f, a, b, tol_days=0.2 / DAY_S)
    return best_margin > 0.0, best_margin, best_jd


# ----------------------------
# 数值搜索
# ----------------------------

def golden_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol_days: float = 1e-6,
) -> Tuple[float, float]:
    """
    寻找何时间 f(t) 取最小值，t 的单位是天，tol_days 是容许的误差范围。
    这样，就找到了日全食发生的“最佳”时间点（虽然可能不是唯一的），以及当时的日月差角。
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while abs(b - a) > tol_days:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = f(x2)

    x = 0.5 * (a + b)
    return x, f(x)


def golden_maximize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol_days: float = 1e-6,
) -> Tuple[float, float]:
    """
    寻找何时 全食裕量取最大值，t 的单位是天，tol_days 是容许的误差范围。
    这样，就找到了该地点日全食发生的“最佳”时间点（虽然可能不是唯一的），以及当时的 totality margin。
    """
    x, y = golden_minimize(lambda t: -f(t), a, b, tol_days=tol_days)
    return x, -y


def find_candidate_solar_eclipse_near_date(
    year: int,
    month: int,
    day: int,
    calendar: str = "auto",
    scan_step_hours: float = 1.0,
    search_pad_days: float = 2.0,
) -> EclipseCandidate:
    """
    已知历史日期，直接在该日期附近搜索一次 geocentric Sun-Moon separation 的极小值。
    """
    jd0 = calendar_to_jd(year, month, day, 0, 0, 0, calendar=calendar) - search_pad_days
    jd1 = calendar_to_jd(year, month, day, 0, 0, 0, calendar=calendar) + 1.0 + search_pad_days

    step = scan_step_hours / 24.0
    grid = np.arange(jd0, jd1 + step, step, dtype=float)
    seps = np.array([geocentric_sun_moon_sep_rad(jd) for jd in grid], dtype=float)

    i = int(np.argmin(seps))
    left = grid[max(0, i - 1)]
    right = grid[min(len(grid) - 1, i + 1)]

    if right <= left:
        left = max(jd0, grid[i] - step)
        right = min(jd1, grid[i] + step)

    jd_min, sep_min_rad = golden_minimize(
        geocentric_sun_moon_sep_rad,
        left,
        right,
        tol_days=0.2 / DAY_S,
    )
    return EclipseCandidate(jd_tt=jd_min, sep_deg=math.degrees(sep_min_rad))


def compress_boolean_runs(xs: np.ndarray, flags: np.ndarray) -> List[Tuple[float, float, int, int]]:
    """
    将布尔数组中的连续 True 区间压缩为 (start_value, end_value, start_index, end_index) 的形式。
    例如，输入 xs=[0, 1, 2, 3, 4] 和 flags=[False, True, True, False, True] 会返回 [(1.0, 2.0, 1, 2), (4.0, 4.0, 4, 4)]。
    """
    runs = []
    n = len(xs)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        runs.append((xs[i], xs[j], i, j))
        i = j + 1
    return runs


def refine_boundary(
    predicate: Callable[[float], bool],
    false_dt: float,
    true_dt: float,
    tol_s: float = 1.0,
    max_iter: int = 80,
) -> float:
    """
    在一个 Delta T 区间上进行二分搜索，找到满足 predicate 的边界点。
    """
    lo = false_dt
    hi = true_dt
    for _ in range(max_iter):
        if hi - lo <= tol_s:
            break
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def refine_boundary_right(
    predicate: Callable[[float], bool],
    true_dt: float,
    false_dt: float,
    tol_s: float = 1.0,
    max_iter: int = 80,
) -> float:
    """
    "提纯"边界值，使得返回的 true_dt 更接近于满足 predicate 的边界点。
    """
    lo = true_dt
    hi = false_dt
    for _ in range(max_iter):
        if hi - lo <= tol_s:
            break
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            lo = mid
        else:
            hi = mid
    return lo


def solve_delta_t_intervals_for_date(
    year: int,
    month: int,
    day: int,
    loc: Location,
    dt_min_s: float,
    dt_max_s: float,
    dt_step_s: float,
    boundary_tol_s: float,
    calendar: str = "auto",
    scan_step_hours: float = 1.0,
    search_pad_days: float = 2.0,
    sep_threshold_deg: float = 1.8,
    event_search_half_window_hours: float = 6.0,
) -> Tuple[EclipseCandidate, List[Tuple[float, float]]]:
    """
    将上述步骤整合在一起，给定一个历史日期和地点，以及 Delta T 搜索区间，返回该日期附近的候选日食事件和满足“该地点会发生日全食”的 Delta T 区间列表。
    """
    event = find_candidate_solar_eclipse_near_date(
        year=year,
        month=month,
        day=day,
        calendar=calendar,
        scan_step_hours=scan_step_hours,
        search_pad_days=search_pad_days,
    )

    if event.sep_deg >= sep_threshold_deg:
        return event, []

    def compatible(dt_s: float) -> bool:
        ok, _, _ = is_total_at_some_time_near_event(
            dt_s,
            event.jd_tt,
            loc,
            search_half_window_hours=event_search_half_window_hours,
        )
        return ok

    dts = np.arange(dt_min_s, dt_max_s + 0.5 * dt_step_s, dt_step_s, dtype=float)
    flags = np.array([compatible(dt) for dt in dts], dtype=bool)
    runs = compress_boolean_runs(dts, flags)

    intervals: List[Tuple[float, float]] = []

    for dt_l, dt_r, i, j in runs:
        left = dt_l
        right = dt_r

        if i > 0:
            left = refine_boundary(
                compatible,
                false_dt=dts[i - 1],
                true_dt=dts[i],
                tol_s=boundary_tol_s,
            )

        if j < len(dts) - 1:
            right = refine_boundary_right(
                compatible,
                true_dt=dts[j],
                false_dt=dts[j + 1],
                tol_s=boundary_tol_s,
            )

        intervals.append((left, right))

    return event, intervals

def build_report_text(
    event: EclipseCandidate,
    intervals: List[Tuple[float, float]],
    time_ranges: List[TotalityTimeRange],
    calendar: str,
    delta_t_now: float = None,
) -> str:
    """
    结果写入txt文件
    """
    lines: List[str] = []

    lines.append("历史日全食 Delta T 反演结果")
    lines.append("")
    lines.append("候选日食事件:")
    lines.append(f"  TT = {format_jd(event.jd_tt, calendar=calendar)}")
    lines.append(f"  sep_min = {event.sep_deg:.6f} deg")
    lines.append("")

    lines.append("满足“该地点会发生日全食”的 Delta T 区间:")
    if not intervals:
        lines.append("  未找到相容区间。")
        return "\n".join(lines)

    for idx, (a, b) in enumerate(intervals, 1):
        lines.append(f"  {idx:02d}. [{a:.2f}, {b:.2f}] s")

    if delta_t_now is not None:
        lines.append("")
        lines.append("相对于当前 Delta T 的差值区间:")
        for idx, (a, b) in enumerate(intervals, 1):
            lines.append(f"  {idx:02d}. [{a - delta_t_now:.2f}, {b - delta_t_now:.2f}] s")

    lines.append("")
    lines.append("由 Delta T 区间反推出的该地点食甚时间范围:")

    for idx, tr in enumerate(time_ranges, 1):
        lines.append(f"  {idx:02d}. 对应 Delta T 区间: [{tr.delta_t_start_s:.2f}, {tr.delta_t_end_s:.2f}] s")
        lines.append(f"      UT1 时间范围: {format_jd(tr.ut1_start_jd, calendar=calendar)}  ~  {format_jd(tr.ut1_end_jd, calendar=calendar)}")
        lines.append(f"      TT  时间范围: {format_jd(tr.tt_start_jd, calendar=calendar)}  ~  {format_jd(tr.tt_end_jd, calendar=calendar)}")
        lines.append(f"      最大 totality margin = {tr.best_margin_rad:.10f} rad")

    return "\n".join(lines)

def write_report_file(report_text: str, output_path: str = None) -> Path:
    """
    同上
    """
    if output_path is None:
        path = Path(__file__).with_name("deltat.txt")
    else:
        path = Path(output_path)

    path.write_text(report_text, encoding="utf-8")
    return path


# ----------------------------
# 主程序
# ----------------------------
import re

DMS_PATTERN = re.compile(
    r'^\s*'
    r'([+-]?\d+(?:\.\d*)?)'          # 度
    r'(?:[°:\s]+(\d+(?:\.\d*)?))?'   # 分，可选
    r'(?:[\'′:\s]+(\d+(?:\.\d*)?))?' # 秒，可选
    r'\s*(?:["″]?)\s*'
    r'([NSEWnsew]?)'                 # 方向字母，可选
    r'\s*$'
)
# 解析输入的“度-分-秒”格式字符串，用以坐标转换。

def parse_dms_angle(text: str, kind: str) -> float:
    """
    解析度分秒字符串，返回十进制度。

    支持示例:
        34 12 30N
        34:12:30N
        34°12'30"N
        108 56 12E
        -34 12 30
    """
    m = DMS_PATTERN.match(text)
    if not m:
        raise ValueError(f"无法解析 {kind}: {text}")

    deg_text, min_text, sec_text, hemi = m.groups()
    deg = float(deg_text)
    minute = float(min_text) if min_text is not None else 0.0
    second = float(sec_text) if sec_text is not None else 0.0
    hemi = hemi.upper()

    if minute < 0.0 or minute >= 60.0:
        raise ValueError(f"{kind} 的分必须在 [0, 60) 内: {text}")
    if second < 0.0 or second >= 60.0:
        raise ValueError(f"{kind} 的秒必须在 [0, 60) 内: {text}")

    if hemi and deg < 0.0:
        raise ValueError(f"{kind} 不能同时使用负号和方向字母: {text}")

    abs_value = abs(deg) + minute / 60.0 + second / 3600.0

    if hemi:
        if hemi in ("S", "W"):
            value = -abs_value
        else:
            value = abs_value
    else:
        value = -abs_value if deg < 0.0 else abs_value

    if kind == "lat":
        if hemi and hemi not in ("N", "S"):
            raise ValueError(f"纬度只能使用 N 或 S: {text}")
        if not (-90.0 <= value <= 90.0):
            raise ValueError(f"纬度超出范围 [-90, 90]: {text}")
    elif kind == "lon":
        if hemi and hemi not in ("E", "W"):
            raise ValueError(f"经度只能使用 E 或 W: {text}")
        if not (-180.0 <= value <= 180.0):
            raise ValueError(f"经度超出范围 [-180, 180]: {text}")
    else:
        raise ValueError(f"未知角度类型: {kind}")

    return value



def main() -> None:
    """
    主函数。读取输入。
    """
    parser = argparse.ArgumentParser(description="根据历史日全食日期反演 Delta T 区间")
    parser.add_argument("--kernel", required=True, help="DE441 SPICE 文件路径，例如 files/de441_part-1.bsp")

    parser.add_argument("--year", required=True, type=int, help="历史年份，支持天文学年份编号")
    parser.add_argument("--month", required=True, type=int, help="月份 1-12")
    parser.add_argument("--day", required=True, type=int, help="日期 1-31")

    parser.add_argument(
        "--min-magnitude",
        type=float,
        default=1.0,
        help="Minimum eclipse magnitude required. Use 1.0 for totality, 0.95 for deep partial eclipse."
    )

    parser.add_argument(
        "--lat",
        required=True,
        type=str,
        help='纬度，度分秒，例如 "34 12 30N"',
    )
    parser.add_argument(
        "--lon",
        required=True,
        type=str,
        help='经度，度分秒，例如 "108 56 12E"',
    )
    parser.add_argument("--elev-m", default=0.0, type=float, help="高程，单位米")

    parser.add_argument("--delta-t-min", required=True, type=float, help="搜索下界，单位秒")
    parser.add_argument("--delta-t-max", required=True, type=float, help="搜索上界，单位秒")
    parser.add_argument("--delta-t-step", default=10.0, type=float, help="粗扫描步长，单位秒")
    parser.add_argument("--boundary-tol", default=1.0, type=float, help="边界细化精度，单位秒")

    parser.add_argument(
        "--calendar",
        choices=["auto", "julian", "gregorian"],
        default="auto",
        help="输入日期采用的历法",
    )
    parser.add_argument(
        "--scan-step-hours",
        default=1.0,
        type=float,
        help="日期附近寻找候选新月时的粗步长，单位小时",
    )
    parser.add_argument(
        "--search-pad-days",
        default=2.0,
        type=float,
        help="在目标日期前后额外搜索的缓冲天数",
    )
    parser.add_argument(
        "--sep-threshold-deg",
        default=1.8,
        type=float,
        help="地心日月最小角距小于该值才视作候选日食，单位度",
    )
    parser.add_argument(
        "--event-search-half-window-hours",
        default=6.0,
        type=float,
        help="围绕候选事件搜索该地点全食发生时刻的半窗，单位小时",
    )
    parser.add_argument(
        "--delta-t-now",
        default=None,
        type=float,
        help="可选。若提供当前 Delta T，则同时输出相对当前的差值区间",
    )
    parser.add_argument(
        "--output",
        default="deltat.txt",
        help="输出结果文件名，默认 deltat.txt",
    )
    parser.add_argument(
        "--time-range-step",
        default=5.0,
        type=float,
        help="由 Delta T 区间反推食甚时间范围时的采样步长，单位秒",
    )

    args = parser.parse_args()
    global MIN_MAGNITUDE
    MIN_MAGNITUDE = args.min_magnitude

    if not (1 <= args.month <= 12):
        raise ValueError("month must be in 1..12")
    if not (1 <= args.day <= 31):
        raise ValueError("day must be in 1..31")
    if args.delta_t_max < args.delta_t_min:
        raise ValueError("delta-t-max must be >= delta-t-min")
    if args.delta_t_step <= 0.0:
        raise ValueError("delta-t-step must be > 0")
    if args.boundary_tol <= 0.0:
        raise ValueError("boundary-tol must be > 0")

    lat_deg = parse_dms_angle(args.lat, kind="lat")
    lon_deg = parse_dms_angle(args.lon, kind="lon")
    loc = Location(lat_deg=lat_deg, lon_deg=lon_deg, elev_m=args.elev_m)

    sp.kclear()
    sp.furnsh(args.kernel)

    try:
        event, intervals = solve_delta_t_intervals_for_date(
            year=args.year,
            month=args.month,
            day=args.day,
            loc=loc,
            dt_min_s=args.delta_t_min,
            dt_max_s=args.delta_t_max,
            dt_step_s=args.delta_t_step,
            boundary_tol_s=args.boundary_tol,
            calendar=args.calendar,
            scan_step_hours=args.scan_step_hours,
            search_pad_days=args.search_pad_days,
            sep_threshold_deg=args.sep_threshold_deg,
            event_search_half_window_hours=args.event_search_half_window_hours,
        )

        print("开始计算，请耐心等候...")
        print()
        print("给定日期附近的候选日食事件:")
        print(f"  TT = {format_jd(event.jd_tt, calendar=args.calendar)}")
        print(f"  sep_min = {event.sep_deg:.6f} deg")

        if event.sep_deg >= args.sep_threshold_deg:
            print()
            print("该日期附近未找到足够接近的新月候选事件。")
            print("可尝试:")
            print("  1. 增大 --search-pad-days")
            print("  2. 放宽 --sep-threshold-deg")
            print("  3. 检查日期或历法是否正确")
            return

        print()
        print("满足“该地点会发生日全食”的历史 Delta T 区间:")
        if not intervals:
            print("  在给定搜索范围内未找到相容的 Delta T。")
            print("  可能原因:")
            print("    1. 该日期该地点并没有发生日全食")
            print("    2. Delta T 搜索范围太窄")
            print("    3. 需要更小的 delta-t-step")
            return

        for idx, (a, b) in enumerate(intervals, 1):
            print(f"  {idx:02d}. [{a:.2f}, {b:.2f}] s")

        time_ranges = infer_totality_time_ranges(
            event_jd_tt=event.jd_tt,
            loc=loc,
            intervals=intervals,
            sample_step_s=args.time_range_step,
            search_half_window_hours=args.event_search_half_window_hours,
        )

        print()
        print("由 Delta T 区间反推出的该地点食甚时间范围:")
        for idx, tr in enumerate(time_ranges, 1):
            print(f"  {idx:02d}. UT1: {format_jd(tr.ut1_start_jd, calendar=args.calendar)}  ~  {format_jd(tr.ut1_end_jd, calendar=args.calendar)}")
            print(f"      TT : {format_jd(tr.tt_start_jd, calendar=args.calendar)}  ~  {format_jd(tr.tt_end_jd, calendar=args.calendar)}")

        report_text = build_report_text(
            event=event,
            intervals=intervals,
            time_ranges=time_ranges,
            calendar=args.calendar,
            delta_t_now=args.delta_t_now,
        )
        report_path = write_report_file(report_text, args.output)

        print()
        print(f"结果已写入文件: {report_path}")

        if args.delta_t_now is not None:
            print()
            print("相对于当前 Delta T 的差值区间:")
            for idx, (a, b) in enumerate(intervals, 1):
                da = a - args.delta_t_now
                db = b - args.delta_t_now
                print(f"  {idx:02d}. [{da:.2f}, {db:.2f}] s")

    finally:
        sp.kclear()


if __name__ == "__main__":
    main()
