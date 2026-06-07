
import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import spiceypy as sp


'''
本段程序读取各观测地点已经计算出的 Delta T 允许区间，并将每个地点的区间画成竖向范围线。
图中同时计算并标出两个地点组合的公共交集：一组排除 Jiangzhou、Shucheng、Jiezhou，
另一组仅排除 Jiezhou。该图用于比较不同历史观测地点对 1542 年日食 Delta T 的约束强弱，
并直观看出哪些地点组合能够给出共同相容的 Delta T 范围。

请确保你已经运行了 exo-deltaT.py 和 exo-deltaT-magnitude.py 两个程序，并将它们的输出文件放在 WORKDIR 目录下。
每个输出文件应该包含一行类似于：  
对应 Delta T 区间: [lower, upper] s
其中 lower 和 upper 是该地点对应的 Delta T 约束区间的下限和上限（单位为秒）。
程序会自动从这些文件中提取 Delta T 区间，并在图上标出每个地点的区间范围。
同时，程序会计算两个地点组合的交集区间，并在图上以不同颜色的带状区域标出这些交集区间。
请根据实际情况修改 WORKDIR 和文件名列表，确保程序能够正确读取数据。
注意：如果某个文件中没有找到 Delta T 区间，程序会发出警告并跳过该文件。
最后，程序会将生成的图像保存到 WORKDIR 目录下，文件名为 deltaT_ranges_1542_sites_with_shucheng_intersections.png。
'''

# ========== 1. 设置文件夹路径 ==========
WORKDIR = Path(__file__).resolve().parent / "files"

# ========== 2. 设置要读取的文件 ==========
# 注意：这里加入了 Changzhi
# 请修改成你在运行 exo-deltaT.py 后实际生成的文件名
files = {
    "Xiaoyi": "xiaoyi_1542_google.txt",
    "Jiexiu": "jiexiu_1542_google.txt",
    "Licheng": "licheng_1542_google.txt",
    "Pingshun": "pingshun_1542_google.txt",
    "Jiangzhou": "xinjiang_jiangzhou_1542_google.txt",
    "Jiezhou": "jiezhou_1542_google.txt",
    "Qinzhou": "qinxian_qinzhou_1542_google.txt",
    "Changzhi": "changzhi_1542_google.txt",
    "Lucheng": "lucheng_1542_google.txt",
}

# ========== 3. 提取 Delta T 区间 ==========
pattern = re.compile(
    r"对应\s*Delta\s*T\s*区间:\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*s"
)

results = []

for site, filename in files.items():
    path = WORKDIR / filename

    if not path.exists():
        print(f"[WARNING] File not found: {path}")
        continue

    text = path.read_text(encoding="utf-8", errors="ignore")

    match = pattern.search(text)

    if match is None:
        print(f"[WARNING] No Delta T interval found in: {filename}")
        continue

    lower = float(match.group(1))
    upper = float(match.group(2))

    results.append({
        "site": site,
        "lower": lower,
        "upper": upper,
        "source": filename,
    })

# ========== 4. 手动加入 Hayakawa 的舒城结果 ==========
# Hayakawa et al. 2026: Shucheng, 1542 eclipse, -328 s <= Delta T <= 332 s
results.append({
    "site": "Shucheng",
    "lower": -328.0,
    "upper": 332.0,
    "source": "Hayakawa et al. 2026",
})


# ========== 5. 检查读取结果 ==========
if not results:
    raise RuntimeError("No valid Delta T intervals were found.")

print("Extracted Delta T intervals:")
for r in results:
    print(f"{r['site']:10s}: [{r['lower']:.2f}, {r['upper']:.2f}] s   ({r['source']})")


# ========== 6. 工具函数：计算一组地点的交集 ==========
def compute_intersection(selected_results):
    if not selected_results:
        return None

    lower = max(r["lower"] for r in selected_results)
    upper = min(r["upper"] for r in selected_results)

    if lower <= upper:
        return lower, upper
    return None


def print_intersection(name, selected_results):
    interval = compute_intersection(selected_results)
    site_names = [r["site"] for r in selected_results]

    print(f"\n{name}:")
    print("Sites:", ", ".join(site_names))

    if interval is None:
        print("No overlap.")
    else:
        print(f"[{interval[0]:.2f}, {interval[1]:.2f}] s")

    return interval


# ========== 7. 计算你要的两个交集 ==========

# 1. 除去 Jiangzhou, Shucheng, Jiezhou 后，剩下七个地点
exclude_main = {"Jiangzhou", "Shucheng", "Jiezhou"}
main_group = [r for r in results if r["site"] not in exclude_main]

main_interval = print_intersection(
    "Intersection 1: excluding Jiangzhou, Shucheng, and Jiezhou",
    main_group
)

# 2. 除去 Jiezhou 后，其余九个地点
exclude_jiezhou = {"Jiezhou"}
all_except_jiezhou = [r for r in results if r["site"] not in exclude_jiezhou]

all_except_jiezhou_interval = print_intersection(
    "Intersection 2: excluding Jiezhou only",
    all_except_jiezhou
)


# ========== 8. 画图 ==========
sites = [r["site"] for r in results]
x = range(len(results))

plt.figure(figsize=(13, 7))

# 每个地点画一条竖线
for i, r in enumerate(results):
    plt.vlines(
        x=i,
        ymin=r["lower"],
        ymax=r["upper"],
        linewidth=4,
    )
    plt.scatter(i, r["lower"], s=40)
    plt.scatter(i, r["upper"], s=40)

    # 在每条竖线旁标出数值范围
    plt.text(
        i + 0.06,
        0.5 * (r["lower"] + r["upper"]),
        f"[{r['lower']:.0f}, {r['upper']:.0f}]",
        fontsize=8,
        va="center",
    )


# ========== 9. 在图上画两个交集区间 ==========

legend_handles = []

if main_interval is not None:
    lo, hi = main_interval
    plt.axhspan(lo, hi, alpha=0.18)
    plt.axhline(lo, linestyle="--", linewidth=1)
    plt.axhline(hi, linestyle="--", linewidth=1)

    plt.text(
        len(results) - 0.3,
        0.5 * (lo + hi),
        f"Intersection 1\nexcluding Jiangzhou,\nShucheng, Jiezhou\n[{lo:.2f}, {hi:.2f}] s",
        va="center",
        ha="right",
        fontsize=9,
    )

    legend_handles.append(
        Patch(alpha=0.18, label=f"Intersection 1: [{lo:.2f}, {hi:.2f}] s")
    )
else:
    print("\n[WARNING] Intersection 1 has no overlap, so no band is plotted.")


if all_except_jiezhou_interval is not None:
    lo, hi = all_except_jiezhou_interval
    plt.axhspan(lo, hi, alpha=0.28)
    plt.axhline(lo, linestyle="-.", linewidth=1)
    plt.axhline(hi, linestyle="-.", linewidth=1)

    plt.text(
        0.2,
        0.5 * (lo + hi),
        f"Intersection 2\nexcluding Jiezhou only\n[{lo:.2f}, {hi:.2f}] s",
        va="center",
        ha="left",
        fontsize=9,
    )

    legend_handles.append(
        Patch(alpha=0.28, label=f"Intersection 2: [{lo:.2f}, {hi:.2f}] s")
    )
else:
    print("\n[WARNING] Intersection 2 has no overlap, so no band is plotted.")


# ========== 10. 图像格式 ==========
plt.xticks(list(x), sites, rotation=35, ha="right")
plt.ylabel("Delta T interval (s)")
plt.xlabel("Observation site")
plt.title("Delta T constraints from the 1542 total solar eclipse records")

plt.grid(axis="y", linestyle="--", alpha=0.4)

if legend_handles:
    plt.legend(handles=legend_handles, loc="best")

plt.tight_layout()


# ========== 11. 保存图片 ==========
output_path = WORKDIR / "deltaT_ranges_1542_sites_with_shucheng_intersections.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nFigure saved to: {output_path}")


'''
本段程序调用 exo-deltaT.py 中的日食判定函数，在指定经纬度网格上逐点测试给定 Delta T
是否会产生严格全食，以及是否达到设定的食分阈值。随后将全食区域、食分阈值边界和历史
观测地点一起绘制到地图上。该图用于展示不同 Delta T 取值下 1542 年日食路径在空间上的移动，
从而判断候选 Delta T 是否能让全食带覆盖或接近相关历史记录地点。
'''

# ============================================================
# 1. 基本路径设置
# ============================================================

WORKDIR = Path(__file__).resolve().parent

# 原程序：严格全食判据
EXO_TOTAL_SCRIPT = WORKDIR / "exo-deltaT.py"

# 复制版程序：magnitude >= MIN_MAGNITUDE 判据
EXO_MAG_SCRIPT = WORKDIR / "exo-deltaT.py"

KERNEL = Path(__file__).resolve().parent / "files" / "de441_part-1.bsp"

OUTPUT_DIR = WORKDIR / "totality_maps_1542"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. 载入原程序和 magnitude 版本程序
# ============================================================

def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exo_total = load_module("exo_deltaT_total", EXO_TOTAL_SCRIPT)
exo_mag = load_module("exo_deltaT_mag", EXO_MAG_SCRIPT)


# ============================================================
# 3. 载入 DE441 kernel
# ============================================================

sp.kclear()
sp.furnsh(KERNEL)


# ============================================================
# 4. 事件设置：1542 年 8 月 11 日
# ============================================================

YEAR = 1542
MONTH = 8
DAY = 11
CALENDAR = "julian"

# 你可以一次画多个 Delta T
DELTA_T_LIST = [-328,-70,299,252,332]   # 也可以改成 [0.0, 210.0, 252.0, 300.0, 332.0]

# 深偏食阈值
MAG_THRESHOLD = 1


# ============================================================
# 5. 地图范围设置
# ============================================================

LON_MIN, LON_MAX = 108.0, 118.5
LAT_MIN, LAT_MAX = 30.0, 39.0

# 网格分辨率，越小越精细但越慢
DLON = 0.10
DLAT = 0.10

lons = np.arange(LON_MIN, LON_MAX + DLON, DLON)
lats = np.arange(LAT_MIN, LAT_MAX + DLAT, DLAT)
LON_GRID, LAT_GRID = np.meshgrid(lons, lats)


# ============================================================
# 6. 需要标注的地点
# ============================================================

sites = {
    "Xiaoyi": (111 + 46 / 60 + 44.37 / 3600, 37 + 8 / 60 + 47.05 / 3600),
    "Jiexiu": (111 + 55 / 60 + 1.68 / 3600, 37 + 1 / 60 + 39.62 / 3600),
    "Licheng": (113 + 19 / 60 + 20.95 / 3600, 36 + 27 / 60 + 56.98 / 3600),
    "Pingshun": (113 + 26 / 60 + 25.67 / 3600, 36 + 11 / 60 + 53.27 / 3600),
    "Jiangzhou": (111 + 14 / 60 + 27.84 / 3600, 35 + 36 / 60 + 35.97 / 3600),
    "Jiezhou": (110 + 53 / 60 + 20.01 / 3600, 34 + 54 / 60 + 20.09 / 3600),
    "Qinzhou": (112 + 42 / 60 + 17.39 / 3600, 36 + 46 / 60 + 8.48 / 3600),
    "Changzhi": (113 + 7 / 60 + 57.30 / 3600, 36 + 11 / 60 + 16.51 / 3600),
    "Lucheng": (113 + 13 / 60 + 26.37 / 3600, 36 + 19 / 60 + 49.38 / 3600),
    "Shucheng": (116 + 57 / 60, 31 + 28 / 60),
}


# ============================================================
# 7. 找候选日食事件
# ============================================================

print("Finding candidate eclipse event...")

event = exo_total.find_candidate_solar_eclipse_near_date(
    YEAR,
    MONTH,
    DAY,
    calendar=CALENDAR,
)

print("Candidate eclipse:")
print(event)


# ============================================================
# 8. 构造地点对象
# ============================================================

def make_location(module, lat_deg, lon_deg, elev_m=0.0):
    try:
        return module.Location(lat_deg=lat_deg, lon_deg=lon_deg, elev_m=elev_m)
    except TypeError:
        return module.Location(lat_deg, lon_deg, elev_m)


# ============================================================
# 9. 判断严格全食
# ============================================================

def is_total_at_site(lon_deg, lat_deg, delta_t_s):
    loc = make_location(exo_total, lat_deg=lat_deg, lon_deg=lon_deg, elev_m=0.0)

    result = exo_total.is_total_at_some_time_near_event(
        delta_t_s,
        event.jd_tt,
        loc,
    )

    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


# ============================================================
# 10. 判断 magnitude >= threshold
# ============================================================

def is_magnitude_at_site(lon_deg, lat_deg, delta_t_s, threshold=0.95):
    loc = make_location(exo_mag, lat_deg=lat_deg, lon_deg=lon_deg, elev_m=0.0)

    # 设置 exo-deltaT_mag.py 里的全局阈值
    exo_mag.MIN_MAGNITUDE = threshold

    result = exo_mag.is_total_at_some_time_near_event(
        delta_t_s,
        event.jd_tt,
        loc,
    )

    if isinstance(result, tuple):
        return bool(result[0])
    return bool(result)


# ============================================================
# 11. 计算并画图
# ============================================================

for delta_t in DELTA_T_LIST:
    print(f"\nComputing grids for Delta T = {delta_t:.1f} s")

    totality = np.zeros_like(LON_GRID, dtype=int)
    mag095 = np.zeros_like(LON_GRID, dtype=int)

    n_lat, n_lon = LON_GRID.shape
    total_points = n_lat * n_lon
    count = 0

    for i in range(n_lat):
        for j in range(n_lon):
            lon = LON_GRID[i, j]
            lat = LAT_GRID[i, j]

            ok_total = is_total_at_site(lon, lat, delta_t)
            ok_mag095 = is_magnitude_at_site(
                lon,
                lat,
                delta_t,
                threshold=MAG_THRESHOLD,
            )

            totality[i, j] = 1 if ok_total else 0
            mag095[i, j] = 1 if ok_mag095 else 0

            count += 1
            if count % 200 == 0:
                print(f"  processed {count}/{total_points}")

    # 只保留深偏食条带，不重复覆盖全食区
    partial095_band = np.logical_and(mag095 == 1, totality == 0).astype(int)

    # ========================================================
    # 画图
    # ========================================================

    plt.figure(figsize=(10, 8))

    # 先画 magnitude >= 0.95 但不是全食的区域
    plt.contourf(
        LON_GRID,
        LAT_GRID,
        partial095_band,
        levels=[0.5, 1.5],
        alpha=0.25,
    )

    # 画 magnitude = 0.95 的边界
    plt.contour(
        LON_GRID,
        LAT_GRID,
        mag095,
        levels=[0.5],
        linewidths=1.2,
        linestyles="--",
    )

    # 再画严格全食区域
    plt.contourf(
        LON_GRID,
        LAT_GRID,
        totality,
        levels=[0.5, 1.5],
        alpha=0.45,
    )

    # 画严格全食边界
    plt.contour(
        LON_GRID,
        LAT_GRID,
        totality,
        levels=[0.5],
        linewidths=1.8,
    )

    # 标注地点
    for name, (lon, lat) in sites.items():
        plt.scatter(lon, lat, s=35)
        plt.text(lon + 0.05, lat + 0.05, name, fontsize=8)

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(
        f"Totality Path and Magnitude ≥ {MAG_THRESHOLD:.2f} Band\n"
        f"1542-08-11, Delta T = {delta_t:.1f} s"
    )

    plt.xlim(LON_MIN, LON_MAX)
    plt.ylim(LAT_MIN, LAT_MAX)
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_png = OUTPUT_DIR / (
        f"totality_and_mag{MAG_THRESHOLD:.2f}_1542_deltaT_{delta_t:.0f}s.png"
    )

    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"Saved figure: {out_png}")
